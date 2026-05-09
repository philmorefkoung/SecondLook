"""LoRA SFT of Qwen2.5-VL-7B-Instruct on the EvidenceCard verdict task.

Input format: data/sft_combined/{train,val}.jsonl in Anthropic-style messages
format (with base64 PNG image content blocks). Each record's assistant message
is the structured-JSON target verdict.

Pipeline:
  - Load Qwen2.5-VL-7B-Instruct + processor
  - Wrap with PEFT LoRA (rank=16) on the language attention projections
  - Per-example: convert Anthropic messages -> Qwen messages with PIL.Image
  - Tokenize via processor.apply_chat_template(..., tokenize=True)
  - Mask labels: loss only on the assistant tokens
  - Train with HF Trainer (AdamW, cosine LR, gradient checkpointing)
  - Save adapter to ckpts/sft_qwen_vl_v1/

Smoke run (verify pipeline on 50 examples, 1 epoch):
  python scripts/sft_qwen_vl.py \
      --train data/sft_combined/train.jsonl --val data/sft_combined/val.jsonl \
      --out ckpts/sft_qwen_vl_smoke --epochs 1 --train-subset 50 --val-subset 8
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def _decode_b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _anthropic_to_qwen_messages(anthropic_msgs: list[dict]) -> list[dict]:
    """Convert {"type":"image","source":{base64}} blocks to Qwen-style
    {"type":"image","image":<PIL>} blocks; pass text blocks through."""
    out = []
    for m in anthropic_msgs:
        new_content = []
        for block in m["content"]:
            t = block.get("type")
            if t == "image":
                src = block["source"]
                if src.get("type") != "base64":
                    raise ValueError(f"unsupported image source type: {src.get('type')}")
                new_content.append({"type": "image", "image": _decode_b64_to_pil(src["data"])})
            elif t == "text":
                new_content.append({"type": "text", "text": block["text"]})
            else:
                raise ValueError(f"unsupported content block type: {t}")
        out.append({"role": m["role"], "content": new_content})
    return out


@dataclass
class _Example:
    messages: list[dict]   # Qwen-style with PIL images
    target_text: str       # the assistant's response text


class VerdictSFTDataset(Dataset):
    """Streams examples from a messages-format JSONL file. Heavy lifting
    (image decode + processor encoding) is deferred to the collator so this
    class is cheap to len() and slice().
    """
    def __init__(self, jsonl_path: str | Path, limit: int | None = None):
        self.records: list[dict] = []
        with Path(jsonl_path).open() as f:
            for line in f:
                self.records.append(json.loads(line))
        if limit:
            self.records = self.records[:limit]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx) -> _Example:
        r = self.records[idx]
        msgs = r["messages"]
        # Split user (input) and assistant (target)
        user_msgs = [m for m in msgs if m["role"] != "assistant"]
        assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
        if not assistant_msgs:
            raise ValueError(f"record {idx} has no assistant response")
        target_text = "".join(b.get("text", "") for b in assistant_msgs[0]["content"])
        return _Example(
            messages=_anthropic_to_qwen_messages(user_msgs),
            target_text=target_text,
        )


def make_collator(processor):
    """Builds a DataCollator that:
       1. Renders the user-only conversation (with image) as the chat-template prefix
       2. Concatenates the target text (the assistant's structured-JSON response)
       3. Returns input_ids/attention_mask/pixel_values/image_grid_thw plus labels
          where label tokens for the user prefix are -100 (loss masked).
    """
    from qwen_vl_utils import process_vision_info

    def collate(batch: list[_Example]) -> dict:
        # Build the prompt-only text (user turn + assistant generation prompt)
        prompts, targets, image_inputs_all = [], [], []
        for ex in batch:
            prompt_text = processor.apply_chat_template(
                ex.messages, tokenize=False, add_generation_prompt=True,
            )
            prompts.append(prompt_text)
            targets.append(ex.target_text + processor.tokenizer.eos_token)
            imgs, _ = process_vision_info(ex.messages)
            image_inputs_all.append(imgs)

        # Tokenize the full sequence (prompt + target) and the prompt-only
        # parts separately so we know how many tokens to mask.
        full_texts = [p + t for p, t in zip(prompts, targets)]
        full = processor(
            text=full_texts,
            images=[img for imgs in image_inputs_all for img in (imgs or [])] or None,
            return_tensors="pt", padding=True,
        )
        # Tokenize prompts alone (no images, just text) to measure prompt length.
        # We use the same tokenizer to ensure tokenization is consistent.
        prompt_ids = processor.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False,
        )["input_ids"]

        labels = full["input_ids"].clone()
        # Mask user-prefix tokens; loss only on assistant target tokens
        for i in range(len(batch)):
            n_prompt = (prompt_ids[i] != processor.tokenizer.pad_token_id).sum().item()
            labels[i, :n_prompt] = -100
        # Also mask pad tokens in label
        labels[full["input_ids"] == processor.tokenizer.pad_token_id] = -100

        full["labels"] = labels
        return full

    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", default="ckpts/sft_qwen_vl_v1")
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--train-subset", type=int, default=None)
    ap.add_argument("--val-subset", type=int, default=None)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id} ...")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    # PEFT LoRA on the language-side attention projections
    from peft import LoraConfig, get_peft_model, TaskType
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = VerdictSFTDataset(args.train, limit=args.train_subset)
    val_ds = VerdictSFTDataset(args.val, limit=args.val_subset)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    collate = make_collator(processor)

    from transformers import TrainingArguments, Trainer
    train_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        eval_strategy="steps", eval_steps=args.eval_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],   # no wandb/tb by default
        dataloader_num_workers=0,   # Windows-safe
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model, args=train_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collate,
    )
    trainer.train()

    # Save final adapter
    model.save_pretrained(out_dir / "adapter_final")
    processor.save_pretrained(out_dir / "adapter_final")
    print(f"Saved adapter to {out_dir / 'adapter_final'}")


if __name__ == "__main__":
    main()
