"""DPO fine-tuning of Qwen2.5-VL-7B (continuing from SFT v2 LoRA).

Loads the SFT v2 adapter as the starting policy. Reference model = same
base + SFT v2 (frozen) so the KL term anchors at SFT v2, not at base
(which would undo the SFT learning).

Memory note: with two model copies (~14 GB each in bf16) + activations,
total is ~30-32 GB on a 32 GB GPU. Falls back to precompute_ref_log_probs
if OOM.

Usage (smoke):
  python scripts/dpo_qwen_vl.py \
      --train data/dpo_v1/train_pairs.jsonl --val data/dpo_v1/val_pairs.jsonl \
      --base-adapter ckpts/sft_qwen_vl_v2/adapter_final \
      --out ckpts/dpo_qwen_vl_v1 --epochs 1 --train-subset 8 --val-subset 4
"""
from __future__ import annotations
import argparse
import base64
import io
import json
from pathlib import Path

import torch
from datasets import Dataset
from PIL import Image


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def _decode_b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def load_pairs(jsonl_path: str | Path, limit: int | None = None) -> list[dict]:
    out = []
    for line in Path(jsonl_path).open():
        out.append(json.loads(line))
        if limit and len(out) >= limit:
            break
    return out


def to_dpo_records(pairs: list[dict]) -> list[dict]:
    """Convert our pair format to TRL DPO format with PIL images.
    Each record:
      prompt:   [{"role":"user","content":[{"type":"image"},{"type":"text","text":...}]}]
      chosen:   [{"role":"assistant","content":[{"type":"text","text":...}]}]
      rejected: [{"role":"assistant","content":[{"type":"text","text":...}]}]
      images:   [PIL]
    """
    records = []
    for p in pairs:
        img = _decode_b64_to_pil(p["image_b64"])
        records.append({
            "prompt": [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": p["prompt"]},
                ]},
            ],
            "chosen": [
                {"role": "assistant", "content": [{"type": "text", "text": p["chosen"]}]},
            ],
            "rejected": [
                {"role": "assistant", "content": [{"type": "text", "text": p["rejected"]}]},
            ],
            "images": [img],
        })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--base-adapter", required=True,
                    help="Path to SFT v2 LoRA adapter (used as both starting policy + ref).")
    ap.add_argument("--out", default="ckpts/dpo_qwen_vl_v1")
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max-length", type=int, default=6144)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--train-subset", type=int, default=None)
    ap.add_argument("--val-subset", type=int, default=None)
    ap.add_argument("--logging-steps", type=int, default=2)
    ap.add_argument("--eval-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=20)
    ap.add_argument("--no-eval", action="store_true",
                    help="Disable mid-training eval (use when val pairs cause OOM).")
    ap.add_argument("--precompute-ref", action="store_true",
                    help="Use precompute_ref_log_probs (saves memory; ref model loaded only once).")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from latest checkpoint in --out dir.")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading processor + model {args.model_id}")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    processor = AutoProcessor.from_pretrained(args.model_id)

    # Policy: base + SFT v2 LoRA (trainable)
    policy_base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    policy = PeftModel.from_pretrained(policy_base, args.base_adapter, is_trainable=True)
    policy.print_trainable_parameters()

    # Reference: base + SFT v2 LoRA (frozen). Same weights as policy at init.
    if args.precompute_ref:
        ref_model = None  # TRL will load ref temporarily, precompute, free.
    else:
        ref_base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_id, torch_dtype=torch.bfloat16, device_map="auto",
        )
        ref_model = PeftModel.from_pretrained(ref_base, args.base_adapter)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    # Build datasets
    train_pairs = load_pairs(args.train, args.train_subset)
    val_pairs = load_pairs(args.val, args.val_subset)
    train_records = to_dpo_records(train_pairs)
    val_records = to_dpo_records(val_pairs)
    print(f"Train DPO pairs: {len(train_records)}  Val DPO pairs: {len(val_records)}")

    train_ds = Dataset.from_list(train_records)
    val_ds = Dataset.from_list(val_records)

    from trl import DPOTrainer, DPOConfig
    dpo_args = DPOConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        eval_strategy="no" if args.no_eval else "steps", eval_steps=args.eval_steps,
        report_to=[],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        precompute_ref_log_probs=args.precompute_ref,
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor,
    )
    trainer.train(resume_from_checkpoint=args.resume)

    policy.save_pretrained(out_dir / "adapter_final")
    processor.save_pretrained(out_dir / "adapter_final")
    print(f"Saved DPO adapter to {out_dir / 'adapter_final'}")


if __name__ == "__main__":
    main()
