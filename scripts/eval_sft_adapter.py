"""Evaluate a Qwen-VL LoRA adapter on a verdict-task val.jsonl.

For each val record:
  - Build the user-only prompt (image + verifier text)
  - Generate a response from base model + adapter
  - Parse the JSON
  - Compare predicted decision (accept/reject/uncertain) against the GT
    verdict (which we recover from the record's meta.gt_label.verdict)

Reports:
  - JSON parse rate
  - Verdict accuracy (binary: accept vs reject; uncertain folded in optionally)
  - Confusion matrix
  - Per-philosophy breakdown

Usage:
  python scripts/eval_sft_adapter.py \
      --val data/sft_combined/val.jsonl \
      --adapter ckpts/sft_qwen_vl_v1/adapter_final \
      --out runs/eval_sft_adapter_v1.json
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


def _decode_b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _anthropic_to_qwen_messages(anthropic_msgs):
    out = []
    for m in anthropic_msgs:
        new_content = []
        for block in m["content"]:
            if block.get("type") == "image":
                new_content.append({"type": "image", "image": _decode_b64_to_pil(block["source"]["data"])})
            elif block.get("type") == "text":
                new_content.append({"type": "text", "text": block["text"]})
        out.append({"role": m["role"], "content": new_content})
    return out


def _parse_decision(text: str) -> str | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    d = str(obj.get("decision", "")).lower()
    if d.startswith("acc") or d.startswith("conf"):
        return "accept"
    if d.startswith("rej"):
        return "reject"
    if d.startswith("unc"):
        return "uncertain"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--adapter", default=None,
                    help="Path to LoRA adapter dir (saved via PEFT save_pretrained). "
                         "Omit to evaluate the base model zero-shot.")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out", default="runs/eval_sft_adapter.json")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print(f"Loading base {args.base_model}" + (f" + adapter {args.adapter}" if args.adapter else " (zero-shot baseline)"))
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(args.base_model)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, args.adapter)
    else:
        model = base
    model.eval()

    val_records = []
    with Path(args.val).open() as f:
        for line in f:
            val_records.append(json.loads(line))
    if args.limit:
        val_records = val_records[:args.limit]
    print(f"Val examples: {len(val_records)}")

    from qwen_vl_utils import process_vision_info
    rows = []
    n_parse_ok = 0
    confusion = defaultdict(int)            # (gt, pred) -> count
    by_philosophy = defaultdict(lambda: [0, 0])   # phil -> [correct, total]

    for r in tqdm(val_records, desc="eval"):
        meta = r["meta"]
        gt = meta["gt_label"]["verdict"]    # "confirm" or "reject"
        gt_decision = "accept" if gt == "confirm" else "reject"
        philosophy = meta.get("philosophy", "?")

        user_msgs = [m for m in r["messages"] if m["role"] != "assistant"]
        qmsgs = _anthropic_to_qwen_messages(user_msgs)
        text = processor.apply_chat_template(qmsgs, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(qmsgs)
        inputs = processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
        gen_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

        pred_decision = _parse_decision(gen_text)
        if pred_decision is not None:
            n_parse_ok += 1
        correct = (pred_decision == gt_decision)
        confusion[(gt_decision, pred_decision or "(parse_fail)")] += 1
        by_philosophy[philosophy][1] += 1
        if correct:
            by_philosophy[philosophy][0] += 1
        rows.append({
            "study_id": meta.get("study_id"),
            "coord_zyx": meta.get("coord_zyx"),
            "gt_decision": gt_decision,
            "pred_decision": pred_decision,
            "correct": correct,
            "philosophy": philosophy,
            "gen_text_head": gen_text[:200],
        })

    n = len(rows)
    summary = {
        "n_val": n,
        "json_parse_rate": n_parse_ok / max(1, n),
        "binary_verdict_accuracy": sum(r["correct"] for r in rows) / max(1, n),
        "confusion": {f"{gt}->{pred}": c for (gt, pred), c in sorted(confusion.items())},
        "by_philosophy": {
            phil: {"accuracy": cnts[0] / max(1, cnts[1]), "n": cnts[1]}
            for phil, cnts in by_philosophy.items()
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "examples": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
