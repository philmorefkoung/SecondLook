"""Mine SFT v2's mistakes on a held-out (or train) SFT data slice.

Loads base + SFT v2 adapter, runs verdict prediction on each example, and
saves a JSONL of (image, gt_target, sft_v2_prediction) where sft_v2 was
WRONG. These become DPO preference pairs:
  chosen = GT-aligned target
  rejected = SFT v2's wrong prediction

Usage:
  python scripts/mine_sft_v2_mistakes.py \
      --metadata data/sft_v3_train/metadata.jsonl \
      --in-dir data/sft_v3_train \
      --adapter ckpts/sft_qwen_vl_v2/adapter_final \
      --out data/dpo_v2/mined_pairs.jsonl \
      --limit 2000
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import re
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


def _verifier_user_prompt(coord, prob) -> str:
    return (
        "You are an auditable second-finding verifier for brain MRI metastases.\n"
        "Your job is to judge whether the candidate at the centre of every tile is a real "
        "additional metastasis (not a vessel, CSF, artefact, or duplicate of the seed).\n"
        "\n"
        "The single PNG below is laid out as a 4 row x N modality grid:\n"
        "  Row 0: candidate AXIAL crops (modalities in order)\n"
        "  Row 1: candidate CORONAL crops\n"
        "  Row 2: candidate SAGITTAL crops\n"
        "  Row 3: SEED lesion AXIAL crops (for direct comparison)\n"
        "Red outline (rows 0-2) marks the proposal model's predicted candidate extent.\n"
        "Green outline (row 3) marks the seed lesion segmentation.\n"
        "The candidate is at the EXACT CENTRE of every candidate tile - localize before reasoning.\n"
        f"Candidate centroid (z,y,x)={coord}, detector probability={prob:.3f}.\n"
        "\n"
        "Respond with JSON only, matching this schema EXACTLY:\n"
        "{\n"
        '  "decision": "accept" | "reject" | "uncertain",\n'
        '  "evidence_for": [string, ...],\n'
        '  "evidence_against": [string, ...],\n'
        '  "seed_similarity": number in [0,1],\n'
        '  "mimic_risk": "low" | "medium" | "high",\n'
        '  "confidence": number in [0,1],\n'
        '  "reason": string (1-2 sentences)\n'
        "}\n"
    )


def _gt_target_json(gt_verdict: str) -> str:
    target = {
        "decision": "accept" if gt_verdict == "confirm" else "reject",
        "evidence_for": [], "evidence_against": [],
        "seed_similarity": 0.0, "mimic_risk": "low",
        "confidence": 0.95,
        "reason": "Ground-truth label.",
    }
    return json.dumps(target)


def _parse_decision(text: str) -> str:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return "uncertain"
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return "uncertain"
    d = str(obj.get("decision", "")).lower()
    if d.startswith("acc") or d.startswith("conf"):
        return "accept"
    if d.startswith("rej"):
        return "reject"
    return "uncertain"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out", default="data/dpo_v2/mined_pairs.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.base_model} + adapter {args.adapter}")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    processor = AutoProcessor.from_pretrained(args.base_model)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    in_dir = Path(args.in_dir)
    records = [json.loads(l) for l in Path(args.metadata).open()]
    if args.limit:
        records = records[:args.limit]
    print(f"Examining {len(records)} examples")

    n_total = 0
    n_correct = 0
    n_pairs_written = 0
    from qwen_vl_utils import process_vision_info
    with out_path.open("w") as fout:
        for r in tqdm(records, desc="mining"):
            gt_v = r["gt_label"]["verdict"]
            if gt_v not in ("confirm", "reject"):
                continue
            img_path = in_dir / r["image_path"]
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            prompt = _verifier_user_prompt(r["coord_zyx"], r["detector_prob"])
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
            gen_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
            sft_decision = _parse_decision(gen_text)
            gt_decision = "accept" if gt_v == "confirm" else "reject"

            n_total += 1
            if sft_decision == gt_decision:
                n_correct += 1
                continue

            # Disagreement => DPO preference pair
            with img_path.open("rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("ascii")
            chosen = _gt_target_json(gt_v)
            # rejected = the actual SFT v2 generation (raw text)
            # Trim to a reasonable length and ensure it's valid JSON-ish
            rejected = gen_text.strip()[:600]
            pair = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "image_b64": img_b64,
                "meta": {
                    "study_id": r["study_id"],
                    "coord_zyx": r["coord_zyx"],
                    "detector_prob": r["detector_prob"],
                    "gt_verdict": gt_v,
                    "sft_v2_prediction": sft_decision,
                },
            }
            fout.write(json.dumps(pair) + "\n")
            n_pairs_written += 1

    print(f"\nTotal examined: {n_total}")
    print(f"SFT v2 correct:  {n_correct} ({n_correct/max(1,n_total):.1%})")
    print(f"DPO pairs written: {n_pairs_written} ({n_pairs_written/max(1,n_total):.1%})")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
