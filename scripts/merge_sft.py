"""Merge previously-finalized SFT records with raw metadata.jsonl files.

Saves another corrector run on already-corrected disagreements: the v1
finalized records already contain rich reasoning targets (philosophy C),
so we reuse them as-is and only convert NEW raw records (proposal-only,
no VLM judgement) to gt_only targets on the fly.

Inputs auto-detected:
  - "*.jsonl" with a "messages" key per record  -> already finalized; passthrough
  - "metadata.jsonl" with "gt_label" key per record -> raw; convert to gt_only

Usage:
  python scripts/merge_sft.py \
      --inputs data/sft_v1_finalized/train.jsonl data/sft_v2_train/metadata.jsonl \
      --val-passthrough data/sft_v1_finalized/val.jsonl \
      --out data/sft_combined \
      --neg-pos-ratio 2.0
"""
from __future__ import annotations
import argparse
import base64
import json
import random
from collections import Counter
from pathlib import Path


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


def _convert_raw_to_messages(raw_path: Path, base_dir: Path) -> list[dict]:
    """Convert a metadata.jsonl from build_sft_dataset/generate_sft_data into
    messages-format records using gt_only targets (philosophy B)."""
    out = []
    for line in raw_path.open():
        r = json.loads(line)
        gt_v = r["gt_label"]["verdict"]
        if gt_v not in ("confirm", "reject"):
            continue
        img_path = base_dir / r["image_path"]
        if not img_path.exists():
            continue
        img_b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
        target = {
            "decision": "accept" if gt_v == "confirm" else "reject",
            "evidence_for": [], "evidence_against": [],
            "seed_similarity": 0.0, "mimic_risk": "low",
            "confidence": 0.95,
            "reason": "Ground-truth label; reasoning omitted.",
        }
        user_prompt = _verifier_user_prompt(r["coord_zyx"], r["detector_prob"])
        out.append({
            "messages": [
                {"role": "user", "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                    {"type": "text", "text": user_prompt},
                ]},
                {"role": "assistant",
                 "content": [{"type": "text", "text": json.dumps(target)}]},
            ],
            "meta": {
                "study_id": r["study_id"],
                "gt_label": r["gt_label"],
                "vlm_label": r["vlm_label"],
                "philosophy": "proposal_only_gt",
                "coord_zyx": r["coord_zyx"],
                "detector_prob": r["detector_prob"],
            },
        })
    return out


def _load_input(p: Path) -> list[dict]:
    """Auto-detect format: messages-jsonl (passthrough) or metadata.jsonl (convert)."""
    with p.open() as f:
        first = f.readline()
    sample = json.loads(first) if first.strip() else {}
    if "messages" in sample:
        return [json.loads(l) for l in p.open()]
    if "gt_label" in sample:
        return _convert_raw_to_messages(p, p.parent)
    raise ValueError(f"unrecognised input format in {p}")


def _balance(records: list[dict], neg_pos_ratio: float, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    pos = [r for r in records if r["meta"]["gt_label"]["verdict"] == "confirm"]
    neg = [r for r in records if r["meta"]["gt_label"]["verdict"] == "reject"]
    if not pos or not neg:
        return records
    target_neg = int(round(len(pos) * neg_pos_ratio))
    if len(neg) > target_neg:
        rng.shuffle(neg)
        neg = neg[:target_neg]
    out = pos + neg
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="One or more input jsonl files (auto-detected format).")
    ap.add_argument("--val-passthrough", default=None,
                    help="Optional already-finalized val.jsonl to pass through.")
    ap.add_argument("--out", default="data/sft_combined")
    ap.add_argument("--neg-pos-ratio", type=float, default=2.0)
    ap.add_argument("--no-dedup", action="store_true",
                    help="Skip dedup by (study_id, coord_zyx). By default we keep the "
                         "FIRST occurrence (so earlier inputs win - put preferred-philosophy "
                         "files first).")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_recs: list[dict] = []
    for inp in args.inputs:
        p = Path(inp)
        recs = _load_input(p)
        print(f"  loaded {len(recs)} records from {p}")
        train_recs.extend(recs)
    print(f"Total train pre-dedup: {len(train_recs)}")

    if not args.no_dedup:
        seen: set[tuple] = set()
        dedup = []
        for r in train_recs:
            key = (r["meta"]["study_id"], tuple(int(c) for c in r["meta"]["coord_zyx"]))
            if key in seen:
                continue
            seen.add(key)
            dedup.append(r)
        n_dropped = len(train_recs) - len(dedup)
        print(f"Dedup: dropped {n_dropped} duplicates; {len(dedup)} unique records remain")
        train_recs = dedup
    print(f"Total train pre-balance: {len(train_recs)}")
    print(f"  philosophy mix: "
          f"{Counter(r['meta']['philosophy'] for r in train_recs).most_common()}")
    print(f"  GT mix:         "
          f"{Counter(r['meta']['gt_label']['verdict'] for r in train_recs).most_common()}")

    train_recs = _balance(train_recs, args.neg_pos_ratio)
    print(f"After balancing (neg:pos = {args.neg_pos_ratio}): {len(train_recs)} train")
    print(f"  philosophy mix: "
          f"{Counter(r['meta']['philosophy'] for r in train_recs).most_common()}")
    print(f"  GT mix:         "
          f"{Counter(r['meta']['gt_label']['verdict'] for r in train_recs).most_common()}")

    train_out = out_dir / "train.jsonl"
    with train_out.open("w") as f:
        for r in train_recs:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {train_out} ({train_out.stat().st_size//1024} KB)")

    if args.val_passthrough:
        val_recs = [json.loads(l) for l in Path(args.val_passthrough).open()]
        val_out = out_dir / "val.jsonl"
        with val_out.open("w") as f:
            for r in val_recs:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {val_out} (passthrough; {len(val_recs)} records, "
              f"{val_out.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
