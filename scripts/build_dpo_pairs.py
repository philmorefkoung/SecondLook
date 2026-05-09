"""Build (chosen, rejected) preference pairs from cached states for VLM DPO.

For each disagreement in data/sft_v1/metadata.jsonl (Sonnet's verdict != GT):
  - chosen   = the GT-aligned corrected target JSON (from data/sft_v1_finalized
               via the philosophy='disagreement_corrected' records)
  - rejected = the original VLM's wrong verdict, reconstructed as the same JSON
               schema from metadata.vlm_label fields
  - prompt   = the verifier user prompt + image (same as SFT format)

Output: data/dpo_v1/pairs.jsonl in TRL-compatible format
        (per record: prompt list, chosen list, rejected list, images list).

Usage:
  python scripts/build_dpo_pairs.py \
      --metadata data/sft_v1/metadata.jsonl \
      --finalized-train data/sft_v1_finalized/train.jsonl \
      --finalized-val   data/sft_v1_finalized/val.jsonl \
      --in-dir data/sft_v1 \
      --out data/dpo_v1
"""
from __future__ import annotations
import argparse
import base64
import json
from collections import defaultdict
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


def _vlm_label_to_target_json(vlm_label: dict) -> str:
    """Reconstruct the JSON the original VLM emitted from cached vlm_label fields."""
    decision_map = {"confirm": "accept", "reject": "reject", "uncertain": "uncertain"}
    target = {
        "decision": decision_map.get(vlm_label.get("verdict") or "uncertain", "uncertain"),
        "evidence_for": list(vlm_label.get("evidence_for") or []),
        "evidence_against": list(vlm_label.get("evidence_against") or []),
        "seed_similarity": float(vlm_label.get("seed_similarity") or 0.0),
        "mimic_risk": vlm_label.get("mimic_risk") or "low",
        "confidence": float(vlm_label.get("confidence") or 0.5),
        "reason": (vlm_label.get("rationale") or "")[:300],
    }
    return json.dumps(target, ensure_ascii=False)


def _key(study_id, coord_zyx) -> tuple:
    return (study_id, tuple(int(c) for c in coord_zyx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True,
                    help="Raw metadata.jsonl from build_sft_dataset.py (has both gt + original vlm).")
    ap.add_argument("--finalized-train", required=True,
                    help="Messages-format train.jsonl with corrected targets.")
    ap.add_argument("--finalized-val", default=None,
                    help="Messages-format val.jsonl (will also build val pairs if provided).")
    ap.add_argument("--in-dir", required=True, help="Image dir from build_sft_dataset.")
    ap.add_argument("--out", default="data/dpo_v1")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load raw metadata - look up by (study_id, coord) -> vlm_label
    raw_meta_by_key: dict[tuple, dict] = {}
    n_raw = 0
    for line in Path(args.metadata).open():
        r = json.loads(line)
        if r["vlm_label"]["verdict"] is None:
            continue
        if r["agreement"]:
            continue
        raw_meta_by_key[_key(r["study_id"], r["coord_zyx"])] = r
        n_raw += 1
    print(f"Loaded {n_raw} disagreement records from {args.metadata}")

    def build_pairs_from_finalized(finalized_path: Path) -> list[dict]:
        pairs = []
        n_examined = 0
        n_skipped_phil = 0
        n_skipped_no_raw = 0
        for line in finalized_path.open():
            r = json.loads(line)
            n_examined += 1
            meta = r["meta"]
            if meta.get("philosophy") != "disagreement_corrected":
                n_skipped_phil += 1
                continue
            key = _key(meta["study_id"], meta["coord_zyx"])
            raw = raw_meta_by_key.get(key)
            if raw is None:
                n_skipped_no_raw += 1
                continue
            chosen_text = ""
            for c in r["messages"][1]["content"]:
                if c.get("type") == "text":
                    chosen_text += c["text"]
            rejected_text = _vlm_label_to_target_json(raw["vlm_label"])
            user_prompt = _verifier_user_prompt(meta["coord_zyx"], meta["detector_prob"])
            # Get the image base64 from the user content
            user_image_b64 = None
            for c in r["messages"][0]["content"]:
                if c.get("type") == "image":
                    user_image_b64 = c["source"]["data"]
                    break
            if user_image_b64 is None:
                continue
            pairs.append({
                "prompt": user_prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
                "image_b64": user_image_b64,
                "meta": {
                    "study_id": meta["study_id"],
                    "coord_zyx": meta["coord_zyx"],
                    "detector_prob": meta["detector_prob"],
                    "gt_verdict": meta["gt_label"]["verdict"],
                    "rejected_vlm_verdict": raw["vlm_label"]["verdict"],
                },
            })
        print(f"  examined={n_examined} skipped_philosophy={n_skipped_phil} "
              f"skipped_no_raw_match={n_skipped_no_raw} -> {len(pairs)} pairs")
        return pairs

    print("Building train pairs:")
    train_pairs = build_pairs_from_finalized(Path(args.finalized_train))
    train_out = out_dir / "train_pairs.jsonl"
    with train_out.open("w") as f:
        for p in train_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {train_out} ({train_out.stat().st_size//1024} KB, {len(train_pairs)} pairs)")

    if args.finalized_val:
        print("Building val pairs:")
        val_pairs = build_pairs_from_finalized(Path(args.finalized_val))
        val_out = out_dir / "val_pairs.jsonl"
        with val_out.open("w") as f:
            for p in val_pairs:
                f.write(json.dumps(p) + "\n")
        print(f"Wrote {val_out} ({val_out.stat().st_size//1024} KB, {len(val_pairs)} pairs)")


if __name__ == "__main__":
    main()
