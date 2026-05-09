"""Bulk-generate SFT examples from the TRAIN split with proposal-only labelling.

Per train study:
  - Load study, characterize seed (largest GT lesion).
  - Run trained Swin-UNETR proposal at threshold=0.3 to get all candidates.
  - For each candidate: build EvidenceCard composite, derive GT label by
    matching candidate centroid to non-seed GT lesions (within match_radius).
  - Save (PNG + label JSON) per candidate. NO VLM call.

Output schema matches `build_sft_dataset.py` so the same finalize_sft_targets.py
can merge cached eval data + this bulk train data into one dataset.

Usage:
  python scripts/generate_sft_data.py \
      --root C:/Users/User/Documents/UCSF_BMSR/UCSF_BrainMetastases_TRAIN \
      --splits brain_mets_agent/data/splits.csv \
      --split train \
      --ckpt ckpts/swin_v1/best.pt \
      --limit 30 \
      --threshold 0.3 \
      --out data/sft_v2_train
"""
from __future__ import annotations
import argparse
import base64
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from brain_mets_agent.data import (
    load_study, extract_instances, select_seed, MODALITIES,
)
from brain_mets_agent.data.phenotype import characterize_seed
from brain_mets_agent.data.splits import read_splits_csv
from brain_mets_agent.models.proposal import probmap_to_candidates
from brain_mets_agent.models import LesionProposalModel
from brain_mets_agent.orchestrator.tools.viewer import ViewerTool
from brain_mets_agent.orchestrator.evidence import build_evidence_card


def _candidate_match(coord_vox, non_seed_gt, max_dist_vox: float):
    """Return (matched, min_dist_vox, matched_gt_idx)."""
    if not non_seed_gt:
        return False, float("inf"), -1
    coord = np.asarray(coord_vox, dtype=np.float64)
    dists = [
        float(np.linalg.norm(coord - np.asarray(g.centroid_vox, dtype=np.float64)))
        for g in non_seed_gt
    ]
    j = int(np.argmin(dists))
    return dists[j] <= max_dist_vox, dists[j], j


def _is_duplicate_of_seed(coord_vox, seed_centroid_vox, dist_threshold_vox: float = 5.0) -> bool:
    return (np.linalg.norm(np.asarray(coord_vox, dtype=np.float64)
                            - np.asarray(seed_centroid_vox, dtype=np.float64))
            <= dist_threshold_vox)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="data/sft_v2_train")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--min-voxels", type=int, default=10)
    ap.add_argument("--match-radius", type=float, default=10.0)
    ap.add_argument("--max-cands-per-study", type=int, default=64,
                    help="Cap candidates per study to keep dataset balanced.")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = use all studies in the split")
    ap.add_argument("--seed-strategy", default="largest")
    args = ap.parse_args()

    splits = read_splits_csv(args.splits)
    study_ids = splits[args.split]
    if args.limit:
        study_ids = study_ids[:args.limit]
    print(f"Generating SFT data from {len(study_ids)} {args.split} studies")

    out_dir = Path(args.out)
    img_root = out_dir / "images"
    img_root.mkdir(parents=True, exist_ok=True)
    metadata_fp = (out_dir / "metadata.jsonl").open("w")

    model = LesionProposalModel()
    model.load_state(args.ckpt)
    print(f"Loaded {args.ckpt}")

    n_total = 0
    n_pos = 0
    n_per_study = {}
    for sid in study_ids:
        try:
            study = load_study(args.root, sid)
        except FileNotFoundError as e:
            print(f"  [skip] {sid}: {e}")
            continue
        if study.seg is None or not (study.seg > 0).any():
            continue
        gt_inst = extract_instances(study.seg, study.spacing, study.affine)
        if not gt_inst:
            continue
        seed, others = select_seed(gt_inst, args.seed_strategy)
        if not others:
            continue   # nothing to find beyond the seed; skip

        ph = characterize_seed(seed.mask, study.images,
                                affine=study.affine, spacing_mm=study.spacing)
        sp_dict = ph.to_dict()
        viewer = ViewerTool(study.images)

        multi_modal = np.stack([study.images[m] for m in MODALITIES], axis=0).astype(np.float32)
        prob = model.predict_probmap(multi_modal)
        cands = probmap_to_candidates(prob, threshold=args.threshold,
                                       min_voxels=args.min_voxels)

        seed_centroid = tuple(int(c) for c in seed.centroid_vox)
        per_study_dir = img_root / sid
        per_study_dir.mkdir(exist_ok=True)
        idx_in_study = 0

        # Sort by detector prob desc; cap to max_cands_per_study to control balance
        cands_sorted = sorted(cands, key=lambda c: -c.prob)[:args.max_cands_per_study]

        for cand in cands_sorted:
            # skip seed-duplicates (no useful SFT signal: it's the GIVEN seed)
            if _is_duplicate_of_seed(cand.coord_vox, seed_centroid):
                continue

            matched, min_dist, gt_idx = _candidate_match(
                cand.coord_vox, others, args.match_radius,
            )
            gt_verdict = "confirm" if matched else "reject"

            card = build_evidence_card(
                viewer=viewer,
                candidate_coord_vox=cand.coord_vox,
                candidate_voxel_count=cand.voxel_count,
                detector_prob=cand.prob,
                seed_coord_vox=seed_centroid,
                seed_phenotype=sp_dict,
                modalities=MODALITIES,
                candidate_mask=cand.mask,
                seed_mask=seed.mask,
                tile_size=96,
            )

            img_path = per_study_dir / f"{idx_in_study:04d}.png"
            img_path.write_bytes(base64.b64decode(card.composite_png_b64))

            rec = {
                "image_path": str(img_path.relative_to(out_dir)).replace("\\", "/"),
                "study_id": sid,
                "candidate_index_in_study": idx_in_study,
                "coord_zyx": list(int(c) for c in cand.coord_vox),
                "detector_prob": float(cand.prob),
                "voxel_count": int(cand.voxel_count),
                "pass_idx": 2,    # all from first-pass proposal
                "had_recovered_mask": True,
                "gt_label": {
                    "verdict": gt_verdict,
                    "match_dist_vox": round(float(min_dist), 3),
                    "matched_gt_index": gt_idx,
                    "confidence": 1.0,
                },
                "vlm_label": {       # No VLM call - empty placeholder.
                    "verdict": None, "confidence": 0.0,
                    "evidence_for": [], "evidence_against": [],
                    "seed_similarity": 0.0, "mimic_risk": "low",
                    "rationale": "",
                },
                "metadata_text": card.metadata_text,
                "source_pickle": f"proposal_only::{args.split}",
            }
            rec["agreement"] = False   # no VLM verdict to agree with
            metadata_fp.write(json.dumps(rec) + "\n")

            n_total += 1
            if gt_verdict == "confirm":
                n_pos += 1
            idx_in_study += 1

        n_per_study[sid] = idx_in_study
        print(f"  {sid}: {idx_in_study} candidates ({len(others)} non-seed GT)")

    metadata_fp.close()

    print()
    print(f"Total examples: {n_total}")
    print(f"  positive (matched a GT lesion): {n_pos}")
    print(f"  negative:                       {n_total - n_pos}")
    print(f"Wrote {out_dir}/metadata.jsonl + images/")
    print()
    print("Next: run scripts/finalize_sft_targets.py with --no-correct-disagreements")
    print("(these examples have no VLM verdict to compare so they go into the "
          "'disagreement_gt_only' philosophy bucket).")


if __name__ == "__main__":
    main()
