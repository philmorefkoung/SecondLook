"""Build a VLM-SFT dataset from cached agent state pickles.

Each cached pickle has, per study:
  - candidates with coord_vox, prob, voxel_count, the VLM's verdict + confidence
  - non_seed_gt with masks + centroids
We re-run the proposal model on each unique study to recover candidate masks,
then for each candidate:
  - regenerate the EvidenceCard composite (matches what the VLM saw at eval time)
  - derive a ground-truth label by matching candidate.coord_vox to non_seed_gt
  - record both labels: gt_label (the SFT target) and vlm_label (what the model said)

Output layout:
  out_dir/
    metadata.jsonl                     # one record per example
    images/<study_id>/<cand_idx>.png   # composite

Each metadata record:
  {
    "image_path": "images/100109A/0042.png",
    "study_id": "100109A",
    "candidate_id": 42,
    "coord_zyx": [..],
    "detector_prob": 0.87,
    "voxel_count": 156,
    "gt_label":   {"verdict": "confirm" | "reject", "match_dist_vox": 3.4, ...},
    "vlm_label":  {"verdict": "confirm" | "uncertain" | "reject", "confidence": ..., ...},
    "agreement":  true / false,
    "source_state_dir": "runs/state_v6_15studies",
  }

A separate `mistakes.jsonl` collects only the disagreements (gt != vlm) - the
hard cases for SFT to focus on.
"""
from __future__ import annotations
import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from brain_mets_agent.data import load_study, MODALITIES
from brain_mets_agent.data.phenotype import characterize_seed
from brain_mets_agent.models.proposal import probmap_to_candidates
from brain_mets_agent.models import LesionProposalModel
from brain_mets_agent.orchestrator.tools.viewer import ViewerTool
from brain_mets_agent.orchestrator.evidence import build_evidence_card


CONFIRM_DIST_VOX = 10.0  # if candidate centroid is within this of any non-seed GT, confirm


def _candidate_match(coord_vox, non_seed_gt) -> tuple[bool, float, int]:
    """Return (matched, min_dist_vox, matched_gt_idx)."""
    if not non_seed_gt:
        return False, float("inf"), -1
    coord = np.asarray(coord_vox, dtype=np.float64)
    dists = [
        float(np.linalg.norm(coord - np.asarray(g.centroid_vox, dtype=np.float64)))
        for g in non_seed_gt
    ]
    idx = int(np.argmin(dists))
    return dists[idx] <= CONFIRM_DIST_VOX, dists[idx], idx


def _collect_unique_studies(state_dirs: list[Path]) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = defaultdict(list)
    for d in state_dirs:
        for p in sorted(d.glob("*.pkl")):
            out[p.stem].append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dirs", nargs="+", required=True,
                    help="One or more cached state directories.")
    ap.add_argument("--root", required=True,
                    help="UCSF-BMSR root (to load source studies).")
    ap.add_argument("--ckpt", required=True,
                    help="Trained Swin-UNETR checkpoint (to recover candidate masks).")
    ap.add_argument("--out", default="data/sft_v1")
    ap.add_argument("--threshold", type=float, default=0.3,
                    help="Probmap binarisation threshold for CC extraction.")
    ap.add_argument("--match-radius", type=float, default=8.0,
                    help="Max coord distance (vox) to consider a cached candidate"
                         " the same as a freshly-extracted CC.")
    args = ap.parse_args()

    state_dirs = [Path(d) for d in args.state_dirs]
    out_dir = Path(args.out)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    metadata_fp = (out_dir / "metadata.jsonl").open("w")
    mistakes_fp = (out_dir / "mistakes.jsonl").open("w")

    by_study = _collect_unique_studies(state_dirs)
    print(f"Found {len(by_study)} unique studies across {len(state_dirs)} dirs")

    model = LesionProposalModel()
    model.load_state(args.ckpt)
    print(f"Loaded model from {args.ckpt}")

    n_total = 0
    n_agreement = 0
    n_per_study = {}
    n_per_label = defaultdict(int)
    for sid, pickles in by_study.items():
        try:
            study = load_study(args.root, sid)
        except FileNotFoundError as e:
            print(f"  [skip] {sid}: {e}")
            continue
        if study.seg is None or not (study.seg > 0).any():
            continue

        # Re-run the proposal to recover candidate masks
        multi_modal = np.stack([study.images[m] for m in MODALITIES], axis=0).astype(np.float32)
        prob = model.predict_probmap(multi_modal)
        fresh_cands = probmap_to_candidates(prob, threshold=args.threshold, min_voxels=10)
        # build coord -> mask lookup (centroid is unique enough at this density)
        fresh_coords = np.array([c.coord_vox for c in fresh_cands], dtype=np.float64)

        viewer = ViewerTool(study.images)
        # phenotype: pick the largest GT lesion as seed (matches the eval convention)
        from brain_mets_agent.data import extract_instances, select_seed
        gt_inst = extract_instances(study.seg, study.spacing, study.affine)
        if not gt_inst:
            continue
        seed, others = select_seed(gt_inst, "largest")
        if seed is None:
            continue
        ph = characterize_seed(seed.mask, study.images,
                                affine=study.affine, spacing_mm=study.spacing)
        sp_dict = ph.to_dict()

        seen_coords: set[tuple[int, int, int]] = set()
        per_study_dir = img_dir / sid
        per_study_dir.mkdir(exist_ok=True)
        idx_in_study = 0

        for pkl in pickles:
            with pkl.open("rb") as f:
                rec = pickle.load(f)
            cands = rec["candidates"]
            for cand in cands:
                # dedup across pickles
                key = tuple(int(c) for c in cand.coord_vox)
                if key in seen_coords:
                    continue
                seen_coords.add(key)

                # find the freshest mask via nearest fresh candidate
                cand_mask = None
                if len(fresh_cands) > 0:
                    target = np.asarray(cand.coord_vox, dtype=np.float64)
                    dists = np.linalg.norm(fresh_coords - target, axis=1)
                    j = int(np.argmin(dists))
                    if float(dists[j]) <= args.match_radius:
                        cand_mask = fresh_cands[j].mask

                # GT label
                matched, min_dist, gt_idx = _candidate_match(cand.coord_vox, others)
                gt_verdict = "confirm" if matched else "reject"

                # Build the evidence card the VLM would have seen
                card = build_evidence_card(
                    viewer=viewer,
                    candidate_coord_vox=cand.coord_vox,
                    candidate_voxel_count=cand.voxel_count,
                    detector_prob=cand.prob,
                    seed_coord_vox=tuple(int(c) for c in seed.centroid_vox),
                    seed_phenotype=sp_dict,
                    modalities=MODALITIES,
                    candidate_mask=cand_mask,
                    seed_mask=seed.mask,
                    tile_size=96,
                )

                # Save image
                img_path = per_study_dir / f"{idx_in_study:04d}.png"
                import base64
                img_path.write_bytes(base64.b64decode(card.composite_png_b64))

                rec_out = {
                    "image_path": str(img_path.relative_to(out_dir)).replace("\\", "/"),
                    "study_id": sid,
                    "candidate_index_in_study": idx_in_study,
                    "coord_zyx": list(int(c) for c in cand.coord_vox),
                    "detector_prob": float(cand.prob),
                    "voxel_count": int(cand.voxel_count),
                    "pass_idx": int(cand.pass_idx),
                    "had_recovered_mask": cand_mask is not None,
                    "gt_label": {
                        "verdict": gt_verdict,
                        "match_dist_vox": round(float(min_dist), 3),
                        "matched_gt_index": gt_idx,
                        "confidence": 1.0,  # ground truth
                    },
                    "vlm_label": {
                        "verdict": cand.verdict,
                        "confidence": round(float(cand.vlm_conf), 4),
                        "evidence_for": list(cand.evidence_for),
                        "evidence_against": list(cand.evidence_against),
                        "seed_similarity": round(float(cand.seed_similarity), 4),
                        "mimic_risk": cand.mimic_risk,
                        "rationale": (cand.rationale or "")[:400],
                    },
                    "metadata_text": card.metadata_text,
                    "source_pickle": str(pkl.relative_to(pkl.parent.parent)).replace("\\", "/"),
                }
                rec_out["agreement"] = (
                    rec_out["gt_label"]["verdict"] == rec_out["vlm_label"]["verdict"]
                )
                metadata_fp.write(json.dumps(rec_out) + "\n")
                if not rec_out["agreement"]:
                    mistakes_fp.write(json.dumps(rec_out) + "\n")

                n_total += 1
                if rec_out["agreement"]:
                    n_agreement += 1
                n_per_label[(rec_out["gt_label"]["verdict"], rec_out["vlm_label"]["verdict"])] += 1
                idx_in_study += 1

        n_per_study[sid] = idx_in_study
        print(f"  {sid}: {idx_in_study} unique candidates")

    metadata_fp.close()
    mistakes_fp.close()

    print()
    print(f"Total examples: {n_total}")
    print(f"Agreement rate: {n_agreement}/{n_total} = {n_agreement/max(n_total,1):.2%}")
    print()
    print("Confusion (gt -> vlm):")
    for (gt, vl), n in sorted(n_per_label.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        print(f"  {str(gt):>12} -> {str(vl):>12}: {n}")
    print()
    print(f"Per-study counts: {n_per_study}")
    print()
    print(f"Wrote {out_dir}/metadata.jsonl + mistakes.jsonl + images/")


if __name__ == "__main__":
    main()
