"""How many of the current DETECTOR_MISS cases would be rescued if we
lowered min_voxels=10 -> 3 in probmap_to_candidates?

For each GT lesion, find the nearest baseline-candidate centroid distance
under both extraction settings:
  (A) probmap_to_candidates(threshold=0.3, min_voxels=10)   <- current
  (B) probmap_to_candidates(threshold=0.3, min_voxels=3)    <- proposed

A "rescue" = (no nearby CC at min_voxels=10) AND (nearby CC at min_voxels=3),
i.e. the detector did find something at the GT location, we just filtered
it. Upper bound on how much a min_voxels=3 re-eval could improve recall.
"""
from __future__ import annotations
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brain_mets_agent.models import NNUNetProbmapCache
from brain_mets_agent.models.proposal import probmap_to_candidates


@dataclass
class _GTLite:
    centroid_vox: tuple
    voxel_count: int
    volume_mm3: float
    mask: object


NEAR = 10.0   # match_predictions' dist_vox_threshold


def nearest_dist(gt, cands) -> float:
    if not cands:
        return float("inf")
    g = np.asarray(gt.centroid_vox, dtype=np.float64)
    return min(float(np.linalg.norm(np.asarray(c.coord_vox, dtype=np.float64) - g))
               for c in cands)


def size_bucket(vol_mm3: float) -> str:
    if vol_mm3 < 30:    return "<30 mm3"
    if vol_mm3 < 100:   return "30-100"
    if vol_mm3 < 300:   return "100-300"
    if vol_mm3 < 1000:  return "300-1000"
    return ">1000"


COHORTS = [
    ("UCSF-test", "runs/state_test_nnunet_dpo_v4", "runs/nnunet_probmap_test"),
    ("Stanford",  "runs/state_stanford_dpo_v4",    "runs/nnunet_probmap_stanford"),
]


def run_one(label, state_dir, probmap_dir):
    cache = NNUNetProbmapCache(probmap_dir)
    rows = []
    for p in sorted(Path(state_dir).glob("*.pkl")):
        with p.open("rb") as f:
            rec = pickle.load(f)
        rec["non_seed_gt"] = [_GTLite(**g) for g in rec["non_seed_gt"]]
        for g in rec["non_seed_gt"]:
            g.mask = None
        prob = cache.predict_probmap_for(rec["study_id"])
        b10 = probmap_to_candidates(prob, threshold=0.3, min_voxels=10)
        b3  = probmap_to_candidates(prob, threshold=0.3, min_voxels=3)
        for gt in rec["non_seed_gt"]:
            d10 = nearest_dist(gt, b10)
            d3  = nearest_dist(gt, b3)
            rows.append({
                "study_id": rec["study_id"],
                "volume_mm3": float(gt.volume_mm3),
                "voxel_count": int(gt.voxel_count),
                "size_bucket": size_bucket(gt.volume_mm3),
                "d_minvox10": d10,
                "d_minvox3":  d3,
                "near_at_10": d10 <= NEAR,
                "near_at_3":  d3  <= NEAR,
            })
    return rows


def report(label, rows):
    n = len(rows)
    near10 = sum(1 for r in rows if r["near_at_10"])
    near3  = sum(1 for r in rows if r["near_at_3"])
    rescue = sum(1 for r in rows if (not r["near_at_10"]) and r["near_at_3"])
    print(f"\n=== {label}  (n={n} GT lesions) ===")
    print(f"  has nearby baseline CC at min_voxels=10: {near10:>4}  ({100*near10/n:>5.1f}%)")
    print(f"  has nearby baseline CC at min_voxels=3:  {near3:>4}  ({100*near3/n:>5.1f}%)")
    print(f"  RESCUED by lowering filter:              {rescue:>4}  ({100*rescue/n:>5.1f}%)")
    if rescue == 0:
        return
    # By size bucket
    SIZES = ["<30 mm3", "30-100", "100-300", "300-1000", ">1000"]
    print(f"\n  Rescues by size bucket:")
    print(f"  {'bucket':<12s}  {'n_GT':>5}  {'n_DET_MISS@10':>14}  {'rescued@3':>10}  {'rescue_rate':>11}")
    for sb in SIZES:
        sub = [r for r in rows if r["size_bucket"] == sb]
        if not sub: continue
        miss10 = sum(1 for r in sub if not r["near_at_10"])
        rescued = sum(1 for r in sub if (not r["near_at_10"]) and r["near_at_3"])
        rate = 100 * rescued / miss10 if miss10 > 0 else 0.0
        print(f"  {sb:<10s}    {len(sub):>5}  {miss10:>14}  {rescued:>10}  {rate:>10.1f}%")


all_rows = []
for label, sd, pd_ in COHORTS:
    rows = run_one(label, sd, pd_)
    all_rows.extend([dict(r, cohort=label) for r in rows])
    report(label, rows)

print()
report("POOLED", all_rows)
