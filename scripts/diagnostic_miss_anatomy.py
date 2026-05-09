"""Where do our missed GT lesions live?

Loads cached agent state for both UCSF-test and Stanford-external cohorts and,
for every GT lesion, classifies it as one of:

  * MATCHED       agent surfaced it within FP/study budget
  * DETECTOR_MISS no baseline candidate (raw probmap CC) within 10 vox of GT
                  centroid -> the detector itself failed; agent has nothing to
                  rank
  * RANKER_MISS   detector found a nearby candidate but agent did not place
                  any TP for this GT inside the FP budget

Then aggregates the miss breakdown by:
  - lesion size (volume_mm3 buckets)
  - z-position bucket (crude posterior-fossa proxy: bottom-third of brain)

Output: stdout table + per-lesion CSV, so we can decide whether an MNI-atlas
prior is worth the multi-day pipeline (it is, iff misses cluster anatomically).

Run from project root:
  python scripts/diagnostic_miss_anatomy.py
"""
from __future__ import annotations
import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nib

from brain_mets_agent.models import NNUNetProbmapCache
from brain_mets_agent.models.proposal import probmap_to_candidates
from brain_mets_agent.orchestrator.tools.ranker import rank_candidates
from brain_mets_agent.eval.metrics import match_predictions


@dataclass
class _GTLite:
    centroid_vox: tuple
    voxel_count: int
    volume_mm3: float
    mask: object  # set to None for fast distance-only matching


# Sweep-best ranker config that won on both cohorts.
RANK_KW = dict(
    weights=(0.45, 0.20, 0.10, 0.25),
    prob_source="vlm_else_detector",
    prefer_confirmed=False,
)
FP_BUDGET = 10
NEAR_THRESHOLD_VOX = 10.0   # same as match_predictions' dist_vox_threshold


COHORTS = [
    {
        "label": "UCSF-test",
        "state_dir": "runs/state_test_nnunet_dpo_v4",
        "probmap_dir": "runs/nnunet_probmap_test",
        "image_root": "C:/Users/User/Documents/UCSF_BMSR/UCSF_BrainMetastases_TRAIN",
        "image_filename": "{sid}/{sid}_T1post.nii.gz",
    },
    {
        "label": "Stanford",
        "state_dir": "runs/state_stanford_dpo_v4",
        "probmap_dir": "runs/nnunet_probmap_stanford",
        "image_root": "C:/Users/User/Documents/brain_mets_agent/runs/stanford_staged",
        "image_filename": "{sid}/{sid}_T1post.nii.gz",
    },
]


def kept_at_fp_budget(preds, gt, fp_budget: float) -> list:
    """Replicates additional_lesion_recall_at_fp's accept rule."""
    kept: list = []
    for p in list(preds):
        kept.append(p)
        tp = len(match_predictions(kept, gt).matched_gt)
        fp = len(kept) - tp
        if fp > fp_budget:
            kept.pop()
            break
    return kept


def brain_z_bounds(t1post_path: Path) -> tuple[int, int]:
    """Return (z_min, z_max) along NIfTI axis 2 (slice axis): bounds where
    per-slice signal is non-trivial. We use a permissive 1% threshold rather
    than 5% because the vertex of the brain has lower per-slice tissue volume
    than the temporal lobes / mid-brain and was getting clipped at 5%."""
    img = nib.load(str(t1post_path))
    data = np.asanyarray(img.dataobj).astype(np.float32)
    per_slice = data.sum(axis=(0, 1))
    threshold = 0.01 * per_slice.max()
    nonzero = np.where(per_slice > threshold)[0]
    if len(nonzero) == 0:
        return 0, data.shape[2]
    return int(nonzero.min()), int(nonzero.max())


def axis_orientation_z(affine: np.ndarray) -> int:
    """Return +1 if increasing voxel z is superior (typical for axial DICOM
    after standard reorientation), -1 if inferior. Used to know whether
    'low z' means inferior (posterior fossa) or superior."""
    return 1 if affine[2, 2] > 0 else -1


def classify_miss(
    gt: _GTLite,
    baseline_cands,
    agent_kept,
    z_norm: float,
) -> str:
    """Return one of: MATCHED, DETECTOR_MISS, RANKER_MISS."""
    # Did agent kept-list match this GT?
    matched_idx = match_predictions(agent_kept, [gt]).matched_gt
    if 0 in matched_idx:
        return "MATCHED"
    # Did the BASELINE detector have any CC near the GT centroid?
    g = np.asarray(gt.centroid_vox, dtype=np.float64)
    nearest = float("inf")
    for c in baseline_cands:
        d = float(np.linalg.norm(np.asarray(c.coord_vox, dtype=np.float64) - g))
        if d < nearest:
            nearest = d
    if nearest > NEAR_THRESHOLD_VOX:
        return "DETECTOR_MISS"
    return "RANKER_MISS"


def size_bucket(volume_mm3: float) -> str:
    if volume_mm3 < 30:    return "<30 mm3 (~<3.5 mm dia)"
    if volume_mm3 < 100:   return "30-100"
    if volume_mm3 < 300:   return "100-300"
    if volume_mm3 < 1000:  return "300-1000"
    return ">1000 (~>12 mm dia)"


def z_bucket(z_norm_inferior_to_superior: float) -> str:
    """z normalized to [0, 1] with 0 = inferior (cerebellum/brainstem),
    1 = superior (vertex). The bottom ~20% of brain extent is the
    posterior-fossa proxy (cerebellum + brainstem). Above that is split
    into lower mid (basal ganglia / thalamus / temporal), upper mid
    (centrum semiovale / corona radiata), and vertex."""
    if z_norm_inferior_to_superior < 0.20:    return "infratentorial (0-20%)"
    if z_norm_inferior_to_superior < 0.50:    return "lower mid (20-50%)"
    if z_norm_inferior_to_superior < 0.80:    return "upper mid (50-80%)"
    return "vertex (80-100%)"


def run_cohort(cohort: dict, csv_writer):
    state_dir = Path(cohort["state_dir"])
    probmap_dir = Path(cohort["probmap_dir"])
    image_root = Path(cohort["image_root"])
    cache = NNUNetProbmapCache(probmap_dir)

    rows: list[dict] = []
    for p in sorted(state_dir.glob("*.pkl")):
        with p.open("rb") as f:
            rec = pickle.load(f)
        rec["non_seed_gt"] = [_GTLite(**g) for g in rec["non_seed_gt"]]
        sid = rec["study_id"]
        # strip GT masks for fast distance-only matching
        for g in rec["non_seed_gt"]:
            g.mask = None

        # Baseline candidates (raw probmap CCs, masks stripped for speed)
        prob = cache.predict_probmap_for(sid)
        bcands = probmap_to_candidates(prob, threshold=0.3, min_voxels=10)
        for c in bcands:
            c.mask = None

        # Agent ranked + budget-capped
        ranked = rank_candidates(rec["candidates"], **RANK_KW)
        kept = kept_at_fp_budget(ranked, rec["non_seed_gt"], FP_BUDGET)

        # Brain z-extent for this study
        t1post = image_root / cohort["image_filename"].format(sid=sid)
        try:
            zmin, zmax = brain_z_bounds(t1post)
            zspan = max(zmax - zmin, 1)
            affine = nib.load(str(t1post)).affine
            up = axis_orientation_z(affine)
        except FileNotFoundError:
            zmin, zmax, zspan, up = 0, prob.shape[2], prob.shape[2], 1

        for gi, gt in enumerate(rec["non_seed_gt"]):
            # `centroid_vox` is (axis0, axis1, axis2) -- axis 2 is the slice
            # axis (Z in NIfTI convention) despite being named `cx` in
            # data/lesions.py. Confirmed by inspection: dim-0 ranges to ~256
            # (in-plane X), dim-2 matches the slice count (~100-150).
            cz = float(gt.centroid_vox[2])
            # Convert raw axis-2 index to fraction along brain extent
            f_along = max(0.0, min(1.0, (cz - zmin) / zspan))
            # If voxel-axis-2 grows toward superior, f_along=0 is inferior -> use directly.
            # If voxel-axis-2 grows toward inferior, flip.
            f_inf_to_sup = f_along if up == 1 else 1.0 - f_along

            cls = classify_miss(gt, bcands, kept, f_inf_to_sup)
            row = {
                "cohort": cohort["label"],
                "study_id": sid,
                "gt_idx": gi,
                "volume_mm3": float(gt.volume_mm3),
                "voxel_count": int(gt.voxel_count),
                "z_centroid_vox": cz,
                "z_inf_to_sup": f_inf_to_sup,
                "size_bucket": size_bucket(gt.volume_mm3),
                "z_bucket": z_bucket(f_inf_to_sup),
                "classification": cls,
            }
            rows.append(row)
            csv_writer.writerow(row)

    return rows


def report(label: str, rows: list[dict]):
    print(f"\n=================== {label} ===================")
    print(f"Total GT lesions: {len(rows)}")
    cls_counts: dict = {}
    for r in rows:
        cls_counts[r["classification"]] = cls_counts.get(r["classification"], 0) + 1
    for k, v in cls_counts.items():
        print(f"  {k:<15s}  {v:>4}  ({100*v/len(rows):>5.1f}%)")

    SIZES = ["<30 mm3 (~<3.5 mm dia)", "30-100", "100-300",
             "300-1000", ">1000 (~>12 mm dia)"]
    Z_BUCKETS = ["infratentorial (0-20%)", "lower mid (20-50%)",
                  "upper mid (50-80%)", "vertex (80-100%)"]

    print("\n--- MISS RATE BY SIZE (DETECTOR_MISS / total) ---")
    print(f"{'bucket':<28s}  {'n':>4}  {'matched':>8}  {'det_miss':>8}  {'rnk_miss':>8}  {'det_miss_rate':>14}")
    for sb in SIZES:
        sub = [r for r in rows if r["size_bucket"] == sb]
        if not sub:
            continue
        n = len(sub)
        m = sum(1 for r in sub if r["classification"] == "MATCHED")
        dm = sum(1 for r in sub if r["classification"] == "DETECTOR_MISS")
        rm = sum(1 for r in sub if r["classification"] == "RANKER_MISS")
        print(f"  {sb:<26s}  {n:>4}  {m:>8}  {dm:>8}  {rm:>8}  {100*dm/n:>13.1f}%")

    print("\n--- MISS RATE BY Z-BUCKET (inferior=0, superior=1) ---")
    print(f"{'bucket':<28s}  {'n':>4}  {'matched':>8}  {'det_miss':>8}  {'rnk_miss':>8}  {'det_miss_rate':>14}")
    for zb in Z_BUCKETS:
        sub = [r for r in rows if r["z_bucket"] == zb]
        if not sub:
            continue
        n = len(sub)
        m = sum(1 for r in sub if r["classification"] == "MATCHED")
        dm = sum(1 for r in sub if r["classification"] == "DETECTOR_MISS")
        rm = sum(1 for r in sub if r["classification"] == "RANKER_MISS")
        print(f"  {zb:<26s}  {n:>4}  {m:>8}  {dm:>8}  {rm:>8}  {100*dm/n:>13.1f}%")

    # Cross-table: small lesions in inferior z (the "posterior fossa micromet" cell)
    print("\n--- 2D: SIZE x Z (DETECTOR_MISS rate) ---")
    print(f"{'size':<28s}  {'infrat.':>14}  {'lower mid':>14}  {'upper mid':>14}  {'vertex':>14}")
    for sb in SIZES:
        cells = []
        for zb in Z_BUCKETS:
            sub = [r for r in rows if r["size_bucket"] == sb and r["z_bucket"] == zb]
            if not sub:
                cells.append("       -      ")
            else:
                dm = sum(1 for r in sub if r["classification"] == "DETECTOR_MISS")
                cells.append(f"{100*dm/len(sub):>5.1f}% n={len(sub):>3}")
        print(f"  {sb:<26s}  {cells[0]:>14}  {cells[1]:>14}  {cells[2]:>14}  {cells[3]:>14}")


def main():
    out_csv_path = Path("runs/diagnostic_miss_anatomy.csv")
    all_rows: list[dict] = []
    with out_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "cohort", "study_id", "gt_idx",
            "volume_mm3", "voxel_count",
            "z_centroid_vox", "z_inf_to_sup",
            "size_bucket", "z_bucket", "classification",
        ])
        writer.writeheader()
        for cohort in COHORTS:
            rows = run_cohort(cohort, writer)
            all_rows.extend(rows)
            report(cohort["label"], rows)
    print(f"\nWrote {out_csv_path}")
    print(f"\n=================== POOLED (UCSF + Stanford) ===================")
    report("POOLED", all_rows)


if __name__ == "__main__":
    main()
