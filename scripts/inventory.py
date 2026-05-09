"""Inventory the extracted UCSF-BMSR dataset.

Usage:
  python scripts/inventory.py \
      --root C:/Users/User/Documents/UCSF_BMSR/UCSF_BrainMetastases_TRAIN \
      --out brain_mets_agent/data/inventory.json
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from tqdm import tqdm

from brain_mets_agent.data.splits import make_patient_grouped_splits, write_splits_csv
from brain_mets_agent.data.ucsf_bmsr import patient_id_of


SUFFIXES = ("T1pre", "T1post", "FLAIR", "subtraction", "T2Synth", "seg", "BraTS-seg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="brain_mets_agent/data/inventory.json")
    ap.add_argument("--splits-out", default="brain_mets_agent/data/splits.csv")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--min-voxels", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.root)
    studies = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if args.limit:
        studies = studies[:args.limit]

    rows = []
    modality_present: Counter = Counter()
    multi_focal = 0
    lesion_counts: list[int] = []
    lesion_sizes_mm3: list[float] = []
    shapes: Counter = Counter()
    spacings: Counter = Counter()

    for sid in tqdm(studies, desc="inventory"):
        sdir = root / sid
        present = {s: (sdir / f"{sid}_{s}.nii.gz").exists() for s in SUFFIXES}
        for s, ok in present.items():
            if ok:
                modality_present[s] += 1

        n_les = 0
        sizes_mm3: list[float] = []
        shape = None
        spacing = None
        if present["seg"]:
            seg_img = nib.load(str(sdir / f"{sid}_seg.nii.gz"))
            seg = np.asanyarray(seg_img.dataobj).astype(np.int32)
            shape = tuple(int(x) for x in seg.shape)
            spacing = tuple(round(float(s), 3) for s in seg_img.header.get_zooms()[:3])
            shapes[shape] += 1
            spacings[spacing] += 1
            labeled, n = ndi.label(seg > 0, structure=ndi.generate_binary_structure(3, 1))
            counts = np.bincount(labeled.ravel())[1:]
            voxel_vol = float(np.prod(spacing))
            sizes_mm3 = [float(c * voxel_vol) for c in counts if c >= args.min_voxels]
            n_les = len(sizes_mm3)
            if n_les >= 2:
                multi_focal += 1
            lesion_counts.append(n_les)
            lesion_sizes_mm3.extend(sizes_mm3)

        rows.append({
            "study_id": sid,
            "patient_id": patient_id_of(sid),
            "shape": list(shape) if shape else None,
            "spacing": list(spacing) if spacing else None,
            "n_lesions": n_les,
            "lesion_sizes_mm3_top50": sorted(sizes_mm3, reverse=True)[:50],
            **{f"has_{s}": present[s] for s in SUFFIXES},
        })

    summary = {
        "n_studies": len(rows),
        "n_patients": len({r["patient_id"] for r in rows}),
        "modality_completeness": dict(modality_present),
        "multi_focal_studies": multi_focal,
        "single_lesion_studies": sum(1 for r in rows if r["n_lesions"] == 1),
        "no_lesion_studies": sum(1 for r in rows if r["n_lesions"] == 0),
        "lesions_per_study": {
            "mean": float(np.mean(lesion_counts)) if lesion_counts else 0.0,
            "median": float(np.median(lesion_counts)) if lesion_counts else 0.0,
            "max": int(np.max(lesion_counts)) if lesion_counts else 0,
            "total": int(np.sum(lesion_counts)) if lesion_counts else 0,
        },
        "lesion_size_mm3": {
            "p10": float(np.percentile(lesion_sizes_mm3, 10)) if lesion_sizes_mm3 else 0.0,
            "p50": float(np.percentile(lesion_sizes_mm3, 50)) if lesion_sizes_mm3 else 0.0,
            "p90": float(np.percentile(lesion_sizes_mm3, 90)) if lesion_sizes_mm3 else 0.0,
        },
        "top_shapes": shapes.most_common(5),
        "top_spacings": spacings.most_common(5),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "studies": rows}, indent=2))
    print(json.dumps(summary, indent=2))

    splits = make_patient_grouped_splits([r["study_id"] for r in rows])
    write_splits_csv(splits, args.splits_out)
    print(f"Wrote splits: {args.splits_out} "
          f"(train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])})")


if __name__ == "__main__":
    main()
