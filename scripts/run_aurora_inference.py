"""Run BrainLesion AURORA brain-mets segmentation on a study split.

AURORA (https://github.com/BrainLesion/AURORA) is a pure-Python brain-tumor /
brain-mets segmentation model. Pip-installable, no Docker required (works on
Windows). Trained on multi-site brain mets data; explicitly multi-modal with
auto-selection of weight variants when modalities are missing.

Modality mapping (UCSF / Stanford -> AURORA):
  UCSF:     T1pre -> t1, T1post -> t1c, T2Synth -> t2, FLAIR -> fla
  Stanford: t1_pre -> t1, t1_gd -> t1c,  (no T2 -> omit), flair -> fla

Output: per-study segmentation NIfTI in original image space, multi-label
(0 = bg, 1 = enhancing tumor, 2 = surrounding non-enhancing FLAIR
hyperintensity / edema). Our `NNUNetProbmapCache` adapter reads any binary
seg via `>0` test, so the same cache works for AURORA outputs.

Usage:
  PYTHONIOENCODING=utf-8 python scripts/run_aurora_inference.py \\
      --root C:/Users/User/Documents/UCSF_BMSR/UCSF_BrainMetastases_TRAIN \\
      --splits brain_mets_agent/data/splits.csv --split test \\
      --modality-template "ucsf" \\
      --out runs/aurora_probmap_test
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from pathlib import Path


MODALITY_TEMPLATES = {
    # UCSF-BMSR-staged style: <root>/<sid>/<sid>_<modality>.nii.gz
    "ucsf": {
        "t1":  "{sid}/{sid}_T1pre.nii.gz",
        "t1c": "{sid}/{sid}_T1post.nii.gz",
        "t2":  "{sid}/{sid}_T2Synth.nii.gz",
        "fla": "{sid}/{sid}_FLAIR.nii.gz",
    },
    # Stanford-staged: same UCSF naming after our stage_stanford.py, but no T2
    "stanford": {
        "t1":  "{sid}/{sid}_T1pre.nii.gz",
        "t1c": "{sid}/{sid}_T1post.nii.gz",
        # T2 absent; AURORA auto-selects a 3-modality model variant
        "fla": "{sid}/{sid}_FLAIR.nii.gz",
    },
}


def read_split_ids(splits_csv: Path, split: str) -> list[str]:
    with splits_csv.open() as f:
        return [row["study_id"].strip() for row in csv.DictReader(f)
                if row["split"].strip() == split]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--modality-template", choices=list(MODALITY_TEMPLATES),
                    required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tta", action="store_true",
                    help="Enable TTA (slower).")
    args = ap.parse_args()

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    from brainles_aurora.inferer import AuroraInferer, AuroraInfererConfig

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    template = MODALITY_TEMPLATES[args.modality_template]

    sids = read_split_ids(Path(args.splits), args.split)
    if args.limit:
        sids = sids[:args.limit]
    print(f"Aurora inference on {len(sids)} {args.split} studies "
          f"(template={args.modality_template}, tta={args.tta})")

    cfg = AuroraInfererConfig(tta=args.tta)
    inferer = AuroraInferer(cfg)

    t_start = time.time()
    skipped: list[str] = []
    succeeded: list[str] = []
    for i, sid in enumerate(sids, start=1):
        out_seg = out / f"{sid}.nii.gz"
        if out_seg.exists():
            succeeded.append(sid)
            continue
        kwargs: dict[str, str] = {}
        ok = True
        for mod, rel in template.items():
            p = root / rel.format(sid=sid)
            if not p.exists():
                print(f"  [skip] {sid}: missing {mod} at {p}")
                ok = False
                break
            kwargs[mod] = str(p)
        if not ok:
            skipped.append(sid)
            continue
        try:
            inferer.infer(segmentation_file=str(out_seg), **kwargs)
            succeeded.append(sid)
        except Exception as e:
            print(f"  [error] {sid}: {e}")
            skipped.append(sid)
            continue
        if i % 5 == 0 or i == len(sids):
            elapsed = time.time() - t_start
            print(f"  [{i}/{len(sids)}] {sid}  "
                  f"({elapsed:.0f}s elapsed, "
                  f"{elapsed/i:.1f}s/study avg)")

    print(f"\nDone. Succeeded: {len(succeeded)}, Skipped: {len(skipped)}")
    if skipped:
        print("Skipped:", skipped[:10], "..." if len(skipped) > 10 else "")
    print(f"Outputs in: {out}")


if __name__ == "__main__":
    main()
