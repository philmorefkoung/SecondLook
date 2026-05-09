"""Stage Stanford BrainMetShare into UCSF-style folder layout for our pipeline.

Stanford BrainMetShare ships per-study folders `Mets_XXX/` containing:
  t1_pre.nii.gz, t1_gd.nii.gz, flair.nii.gz, bravo.nii.gz, seg.nii.gz (105 of 156)

Our existing data loader expects UCSF-style paths:
  <root>/<sid>/<sid>_T1pre.nii.gz, _T1post.nii.gz, _FLAIR.nii.gz,
              _subtraction.nii.gz, _seg.nii.gz

Modality mapping:
  Stanford t1_pre -> T1pre
  Stanford t1_gd  -> T1post     (T1 with gadolinium = post-contrast)
  Stanford flair  -> FLAIR
  Stanford bravo  -> (unused, nnU-Net Task115 takes only the 4 above)
  computed        -> subtraction = T1post - T1pre  (modalities are co-registered;
                                                    raw subtraction matches UCSF's
                                                    pre-computed `_subtraction`
                                                    file convention)
  Stanford seg    -> seg                          (binary {0, 1})

Studies without seg.nii.gz are skipped (we need GT for evaluation metrics).

Usage:
  python scripts/stage_stanford.py \\
      --in C:/Users/User/Documents/BrainMetShare \\
      --out runs/stanford_staged
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np


MODALITY_MAP = [
    ("t1_pre.nii.gz", "T1pre.nii.gz"),
    ("t1_gd.nii.gz",  "T1post.nii.gz"),
    ("flair.nii.gz",  "FLAIR.nii.gz"),
    ("seg.nii.gz",    "seg.nii.gz"),
]


def stage_study(src_dir: Path, dst_dir: Path, sid: str) -> bool:
    """Stage one study. Returns True if seg present and full set staged."""
    seg_src = src_dir / "seg.nii.gz"
    if not seg_src.exists():
        return False
    for mod in ("t1_pre.nii.gz", "t1_gd.nii.gz", "flair.nii.gz"):
        if not (src_dir / mod).exists():
            return False

    dst_dir.mkdir(parents=True, exist_ok=True)
    for src_name, dst_suffix in MODALITY_MAP:
        src = src_dir / src_name
        dst = dst_dir / f"{sid}_{dst_suffix}"
        if not dst.exists():
            shutil.copy2(src, dst)

    sub_path = dst_dir / f"{sid}_subtraction.nii.gz"
    if not sub_path.exists():
        t1pre  = nib.load(str(src_dir / "t1_pre.nii.gz"))
        t1post = nib.load(str(src_dir / "t1_gd.nii.gz"))
        sub_data = (np.asanyarray(t1post.dataobj).astype(np.float32)
                    - np.asanyarray(t1pre.dataobj).astype(np.float32))
        sub_img = nib.Nifti1Image(sub_data, t1post.affine, t1post.header)
        nib.save(sub_img, str(sub_path))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src_root", required=True)
    ap.add_argument("--out", dest="dst_root", required=True)
    args = ap.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    study_dirs = sorted(p for p in src_root.iterdir()
                         if p.is_dir() and p.name.startswith("Mets_"))
    print(f"Found {len(study_dirs)} study folders under {src_root}")

    staged = []
    skipped = []
    for src_dir in study_dirs:
        sid = src_dir.name
        dst_dir = dst_root / sid
        ok = stage_study(src_dir, dst_dir, sid)
        if ok:
            staged.append(sid)
            if len(staged) % 20 == 0:
                print(f"  staged {len(staged)} so far ({sid})")
        else:
            skipped.append(sid)
    print(f"\nStaged: {len(staged)}  ·  Skipped (no seg): {len(skipped)}")
    print(f"Output: {dst_root}")

    splits_addendum = dst_root / "stanford_split.csv"
    with splits_addendum.open("w", newline="") as f:
        f.write("study_id,patient_id,split\n")
        for sid in staged:
            f.write(f"{sid},{sid},stanford_external\n")
    print(f"Wrote split CSV: {splits_addendum} ({len(staged)} rows)")


if __name__ == "__main__":
    main()
