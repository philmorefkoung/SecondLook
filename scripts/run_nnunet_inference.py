"""Pre-batch BMSR-paper nnU-Net v1 inference on a study split.

Stages each study's 4 modalities into nnUNet v1's expected file naming
(<id>_0000=T1post, _0001=T1pre, _0002=FLAIR, _0003=subtraction), then runs
`nnUNet_predict -z` via subprocess into the `nnunet1` conda env. Output is
the per-study softmax .npz saved alongside the segmentation .nii.gz, which
`brain_mets_agent.models.NNUNetProbmapCache` reads on demand inside the
agent eval flow.

Run nnUNet_install_pretrained_model_from_zip ONCE before first use:

    conda run -n nnunet1 \\
        env RESULTS_FOLDER=ckpts/bmsr_nnunet/results \\
        nnUNet_install_pretrained_model_from_zip \\
        ckpts/bmsr_nnunet/Metastases_T1post_T1pre_FLAIR_Subtraction.zip

Then:

    python scripts/run_nnunet_inference.py \\
        --root C:/Users/User/Documents/UCSF_BMSR/UCSF_BrainMetastases_TRAIN \\
        --splits brain_mets_agent/data/splits.csv --split test \\
        --task Task115_Metastases_All \\
        --results-folder ckpts/bmsr_nnunet/results \\
        --staging runs/nnunet_input_test \\
        --out runs/nnunet_probmap_test
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Modality-to-channel-index per the BMSR-paper model:
#   _0000 T1post, _0001 T1pre, _0002 FLAIR, _0003 subtraction
CHANNEL_MAP = [
    ("T1post",      "0000"),
    ("T1pre",       "0001"),
    ("FLAIR",       "0002"),
    ("subtraction", "0003"),
]


def read_split_ids(splits_csv: Path, split: str) -> list[str]:
    import csv
    out = []
    with splits_csv.open() as f:
        for row in csv.DictReader(f):
            if row["split"].strip() == split:
                out.append(row["study_id"].strip())
    return out


def stage_studies(root: Path, study_ids: list[str], staging: Path) -> list[str]:
    """Copy each study's 4 modality files into nnUNet v1 input naming."""
    staging.mkdir(parents=True, exist_ok=True)
    staged = []
    for sid in study_ids:
        sdir = root / sid
        if not sdir.exists():
            print(f"  [skip] {sid}: not found at {sdir}")
            continue
        ok = True
        for mod, ch in CHANNEL_MAP:
            src = sdir / f"{sid}_{mod}.nii.gz"
            if not src.exists():
                print(f"  [skip] {sid}: missing {mod}")
                ok = False
                break
            dst = staging / f"{sid}_{ch}.nii.gz"
            if not dst.exists():
                shutil.copy2(src, dst)
        if ok:
            staged.append(sid)
    return staged


def run_predict(staging: Path, out: Path, task: str, results_folder: Path,
                conda_env: str, model: str = "3d_fullres", folds: str = "all",
                use_tta: bool = False):
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["RESULTS_FOLDER"] = str(results_folder)
    env["nnUNet_raw_data_base"] = str(staging.parent / "_nnunet_raw_unused")
    env["nnUNet_preprocessed"] = str(staging.parent / "_nnunet_pre_unused")
    cmd = [
        "conda", "run", "--no-capture-output", "-n", conda_env,
        "nnUNet_predict",
        "-i", str(staging),
        "-o", str(out),
        "-t", task,
        "-m", model,
        "-f", folds,
        "-z",
        # Windows multiprocessing.spawn doesn't survive nnUNet v1's pool;
        # serial preprocessing is the safe path. Lambdas in nnUNet's
        # network code (nd_softmax.softmax_helper, nnUNetTrainerV2's
        # initialize_network, generic_UNet.upscale_logits_ops) had to be
        # converted to nn.Identity() / def helpers in the installed
        # package for pickling to succeed across Windows process spawn.
        "--num_threads_preprocessing", "1",
        "--num_threads_nifti_save", "1",
    ]
    if not use_tta:
        # Default off: TTA = mirror augmentation along all 3 axes (8x slower
        # but typically +1-3pt small-lesion sensitivity).
        cmd.append("--disable_tta")
    print("  $ " + " ".join(cmd))
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        sys.exit(f"nnUNet_predict failed (exit {res.returncode})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="UCSF-BMSR dataset root (contains study subdirs).")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--task", default="Task115_Metastases_All")
    ap.add_argument("--results-folder", required=True,
                    help="nnUNet RESULTS_FOLDER (where the model was installed).")
    ap.add_argument("--staging", required=True,
                    help="Temp dir for staged input volumes (nnUNet naming).")
    ap.add_argument("--out", required=True,
                    help="Output dir; <study_id>.npz softmax will land here.")
    ap.add_argument("--conda-env", default="nnunet1")
    ap.add_argument("--model", default="3d_fullres")
    ap.add_argument("--folds", default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--enable-tta", action="store_true",
                    help="Enable test-time augmentation (mirror along all 3 axes). "
                         "8x slower but typically +1-3pt small-lesion sensitivity.")
    args = ap.parse_args()

    splits_csv = Path(args.splits)
    root = Path(args.root)
    staging = Path(args.staging)
    out = Path(args.out)

    ids = read_split_ids(splits_csv, args.split)
    if args.limit:
        ids = ids[:args.limit]
    print(f"Split {args.split}: {len(ids)} studies")

    staged = stage_studies(root, ids, staging)
    print(f"Staged {len(staged)} studies into {staging}")

    run_predict(staging, out, args.task, Path(args.results_folder),
                args.conda_env, model=args.model, folds=args.folds,
                use_tta=args.enable_tta)
    print(f"Done. Softmax .npz files in {out}")


if __name__ == "__main__":
    main()
