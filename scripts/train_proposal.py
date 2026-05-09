"""Train Swin-UNETR as the lesion-proposal model on UCSF-BMSR.

Smoke run (verify the pipeline on a small subset):
  python scripts/train_proposal.py \
      --root C:/Users/User/Documents/UCSF_BMSR/UCSF_BrainMetastases_TRAIN \
      --splits brain_mets_agent/data/splits.csv \
      --epochs 1 --train-subset 4 --val-subset 1 \
      --batch 1 --patch 64 64 64 \
      --out ckpts/swin_smoke

Full run (default hyperparams):
  python scripts/train_proposal.py --root <root> --splits <csv> --epochs 200
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast, GradScaler

from brain_mets_agent.data.training import make_loaders


def lesion_recall_from_pred(pred_prob: np.ndarray, label: np.ndarray,
                            threshold: float = 0.5, min_voxels: int = 10) -> float:
    """Per-volume lesion-instance recall (CC-level, distance-based)."""
    from scipy import ndimage as ndi
    pred = pred_prob >= threshold
    if not pred.any() and not label.any():
        return 1.0
    if not label.any():
        return 0.0
    gt_lab, n_gt = ndi.label(label > 0, structure=ndi.generate_binary_structure(3, 1))
    pr_lab, _n_pr = ndi.label(pred, structure=ndi.generate_binary_structure(3, 1))
    matched = 0
    for gi in range(1, n_gt + 1):
        gt_mask = gt_lab == gi
        if gt_mask.sum() < min_voxels:
            n_gt -= 1
            continue
        if (pr_lab[gt_mask] > 0).any():
            matched += 1
    return matched / n_gt if n_gt > 0 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", default="ckpts/swin")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--patch", type=int, nargs=3, default=[96, 96, 96])
    ap.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.5])
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--feature-size", type=int, default=48)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--train-subset", type=int, default=None)
    ap.add_argument("--val-subset", type=int, default=None)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    best_ckpt = out_dir / "best.pt"
    last_ckpt = out_dir / "last.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else ''})")

    train_loader, val_loader = make_loaders(
        root=args.root, splits_csv=args.splits,
        spacing=tuple(args.spacing), patch=tuple(args.patch),
        train_batch=args.batch, num_workers=args.num_workers,
        train_subset=args.train_subset, val_subset=args.val_subset,
    )
    print(f"Train studies: {len(train_loader.dataset)}  Val studies: {len(val_loader.dataset)}")

    from monai.networks.nets import SwinUNETR
    from monai.losses import DiceCELoss
    from monai.inferers import sliding_window_inference

    model = SwinUNETR(
        in_channels=4, out_channels=2, feature_size=args.feature_size,
    ).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = GradScaler("cuda", enabled=args.amp)

    history = []
    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=args.amp):
                logits = model(image)
                loss = loss_fn(logits, label)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        sched.step()
        train_loss = float(np.mean(losses)) if losses else float("nan")
        epoch_time = time.time() - t0
        print(f"[ep {epoch:03d}] train_loss={train_loss:.4f}  lr={optim.param_groups[0]['lr']:.2e}  t={epoch_time:.1f}s")

        record = {"epoch": epoch, "train_loss": train_loss, "epoch_time_s": epoch_time}

        if epoch % args.val_every == 0 or epoch == args.epochs:
            model.eval()
            recalls, dices = [], []
            with torch.no_grad():
                for batch in val_loader:
                    image = batch["image"].to(device, non_blocking=True)
                    label = batch["label"].to(device, non_blocking=True)
                    with autocast("cuda", enabled=args.amp):
                        logits = sliding_window_inference(
                            inputs=image, roi_size=tuple(args.patch),
                            sw_batch_size=2, predictor=model, overlap=0.5,
                        )
                        prob = torch.softmax(logits, dim=1)[:, 1]
                    pred = (prob >= 0.5).float()
                    inter = (pred * label[:, 0]).sum()
                    denom = pred.sum() + label[:, 0].sum()
                    dice = float((2 * inter / (denom + 1e-6)).cpu()) if denom > 0 else 1.0
                    dices.append(dice)
                    recalls.append(lesion_recall_from_pred(
                        prob[0].cpu().numpy(), label[0, 0].cpu().numpy(),
                    ))
            mean_dice = float(np.mean(dices)) if dices else 0.0
            mean_recall = float(np.mean(recalls)) if recalls else 0.0
            record.update({"val_dice": mean_dice, "val_lesion_recall": mean_recall})
            print(f"           val_dice={mean_dice:.4f}  val_lesion_recall={mean_recall:.4f}")

            score = mean_recall
            if score > best_val:
                best_val = score
                torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                            "val_dice": mean_dice, "val_lesion_recall": mean_recall},
                           best_ckpt)
                print(f"           -> saved {best_ckpt}")

        history.append(record)
        torch.save({"epoch": epoch, "state_dict": model.state_dict()}, last_ckpt)
        metrics_path.write_text(json.dumps(history, indent=2))

    print(f"Done. Best val_lesion_recall={best_val:.4f}. Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
