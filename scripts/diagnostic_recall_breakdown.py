"""Stratified recall diagnostic: per-study by GT count, per-lesion by size.

Loads cached agent state, applies the configured ranker, then for each study
asks: at FP=5/study budget, which non-seed GT lesions were matched and which
were missed? Aggregates two ways:
  1. Per-study recall, bucketed by total non-seed GT count
  2. Per-lesion match status (binary) bucketed by lesion volume_mm3

Reports stratified means + plots, plus a CSV per-lesion table.

Usage:
  python scripts/diagnostic_recall_breakdown.py \
      --state-dir runs/state_test_dpo_v4 \
      --out-prefix runs/diagnostic_test
"""
from __future__ import annotations
import argparse
import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brain_mets_agent.orchestrator.tools.ranker import rank_candidates
from brain_mets_agent.eval.metrics import match_predictions


@dataclass
class _GTLite:
    centroid_vox: tuple
    voxel_count: int
    volume_mm3: float
    mask: object


def _load_states(state_dir: Path) -> list[dict]:
    out = []
    for p in sorted(state_dir.glob("*.pkl")):
        with p.open("rb") as f:
            rec = pickle.load(f)
        rec["non_seed_gt"] = [_GTLite(**g) for g in rec["non_seed_gt"]]
        out.append(rec)
    return out


def _kept_at_fp_budget(preds, gt, fp_budget: float) -> list:
    """Return the list of candidates kept under the FP budget."""
    kept = []
    for p in list(preds):
        kept.append(p)
        tp = len(match_predictions(kept, gt).matched_gt)
        fp = len(kept) - tp
        if fp > fp_budget:
            kept.pop()
            break
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--out-prefix", default="runs/diagnostic")
    ap.add_argument("--rank-config",
                    default="prefer_confirmed=False,weights=verdict_heavy,prob_source=detector_only")
    ap.add_argument("--fp-budget", type=float, default=5.0)
    args = ap.parse_args()

    weight_presets = {
        "verdict_heavy": (0.45, 0.20, 0.10, 0.25),
        "balanced": (0.30, 0.30, 0.15, 0.25),
        "prob_heavy": (0.20, 0.50, 0.10, 0.20),
        "very_prob_heavy": (0.10, 0.70, 0.10, 0.10),
    }
    rank_kwargs: dict = {}
    for kv in args.rank_config.split(","):
        k, v = kv.split("=", 1)
        if k == "weights":
            rank_kwargs["weights"] = weight_presets.get(v, weight_presets["verdict_heavy"])
        elif k == "prefer_confirmed":
            rank_kwargs["prefer_confirmed"] = (v.lower() == "true")
        else:
            rank_kwargs[k] = v
    print(f"Rank config: {rank_kwargs}")
    print(f"FP budget: {args.fp_budget}")

    states = _load_states(Path(args.state_dir))
    print(f"Loaded {len(states)} cached studies")

    per_study_rows = []
    per_lesion_rows = []
    for rec in states:
        gt = rec["non_seed_gt"]
        if not gt:
            continue
        ranked = rank_candidates(rec["candidates"], **rank_kwargs)
        kept = _kept_at_fp_budget(ranked, gt, args.fp_budget)
        m = match_predictions(kept, gt)
        matched_gt_idx = set(m.matched_gt)
        n_total = len(gt)
        n_match = len(matched_gt_idx)
        per_study_rows.append({
            "study_id": rec["study_id"],
            "n_gt": n_total,
            "n_matched": n_match,
            "recall": n_match / n_total,
            "n_kept_preds": len(kept),
        })
        for gi, g in enumerate(gt):
            per_lesion_rows.append({
                "study_id": rec["study_id"],
                "gt_idx": gi,
                "volume_mm3": float(g.volume_mm3),
                "voxel_count": int(g.voxel_count),
                "matched": (gi in matched_gt_idx),
            })

    # Per-study by burden bucket
    by_burden: dict[str, list[float]] = {}
    burden_buckets = [(1, 1, "1"), (2, 3, "2-3"), (4, 6, "4-6"),
                      (7, 12, "7-12"), (13, 25, "13-25"), (26, 1000, "26+")]
    for r in per_study_rows:
        for lo, hi, label in burden_buckets:
            if lo <= r["n_gt"] <= hi:
                by_burden.setdefault(label, []).append(r["recall"])
                break

    # Per-lesion by size bucket
    by_size: dict[str, list[bool]] = {}
    size_buckets = [(0, 30, "<30 mm3 (~<3.5 mm dia)"),
                    (30, 100, "30-100"),
                    (100, 300, "100-300"),
                    (300, 1000, "300-1000"),
                    (1000, 1e9, ">1000 (~>12 mm dia)")]
    for r in per_lesion_rows:
        for lo, hi, label in size_buckets:
            if lo <= r["volume_mm3"] < hi:
                by_size.setdefault(label, []).append(r["matched"])
                break

    print("\n=== PER-STUDY recall, bucketed by total non-seed GT count ===")
    print(f"  {'bucket':>10}  {'n_studies':>10}  {'mean_recall':>12}  {'median_n_gt':>11}")
    for _, _, label in burden_buckets:
        recalls = by_burden.get(label, [])
        if not recalls: continue
        ns = [r["n_gt"] for r in per_study_rows
              if any(b == label for b in [_b[2] for _b in burden_buckets if _b[0] <= r["n_gt"] <= _b[1]])]
        print(f"  {label:>10}  {len(recalls):>10}  {np.mean(recalls):>12.3f}  "
              f"{int(np.median(ns)):>11}")

    print("\n=== PER-LESION match rate, bucketed by lesion volume ===")
    print(f"  {'bucket':>30}  {'n_lesions':>10}  {'match_rate':>11}")
    for _, _, label in size_buckets:
        ms = by_size.get(label, [])
        if not ms: continue
        print(f"  {label:>30}  {len(ms):>10}  {np.mean(ms):>11.3f}")

    # Plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Recall vs burden
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [b[2] for b in burden_buckets if b[2] in by_burden]
    means = [np.mean(by_burden[l]) for l in labels]
    counts = [len(by_burden[l]) for l in labels]
    bars = ax.bar(labels, means, color="#2c7fb8")
    ax.set_xlabel("Total non-seed GT lesions per study")
    ax.set_ylabel(f"Mean recall @ FP={args.fp_budget}/study")
    ax.set_ylim(0, 1)
    ax.set_title(f"Recall vs lesion burden (n={sum(counts)} studies)")
    for b, n, m in zip(bars, counts, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.02,
                f"n={n}\n{m:.2f}", ha="center", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_recall_by_burden.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_prefix}_recall_by_burden.png")

    # Match rate vs lesion size
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [b[2] for b in size_buckets if b[2] in by_size]
    rates = [np.mean(by_size[l]) for l in labels]
    counts = [len(by_size[l]) for l in labels]
    bars = ax.bar(labels, rates, color="#7fbc41")
    ax.set_xlabel("Lesion volume (mm^3)")
    ax.set_ylabel(f"Per-lesion match rate @ FP={args.fp_budget}/study")
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-lesion sensitivity vs size (n={sum(counts)} lesions)")
    for b, n, r in zip(bars, counts, rates):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.02,
                f"n={n}\n{r:.2f}", ha="center", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_match_by_size.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_prefix}_match_by_size.png")

    # Save CSVs + summary JSON
    csv_path = Path(f"{out_prefix}_per_lesion.csv")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["study_id", "gt_idx", "volume_mm3",
                                            "voxel_count", "matched"])
        w.writeheader()
        for r in per_lesion_rows:
            w.writerow(r)
    print(f"Saved {csv_path}")

    summary = {
        "n_studies": len(per_study_rows),
        "n_lesions": len(per_lesion_rows),
        "fp_budget": args.fp_budget,
        "rank_config": args.rank_config,
        "by_burden": {l: {"n_studies": len(by_burden[l]),
                          "mean_recall": float(np.mean(by_burden[l]))}
                       for l in by_burden},
        "by_size": {l: {"n_lesions": len(by_size[l]),
                        "match_rate": float(np.mean(by_size[l]))}
                     for l in by_size},
    }
    json_path = Path(f"{out_prefix}_summary.json")
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
