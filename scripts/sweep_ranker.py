"""Sweep ranker hyperparameters offline using cached per-study agent state.

Workflow:
  1. Run scripts/evaluate.py with --save-state-dir runs/state_v3/ once.
     This caches per-study TrackedCandidate state (after all VLM calls are made).
  2. Run this script. It loads each pickle and re-ranks under each config in
     the sweep grid; metrics (FROC, recall@FP, top-k, MRR) are computed per
     config without any new VLM call.

Usage:
  python scripts/sweep_ranker.py \
      --state-dir runs/state_v3 \
      --out runs/sweep_ranker_v3.json
"""
from __future__ import annotations
import argparse
import json
import pickle
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

from brain_mets_agent.orchestrator.tools.ranker import rank_candidates
from brain_mets_agent.eval.metrics import (
    additional_lesion_recall_at_fp, froc, top_k_recall, mean_reciprocal_rank,
    match_predictions,
)


# ---------- gt-shaped wrapper ----------

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
        # convert non_seed_gt dicts back to objects with attributes
        rec["non_seed_gt"] = [_GTLite(**g) for g in rec["non_seed_gt"]]
        out.append(rec)
    return out


# ---------- sweep ----------

def _eval_config(states, cfg) -> dict:
    per_study = []
    for rec in states:
        ranked = rank_candidates(
            rec["candidates"],
            prefer_confirmed=cfg.get("prefer_confirmed", True),
            weights=cfg.get("weights", (0.45, 0.20, 0.10, 0.25)),
            verdict_scores={"confirm": cfg["verdict_confirm"],
                            "uncertain": cfg["verdict_uncertain"],
                            "reject": cfg.get("verdict_reject", 0.0)},
            mimic_penalties={"low": 0.0,
                              "medium": cfg["mimic_medium"],
                              "high": cfg["mimic_high"]},
            prob_source=cfg.get("prob_source", "vlm_else_detector"),
        )
        per_study.append({"study_id": rec["study_id"],
                          "preds": ranked,
                          "non_seed_gt": rec["non_seed_gt"]})
    metrics = {
        "recall_at_fp_0_5": additional_lesion_recall_at_fp(per_study, 0.5),
        "recall_at_fp_1": additional_lesion_recall_at_fp(per_study, 1),
        "recall_at_fp_2": additional_lesion_recall_at_fp(per_study, 2),
        "recall_at_fp_5": additional_lesion_recall_at_fp(per_study, 5),
        "recall_at_fp_10": additional_lesion_recall_at_fp(per_study, 10),
        "top5_recall": top_k_recall(per_study, 5),
        "top10_recall": top_k_recall(per_study, 10),
        "mrr": mean_reciprocal_rank(per_study),
    }
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--out", default="runs/sweep_ranker.json")
    ap.add_argument("--rank-by", default="recall_at_fp_2",
                    help="Sort the result table by this metric.")
    args = ap.parse_args()

    states = _load_states(Path(args.state_dir))
    print(f"Loaded {len(states)} cached studies from {args.state_dir}")
    if not states:
        raise SystemExit("no .pkl found")

    # Grid: focuses on the levers that actually move recall on a small slice.
    # `prefer_confirmed=True` filters all rejected candidates; with `False` they
    # stay in the ranking at low verdict_score but their detector prob still
    # contributes - so VLM-false-rejects can still be surfaced at high FP budget.
    weight_presets = {
        "verdict_heavy":    (0.45, 0.20, 0.10, 0.25),   # current default
        "balanced":         (0.30, 0.30, 0.15, 0.25),
        "prob_heavy":       (0.20, 0.50, 0.10, 0.20),
        "very_prob_heavy":  (0.10, 0.70, 0.10, 0.10),
    }
    grid = []
    for prefer, vr, vu, mh, w_name, ps in product(
        [True, False],                             # prefer_confirmed
        [0.0, 0.25],                               # verdict_reject (only used when prefer=False)
        [0.5, 0.85],                               # verdict_uncertain
        [0.0, 0.20, 0.40],                         # mimic_high
        list(weight_presets),
        ["vlm_else_detector", "detector_only", "average"],   # prob_source
    ):
        if prefer and vr != 0.0:
            continue   # vr unused when prefer=True; dedupe
        grid.append({"prefer_confirmed": prefer,
                      "verdict_confirm": 1.0,
                      "verdict_uncertain": vu,
                      "verdict_reject": vr,
                      "mimic_medium": mh / 2,
                      "mimic_high": mh,
                      "weights": weight_presets[w_name],
                      "weights_name": w_name,
                      "prob_source": ps})
    print(f"Sweeping {len(grid)} configs")

    rows = []
    for cfg in grid:
        m = _eval_config(states, cfg)
        rows.append({**cfg, **m})

    rows.sort(key=lambda r: -r[args.rank_by])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"n_states": len(states),
                                     "n_configs": len(rows),
                                     "rank_by": args.rank_by,
                                     "results": rows}, indent=2, default=str))

    print(f"\nTop 15 by {args.rank_by}:")
    hdr = f"  {'pref':>4} {'wts':>16} {'prob':>16} {'vu':>4} {'mh':>4} | {'r@.5':>5} {'r@1':>5} {'r@2':>5} {'r@5':>5} {'r@10':>5} {'top5':>5} {'top10':>5} {'mrr':>5}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows[:15]:
        print(f"  {str(r['prefer_confirmed'])[0]:>4} {r['weights_name']:>16} {r['prob_source']:>16} "
              f"{r['verdict_uncertain']:>4} {r['mimic_high']:>4} | "
              f"{r['recall_at_fp_0_5']:>5.3f} {r['recall_at_fp_1']:>5.3f} "
              f"{r['recall_at_fp_2']:>5.3f} {r['recall_at_fp_5']:>5.3f} "
              f"{r['recall_at_fp_10']:>5.3f} {r['top5_recall']:>5.3f} "
              f"{r['top10_recall']:>5.3f} {r['mrr']:>5.3f}")

    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
