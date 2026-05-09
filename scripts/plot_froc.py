"""Plot FROC curves from cached agent state pickles.

For each input "run" (cached state directory + label), computes per-study
recall at a dense set of FP/study thresholds and plots:
  - mean recall (line)
  - 25th/75th percentile recall across studies (shaded band)

Multiple runs go on the same figure for direct visual comparison
(baseline vs SFT v2 vs DPO v1 vs DPO v4 vs ...). The detector-only
baseline is computed inline from each cached state's `non_seed_gt`
+ a fresh per-study probmap (or, if a baseline state dir is provided,
read directly from it).

Usage:
  python scripts/plot_froc.py \
      --runs "DPO v4 (test):runs/state_test_dpo_v4" \
             "DPO v4 (val 16-30):runs/state_v17_dpo_v4" \
             "DPO v1 (val 16-30):runs/state_v13_dpo" \
      --out runs/froc_compare.png
"""
from __future__ import annotations
import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brain_mets_agent.orchestrator.tools.ranker import rank_candidates
from brain_mets_agent.eval.metrics import additional_lesion_recall_at_fp


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


def _per_study_recall_at_fp(per_study, fp: float) -> list[float]:
    """Returns list of per-study recalls at the given FP threshold."""
    from brain_mets_agent.eval.metrics import match_predictions
    out = []
    for r in per_study:
        gt = r["non_seed_gt"]
        if not gt:
            continue
        kept = []
        for p in list(r["preds"]):
            kept.append(p)
            tp = len(match_predictions(kept, gt).matched_gt)
            fp_count = len(kept) - tp
            if fp_count > fp:
                kept.pop()
                break
        tp_final = len(match_predictions(kept, gt).matched_gt)
        out.append(tp_final / len(gt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="One or more 'label:state_dir' specs.")
    ap.add_argument("--out", default="runs/froc_curve.png")
    ap.add_argument("--rank-config",
                    default="prefer_confirmed=False,weights=verdict_heavy,prob_source=detector_only",
                    help="Comma-separated key=value overrides for rank_candidates.")
    ap.add_argument("--fps", default="0.1,0.25,0.5,1,2,4,8,16,32",
                    help="Comma-separated FP/study thresholds for the curve.")
    ap.add_argument("--title", default="FROC: agent recall vs allowed FP/study")
    ap.add_argument("--baseline-jsons", nargs="*", default=[],
                    help="Optional eval_*.json file(s); their `baseline.froc` "
                         "will be overlaid as dashed reference lines.")
    args = ap.parse_args()

    fps = [float(x) for x in args.fps.split(",")]

    weight_presets = {
        "verdict_heavy":    (0.45, 0.20, 0.10, 0.25),
        "balanced":         (0.30, 0.30, 0.15, 0.25),
        "prob_heavy":       (0.20, 0.50, 0.10, 0.20),
        "very_prob_heavy":  (0.10, 0.70, 0.10, 0.10),
    }
    rank_kwargs: dict = {}
    for kv in args.rank_config.split(","):
        k, v = kv.split("=", 1)
        if k == "weights":
            rank_kwargs["weights"] = weight_presets.get(v, weight_presets["very_prob_heavy"])
        elif k == "prefer_confirmed":
            rank_kwargs["prefer_confirmed"] = (v.lower() == "true")
        else:
            rank_kwargs[k] = v
    print(f"Rank config: {rank_kwargs}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))

    all_summaries = {}
    for spec in args.runs:
        if ":" not in spec:
            print(f"  [skip] {spec}: missing 'label:dir'")
            continue
        label, sd = spec.split(":", 1)
        states = _load_states(Path(sd))
        if not states:
            print(f"  [skip] {label}: no .pkl in {sd}")
            continue
        per_study = []
        for rec in states:
            ranked = rank_candidates(rec["candidates"], **rank_kwargs)
            per_study.append({"study_id": rec["study_id"],
                              "preds": ranked,
                              "non_seed_gt": rec["non_seed_gt"]})

        means = []
        p25s = []
        p75s = []
        for fp in fps:
            recalls = _per_study_recall_at_fp(per_study, fp)
            means.append(float(np.mean(recalls)) if recalls else 0.0)
            p25s.append(float(np.percentile(recalls, 25)) if recalls else 0.0)
            p75s.append(float(np.percentile(recalls, 75)) if recalls else 0.0)

        ax.plot(fps, means, marker="o", label=f"{label} (n={len(per_study)})")
        ax.fill_between(fps, p25s, p75s, alpha=0.15)
        all_summaries[label] = {"mean": means, "p25": p25s, "p75": p75s, "n": len(per_study)}
        print(f"  {label} (n={len(per_study)}): "
              f"r@1={means[fps.index(1)]:.3f}, r@2={means[fps.index(2)]:.3f}, "
              f"r@5={means[fps.index(8) if 5 not in fps else fps.index(5)]:.3f}")

    # Overlay baseline FROC curves from eval_*.json files
    for bj in args.baseline_jsons:
        try:
            d = json.loads(Path(bj).read_text())
        except Exception as e:
            print(f"  [skip] {bj}: {e}")
            continue
        froc = d.get("baseline", {}).get("froc", {})
        n = d.get("n_studies_evaluated", "?")
        if not froc:
            continue
        bfps = sorted(float(k) for k in froc)
        bvals = [froc[str(k) if str(k) in froc else (str(int(k)) if k == int(k) else str(k))] for k in bfps]
        # Label from filename
        label = f"Detector baseline ({Path(bj).stem}, n={n})"
        ax.plot(bfps, bvals, "--", marker="x", label=label, alpha=0.8)

    ax.set_xscale("log")
    ax.set_xlabel("Allowed false positives per study (log scale)")
    ax.set_ylabel("Mean per-study recall")
    ax.set_title(args.title)
    ax.set_ylim(0, 1)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")

    # Also dump JSON summary
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps({"fps": fps, "runs": all_summaries}, indent=2))
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
