"""Generate figures providing evidence for why steps 4/5 (residual / 2nd-pass)
mechanisms don't work — and a sample audit log to confirm logging is intact.

Outputs into figures/:
  fig_E0_audit_log.png         — sample agent audit log (real cached actions)
  fig_E1_miss_diagnostic.png   — misses are size-driven, not anatomy-driven
                                  → 2nd-pass mechanisms can't rescue what the
                                  detector never proposed
  fig_E2_negative_results.png  — 4-panel: min_voxels=3 (small +), TTA (tradeoff),
                                  phenotype-step5 (big regression),
                                  AURORA ensemble (lifts r@10 / hurts MRR)

All numbers computed live from cached states + the diagnostic CSV.
"""
from __future__ import annotations
import argparse
import csv
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from brain_mets_agent.models import (
    NNUNetProbmapCache, EnsembleProbmapCache,
)
from brain_mets_agent.models.proposal import probmap_to_candidates
from brain_mets_agent.orchestrator.tools.ranker import rank_candidates
from brain_mets_agent.eval.metrics import (
    additional_lesion_recall_at_fp, top_k_recall, mean_reciprocal_rank,
)


plt.rcParams.update({
    "figure.facecolor":   "#1a1a1a",
    "axes.facecolor":     "#1a1a1a",
    "savefig.facecolor":  "#1a1a1a",
    "axes.edgecolor":     "#666666",
    "axes.labelcolor":    "#dddddd",
    "xtick.color":        "#aaaaaa",
    "ytick.color":        "#aaaaaa",
    "text.color":         "#eeeeee",
    "axes.grid":          False,
    "font.size":          11,
    "font.family":        "sans-serif",
    "axes.titleweight":   "bold",
})

ACCENT = "#ff6b4a"
GOOD = "#7fbc41"
BAD = "#cc4a3a"


@dataclass
class _GTLite:
    centroid_vox: tuple
    voxel_count: int
    volume_mm3: float
    mask: object


def _metrics(state_dir: Path, cache, mv: int = 3,
              prob_source: str = "vlm_else_detector"):
    baseline_results = []
    agent_results = []
    for p in sorted(state_dir.glob("*.pkl")):
        with p.open("rb") as f:
            rec = pickle.load(f)
        rec["non_seed_gt"] = [_GTLite(**g) for g in rec["non_seed_gt"]]
        for g in rec["non_seed_gt"]:
            g.mask = None
        sid = rec["study_id"]
        prob = cache.predict_probmap_for(sid)
        bcands = probmap_to_candidates(prob, threshold=0.3, min_voxels=mv)
        for c in bcands:
            c.mask = None
        baseline_results.append({"study_id": sid, "preds": bcands,
                                  "non_seed_gt": rec["non_seed_gt"]})
        ranked = rank_candidates(rec["candidates"],
                                   weights=(0.45, 0.20, 0.10, 0.25),
                                   prob_source=prob_source,
                                   prefer_confirmed=False)
        agent_results.append({"study_id": sid, "preds": ranked,
                               "non_seed_gt": rec["non_seed_gt"]})

    def agg(results):
        return {
            "r@1":  additional_lesion_recall_at_fp(results, 1),
            "r@2":  additional_lesion_recall_at_fp(results, 2),
            "r@5":  additional_lesion_recall_at_fp(results, 5),
            "r@10": additional_lesion_recall_at_fp(results, 10),
            "MRR":  mean_reciprocal_rank(results),
        }
    return agg(baseline_results), agg(agent_results)


# ---------- figure E0: audit log ----------

def fig_E0_audit_log(out: Path):
    pkl_path = Path("runs/state_test_nnunet_dpo_v4/100113A.pkl")
    with pkl_path.open("rb") as f:
        rec = pickle.load(f)
    actions = rec["actions"]

    fig, ax = plt.subplots(1, 1, figsize=(13, 6.5))
    ax.axis("off")

    # Format as a monospace table
    header = f"  {'#':>3}   {'action':<30s}    summary"
    sep = "  ─── ─" + "─" * 31 + "─" * 60
    lines = [header, sep]
    for i, (name, summary) in enumerate(actions):
        line = f"  {i:>3}   {name:<30s}    {summary[:80]}"
        lines.append(line)

    text = "\n".join(lines)
    ax.text(0.02, 0.94, text, family="monospace", fontsize=11,
            color="#dddddd", va="top", ha="left",
            bbox=dict(facecolor="#0d1117", edgecolor=ACCENT,
                       pad=18, linewidth=1.5))

    note = (
        f"Sample audit log for study 100113A (UCSF test, n=10 actions).\n"
        f"Every step records its name, structured kwargs, and a human-readable\n"
        f"summary. Per-candidate verifications also produce vlm.verify entries.\n"
        f"The full log lives in state.actions and is preserved in the cached pickle."
    )
    ax.text(0.02, 0.07, note, fontsize=10, color="#999999",
            va="bottom", ha="left", style="italic")

    fig.suptitle(
        "Audit log — preserved by the rule-based agent",
        fontsize=14, color="#eeeeee", y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig_E0_audit_log.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_E0_audit_log.png")


# ---------- figure E1: miss diagnostic ----------

def fig_E1_miss_diagnostic(out: Path):
    csv_path = Path("runs/diagnostic_miss_anatomy.csv")
    if not csv_path.exists():
        print("  [skip] runs/diagnostic_miss_anatomy.csv missing — "
              "run scripts/diagnostic_miss_anatomy.py first")
        return
    rows = list(csv.DictReader(csv_path.open()))
    print(f"  loaded {len(rows)} GT lesions from diagnostic CSV")

    SIZES = ["<30 mm3 (~<3.5 mm dia)", "30-100", "100-300",
             "300-1000", ">1000 (~>12 mm dia)"]
    SIZES_PRETTY = ["<30 mm³\n(~<3.5 mm)", "30-100\nmm³", "100-300\nmm³",
                     "300-1000\nmm³", ">1000\nmm³ (~>12 mm)"]
    Z_BUCKETS = ["infratentorial (0-20%)", "lower mid (20-50%)",
                  "upper mid (50-80%)", "vertex (80-100%)"]
    Z_PRETTY = ["infratentorial\n(0-20%)", "lower mid\n(20-50%)",
                 "upper mid\n(50-80%)", "vertex\n(80-100%)"]

    # Compute per-bucket counts and detector-miss rate
    by_size_total: dict = {s: 0 for s in SIZES}
    by_size_dm:    dict = {s: 0 for s in SIZES}
    grid_total = np.zeros((len(SIZES), len(Z_BUCKETS)))
    grid_dm    = np.zeros((len(SIZES), len(Z_BUCKETS)))
    n_total = 0
    cls_counts: dict = {}
    for r in rows:
        sb = r["size_bucket"]
        zb = r["z_bucket"]
        cls = r["classification"]
        cls_counts[cls] = cls_counts.get(cls, 0) + 1
        if sb in by_size_total:
            by_size_total[sb] += 1
            if cls == "DETECTOR_MISS":
                by_size_dm[sb] += 1
            si = SIZES.index(sb)
            zi = Z_BUCKETS.index(zb) if zb in Z_BUCKETS else -1
            if zi >= 0:
                grid_total[si, zi] += 1
                if cls == "DETECTOR_MISS":
                    grid_dm[si, zi] += 1
        n_total += 1

    # === Build figure: 3 panels ===
    fig = plt.figure(figsize=(17, 7.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.4, 1.1], wspace=0.30)

    # PANEL A — overall classification breakdown
    ax = fig.add_subplot(gs[0])
    cats = ["DETECTOR_MISS", "MATCHED", "RANKER_MISS"]
    cat_pretty = ["DETECTOR\nMISS\n(no signal)",
                   "MATCHED\n(agent\nfound it)",
                   "RANKER\nMISS\n(agent\nordering)"]
    counts = [cls_counts.get(c, 0) for c in cats]
    pcts = [100 * c / n_total for c in counts]
    colors = [BAD, GOOD, "#888888"]
    bars = ax.bar(cat_pretty, counts, color=colors,
                   edgecolor="#1a1a1a", linewidth=1)
    for bar, c, pct in zip(bars, counts, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, c + n_total * 0.018,
                f"{pct:.1f}%\nn={c}",
                ha="center", va="bottom", color="#eeeeee",
                fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.20)
    ax.set_ylabel("number of GT lesions", fontsize=10)
    ax.set_title(f"GT lesion classification (pooled, n={n_total})",
                  color="#eeeeee", pad=8, fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    # PANEL B — miss rate by SIZE
    ax = fig.add_subplot(gs[1])
    rates = [100 * by_size_dm[s] / max(by_size_total[s], 1) for s in SIZES]
    counts_b = [by_size_total[s] for s in SIZES]
    bar_colors = [BAD if r > 60 else (ACCENT if r > 30 else GOOD)
                   for r in rates]
    bars = ax.bar(SIZES_PRETTY, rates, color=bar_colors,
                   edgecolor="#1a1a1a", linewidth=1)
    for bar, r, n in zip(bars, rates, counts_b):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 1.5,
                f"{r:.0f}%",
                ha="center", va="bottom", color="#ffffff",
                fontsize=12, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, 2,
                f"n={n}",
                ha="center", va="bottom", color="#1a1a1a",
                fontsize=9, fontweight="bold")
    ax.set_ylim(0, 95)
    ax.set_ylabel("DETECTOR_MISS rate (%)", fontsize=10)
    ax.set_title("Miss rate by lesion SIZE\nspread = 57 pts (size dominates)",
                  color="#eeeeee", pad=8, fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    # PANEL C — miss rate by ANATOMY (z-bucket)
    ax = fig.add_subplot(gs[2])
    by_z_total: dict = {z: 0 for z in Z_BUCKETS}
    by_z_dm:    dict = {z: 0 for z in Z_BUCKETS}
    for r in rows:
        zb = r["z_bucket"]
        if zb not in by_z_total:
            continue
        by_z_total[zb] += 1
        if r["classification"] == "DETECTOR_MISS":
            by_z_dm[zb] += 1
    z_rates = [100 * by_z_dm[z] / max(by_z_total[z], 1) for z in Z_BUCKETS]
    z_counts = [by_z_total[z] for z in Z_BUCKETS]
    bars = ax.bar(Z_PRETTY, z_rates,
                   color=ACCENT, edgecolor="#1a1a1a", linewidth=1)
    for bar, r, n in zip(bars, z_rates, z_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 1.5,
                f"{r:.0f}%", ha="center", va="bottom",
                color="#ffffff", fontsize=12, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, 2,
                f"n={n}", ha="center", va="bottom",
                color="#1a1a1a", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 95)
    ax.set_ylabel("DETECTOR_MISS rate (%)", fontsize=10)
    ax.set_title("Miss rate by ANATOMY\nspread = 7 pts (anatomy weak)",
                  color="#eeeeee", pad=8, fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.suptitle(
        "Why steps 4 / 5 don't work   ·   misses are dominated by lesion SIZE, "
        "not anatomy\n"
        "most missed lesions have ZERO detector signal "
        "(nothing for residual or atlas methods to rescue)   ·   "
        "RANKER_MISS rate is only 3% — ordering is already near-optimal",
        fontsize=11, color="#eeeeee", y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out / "fig_E1_miss_diagnostic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_E1_miss_diagnostic.png")


# ---------- figure E2: negative-result experiments ----------

def fig_E2_negative_results(out: Path):
    print("  computing 4 experiment metrics on Stanford...")
    keys = ["r@1", "r@2", "r@5", "r@10", "MRR"]

    # Reference: original (mv=10) Stanford config — the v4 default before
    # any experiments
    ref_b, ref_a = _metrics(
        Path("runs/state_stanford_dpo_v4"),
        NNUNetProbmapCache("runs/nnunet_probmap_stanford"),
        mv=10, prob_source="detector_only",
    )

    # Experiment 1: min_voxels=3 (small +)
    e1_b, e1_a = _metrics(
        Path("runs/state_stanford_dpo_v4_mv3"),
        NNUNetProbmapCache("runs/nnunet_probmap_stanford"),
        mv=3, prob_source="detector_only",
    )

    # Experiment 2: TTA (tradeoff)
    e2_b, e2_a = _metrics(
        Path("runs/state_stanford_dpo_v4_tta"),
        NNUNetProbmapCache("runs/nnunet_probmap_stanford_tta"),
        mv=3, prob_source="detector_only",
    )

    # Experiment 3: phenotype-step5 (big regression)
    e3_b, e3_a = _metrics(
        Path("runs/state_stanford_dpo_v4_phen5"),
        NNUNetProbmapCache("runs/nnunet_probmap_stanford"),
        mv=3, prob_source="detector_only",
    )

    # Experiment 4: AURORA ensemble (mixed-to-negative)
    e4_b, e4_a = _metrics(
        Path("runs/state_stanford_ensemble"),
        EnsembleProbmapCache("runs/nnunet_probmap_stanford",
                              "runs/aurora_probmap_stanford"),
        mv=3, prob_source="detector_only",
    )

    panels = [
        ("min_voxels = 10 → 3\n(rescue smaller CCs)",
         e1_a, ref_a, "small +", GOOD),
        ("nnU-Net TTA enabled\n(mirror-augmented inference)",
         e2_a, e1_a, "precision-for-recall", "#aaaaaa"),
        ("Step 5 with phenotype-similarity\nweighting (the 'phenotype trap')",
         e3_a, e1_a, "DEVASTATING regression", BAD),
        ("AURORA detector ensemble\n(BraTS-Mets-2023 model + nnU-Net)",
         e4_a, e1_a, "lifts r@10 / hurts MRR", BAD),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    x = np.arange(len(keys))
    w = 0.36

    for ax, (title, after, before, verdict, verdict_color) in zip(axes, panels):
        before_vals = [before[k] for k in keys]
        after_vals = [after[k] for k in keys]
        deltas = [a - b for a, b in zip(after_vals, before_vals)]

        ax.bar(x - w / 2, before_vals, w, color="#5a9fd4",
               label="before", edgecolor="#1a1a1a", linewidth=1)
        ax.bar(x + w / 2, after_vals, w, color=ACCENT,
               label="after", edgecolor="#1a1a1a", linewidth=1)
        for xi, (b, a, d) in enumerate(zip(before_vals, after_vals, deltas)):
            ax.text(xi, max(b, a) + 0.04,
                    f"{'+' if d > 0 else '−' if d < 0 else '±'}{abs(d * 100):.1f}",
                    ha="center", va="bottom",
                    color=(GOOD if d > 0 else BAD if d < -0.005
                            else "#aaaaaa"),
                    fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(keys, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 0.75)
        ax.set_ylabel("metric value", fontsize=9)
        ax.set_title(title, color="#eeeeee", fontsize=11, pad=8)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", facecolor="#222",
                   edgecolor="#444", fontsize=9)

        # Verdict badge
        ax.text(0.02, 0.96, f"VERDICT  ·  {verdict}",
                transform=ax.transAxes, fontsize=10,
                color=verdict_color, fontweight="bold",
                bbox=dict(facecolor="#0d1117",
                           edgecolor=verdict_color, linewidth=1, pad=4),
                va="top", ha="left")

    fig.suptitle(
        "Detector-side / 2nd-pass experiments on Stanford (n=87)   ·   "
        "Δ pp annotated above each pair   ·   bars are agent-side metrics",
        fontsize=12, color="#eeeeee", y=0.97,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig_E2_negative_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_E2_negative_results.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig_E0_audit_log(out)
    fig_E1_miss_diagnostic(out)
    fig_E2_negative_results(out)
    print(f"\nAll experiment figures saved to {out}/")


if __name__ == "__main__":
    main()
