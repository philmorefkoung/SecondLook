"""Generate results-section figures from cached agent state.

Produces 5 PNGs into figures/:

  fig_R1_headline.png       Stanford agent vs detector baseline (5 metrics)
  fig_R2_froc.png            FROC curves on both cohorts
  fig_R3_size_sensitivity.png Per-lesion match-rate by size (Stanford)
  fig_R4_topk_distribution.png Rank-of-first-TP distribution (Stanford)
  fig_R5_cross_site.png      UCSF vs Stanford retention table

All numbers are computed live from cached state pickles + nnU-Net probmap
caches; numbers stay in sync if any underlying experiment is re-run.
"""
from __future__ import annotations
import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from brain_mets_agent.models import NNUNetProbmapCache
from brain_mets_agent.models.proposal import probmap_to_candidates
from brain_mets_agent.orchestrator.tools.ranker import rank_candidates
from brain_mets_agent.eval.metrics import (
    additional_lesion_recall_at_fp, top_k_recall, mean_reciprocal_rank,
    match_predictions,
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
    "axes.grid":          True,
    "grid.color":         "#333333",
    "grid.linewidth":     0.5,
    "font.size":          11,
    "font.family":        "sans-serif",
    "axes.titleweight":   "bold",
})

ACCENT = "#ff6b4a"
BASELINE = "#5a9fd4"
SEED = "#5af5a3"


@dataclass
class _GTLite:
    centroid_vox: tuple
    voxel_count: int
    volume_mm3: float
    mask: object


# ---------- compute helpers ----------

def compute_per_study(state_dir: Path, probmap_dir: Path, mv: int = 3,
                       sweep_best: bool = True):
    """Returns (baseline_results, agent_results) — lists of dicts ready
    for the metric functions."""
    cache = NNUNetProbmapCache(probmap_dir)
    baseline, agent = [], []
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
        baseline.append({"study_id": sid, "preds": bcands,
                          "non_seed_gt": rec["non_seed_gt"]})
        if sweep_best:
            ranked = rank_candidates(rec["candidates"],
                                      weights=(0.45, 0.20, 0.10, 0.25),
                                      prob_source="vlm_else_detector",
                                      prefer_confirmed=False)
        else:
            ranked = rank_candidates(rec["candidates"],
                                      weights=(0.45, 0.20, 0.10, 0.25),
                                      prob_source="detector_only",
                                      prefer_confirmed=False)
        agent.append({"study_id": sid, "preds": ranked,
                       "non_seed_gt": rec["non_seed_gt"]})
    return baseline, agent


def metrics(results):
    return {
        "r@1":   additional_lesion_recall_at_fp(results, 1),
        "r@2":   additional_lesion_recall_at_fp(results, 2),
        "r@5":   additional_lesion_recall_at_fp(results, 5),
        "r@10":  additional_lesion_recall_at_fp(results, 10),
        "top-5":  top_k_recall(results, 5),
        "MRR":   mean_reciprocal_rank(results),
    }


def first_tp_rank(preds, gt) -> int | None:
    for r, p in enumerate(preds, start=1):
        if match_predictions([p], gt).matched_gt:
            return r
    return None


# ---------- figure R1: headline bar chart ----------

def fig_R1_headline(out: Path):
    print("Computing Stanford metrics...")
    base_r, agent_r = compute_per_study(
        Path("runs/state_stanford_dpo_v4_mv3"),
        Path("runs/nnunet_probmap_stanford"))
    base_m = metrics(base_r)
    agent_m = metrics(agent_r)

    keys = ["r@1", "r@2", "r@5", "r@10", "top-5", "MRR"]
    base_vals = [base_m[k] for k in keys]
    agent_vals = [agent_m[k] for k in keys]
    deltas = [a - b for a, b in zip(agent_vals, base_vals)]

    fig, ax = plt.subplots(1, 1, figsize=(14, 6.2))
    x = np.arange(len(keys))
    w = 0.36
    ax.bar(x - w / 2, base_vals, w, label="nnU-Net detector (baseline)",
           color=BASELINE, edgecolor="#1a1a1a", linewidth=1)
    ax.bar(x + w / 2, agent_vals, w, label="nnU-Net + Discrepancy-Hunter agent",
           color=ACCENT, edgecolor="#1a1a1a", linewidth=1)
    # Delta annotations live ABOVE each pair (well below ceiling)
    ymax = max(max(base_vals), max(agent_vals))
    delta_y = ymax * 1.08
    for xi, (b, a, d) in enumerate(zip(base_vals, agent_vals, deltas)):
        ax.text(xi - w / 2, b + 0.012, f"{b:.3f}",
                ha="center", va="bottom", color="#aaaaaa", fontsize=10)
        ax.text(xi + w / 2, a + 0.012, f"{a:.3f}",
                ha="center", va="bottom",
                color=("#ffffff" if d > 0 else "#888888"),
                fontsize=10, fontweight="bold")
        sign = "+" if d > 0 else ""
        ax.text(xi, delta_y, f"Δ {sign}{d * 100:.1f} pt",
                ha="center", va="bottom",
                color=(ACCENT if d > 0 else "#888888"),
                fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=12, fontweight="bold")
    ax.set_ylabel("metric value", fontsize=11)
    # Headroom for delta annotations + legend below them
    ax.set_ylim(0, ymax * 1.22)
    # Move legend below x-axis to free up the top of the plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, facecolor="#222", edgecolor="#444",
              fontsize=11)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Stanford BrainMetShare external validation   ·   "
        "n=87 studies   ·   leakage-clean   ·   sweep-best ranker per metric",
        fontsize=13, color="#eeeeee", y=0.97,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig_R1_headline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_R1_headline.png")


# ---------- figure R2: FROC curves ----------

def fig_R2_froc(out: Path):
    print("Computing FROC curves for both cohorts...")
    fps = [0.5, 1, 2, 4, 5, 8, 10, 16]

    def froc_pair(state_dir, probmap_dir):
        b, a = compute_per_study(Path(state_dir), Path(probmap_dir))
        b_curve = [additional_lesion_recall_at_fp(b, fp) for fp in fps]
        a_curve = [additional_lesion_recall_at_fp(a, fp) for fp in fps]
        return b_curve, a_curve

    ucsf_b, ucsf_a = froc_pair("runs/state_test_nnunet_dpo_v4",
                                "runs/nnunet_probmap_test")
    sf_b, sf_a = froc_pair("runs/state_stanford_dpo_v4_mv3",
                            "runs/nnunet_probmap_stanford")

    fig, ax = plt.subplots(1, 1, figsize=(11, 6.5))
    ax.plot(fps, ucsf_b, "--o", color="#888888", linewidth=1.5,
            markersize=6, label="UCSF test — detector baseline (n=47)",
            alpha=0.85)
    ax.plot(fps, ucsf_a, "-o", color="#ffb070", linewidth=2.4,
            markersize=7, label="UCSF test — + agent (n=47)")
    ax.plot(fps, sf_b, "--s", color="#5a9fd4", linewidth=1.5,
            markersize=6, label="Stanford — detector baseline (n=87)",
            alpha=0.85)
    ax.plot(fps, sf_a, "-s", color=ACCENT, linewidth=2.6,
            markersize=7, label="Stanford — + agent (n=87)")

    # Detector ceiling annotations
    ax.axhline(y=ucsf_b[-1], color="#ffb070", alpha=0.3, linestyle=":",
               linewidth=1.2)
    ax.text(fps[-1] * 1.01, ucsf_b[-1], f" UCSF ceiling ≈ {ucsf_b[-1]:.2f}",
            color="#ffb070", fontsize=9, va="center", alpha=0.8)
    ax.axhline(y=sf_b[-1], color=ACCENT, alpha=0.3, linestyle=":",
               linewidth=1.2)
    ax.text(fps[-1] * 1.01, sf_b[-1], f" Stanford ceiling ≈ {sf_b[-1]:.2f}",
            color=ACCENT, fontsize=9, va="center", alpha=0.8)

    ax.set_xscale("log")
    ax.set_xlabel("Allowed false positives per study (log scale)",
                  fontsize=11)
    ax.set_ylabel("Mean per-study recall", fontsize=11)
    ax.set_xticks(fps)
    ax.set_xticklabels([str(f) if f != int(f) else str(int(f)) for f in fps])
    ax.set_xlim(0.4, 22)
    ax.set_ylim(0, 0.72)
    ax.legend(loc="upper left", facecolor="#222", edgecolor="#444",
              fontsize=10)
    ax.grid(True, which="both", alpha=0.25)

    fig.suptitle(
        "FROC curves   ·   recall vs FP / study   ·   nnU-Net + DPO v4 agent "
        "vs detector-only baseline   ·   default ranker config",
        fontsize=12, color="#eeeeee", y=0.96,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig_R2_froc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_R2_froc.png")


# ---------- figure R3: per-lesion sensitivity by size ----------

def fig_R3_size_sensitivity(out: Path):
    print("Computing per-lesion sensitivity by size on Stanford...")
    state_dir = Path("runs/state_stanford_dpo_v4_mv3")
    cache = NNUNetProbmapCache("runs/nnunet_probmap_stanford")

    # Per-GT-lesion: was it matched within FP=10/study budget?
    SIZES = [
        ("<30 mm³ (~<3.5 mm)", 0, 30),
        ("30-100 mm³",         30, 100),
        ("100-300 mm³",        100, 300),
        ("300-1000 mm³ (~8-12 mm)", 300, 1000),
        (">1000 mm³ (~>12 mm)", 1000, 1e9),
    ]
    by_bucket: dict[str, list[bool]] = {b[0]: [] for b in SIZES}

    for p in sorted(state_dir.glob("*.pkl")):
        with p.open("rb") as f:
            rec = pickle.load(f)
        rec["non_seed_gt"] = [_GTLite(**g) for g in rec["non_seed_gt"]]
        for g in rec["non_seed_gt"]:
            g.mask = None
        ranked = rank_candidates(rec["candidates"],
                                   weights=(0.45, 0.20, 0.10, 0.25),
                                   prob_source="vlm_else_detector",
                                   prefer_confirmed=False)
        # Cap at FP=10/study
        kept = []
        for cand in ranked:
            kept.append(cand)
            tp = len(match_predictions(kept, rec["non_seed_gt"]).matched_gt)
            fp = len(kept) - tp
            if fp > 10:
                kept.pop()
                break
        matched = set(match_predictions(kept, rec["non_seed_gt"]).matched_gt)
        for gi, gt in enumerate(rec["non_seed_gt"]):
            for label, lo, hi in SIZES:
                if lo <= gt.volume_mm3 < hi:
                    by_bucket[label].append(gi in matched)
                    break

    labels = []
    rates = []
    counts = []
    for label, _, _ in SIZES:
        ms = by_bucket[label]
        if ms:
            labels.append(label)
            rates.append(np.mean(ms))
            counts.append(len(ms))

    fig, ax = plt.subplots(1, 1, figsize=(11.5, 5.5))
    y = np.arange(len(labels))
    bars = ax.barh(y, rates, color=ACCENT, edgecolor="#1a1a1a", linewidth=1)
    # Color the smallest bar red-tinted, largest green-tinted
    for i, bar in enumerate(bars):
        r = rates[i]
        if r < 0.30:
            bar.set_color("#cc4a3a")
        elif r >= 0.70:
            bar.set_color("#7fbc41")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Per-lesion match rate at FP=10 / study", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    for bar, r, n in zip(bars, rates, counts):
        ax.text(r + 0.012, bar.get_y() + bar.get_height() / 2,
                f"{r:.2f}",
                va="center", color="#ffffff", fontsize=12, fontweight="bold")
        ax.text(0.005, bar.get_y() + bar.get_height() / 2,
                f"n={n}", va="center", color="#1a1a1a",
                fontsize=10, fontweight="bold")

    fig.suptitle(
        "Per-lesion sensitivity by size — Stanford BrainMetShare (n=87)   ·   "
        "agent + sweep-best ranker   ·   FP=10 / study budget",
        fontsize=12, color="#eeeeee", y=0.97,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig_R3_size_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_R3_size_sensitivity.png")


# ---------- figure R4: top-K rank distribution ----------

def fig_R4_topk_distribution(out: Path):
    print("Computing first-TP rank distribution on Stanford...")
    state_dir = Path("runs/state_stanford_dpo_v4_mv3")
    rank1, rank2, rank3p, never = 0, 0, 0, 0
    for p in sorted(state_dir.glob("*.pkl")):
        with p.open("rb") as f:
            rec = pickle.load(f)
        rec["non_seed_gt"] = [_GTLite(**g) for g in rec["non_seed_gt"]]
        for g in rec["non_seed_gt"]:
            g.mask = None
        ranked = rank_candidates(rec["candidates"],
                                   weights=(0.45, 0.20, 0.10, 0.25),
                                   prob_source="vlm_else_detector",
                                   prefer_confirmed=False)
        r = first_tp_rank(ranked, rec["non_seed_gt"])
        if r == 1:
            rank1 += 1
        elif r == 2:
            rank2 += 1
        elif r is not None:
            rank3p += 1
        else:
            never += 1
    n = rank1 + rank2 + rank3p + never

    cats = ["rank 1\n(top-1 hit)", "rank 2", "rank 3+", "no TP found"]
    counts = [rank1, rank2, rank3p, never]
    pct = [100 * c / n for c in counts]
    colors = [ACCENT, "#ffb070", "#888888", "#5a3a3a"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                              gridspec_kw={"width_ratios": [1, 1.2]})
    # Left: pie chart
    ax = axes[0]
    wedges, _ = ax.pie(counts, colors=colors, startangle=90,
                        wedgeprops=dict(edgecolor="#1a1a1a", linewidth=2))
    for i, w in enumerate(wedges):
        ang = (w.theta2 + w.theta1) / 2
        x = np.cos(np.deg2rad(ang)) * 0.65
        y = np.sin(np.deg2rad(ang)) * 0.65
        ax.text(x, y, f"{pct[i]:.0f}%",
                ha="center", va="center",
                color="#1a1a1a", fontsize=14, fontweight="bold")
    ax.set_title(f"Rank of first true positive   ·   n={n} studies",
                  color="#eeeeee", pad=8)

    # Right: cumulative bar (rank-1, rank-≤2, rank-≤K, found anywhere)
    ax = axes[1]
    cumkeys = ["rank 1", "rank ≤ 2", "any rank found"]
    cumvals = [rank1 / n, (rank1 + rank2) / n,
               (rank1 + rank2 + rank3p) / n]
    yp = np.arange(len(cumkeys))
    ax.barh(yp, cumvals, color=[ACCENT, "#ffb070", "#5af5a3"],
            edgecolor="#1a1a1a", linewidth=1)
    for i, v in enumerate(cumvals):
        ax.text(v + 0.012, yp[i],
                f"{v * 100:.1f}%   ({int(v * n)} studies)",
                va="center", color="#ffffff", fontsize=12, fontweight="bold")
    ax.set_yticks(yp)
    ax.set_yticklabels(cumkeys, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("fraction of studies", fontsize=11)
    ax.grid(axis="x", alpha=0.25)

    # Legend on the pie panel
    handles = [mpatches.Patch(color=c, label=f"{l}  ({n_})")
               for c, l, n_ in zip(colors, cats, counts)]
    axes[0].legend(handles=handles, loc="lower center",
                    bbox_to_anchor=(0.5, -0.15),
                    facecolor="#222", edgecolor="#444",
                    fontsize=10, ncol=2)

    fig.suptitle(
        "What rank does the first true positive land at?   ·   "
        "Stanford agent (sweep-best, n=87)",
        fontsize=12, color="#eeeeee", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig_R4_topk_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_R4_topk_distribution.png")


# ---------- figure R5: cross-site retention ----------

def fig_R5_cross_site(out: Path):
    print("Computing cross-site retention table...")
    keys = ["r@1", "r@2", "r@5", "r@10", "MRR"]

    def metrics_for(state_dir, probmap_dir):
        b, a = compute_per_study(Path(state_dir), Path(probmap_dir))
        return metrics(b), metrics(a)

    ucsf_b, ucsf_a = metrics_for("runs/state_test_nnunet_dpo_v4",
                                   "runs/nnunet_probmap_test")
    sf_b, sf_a = metrics_for("runs/state_stanford_dpo_v4_mv3",
                               "runs/nnunet_probmap_stanford")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                              gridspec_kw={"width_ratios": [1, 1.05]})
    # LEFT: grouped bars per metric for both cohorts (delta only)
    ax = axes[0]
    ucsf_d = [ucsf_a[k] - ucsf_b[k] for k in keys]
    sf_d = [sf_a[k] - sf_b[k] for k in keys]
    x = np.arange(len(keys))
    w = 0.36
    ax.bar(x - w / 2, [d * 100 for d in ucsf_d], w,
           label="UCSF test (n=47)", color="#ffb070",
           edgecolor="#1a1a1a", linewidth=1)
    ax.bar(x + w / 2, [d * 100 for d in sf_d], w,
           label="Stanford (n=87)", color=ACCENT,
           edgecolor="#1a1a1a", linewidth=1)
    for xi, d in enumerate(ucsf_d):
        ax.text(xi - w / 2, d * 100 + (0.4 if d > 0 else -0.6),
                f"{'+' if d > 0 else ''}{d * 100:.1f}",
                ha="center",
                va=("bottom" if d > 0 else "top"),
                color="#ffb070", fontsize=10, fontweight="bold")
    for xi, d in enumerate(sf_d):
        ax.text(xi + w / 2, d * 100 + (0.4 if d > 0 else -0.6),
                f"{'+' if d > 0 else ''}{d * 100:.1f}",
                ha="center",
                va=("bottom" if d > 0 else "top"),
                color=ACCENT, fontsize=10, fontweight="bold")
    ax.axhline(y=0, color="#666", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=11, fontweight="bold")
    ax.set_ylabel("Δ vs detector baseline   (percentage points)", fontsize=10)
    ax.legend(loc="upper right", facecolor="#222", edgecolor="#444",
              fontsize=10)
    ax.set_title("Agent uplift over baseline, by cohort",
                  color="#eeeeee", pad=10)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    # RIGHT: full numeric table — baseline / agent / delta for both cohorts
    ax = axes[1]
    ax.axis("off")
    rows = []
    for k in keys:
        u_b, u_a = ucsf_b[k], ucsf_a[k]
        s_b, s_a = sf_b[k], sf_a[k]
        u_d = (u_a - u_b) * 100
        s_d = (s_a - s_b) * 100
        rows.append([
            k,
            f"{u_b:.3f}", f"{u_a:.3f}",
            f"{'+' if u_d > 0 else ('−' if u_d < 0 else '±')}{abs(u_d):.1f}",
            f"{s_b:.3f}", f"{s_a:.3f}",
            f"{'+' if s_d > 0 else ('−' if s_d < 0 else '±')}{abs(s_d):.1f}",
        ])
    headers = ["metric",
                "UCSF base", "UCSF agent", "Δ pt",
                "Stanf. base", "Stanf. agent", "Δ pt"]
    table = ax.table(cellText=rows, colLabels=headers,
                      loc="center", cellLoc="center",
                      colColours=["#2a2a2a"] * len(headers),
                      colWidths=[0.10, 0.13, 0.13, 0.10, 0.13, 0.13, 0.10])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.7)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#444444")
        if r == 0:
            cell.set_text_props(weight="bold", color="#ffffff")
        else:
            cell.set_facecolor("#1a1a1a")
            text = cell.get_text().get_text()
            color = "#dddddd"
            # Δ columns get color based on sign
            if c in (3, 6):
                if text.startswith("+"):
                    color = ACCENT
                elif text.startswith("−"):
                    color = "#888888"
                cell.set_text_props(weight="bold")
            cell.set_text_props(color=color)
    ax.set_title("Numbers side-by-side   ·   baseline → agent → Δ for each cohort",
                  color="#eeeeee", pad=8)

    fig.suptitle(
        "Cross-site retention   ·   "
        "agent uplift on UCSF (with detector leakage caveat) vs "
        "Stanford (leakage-clean)",
        fontsize=12, color="#eeeeee", y=0.97,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig_R5_cross_site.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_R5_cross_site.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig_R1_headline(out)
    fig_R2_froc(out)
    fig_R3_size_sensitivity(out)
    fig_R4_topk_distribution(out)
    fig_R5_cross_site(out)
    print(f"\nAll results figures saved to {out}/")


if __name__ == "__main__":
    main()
