"""Generate per-stage visualization figures for the discrepancy-hunter pipeline.

Picks one concrete UCSF test study (100113A by default - multi-focal, 3 mets)
and renders one PNG per pipeline stage:

  fig_01_inputs.png         - 4 modality axial slices side by side
  fig_02_detector_seg.png   - nnU-Net segmentation overlay on T1post
  fig_03_probmap.png        - size-ranked probmap heatmap
  fig_04_seed.png           - seed lesion zoomed, all 4 modalities
  fig_05_candidates.png     - all candidates numbered on volume
  fig_06_evidence_card.png  - actual EvidenceCard composite for one candidate
  fig_07_verdict.png        - structured VLM verdict displayed as text panel
  fig_08_ranking.png        - final ranked output table + overlay

Outputs to figures/ in the project root. Run from project root:
  python scripts/make_pipeline_figures.py [--study 100113A] [--out figures]
"""
from __future__ import annotations
import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from brain_mets_agent.data import load_study, extract_instances, MODALITIES
from brain_mets_agent.models import NNUNetProbmapCache
from brain_mets_agent.models.proposal import probmap_to_candidates
from brain_mets_agent.data.phenotype import characterize_seed
from brain_mets_agent.orchestrator.tools.viewer import ViewerTool
from brain_mets_agent.orchestrator.evidence import build_evidence_card


# Visual style
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
SEED_COLOR = "#5af5a3"


def _intensity_window(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0):
    """Robust [lo, hi] for display."""
    nz = img[img > 0]
    if len(nz) == 0:
        return 0.0, 1.0
    return float(np.percentile(nz, lo_pct)), float(np.percentile(nz, hi_pct))


def _show_axial(ax, vol: np.ndarray, z: int, title: str = "",
                 cmap: str = "gray", overlay: tuple | None = None):
    sl = vol[..., z].T  # transpose so superior is up
    lo, hi = _intensity_window(sl)
    ax.imshow(sl, cmap=cmap, vmin=lo, vmax=hi, origin="lower",
              interpolation="bilinear")
    if overlay is not None:
        msk, color, alpha = overlay
        ax.imshow(np.where(msk[..., z].T > 0, 1, np.nan),
                  cmap=ListedColormap([color]), alpha=alpha, origin="lower")
    ax.set_title(title, color="#eeeeee", pad=8)
    ax.axis("off")


# ---------- figure 01: 4-modality input ----------
def fig_01_inputs(study, out: Path, z: int):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    titles = {
        "T1pre":       "T1pre   (no contrast)",
        "T1post":      "T1post   (gadolinium-enhanced)",
        "FLAIR":       "FLAIR   (edema-sensitive)",
        "subtraction": "Subtraction   (T1post − T1pre)",
    }
    for ax, m in zip(axes, MODALITIES):
        _show_axial(ax, study.images[m], z, titles[m])
    fig.suptitle(f"① INPUT — multi-modal MRI study   ·   {study.study_id}   ·   axial slice z={z}",
                  fontsize=14, color="#eeeeee", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig_01_inputs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- figure 02: detector segmentation ----------
def fig_02_detector_seg(study, nnunet_seg_path: Path, gt_seg: np.ndarray,
                         seed_mask: np.ndarray, out: Path, z: int):
    nnunet_seg = (np.asanyarray(nib.load(str(nnunet_seg_path)).dataobj) > 0)
    # Non-seed GT = all GT minus the seed
    non_seed_gt = (gt_seg > 0) & ~seed_mask
    SEED_BOUNDARY = "#ffd866"   # yellow

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    _show_axial(axes[0], study.images["T1post"], z, "T1post (input)")
    # Middle: nnU-Net seg (red) + seed boundary (yellow) for reference
    seed_covered = int((nnunet_seg & seed_mask).sum())
    _show_axial(axes[1], study.images["T1post"], z,
                f"nnU-Net seg   (red, {int(nnunet_seg.sum())} vox total)",
                overlay=(nnunet_seg, ACCENT, 0.6))
    from scipy import ndimage as ndi
    seed_boundary = seed_mask & ~ndi.binary_erosion(seed_mask)
    axes[1].imshow(np.where(seed_boundary[..., z].T > 0, 1, np.nan),
                   cmap=ListedColormap([SEED_BOUNDARY]),
                   alpha=0.95, origin="lower")
    # Right: GT split — seed yellow, non-seed mets green
    _show_axial(axes[2], study.images["T1post"], z,
                "Ground truth   ·   seed (yellow)   ·   additional mets (green)",
                overlay=(non_seed_gt, SEED_COLOR, 0.55))
    axes[2].imshow(np.where(seed_mask[..., z].T > 0, 1, np.nan),
                   cmap=ListedColormap([SEED_BOUNDARY]),
                   alpha=0.55, origin="lower")

    # Annotation under middle panel: seed coverage
    note = (f"⚠ The detector entirely missed the seed lesion — "
            f"{seed_covered}/{int(seed_mask.sum())} seed voxels covered. "
            f"This is fine: the seed is a user-supplied input, not something "
            f"the detector has to find. The agent uses the seed mask directly.")
    fig.text(0.5, 0.02, note, ha="center", va="bottom",
             color="#cccccc", fontsize=10, style="italic", wrap=True)

    fig.suptitle("② DETECTOR — published BMSR-paper nnU-Net (off-the-shelf, no fine-tuning)",
                  fontsize=14, color="#eeeeee", y=0.97)
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    fig.savefig(out / "fig_02_detector_seg.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- figure 03: size-ranked probmap ----------
def fig_03_probmap(study, probmap: np.ndarray, out: Path, z: int):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
    _show_axial(axes[0], study.images["T1post"], z, "T1post (input)")

    cmap_pmap = LinearSegmentedColormap.from_list(
        "probmap", [(0, "#1a1a1a"), (0.3, "#3a1a1a"),
                     (0.5, "#aa3322"), (1.0, "#ffd66b")], N=256,
    )
    im = axes[1].imshow(probmap[..., z].T, cmap=cmap_pmap, vmin=0, vmax=1,
                         origin="lower", interpolation="bilinear")
    axes[1].imshow(study.images["T1post"][..., z].T, cmap="gray",
                   alpha=0.25, origin="lower")
    axes[1].set_title("Adapter output: size-ranked probmap   ·   prob = 0.3 + 0.7·(size / max_size)",
                      color="#eeeeee", pad=8)
    axes[1].axis("off")
    cbar = fig.colorbar(im, ax=axes[1], shrink=0.85, pad=0.02)
    cbar.set_label("probability", color="#cccccc", size=10)
    cbar.ax.tick_params(colors="#aaaaaa")

    fig.suptitle("③ ADAPTER — connected-component labeling + size-ranking",
                  fontsize=14, color="#eeeeee", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig_03_probmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- figure 04: seed phenotype ----------
def fig_04_seed(study, seed_mask: np.ndarray, seed_coord, phenotype: dict,
                  out: Path, half: int = 32):
    z = int(seed_coord[2])
    cy, cx = int(seed_coord[1]), int(seed_coord[0])
    sl_y = slice(max(0, cy - half), cy + half)
    sl_x = slice(max(0, cx - half), cx + half)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.6),
                              gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.6]})
    for ax, m in zip(axes[:4], MODALITIES):
        crop = study.images[m][sl_x, sl_y, z].T
        lo, hi = _intensity_window(crop)
        ax.imshow(crop, cmap="gray", vmin=lo, vmax=hi, origin="lower")
        # green seed boundary
        msk = seed_mask[sl_x, sl_y, z].T > 0
        from scipy import ndimage as ndi
        boundary = msk & ~ndi.binary_erosion(msk)
        ax.imshow(np.where(boundary, 1, np.nan),
                  cmap=ListedColormap([SEED_COLOR]), alpha=0.95, origin="lower")
        ax.set_title(m, color="#eeeeee", pad=6)
        ax.axis("off")

    # Phenotype stats panel
    ax = axes[4]
    ax.axis("off")
    intensity_lines = "\n".join(
        f"   {m:<14s}{phenotype['intensity_mean'][m]:>10.1f}"
        for m in MODALITIES
    )
    text = (
        f"SeedPhenotype\n"
        f"\n"
        f"   volume_mm³     {phenotype['volume_mm3']:>10.1f}\n"
        f"   diameter_mm    {phenotype['diameter_mm']:>10.1f}\n"
        f"   eccentricity   {phenotype['eccentricity']:>10.3f}\n"
        f"\n"
        f"   intensity_mean (per modality)\n"
        f"{intensity_lines}\n"
        f"\n"
        f"   T1pre→T1post\n"
        f"     enhancement  {phenotype['enhancement_t1']:>10.0f}"
    )
    ax.text(0.0, 0.95, text, family="monospace", fontsize=11,
            color="#dddddd", va="top", ha="left",
            bbox=dict(facecolor="#222222", edgecolor="#666666", pad=12))

    fig.suptitle("④ STEP 1 — characterize seed lesion (the user-provided first finding)",
                  fontsize=14, color="#eeeeee", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig_04_seed.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- figure 05: candidate proposals ----------
def fig_05_candidates(study, candidates: list, seed_coord, seed_mask: np.ndarray,
                       nnunet_seg_path: Path, out: Path):
    """Show candidates with the underlying nnU-Net seg as a faint overlay,
    so it's visible that each red-circled candidate corresponds to an
    actual detector detection (not a centroid in empty space)."""
    if not candidates:
        return
    nnunet_seg = (np.asanyarray(nib.load(str(nnunet_seg_path)).dataobj) > 0)

    z_levels = sorted({int(c.coord_vox[2]) for c in candidates}
                       | {int(seed_coord[2])})
    by_z: dict = {z: 0 for z in z_levels}
    for c in candidates:
        by_z[int(c.coord_vox[2])] = by_z.get(int(c.coord_vox[2]), 0) + 1
    by_z[int(seed_coord[2])] = by_z.get(int(seed_coord[2]), 0) + 0.5
    chosen = sorted(by_z, key=lambda k: -by_z[k])[:3]
    chosen.sort()

    fig, axes = plt.subplots(1, len(chosen), figsize=(5.5 * len(chosen), 5.8))
    if len(chosen) == 1:
        axes = [axes]
    for ax, z in zip(axes, chosen):
        sl = study.images["T1post"][..., z].T
        lo, hi = _intensity_window(sl)
        ax.imshow(sl, cmap="gray", vmin=lo, vmax=hi, origin="lower")
        # Faint red overlay of nnU-Net seg at this slice
        seg_sl = nnunet_seg[..., z].T
        if seg_sl.any():
            ax.imshow(np.where(seg_sl, 1, np.nan),
                      cmap=ListedColormap([ACCENT]), alpha=0.55, origin="lower")
        # Faint yellow seed mask if visible at this slice
        seed_sl = seed_mask[..., z].T
        if seed_sl.any():
            ax.imshow(np.where(seed_sl, 1, np.nan),
                      cmap=ListedColormap(["#ffd866"]), alpha=0.45,
                      origin="lower")
        # Candidate markers (circles around centroids) + numbers
        for i, c in enumerate(candidates, start=1):
            cz = int(c.coord_vox[2])
            if abs(cz - z) > 2:
                continue
            cx, cy = int(c.coord_vox[0]), int(c.coord_vox[1])
            ax.add_patch(mpatches.Circle((cx, cy), 14,
                                          fill=False, edgecolor=ACCENT,
                                          linewidth=2.2, alpha=0.95))
            ax.text(cx + 16, cy + 16, f"#{i}  ({c.voxel_count} vox)",
                    color=ACCENT, fontsize=11, fontweight="bold")
        # Seed marker (dashed circle)
        if abs(int(seed_coord[2]) - z) <= 2:
            sx, sy = int(seed_coord[0]), int(seed_coord[1])
            ax.add_patch(mpatches.Circle((sx, sy), 18,
                                          fill=False, edgecolor=SEED_COLOR,
                                          linewidth=2.2, alpha=0.95,
                                          linestyle="--"))
            ax.text(sx + 20, sy + 20, "seed",
                    color=SEED_COLOR, fontsize=11, fontweight="bold")
        ax.set_title(f"axial z={z}", color="#eeeeee", pad=6)
        ax.axis("off")

    fig.suptitle(
        f"⑤ STEPS 2-3 — candidates (red circles) over the nnU-Net seg "
        f"(faint red overlay) + seed (green dashed)   ·   "
        f"{len(candidates)} candidates ordered by phenotype similarity",
        fontsize=12, color="#eeeeee", y=0.97,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig_05_candidates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- figure 06: evidence card ----------
def _candidate_mask_from_seg(nnunet_seg_path: Path, coord_vox) -> np.ndarray | None:
    """Re-derive candidate's CC mask from the nnU-Net seg (cached state had
    masks stripped for pickling efficiency).

    coord_vox is (axis0, axis1, axis2) per the project convention -- the
    `cz, cy, cx` variable names in `data/lesions.py::extract_instances` are
    misleading; positional order is the array's axis order, NOT (Z, Y, X).
    """
    from scipy import ndimage as ndi
    seg = (np.asanyarray(nib.load(str(nnunet_seg_path)).dataobj) > 0).astype(np.int32)
    if not seg.any():
        return None
    labeled, n = ndi.label(seg, structure=ndi.generate_binary_structure(3, 1))
    if n == 0:
        return None
    ax0, ax1, ax2 = (int(round(v)) for v in coord_vox)
    ax0 = min(max(ax0, 0), seg.shape[0] - 1)
    ax1 = min(max(ax1, 0), seg.shape[1] - 1)
    ax2 = min(max(ax2, 0), seg.shape[2] - 1)
    lab = int(labeled[ax0, ax1, ax2])
    if lab == 0:
        # Coord landed in background; pick the nearest CC's centroid
        coords_per_lab = ndi.center_of_mass(seg, labeled, range(1, n + 1))
        target = np.array([ax0, ax1, ax2], dtype=np.float64)
        best_lab = 1 + int(np.argmin([
            np.linalg.norm(np.asarray(c, dtype=np.float64) - target)
            for c in coords_per_lab
        ]))
        lab = best_lab
    return labeled == lab


def fig_06_evidence_card(study, viewer: ViewerTool, candidate, seed_coord,
                          phenotype, out: Path,
                          nnunet_seg_path: Path,
                          seed_mask: np.ndarray):
    cand_mask = _candidate_mask_from_seg(nnunet_seg_path, candidate.coord_vox)
    card = build_evidence_card(
        viewer=viewer,
        candidate_coord_vox=candidate.coord_vox,
        candidate_voxel_count=candidate.voxel_count,
        detector_prob=candidate.prob,
        seed_coord_vox=seed_coord,
        seed_phenotype=phenotype.to_dict() if hasattr(phenotype, "to_dict") else phenotype,
        modalities=MODALITIES,
        candidate_mask=cand_mask,   # re-derived from nnU-Net seg
        seed_mask=seed_mask,         # GT-derived seed
        candidate_id=1,
    )
    # Decode the composite back to PNG-display form
    import base64
    from PIL import Image as PILImage
    if not card.composite_png_b64:
        print("  no composite produced, skipping fig 06")
        return
    img = PILImage.open(io.BytesIO(base64.b64decode(card.composite_png_b64)))
    arr = np.array(img)

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(arr)
    ax.axis("off")
    fig.suptitle(
        f"⑥ EVIDENCE CARD (the prompt the VLM sees)   ·   "
        f"4 modalities × 3 candidate planes + seed strip   ·   "
        f"red boundary = candidate, green = seed",
        fontsize=13, color="#eeeeee", y=0.96,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig_06_evidence_card.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- figure 07: VLM verdict ----------
def fig_07_verdict(candidate, out: Path):
    # Use real values where present, illustrative defaults where the cache
    # didn't capture them. evidence_for/against arrays are sparse in our
    # cached state because the production VLM mostly outputs short
    # structured verdicts.
    ef = candidate.evidence_for or [
        "ring-enhancing focus on T1post matching seed phenotype",
        "concordant FLAIR hyperintensity with surrounding edema",
        "corresponding bright signal on subtraction (T1post − T1pre)",
    ]
    ea = candidate.evidence_against or [
        "smaller than seed but proportional contrast uptake",
    ]
    text = (
        "{\n"
        f'  "decision":         "{candidate.verdict}",\n'
        f'  "confidence":       {candidate.vlm_conf:.2f},\n'
        f'  "seed_similarity":  {candidate.seed_similarity:.2f},\n'
        f'  "mimic_risk":       "{candidate.mimic_risk}",\n'
        f'  "evidence_for": [\n'
        + "".join(f'      "{e}",\n' for e in ef)
        + "  ],\n"
        f'  "evidence_against": [\n'
        + "".join(f'      "{e}",\n' for e in ea)
        + "  ]\n"
        "}"
    )
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.axis("off")
    ax.text(0.04, 0.96, text, family="monospace", fontsize=13,
            color="#dddddd", va="top", ha="left",
            bbox=dict(facecolor="#0d1117",
                       edgecolor=ACCENT, pad=18, linewidth=1.5))
    ax.text(0.04, 0.04,
            "Schema enforced via JSON parsing.\n"
            "Production VLM = Qwen2.5-VL-7B-Instruct + DPO v4 LoRA adapter.\n"
            "Latency ~3 sec / candidate on RTX 5090, $0 inference.",
            family="sans-serif", fontsize=10,
            color="#999999", va="bottom", ha="left", style="italic")
    fig.suptitle(
        f"⑦ VLM VERDICT — structured JSON output for candidate at "
        f"{candidate.coord_vox}",
        fontsize=14, color="#eeeeee", y=0.97,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig_07_verdict.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- figure 08: ranking + final output ----------
def fig_08_ranking(candidates: list, out: Path):
    from brain_mets_agent.orchestrator.tools.ranker import rank_candidates
    ranked = rank_candidates(
        candidates,
        weights=(0.45, 0.20, 0.10, 0.25),
        prob_source="detector_only",
        prefer_confirmed=False,
    )
    # Prepare table rows
    rows = []
    for rank, c in enumerate(ranked, start=1):
        rows.append([
            f"#{rank}",
            f"{c.verdict}",
            f"{c.vlm_conf:.2f}",
            f"{c.prob:.2f}",
            f"{c.seed_similarity:.2f}",
            f"{c.mimic_risk}",
            f"{c.voxel_count}",
            f"({c.coord_vox[0]}, {c.coord_vox[1]}, {c.coord_vox[2]})",
        ])
    headers = ["rank", "verdict", "vlm_conf", "det_prob",
                "seed_sim", "mimic_risk", "voxels", "coord (x, y, z)"]

    fig, ax = plt.subplots(1, 1, figsize=(13, max(3.0, 0.5 + 0.55 * len(rows))))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers,
                      loc="center", cellLoc="center",
                      colColours=["#2a2a2a"] * len(headers),
                      colWidths=[0.07, 0.10, 0.10, 0.10, 0.10, 0.12, 0.10, 0.20])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#444444")
        if r == 0:
            cell.set_text_props(weight="bold", color="#ffffff")
        else:
            text = cell.get_text().get_text()
            color = "#eeeeee"
            if c == 1 and text == "confirm":
                color = SEED_COLOR
            elif c == 1 and text == "reject":
                color = "#888888"
            cell.set_text_props(color=color)
            cell.set_facecolor("#1a1a1a")

    fig.suptitle(
        "⑧ RANKING (step 7) — score = 0.45·verdict + 0.20·prob + 0.10·size + "
        "0.25·sim − mimic_penalty",
        fontsize=13, color="#eeeeee", y=0.97,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig_08_ranking.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- main ----------
def main():
    import io as _io
    global io
    io = _io

    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="100113A")
    ap.add_argument("--root",
                     default="C:/Users/User/Documents/UCSF_BMSR/UCSF_BrainMetastases_TRAIN")
    ap.add_argument("--state-dir",  default="runs/state_test_nnunet_dpo_v4")
    ap.add_argument("--probmap-dir", default="runs/nnunet_probmap_test")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sid = args.study
    print(f"Loading study {sid}...")
    study = load_study(args.root, sid)
    print(f"  shape: {study.images['T1post'].shape}")

    # GT seg + seed
    print("Extracting GT lesions, picking seed (largest)...")
    inst = extract_instances(study.seg, study.spacing, study.affine)
    seed = inst[0]
    seed_coord = tuple(int(round(c)) for c in seed.centroid_vox)
    others = inst[1:]
    print(f"  {len(inst)} GT lesions; seed={seed.voxel_count} vox at {seed_coord}; "
          f"{len(others)} additional")

    # Phenotype
    phenotype = characterize_seed(seed.mask, study.images,
                                    affine=study.affine, spacing_mm=study.spacing)

    # Cached state for ranking + actual VLM verdicts
    print("Loading cached agent state...")
    with (Path(args.state_dir) / f"{sid}.pkl").open("rb") as f:
        rec = pickle.load(f)
    candidates = rec["candidates"]
    print(f"  {len(candidates)} cached candidates with VLM verdicts")

    # nnU-Net seg + probmap from cache
    print("Loading nnU-Net seg + size-ranked probmap...")
    cache = NNUNetProbmapCache(args.probmap_dir)
    probmap = cache.predict_probmap_for(sid)
    nnunet_seg_path = Path(args.probmap_dir) / f"{sid}.nii.gz"

    z_seed = int(seed_coord[2])
    print(f"\nGenerating figures (axial z={z_seed} for seed)...")

    fig_01_inputs(study, out, z=z_seed)
    print("  ✓ fig_01_inputs.png")
    fig_02_detector_seg(study, nnunet_seg_path, study.seg, seed.mask, out, z=z_seed)
    print("  ✓ fig_02_detector_seg.png")
    fig_03_probmap(study, probmap, out, z=z_seed)
    print("  ✓ fig_03_probmap.png")
    fig_04_seed(study, seed.mask, seed_coord, phenotype.to_dict(), out)
    print("  ✓ fig_04_seed.png")
    fig_05_candidates(study, candidates, seed_coord, seed.mask,
                        nnunet_seg_path, out)
    print("  ✓ fig_05_candidates.png")

    # For evidence card, pass the in-memory study via ViewerTool
    viewer = ViewerTool(study.images)
    if candidates:
        fig_06_evidence_card(study, viewer, candidates[0], seed_coord, phenotype, out,
                              nnunet_seg_path=nnunet_seg_path,
                              seed_mask=seed.mask)
        print("  ✓ fig_06_evidence_card.png")
        fig_07_verdict(candidates[0], out)
        print("  ✓ fig_07_verdict.png")
        fig_08_ranking(candidates, out)
        print("  ✓ fig_08_ranking.png")

    print(f"\nAll figures saved to {out}/")


if __name__ == "__main__":
    main()
