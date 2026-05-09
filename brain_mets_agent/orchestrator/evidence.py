"""Evidence cards for VLM verification.

Builds a single composite RGB PNG that lays out:
  Row 0: candidate AXIAL crops across modalities (red boundary = candidate mask)
  Row 1: candidate CORONAL crops across modalities
  Row 2: candidate SAGITTAL crops across modalities
  Row 3: SEED axial crops across modalities (green boundary = seed mask)

The candidate is at the exact centre of every candidate tile - so the VLM
prompt can say "the candidate is at the centre" and the model can localize
before reasoning. The red boundary outlines the proposal model's predicted
extent (so the VLM can judge whether the segmentation is plausible). The
green boundary on the seed strip lets the VLM compare candidate phenotype
to the seed's segmentation directly.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
import io
import base64

import numpy as np

from .tools.viewer import ViewerTool


PLANE_ROWS = ("axial", "coronal", "sagittal")
CAND_BOUNDARY_RGB = (255, 80, 80)    # red
SEED_BOUNDARY_RGB = (80, 255, 120)   # green


@dataclass
class EvidenceCard:
    candidate_id: int | None
    coord_vox: tuple[int, int, int]
    voxel_count: int
    detector_prob: float
    panels: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)        # (plane, mod) -> tile
    seed_panels: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    panel_masks: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    seed_panel_masks: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    seed_phenotype: dict | None = None
    composite_png_b64: str = ""
    metadata_text: str = ""


def build_evidence_card(
    *,
    viewer: ViewerTool,
    candidate_coord_vox: tuple[int, int, int],
    candidate_voxel_count: int,
    detector_prob: float,
    seed_coord_vox: tuple[int, int, int],
    seed_phenotype: dict,
    modalities: Sequence[str],
    candidate_mask: np.ndarray | None = None,
    seed_mask: np.ndarray | None = None,
    candidate_id: int | None = None,
    tile_size: int = 96,
) -> EvidenceCard:
    panels: dict[tuple[str, str], np.ndarray] = {}
    panel_masks: dict[tuple[str, str], np.ndarray] = {}
    for plane in PLANE_ROWS:
        # mask is the same per plane regardless of modality - extract once
        mask_tile = viewer.tile_volume(
            candidate_coord_vox, candidate_mask, plane=plane, size=tile_size,
        )
        for m in modalities:
            panels[(plane, m)] = viewer.tile(
                candidate_coord_vox, m, plane=plane, size=tile_size,
            ).image
            if mask_tile is not None:
                panel_masks[(plane, m)] = mask_tile

    seed_panels: dict[tuple[str, str], np.ndarray] = {}
    seed_panel_masks: dict[tuple[str, str], np.ndarray] = {}
    seed_mask_tile = viewer.tile_volume(
        seed_coord_vox, seed_mask, plane="axial", size=tile_size,
    )
    for m in modalities:
        seed_panels[("axial", m)] = viewer.tile(
            seed_coord_vox, m, plane="axial", size=tile_size,
        ).image
        if seed_mask_tile is not None:
            seed_panel_masks[("axial", m)] = seed_mask_tile

    composite = render_composite_rgb(
        panels, panel_masks, seed_panels, seed_panel_masks, modalities, tile_size,
    )
    composite_b64 = composite_to_png_b64(composite)
    meta = render_metadata_text(
        coord=candidate_coord_vox,
        voxel_count=candidate_voxel_count,
        detector_prob=detector_prob,
        seed_phenotype=seed_phenotype,
        candidate_has_mask=candidate_mask is not None,
        seed_has_mask=seed_mask is not None,
    )

    return EvidenceCard(
        candidate_id=candidate_id,
        coord_vox=tuple(int(c) for c in candidate_coord_vox),
        voxel_count=int(candidate_voxel_count),
        detector_prob=float(detector_prob),
        panels=panels,
        seed_panels=seed_panels,
        panel_masks=panel_masks,
        seed_panel_masks=seed_panel_masks,
        seed_phenotype=seed_phenotype,
        composite_png_b64=composite_b64,
        metadata_text=meta,
    )


def render_composite_rgb(
    panels, panel_masks, seed_panels, seed_panel_masks,
    modalities, tile_size: int,
) -> np.ndarray:
    n_mod = len(modalities)
    H = 4 * tile_size + 5
    W = n_mod * tile_size + (n_mod + 1)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    def place(row: int, col: int, img: np.ndarray, mask: np.ndarray | None,
              boundary_rgb: tuple[int, int, int]) -> None:
        a = img.astype(np.float32)
        if a.size == 0:
            return
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) / max(1e-6, hi - lo)
        if a.shape != (tile_size, tile_size):
            a = _resize2d(a, (tile_size, tile_size))
        gray = (a * 255).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)
        if mask is not None and mask.size > 0:
            m = mask.astype(bool)
            if m.shape != (tile_size, tile_size):
                m = _resize2d(m.astype(np.float32), (tile_size, tile_size)) > 0.5
            from scipy import ndimage as ndi
            inner = ndi.binary_erosion(m, iterations=1)
            boundary = m & ~inner
            rgb[boundary] = np.array(boundary_rgb, dtype=np.uint8)
        y0 = row * tile_size + (row + 1)
        x0 = col * tile_size + (col + 1)
        canvas[y0:y0 + tile_size, x0:x0 + tile_size] = rgb

    for r, plane in enumerate(PLANE_ROWS):
        for c, m in enumerate(modalities):
            place(r, c,
                  panels.get((plane, m), np.zeros((tile_size, tile_size))),
                  panel_masks.get((plane, m)),
                  CAND_BOUNDARY_RGB)
    for c, m in enumerate(modalities):
        place(3, c,
              seed_panels.get(("axial", m), np.zeros((tile_size, tile_size))),
              seed_panel_masks.get(("axial", m)),
              SEED_BOUNDARY_RGB)

    return canvas


def _resize2d(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from scipy import ndimage as ndi
    zh = size[0] / max(1, arr.shape[0])
    zw = size[1] / max(1, arr.shape[1])
    return ndi.zoom(arr, (zh, zw), order=1)


def composite_to_png_b64(composite: np.ndarray) -> str:
    from PIL import Image
    if composite.ndim == 3:
        img = Image.fromarray(composite, mode="RGB")
    else:
        img = Image.fromarray(np.clip(composite * 255, 0, 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_metadata_text(
    coord, voxel_count, detector_prob, seed_phenotype,
    candidate_has_mask: bool = False, seed_has_mask: bool = False,
) -> str:
    sp = seed_phenotype or {}
    intens = sp.get("intensity_mean", {})
    overlays = []
    if candidate_has_mask:
        overlays.append("RED outline = proposal model's predicted candidate extent")
    if seed_has_mask:
        overlays.append("GREEN outline (Row 3 only) = seed lesion segmentation")
    overlay_str = ("\n" + "\n".join(overlays)) if overlays else ""
    return (
        f"Candidate centroid (z,y,x): {tuple(int(c) for c in coord)}\n"
        f"Candidate voxel count: {voxel_count} "
        f"({'second-pass single-voxel peak' if voxel_count == 0 else 'first-pass component'})\n"
        f"Detector probability: {detector_prob:.3f}"
        f"{overlay_str}\n"
        "\n"
        "Seed phenotype:\n"
        f"  voxels={sp.get('voxel_count')}, "
        f"volume_mm3={sp.get('volume_mm3', 0.0):.1f}, "
        f"diameter_mm={sp.get('diameter_mm', 0.0):.1f}\n"
        f"  intensity_mean={ {k: round(float(v), 3) for k, v in intens.items()} }\n"
        f"  enhancement_t1={sp.get('enhancement_t1', 0.0):.3f}\n"
        f"  eccentricity={sp.get('eccentricity', 0.0):.3f}"
    )
