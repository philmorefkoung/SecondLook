"""Viewer tool: extract slices, centered 2D tiles, and 3D crops.

Two panel kinds:
  - `slice(coord, modality, plane)`: full 2D slice through the candidate.
  - `tile(coord, modality, plane, size)`: 2D tile centered on the candidate.

VLM verification uses centered tiles by default so the lesion is at the panel
centre - HeuristicVLM expects this, and a real VLM gets a focused view.
The audit trail uses the candidate's own bbox (set by the proposal model),
not the panel bbox, so the choice of slice vs tile does not change audit.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class Panel:
    coord_vox: tuple[int, int, int]
    modality: str
    plane: str                        # "axial" | "coronal" | "sagittal"
    image: np.ndarray                 # 2D float32
    crop_bbox: tuple[slice, slice, slice]


class ViewerTool:
    def __init__(
        self,
        study_images: dict[str, np.ndarray],
        crop_size_vox: tuple[int, int, int] = (48, 48, 48),
        default_tile_size: int = 64,
    ):
        self.imgs = study_images
        self.crop = tuple(crop_size_vox)
        self.default_tile_size = int(default_tile_size)
        any_vol = next(iter(study_images.values()))
        self.shape = any_vol.shape

    def slice(self, coord_vox, modality: str, plane: str = "axial") -> Panel:
        z, y, x = (int(c) for c in coord_vox)
        vol = self.imgs[modality]
        if plane == "axial":
            img = vol[z]
            bbox = (slice(z, z + 1), slice(0, vol.shape[1]), slice(0, vol.shape[2]))
        elif plane == "coronal":
            img = vol[:, y, :]
            bbox = (slice(0, vol.shape[0]), slice(y, y + 1), slice(0, vol.shape[2]))
        elif plane == "sagittal":
            img = vol[:, :, x]
            bbox = (slice(0, vol.shape[0]), slice(0, vol.shape[1]), slice(x, x + 1))
        else:
            raise ValueError(f"unknown plane: {plane}")
        return Panel(
            coord_vox=(z, y, x),
            modality=modality, plane=plane,
            image=img.astype(np.float32),
            crop_bbox=bbox,
        )

    def tile(
        self,
        coord_vox,
        modality: str,
        plane: str = "axial",
        size: int | None = None,
    ) -> Panel:
        z, y, x = (int(c) for c in coord_vox)
        vol = self.imgs[modality]
        s = int(size or self.default_tile_size)
        half = s // 2
        if plane == "axial":
            sl = vol[z]
            yl, yh = max(0, y - half), min(sl.shape[0], y + half)
            xl, xh = max(0, x - half), min(sl.shape[1], x + half)
            img = sl[yl:yh, xl:xh]
            bbox = (slice(z, z + 1), slice(yl, yh), slice(xl, xh))
        elif plane == "coronal":
            sl = vol[:, y, :]
            zl, zh = max(0, z - half), min(sl.shape[0], z + half)
            xl, xh = max(0, x - half), min(sl.shape[1], x + half)
            img = sl[zl:zh, xl:xh]
            bbox = (slice(zl, zh), slice(y, y + 1), slice(xl, xh))
        elif plane == "sagittal":
            sl = vol[:, :, x]
            zl, zh = max(0, z - half), min(sl.shape[0], z + half)
            yl, yh = max(0, y - half), min(sl.shape[1], y + half)
            img = sl[zl:zh, yl:yh]
            bbox = (slice(zl, zh), slice(yl, yh), slice(x, x + 1))
        else:
            raise ValueError(f"unknown plane: {plane}")
        return Panel(
            coord_vox=(z, y, x),
            modality=modality, plane=plane,
            image=img.astype(np.float32),
            crop_bbox=bbox,
        )

    def crop3d(self, coord_vox, modality: str) -> np.ndarray:
        bbox = self._crop_bbox(coord_vox)
        return self.imgs[modality][bbox].astype(np.float32)

    def tile_volume(
        self,
        coord_vox,
        vol_3d: np.ndarray,
        plane: str = "axial",
        size: int | None = None,
    ) -> np.ndarray | None:
        """Same indexing logic as `tile()` but on an arbitrary 3D array (e.g. a
        binary mask). Returns the 2D slice tile (no normalisation, no Panel
        wrapping). Returns None if `vol_3d` is None.
        """
        if vol_3d is None:
            return None
        z, y, x = (int(c) for c in coord_vox)
        s = int(size or self.default_tile_size)
        half = s // 2
        if plane == "axial":
            sl = vol_3d[z]
            yl, yh = max(0, y - half), min(sl.shape[0], y + half)
            xl, xh = max(0, x - half), min(sl.shape[1], x + half)
            return sl[yl:yh, xl:xh]
        if plane == "coronal":
            sl = vol_3d[:, y, :]
            zl, zh = max(0, z - half), min(sl.shape[0], z + half)
            xl, xh = max(0, x - half), min(sl.shape[1], x + half)
            return sl[zl:zh, xl:xh]
        if plane == "sagittal":
            sl = vol_3d[:, :, x]
            zl, zh = max(0, z - half), min(sl.shape[0], z + half)
            yl, yh = max(0, y - half), min(sl.shape[1], y + half)
            return sl[zl:zh, yl:yh]
        raise ValueError(f"unknown plane: {plane}")

    def panel(
        self,
        coord_vox,
        modalities=None,
        plane: str = "axial",
        tile_size: int | None = 64,
    ) -> list[Panel]:
        """Default returns centered 2D tiles. Pass tile_size=None for full slices."""
        mods = list(modalities) if modalities is not None else list(self.imgs)
        if tile_size is None:
            return [self.slice(coord_vox, m, plane) for m in mods]
        return [self.tile(coord_vox, m, plane, size=tile_size) for m in mods]

    def _crop_bbox(self, coord_vox) -> tuple[slice, slice, slice]:
        half = [s // 2 for s in self.crop]
        out = []
        for c, h, dim in zip(coord_vox, half, self.shape):
            lo = max(0, int(c) - h)
            hi = min(dim, int(c) + h)
            out.append(slice(lo, hi))
        return tuple(out)
