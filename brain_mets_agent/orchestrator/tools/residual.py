"""Residual-search-space updater.

Build and update a [0,1] map over the volume that encodes "where the agent
should still look". After verdicts are collected:
  - confirmed regions are strongly suppressed (already accounted for)
  - rejected regions are softly downweighted (don't waste budget revisiting)
  - uncertain regions remain eligible for re-inspection
"""
from __future__ import annotations
from typing import Sequence, Optional

import numpy as np
from scipy import ndimage as ndi


def initial_residual(
    volume_shape: tuple[int, int, int],
    seed_mask: Optional[np.ndarray],
    exclude_radius_vox: int = 8,
) -> np.ndarray:
    r = np.ones(volume_shape, dtype=np.float32)
    if seed_mask is not None:
        suppress = ndi.binary_dilation(seed_mask, iterations=exclude_radius_vox)
        r[suppress] = 0.0
    return r


def update_residual(
    residual: np.ndarray,
    confirmed_masks: Sequence[np.ndarray],
    rejected_masks: Sequence[np.ndarray],
    uncertain_masks: Sequence[np.ndarray],
    confirmed_weight: float = 0.0,
    rejected_weight: float = 0.3,
    uncertain_weight: float = 0.8,
    dilate: int = 4,
) -> np.ndarray:
    out = residual.copy()
    for m in confirmed_masks:
        out[ndi.binary_dilation(m, iterations=dilate)] = confirmed_weight
    for m in rejected_masks:
        out[ndi.binary_dilation(m, iterations=max(1, dilate // 2))] *= rejected_weight
    for m in uncertain_masks:
        out[m] = np.maximum(out[m], uncertain_weight)
    return out


def topk_unexplored(
    residual: np.ndarray,
    prob_map: np.ndarray,
    k: int = 10,
    suppress_radius: int = 6,
    min_value: float = 0.05,
) -> list[tuple[tuple[int, int, int], float]]:
    """Pick top-k local maxima of (residual * prob_map) with NMS."""
    score = (residual * prob_map).astype(np.float32)
    out: list[tuple[tuple[int, int, int], float]] = []
    s = score.copy()
    for _ in range(k):
        peak = float(s.max())
        if peak < min_value:
            break
        idx = np.unravel_index(int(np.argmax(s)), s.shape)
        out.append((tuple(int(i) for i in idx), peak))
        z, y, x = idx
        sl = (
            slice(max(0, z - suppress_radius), z + suppress_radius + 1),
            slice(max(0, y - suppress_radius), y + suppress_radius + 1),
            slice(max(0, x - suppress_radius), x + suppress_radius + 1),
        )
        s[sl] = 0.0
    return out
