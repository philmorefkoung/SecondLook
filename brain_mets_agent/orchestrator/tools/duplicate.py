"""Duplicate checker: candidate vs seed/known lesions."""
from __future__ import annotations
from typing import Sequence, Optional

import numpy as np


def is_duplicate(
    cand_coord: tuple[float, float, float],
    cand_mask: Optional[np.ndarray],
    known_coords: Sequence[tuple[float, float, float]],
    known_masks: Sequence[Optional[np.ndarray]],
    distance_threshold_vox: float = 5.0,
    iou_threshold: float = 0.3,
) -> bool:
    cand = np.asarray(cand_coord, dtype=np.float64)
    for kc, km in zip(known_coords, known_masks):
        d = float(np.linalg.norm(cand - np.asarray(kc, dtype=np.float64)))
        if d <= distance_threshold_vox:
            return True
        if cand_mask is not None and km is not None:
            inter = float(np.logical_and(cand_mask, km).sum())
            if inter == 0:
                continue
            union = float(np.logical_or(cand_mask, km).sum())
            if union > 0 and (inter / union) >= iou_threshold:
                return True
    return False
