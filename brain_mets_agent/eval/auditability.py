"""Auditability: do the agent's cited regions actually intersect GT lesions?"""
from __future__ import annotations

import numpy as np


def cited_panels_overlap(state, gt_seg: np.ndarray) -> dict:
    """For each non-rejected candidate, check whether its bbox intersects gt_seg."""
    n_cited = 0
    n_overlap = 0
    for c in state.candidates:
        if c.verdict not in ("confirm", "uncertain"):
            continue
        n_cited += 1
        if c.bbox is None:
            continue
        if gt_seg[c.bbox].any():
            n_overlap += 1
    return {
        "cited": n_cited,
        "with_gt_overlap": n_overlap,
        "rate": (n_overlap / n_cited) if n_cited else 0.0,
    }
