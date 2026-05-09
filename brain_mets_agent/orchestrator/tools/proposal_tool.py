"""Wrapper exposing a model (or any predictor callable) as a stable tool."""
from __future__ import annotations
from typing import Callable

import numpy as np

from ...models.proposal import probmap_to_candidates


Predictor = Callable[[np.ndarray], np.ndarray]   # multi_modal -> probmap


class ProposalTool:
    """Holds either a real model.predict_probmap or a callable substitute."""
    def __init__(self, predictor: Predictor):
        self.predictor = predictor

    def first_pass(
        self,
        multi_modal: np.ndarray,
        threshold: float = 0.5,
        min_voxels: int = 10,
    ):
        prob = self.predictor(multi_modal)
        cands = probmap_to_candidates(prob, threshold=threshold, min_voxels=min_voxels)
        return prob, cands
