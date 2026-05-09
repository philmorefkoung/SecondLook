"""Final ranker for the additional-lesion list.

Score combines verdict, VLM/detector confidence, candidate size, and the
phenotype-similarity-to-seed feature, then subtracts a mimic-risk penalty.
"""
from __future__ import annotations
from typing import Sequence

from ..state import TrackedCandidate


DEFAULT_VERDICT_SCORE = {"confirm": 1.0, "uncertain": 0.5, "reject": 0.0, None: 0.25}
DEFAULT_MIMIC_PENALTY = {"low": 0.0, "medium": 0.15, "high": 0.4}


def rank_candidates(
    tracked: Sequence[TrackedCandidate],
    prefer_confirmed: bool = True,
    weights: tuple[float, float, float, float] = (0.45, 0.20, 0.10, 0.25),
    verdict_scores: dict | None = None,
    mimic_penalties: dict | None = None,
    prob_source: str = "vlm_else_detector",
) -> list[TrackedCandidate]:
    """Score = w_v*verdict + w_p*prob_term + w_s*size_norm
              + w_sim*seed_similarity - mimic_penalty.

    weights = (verdict, prob, size, similarity).
    `verdict_scores` and `mimic_penalties` override the defaults; missing keys
    fall back. Set `prefer_confirmed=False` to keep rejected candidates in the
    ranked list (they will still get verdict_score 0 by default).

    `prob_source` controls how the prob_term is computed:
      - "vlm_else_detector" (default): vlm_conf if non-zero else detector prob
      - "detector_only": always detector prob (immune to VLM-confidence noise)
      - "vlm_only": always vlm_conf (0 if not verified)
      - "average": 0.5 * vlm_conf + 0.5 * detector prob
    """
    if not tracked:
        return []
    vs = {**DEFAULT_VERDICT_SCORE, **(verdict_scores or {})}
    mp = {**DEFAULT_MIMIC_PENALTY, **(mimic_penalties or {})}
    max_count = max((c.voxel_count for c in tracked), default=1) or 1
    w_v, w_p, w_s, w_sim = weights
    scored: list[tuple[float, TrackedCandidate]] = []
    for c in tracked:
        if prefer_confirmed and c.verdict == "reject":
            continue
        if prob_source == "detector_only":
            prob_term = c.prob
        elif prob_source == "vlm_only":
            prob_term = c.vlm_conf
        elif prob_source == "average":
            prob_term = 0.5 * c.vlm_conf + 0.5 * c.prob if c.vlm_conf else c.prob
        else:  # "vlm_else_detector"
            prob_term = c.vlm_conf if c.vlm_conf else c.prob
        sim = float(getattr(c, "seed_similarity", 0.0) or 0.0)
        mimic = mp.get(getattr(c, "mimic_risk", "low") or "low", 0.0)
        score = (
            w_v * vs[c.verdict]
            + w_p * prob_term
            + w_s * (c.voxel_count / max_count)
            + w_sim * sim
            - mimic
        )
        scored.append((score, c))
    scored.sort(key=lambda kv: kv[0], reverse=True)
    return [c for _, c in scored]
