"""Lesion-level evaluation.

Conventions:
  - `preds` is a list of objects with `.coord_vox`, `.prob`, optionally `.mask`
  - `gt` is a list of LesionInstance (so it has `.centroid_vox`, `.mask`,
    `.voxel_count`)
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class MatchResult:
    matched_gt: list[int]
    matched_pred: list[int]
    pred_to_gt: dict[int, int]
    iou_per_match: list[float]


def _mask_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def match_predictions(
    preds,
    gt,
    iou_threshold: float = 0.1,
    dist_vox_threshold: float = 10.0,
    use_input_order: bool = True,
) -> MatchResult:
    """Greedy matching.

    If `use_input_order` (the default), iterates `preds` in the order given - so
    a ranked list from the agent is respected. Set `use_input_order=False` to
    iterate by detector probability descending (legacy behaviour; useful for
    detector-only baselines whose `.prob` is the canonical score).
    """
    if use_input_order:
        pred_order = list(range(len(preds)))
    else:
        pred_order = sorted(range(len(preds)), key=lambda i: -getattr(preds[i], "prob", 0.0))
    matched_gt: set[int] = set()
    matched_pred: list[int] = []
    pred_to_gt: dict[int, int] = {}
    iou_list: list[float] = []
    for pi in pred_order:
        p = preds[pi]
        best_gi, best_iou, best_dist = None, 0.0, float("inf")
        for gi, g in enumerate(gt):
            if gi in matched_gt:
                continue
            d = float(np.linalg.norm(
                np.asarray(p.coord_vox, dtype=np.float64)
                - np.asarray(g.centroid_vox, dtype=np.float64)
            ))
            iou = _mask_iou(getattr(p, "mask", None), getattr(g, "mask", None))
            if iou >= iou_threshold or d <= dist_vox_threshold:
                if iou > best_iou or (iou == best_iou and d < best_dist):
                    best_gi, best_iou, best_dist = gi, iou, d
        if best_gi is not None:
            matched_gt.add(best_gi)
            matched_pred.append(pi)
            pred_to_gt[pi] = best_gi
            iou_list.append(best_iou)
    return MatchResult(
        matched_gt=sorted(matched_gt),
        matched_pred=matched_pred,
        pred_to_gt=pred_to_gt,
        iou_per_match=iou_list,
    )


def additional_lesion_recall_at_fp(per_study_results, fp_per_study: int = 5) -> float:
    """Mean per-study recall after capping false positives.

    Trusts the order of `r["preds"]` - so the agent's ranker output is respected.
    Detector-only baselines should pass `preds` already sorted by detector prob
    (probmap_to_candidates does this).

    per_study_results: iterable of dicts {"preds": [...], "non_seed_gt": [...]}
    """
    recalls = []
    for r in per_study_results:
        gt = r["non_seed_gt"]
        if not gt:
            continue
        kept = []
        for p in list(r["preds"]):
            kept.append(p)
            tp = len(match_predictions(kept, gt).matched_gt)
            fp = len(kept) - tp
            if fp > fp_per_study:
                kept.pop()
                break
        tp_final = len(match_predictions(kept, gt).matched_gt)
        recalls.append(tp_final / len(gt))
    return float(np.mean(recalls)) if recalls else 0.0


def froc(per_study_results, fp_thresholds=(0.5, 1, 2, 4, 8, 16)) -> dict[float, float]:
    return {fp: additional_lesion_recall_at_fp(per_study_results, fp) for fp in fp_thresholds}


def top_k_recall(per_study_results, k: int = 5) -> float:
    """Mean per-study recall over the first k preds (trusts input order)."""
    recalls = []
    for r in per_study_results:
        gt = r["non_seed_gt"]
        if not gt:
            continue
        topk = list(r["preds"])[:k]
        recalls.append(len(match_predictions(topk, gt).matched_gt) / len(gt))
    return float(np.mean(recalls)) if recalls else 0.0


def mean_reciprocal_rank(per_study_results) -> float:
    """1/rank-of-first-TP (trusts input order)."""
    rrs = []
    for r in per_study_results:
        gt = r["non_seed_gt"]
        if not gt:
            continue
        rr = 0.0
        for rank, p in enumerate(list(r["preds"]), start=1):
            if match_predictions([p], gt).matched_gt:
                rr = 1.0 / rank
                break
        rrs.append(rr)
    return float(np.mean(rrs)) if rrs else 0.0
