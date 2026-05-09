"""One-shot: compute baseline + agent metrics from cached Stanford state pickles.

Used when scripts/evaluate.py crashed post-loop but the per-study state pickles
are intact. Re-derives baseline candidates from raw nnUNet output, re-ranks
agent candidates from the pickle, and aggregates standard metrics.
"""
from __future__ import annotations
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

from brain_mets_agent.models import NNUNetProbmapCache
from brain_mets_agent.models.proposal import probmap_to_candidates
from brain_mets_agent.orchestrator.tools.ranker import rank_candidates
from brain_mets_agent.eval.metrics import (
    additional_lesion_recall_at_fp,
    top_k_recall,
    mean_reciprocal_rank,
)


@dataclass
class _GTLite:
    centroid_vox: tuple
    voxel_count: int
    volume_mm3: float
    mask: object


def main():
    cache = NNUNetProbmapCache("runs/nnunet_probmap_stanford")
    state_dir = Path("runs/state_stanford_dpo_v4")

    baseline_results, agent_results = [], []
    for p in sorted(state_dir.glob("*.pkl")):
        with p.open("rb") as f:
            rec = pickle.load(f)
        rec["non_seed_gt"] = [_GTLite(**g) for g in rec["non_seed_gt"]]
        sid = rec["study_id"]
        prob = cache.predict_probmap_for(sid)
        bcands = probmap_to_candidates(prob, threshold=0.3, min_voxels=10)
        # Strip masks: with full-volume bool masks the metric does O(N^2*K) IoU
        # on 10M-voxel arrays which takes hours on Stanford. Centroid-distance
        # matching with the same dist_vox_threshold=10 the metric uses is
        # near-equivalent and runs in seconds.
        for c in bcands:
            c.mask = None
        for g in rec["non_seed_gt"]:
            g.mask = None
        baseline_results.append({"study_id": sid, "preds": bcands,
                                  "non_seed_gt": rec["non_seed_gt"]})
        ranked = rank_candidates(
            rec["candidates"],
            prefer_confirmed=False,
            weights=(0.45, 0.20, 0.10, 0.25),
            prob_source="detector_only",
        )
        agent_results.append({"study_id": sid, "preds": ranked,
                               "non_seed_gt": rec["non_seed_gt"]})

    def agg(results):
        return {
            "r@.5":  additional_lesion_recall_at_fp(results, 0.5),
            "r@1":   additional_lesion_recall_at_fp(results, 1),
            "r@2":   additional_lesion_recall_at_fp(results, 2),
            "r@4":   additional_lesion_recall_at_fp(results, 4),
            "r@5":   additional_lesion_recall_at_fp(results, 5),
            "r@8":   additional_lesion_recall_at_fp(results, 8),
            "r@10":  additional_lesion_recall_at_fp(results, 10),
            "r@16":  additional_lesion_recall_at_fp(results, 16),
            "top5":  top_k_recall(results, 5),
            "top10": top_k_recall(results, 10),
            "mrr":   mean_reciprocal_rank(results),
        }

    b = agg(baseline_results)
    a = agg(agent_results)

    print(f"Stanford BrainMetShare (n={len(baseline_results)} studies w/ additional GT)")
    print(f"{'metric':<10s}  {'baseline':>10s}  {'agent_def':>10s}  {'delta':>9s}")
    for k in b:
        d = a[k] - b[k]
        sign = "+" if d > 0 else ""
        print(f"  {k:<8s}  {b[k]:>10.3f}  {a[k]:>10.3f}  {sign}{d:>8.3f}")

    Path("runs/eval_stanford_dpo_v4.json").write_text(
        json.dumps({"n_studies": len(baseline_results),
                    "baseline": b, "agent_default": a}, indent=2)
    )
    print("\nSaved runs/eval_stanford_dpo_v4.json")


if __name__ == "__main__":
    main()
