"""Evaluate the discrepancy-hunter on a split.

Compares two configurations on the same studies:
  * baseline: detector-only (raw probmap candidates, no agent)
  * agent:    rule-based or llm-driven orchestrator output (rank_candidates)

Reports per-study and aggregate FROC + recall@FP/study + top-k recall + MRR
+ auditability rate. Saves a JSON report.

Usage:
  python scripts/evaluate.py \
      --root C:/Users/User/Documents/UCSF_BMSR/UCSF_BrainMetastases_TRAIN \
      --splits brain_mets_agent/data/splits.csv --split val \
      --predictor mock \
      --agent rule \
      --limit 5 --out runs/eval_mock.json
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from tqdm import tqdm

from brain_mets_agent.data import (
    load_study, extract_instances, select_seed, MODALITIES,
)
from brain_mets_agent.data.splits import read_splits_csv
from brain_mets_agent.models.proposal import probmap_to_candidates
from brain_mets_agent.orchestrator import (
    DiscrepancyHunterAgent, LLMDrivenAgent, AgentConfig,
    MockLLM, AnthropicBackend,
)
from brain_mets_agent.orchestrator.tools import (
    ViewerTool, ProposalTool, VerifierTool, HeuristicVLM, AnthropicVLM,
)
from brain_mets_agent.orchestrator.tools.verifier import LocalQwenVLM
from brain_mets_agent.orchestrator.tools.ranker import rank_candidates
from brain_mets_agent.eval import (
    additional_lesion_recall_at_fp, froc, top_k_recall,
    mean_reciprocal_rank, cited_panels_overlap,
)


# ---------- predictor and verifier factories ----------

def build_predictor(kind: str, ckpt: str | None,
                     nnunet_probmap_dir: str | None = None):
    """Returns (factory, model) where factory(study) -> predict_fn(multi_modal)->probmap.

    The factory takes the full Study so backends that need non-image context
    (mock needs `seg` to cheat; nnunet needs `study_id` to look up cached
    probmap) can specialize. Backends that don't need it (swin) just ignore.
    """
    if kind == "mock":
        def make(study):
            seg = study.seg
            def predict(multi_modal: np.ndarray) -> np.ndarray:
                return ndi.gaussian_filter((seg > 0).astype(np.float32), sigma=1.0)
            return predict
        return make, None
    if kind == "swin":
        from brain_mets_agent.models import LesionProposalModel
        model = LesionProposalModel()
        if ckpt:
            model.load_state(ckpt)
        return (lambda study: model.predict_probmap), model
    if kind == "nnunet":
        from brain_mets_agent.models import NNUNetProbmapCache
        if not nnunet_probmap_dir:
            raise ValueError("--nnunet-probmap-dir required for --predictor nnunet")
        # Comma-separated list of dirs -> ensemble; single dir -> single source.
        dirs = [d.strip() for d in nnunet_probmap_dir.split(",") if d.strip()]
        if len(dirs) == 1:
            cache = NNUNetProbmapCache(dirs[0])
        else:
            from brain_mets_agent.models import EnsembleProbmapCache
            cache = EnsembleProbmapCache(*dirs)
        def make(study):
            sid = study.study_id
            def predict(multi_modal: np.ndarray) -> np.ndarray:
                return cache.predict_probmap_for(sid)
            return predict
        return make, cache
    raise ValueError(kind)


def build_verifier(kind: str, *, adapter_path: str | None = None,
                   base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
    if kind == "heuristic":
        return VerifierTool(HeuristicVLM())
    if kind == "anthropic":
        return VerifierTool(AnthropicVLM())
    if kind == "local-qwen":
        return VerifierTool(LocalQwenVLM(base_model=base_model, adapter_path=adapter_path))
    raise ValueError(kind)


# ---------- baseline + agent runners ----------

@dataclass
class StudyResult:
    study_id: str
    n_gt: int
    n_seed_lesions: int          # 1 if seed selected
    n_non_seed_gt: int
    baseline_preds: list
    agent_preds: list
    agent_audit: dict[str, Any]
    agent_actions: int
    agent_budget_used: int


def _save_study_state(out_dir: str, study_id: str, state, non_seed_gt, seed):
    """Pickle a per-study record sufficient to re-rank offline.

    Strips heavy mask arrays from candidates (we keep coord + bbox + verdict
    + structured fields, which is everything the ranker reads). For matching
    we keep gt centroids + voxel counts + light masks too.
    """
    import pickle
    from copy import copy
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    light_cands = []
    for c in state.candidates:
        cc = copy(c)
        cc.mask = None  # too large to pickle 5x per study
        light_cands.append(cc)

    light_gt = []
    for g in non_seed_gt:
        light_gt.append({
            "centroid_vox": tuple(g.centroid_vox),
            "voxel_count": int(g.voxel_count),
            "volume_mm3": float(g.volume_mm3),
            "mask": g.mask,   # kept (used for IoU matching) - usually < a few MB compressed
        })

    record = {
        "study_id": study_id,
        "seed_coord_vox": state.seed_coord_vox,
        "seed_voxel_count": int(seed.voxel_count) if seed else 0,
        "candidates": light_cands,
        "non_seed_gt": light_gt,
        "actions": [(a.name, a.result_summary) for a in state.actions],
        "budget_used": AgentConfig().budget_actions - state.budget_remaining,
        "seed_phenotype": state.seed_phenotype,
    }
    with (out / f"{study_id}.pkl").open("wb") as f:
        pickle.dump(record, f)


def run_baseline(study, predict_fn, min_voxels: int = 10) -> list:
    multi_modal = np.stack(
        [study.images[m] for m in MODALITIES], axis=0
    ).astype(np.float32)
    prob = predict_fn(multi_modal)
    return probmap_to_candidates(prob, threshold=0.3, min_voxels=min_voxels)


def run_agent(agent_kind: str, study, seed, predictor_factory, verifier, llm=None,
              ranker_kwargs: dict | None = None, budget_actions: int = 32,
              proposal_threshold: float | None = None,
              skip_step4: bool = False, skip_step5: bool = False,
              min_voxels: int | None = None):
    multi_modal_predictor = predictor_factory(study)
    proposal = ProposalTool(multi_modal_predictor)
    seed_coord = tuple(int(round(c)) for c in seed.centroid_vox)
    cfg_kwargs = {"budget_actions": budget_actions,
                  "skip_step4": skip_step4, "skip_step5": skip_step5}
    if proposal_threshold is not None:
        cfg_kwargs["threshold"] = proposal_threshold
    if min_voxels is not None:
        cfg_kwargs["min_voxels"] = min_voxels
    cfg = AgentConfig(**cfg_kwargs)

    if agent_kind == "rule":
        agent = DiscrepancyHunterAgent(
            proposal=proposal,
            viewer_factory=lambda imgs: ViewerTool(imgs),
            verifier=verifier,
            modalities=MODALITIES,
            cfg=cfg,
        )
    elif agent_kind == "llm":
        agent = LLMDrivenAgent(
            proposal=proposal,
            viewer_factory=lambda imgs: ViewerTool(imgs),
            verifier=verifier,
            modalities=MODALITIES,
            llm=llm,
            cfg=cfg,
        )
    else:
        raise ValueError(agent_kind)

    state = agent.run(
        study_id=study.study_id,
        study_images=study.images,
        seed_mask=seed.mask,
        seed_coord_vox=seed_coord,
        study_meta={"affine": study.affine, "spacing_mm": study.spacing},
    )
    ranked = rank_candidates(state.candidates, **(ranker_kwargs or {}))
    audit = cited_panels_overlap(state, study.seg)
    return state, ranked, audit


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--predictor", choices=["mock", "swin", "nnunet"], default="mock")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--nnunet-probmap-dir", default=None,
                    help="Directory of pre-computed nnU-Net softmax probmaps "
                         "(<study_id>.npz, key=softmax). Required for "
                         "--predictor nnunet. See scripts/run_nnunet_inference.py. "
                         "Pass a comma-separated list of dirs to ensemble multiple "
                         "detector sources via union-of-binary-segs.")
    ap.add_argument("--agent", choices=["rule", "llm"], default="llm",
                    help="Default is the LLM-driven agent (rule-based kept for ablation only).")
    ap.add_argument("--llm-backend", choices=["anthropic", "mock"], default="anthropic",
                    help="Only used when --agent llm.")
    ap.add_argument("--llm-model", default="claude-sonnet-4-6")
    ap.add_argument("--vlm", choices=["heuristic", "anthropic", "local-qwen"],
                    default="heuristic")
    ap.add_argument("--vlm-adapter", default=None,
                    help="LoRA adapter path for --vlm local-qwen.")
    ap.add_argument("--vlm-base-model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                    help="Base model id for --vlm local-qwen.")
    # Ranker tuning (defaults preserve v1 behaviour)
    ap.add_argument("--verdict-confirm", type=float, default=1.0)
    ap.add_argument("--verdict-uncertain", type=float, default=0.5)
    # v7-best defaults (set by sweep on n=13): no mimic penalty, keep rejected.
    ap.add_argument("--mimic-medium", type=float, default=0.0)
    ap.add_argument("--mimic-high", type=float, default=0.0)
    ap.add_argument("--prefer-confirmed", action=argparse.BooleanOptionalAction, default=False,
                    help="Default off: rejected candidates stay in the ranking with "
                         "verdict_score 0 so VLM-false-rejects can still surface via "
                         "detector probability.")
    ap.add_argument("--prob-source",
                    choices=["vlm_else_detector", "detector_only", "vlm_only", "average"],
                    default="detector_only",
                    help="Score formula's probability term. detector_only is the "
                         "current sweep best (vlm_conf appears uncalibrated).")
    ap.add_argument("--weights-preset",
                    choices=["verdict_heavy", "balanced", "prob_heavy", "very_prob_heavy"],
                    default="verdict_heavy",
                    help="Ranker weight tuple preset (verdict, prob, size, similarity). "
                         "verdict_heavy is the v10-best when the VLM is SFT'd and reliable.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0,
                    help="Skip the first N studies in the split (useful to avoid "
                         "studies whose candidates were used in SFT training).")
    ap.add_argument("--seed-strategy", choices=["largest", "random"], default="largest")
    ap.add_argument("--budget-actions", type=int, default=32,
                    help="Verifier budget per study (max number of VLM calls).")
    ap.add_argument("--proposal-threshold", type=float, default=None,
                    help="Override AgentConfig.threshold for step-2 first-pass proposal "
                         "(default 0.5). Try 0.2 for pool expansion.")
    ap.add_argument("--min-voxels", type=int, default=None,
                    help="Override min_voxels for both baseline candidate extraction "
                         "AND AgentConfig.min_voxels in step-2. Default 10. Try 3 to "
                         "admit smaller CCs from the detector (rescue-experiment).")
    ap.add_argument("--skip-step4", action="store_true",
                    help="Skip lower-threshold pass (step 4). Free budget for step 2.")
    ap.add_argument("--skip-step5", action="store_true",
                    help="Skip residual-peaks pass (step 5). Empirically zero recall "
                         "uplift under prob-heavy ranking.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip studies whose pickle already exists in --save-state-dir.")
    ap.add_argument("--inter-study-sleep", type=float, default=0.0,
                    help="Seconds to sleep between studies (helps avoid token-rate "
                         "limits on LLM-driven runs).")
    ap.add_argument("--out", default="runs/eval.json")
    ap.add_argument("--save-state-dir", default=None,
                    help="If set, pickle per-study (state.candidates without masks, "
                         "non_seed_gt centroids, seed metadata) so a sweep script "
                         "can re-rank offline without re-calling the VLM.")
    args = ap.parse_args()

    splits = read_splits_csv(args.splits)
    study_ids = splits[args.split]
    if args.offset:
        study_ids = study_ids[args.offset:]
    if args.limit:
        study_ids = study_ids[:args.limit]
    print(f"Evaluating {len(study_ids)} study(s) from split={args.split} "
          f"(offset={args.offset}, limit={args.limit})")

    predictor_factory, _ = build_predictor(args.predictor, args.ckpt,
                                              nnunet_probmap_dir=args.nnunet_probmap_dir)
    verifier = build_verifier(args.vlm, adapter_path=args.vlm_adapter,
                                base_model=args.vlm_base_model)
    if args.agent == "llm":
        llm = (AnthropicBackend(model=args.llm_model)
               if args.llm_backend == "anthropic" else MockLLM([]))
    else:
        llm = None

    baseline_results = []
    agent_results = []
    audit_rates = []
    actions_per_study = []
    budget_used_per_study = []

    import time
    for sid in tqdm(study_ids, desc="eval"):
        if args.resume and args.save_state_dir:
            existing = Path(args.save_state_dir) / f"{sid}.pkl"
            if existing.exists():
                continue
        study = load_study(args.root, sid)
        if study.seg is None or not (study.seg > 0).any():
            continue
        inst = extract_instances(study.seg, study.spacing, study.affine)
        if not inst:
            continue
        seed, others = select_seed(inst, args.seed_strategy)
        if not others:
            continue   # nothing to find beyond the seed
        if args.inter_study_sleep > 0:
            time.sleep(args.inter_study_sleep)

        # baseline: detector-only
        bcands = run_baseline(study, predictor_factory(study),
                                min_voxels=args.min_voxels if args.min_voxels is not None else 10)
        baseline_results.append({"study_id": sid, "preds": bcands, "non_seed_gt": others})

        # agent
        weight_presets = {
            "verdict_heavy":    (0.45, 0.20, 0.10, 0.25),
            "balanced":         (0.30, 0.30, 0.15, 0.25),
            "prob_heavy":       (0.20, 0.50, 0.10, 0.20),
            "very_prob_heavy":  (0.10, 0.70, 0.10, 0.10),
        }
        ranker_kwargs = {
            "verdict_scores": {"confirm": args.verdict_confirm,
                                "uncertain": args.verdict_uncertain},
            "mimic_penalties": {"low": 0.0, "medium": args.mimic_medium,
                                 "high": args.mimic_high},
            "prob_source": args.prob_source,
            "weights": weight_presets[args.weights_preset],
            "prefer_confirmed": args.prefer_confirmed,
        }
        state, ranked, audit = run_agent(
            args.agent, study, seed, predictor_factory, verifier, llm,
            ranker_kwargs=ranker_kwargs, budget_actions=args.budget_actions,
            proposal_threshold=args.proposal_threshold,
            skip_step4=args.skip_step4, skip_step5=args.skip_step5,
            min_voxels=args.min_voxels,
        )
        agent_results.append({"study_id": sid, "preds": ranked, "non_seed_gt": others})
        audit_rates.append(audit["rate"])
        actions_per_study.append(len(state.actions))
        budget_used_per_study.append(args.budget_actions - state.budget_remaining)

        if args.save_state_dir:
            _save_study_state(args.save_state_dir, sid, state, others, seed)

    def aggregate(results):
        return {
            "n_studies_with_additional": len(results),
            "froc": froc(results),
            "recall_at_fp_5": additional_lesion_recall_at_fp(results, 5),
            "recall_at_fp_2": additional_lesion_recall_at_fp(results, 2),
            "top5_recall": top_k_recall(results, 5),
            "top10_recall": top_k_recall(results, 10),
            "mrr": mean_reciprocal_rank(results),
        }

    report = {
        "config": vars(args),
        "n_studies_evaluated": len(baseline_results),
        "baseline": aggregate(baseline_results),
        "agent": aggregate(agent_results),
        "agent_audit": {
            "mean_overlap_rate": float(np.mean(audit_rates)) if audit_rates else 0.0,
            "mean_actions_per_study": float(np.mean(actions_per_study)) if actions_per_study else 0.0,
            "mean_budget_used": float(np.mean(budget_used_per_study)) if budget_used_per_study else 0.0,
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
