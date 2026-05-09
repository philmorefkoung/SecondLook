"""Run the discrepancy-hunter agent on one study.

Usage (mock backend):
  python scripts/run_agent.py \
      --root C:/Users/User/Documents/UCSF_BMSR/UCSF_BrainMetastases_TRAIN \
      --study 100101A \
      --backend mock
"""
from __future__ import annotations
import argparse
import json

import numpy as np
from scipy import ndimage as ndi

from brain_mets_agent.data import (
    load_study, extract_instances, select_seed, MODALITIES,
)
from brain_mets_agent.orchestrator import (
    DiscrepancyHunterAgent, AgentConfig,
)
from brain_mets_agent.orchestrator.tools import (
    ViewerTool, ProposalTool, VerifierTool, HeuristicVLM, AnthropicVLM,
)
from brain_mets_agent.orchestrator.tools.ranker import rank_candidates
from brain_mets_agent.eval import cited_panels_overlap


def build_proposal(backend: str, study_seg):
    if backend == "mock":
        # gaussian-blurred GT mask -> peaks line up with true lesions
        def predict(multi_modal):
            return ndi.gaussian_filter((study_seg > 0).astype(np.float32), sigma=1.0)
        return ProposalTool(predict)
    if backend == "swin":
        from brain_mets_agent.models import LesionProposalModel
        model = LesionProposalModel()
        return ProposalTool(model.predict_probmap)
    raise ValueError(backend)


def build_verifier(backend: str):
    if backend == "mock" or backend == "heuristic":
        return VerifierTool(HeuristicVLM())
    if backend == "anthropic":
        return VerifierTool(AnthropicVLM())
    raise ValueError(backend)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--study", required=True)
    ap.add_argument("--backend", choices=["mock", "swin"], default="mock")
    ap.add_argument("--vlm", choices=["heuristic", "anthropic"], default="heuristic")
    ap.add_argument("--seed-strategy", choices=["largest", "random"], default="largest")
    args = ap.parse_args()

    study = load_study(args.root, args.study)
    print(f"Loaded {study.study_id}: shape={study.images['T1post'].shape} "
          f"spacing={study.spacing}")

    inst = extract_instances(study.seg, study.spacing, study.affine)
    print(f"GT lesion instances: {len(inst)} (top sizes mm3: "
          f"{[round(i.volume_mm3, 1) for i in inst[:5]]})")
    seed, others = select_seed(inst, args.seed_strategy)
    if seed is None:
        print("No GT lesions found; nothing to seed.")
        return
    print(f"Seed: vol={seed.volume_mm3:.1f}mm3 at vox={tuple(round(c, 1) for c in seed.centroid_vox)}; "
          f"{len(others)} additional GT lesion(s) remain")

    agent = DiscrepancyHunterAgent(
        proposal=build_proposal(args.backend, study.seg),
        viewer_factory=lambda imgs: ViewerTool(imgs),
        verifier=build_verifier(args.vlm),
        modalities=MODALITIES,
        cfg=AgentConfig(),
    )
    state = agent.run(
        study_id=study.study_id,
        study_images=study.images,
        seed_mask=seed.mask,
        seed_coord_vox=tuple(int(round(c)) for c in seed.centroid_vox),
        study_meta={"affine": study.affine, "spacing_mm": study.spacing},
    )

    ranked = rank_candidates(state.candidates)
    audit = cited_panels_overlap(state, study.seg)

    print(json.dumps({
        "summary": state.to_summary(),
        "ranked_top10": [
            {"coord": list(c.coord_vox), "verdict": c.verdict,
             "vlm_conf": round(c.vlm_conf, 3), "prob": round(c.prob, 3),
             "voxels": c.voxel_count, "pass": c.pass_idx}
            for c in ranked[:10]
        ],
        "auditability": audit,
    }, indent=2))


if __name__ == "__main__":
    main()
