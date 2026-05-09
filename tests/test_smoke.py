"""Mock end-to-end test: synthetic 3D phantom with two lesions.

Verifies the orchestrator's plumbing - proposal -> duplicate -> viewer -> VLM
-> residual update -> ranker - without requiring torch/MONAI/Anthropic.
"""
from __future__ import annotations
import numpy as np
import pytest
from scipy import ndimage as ndi

from brain_mets_agent.data import extract_instances, select_seed
from brain_mets_agent.orchestrator import DiscrepancyHunterAgent, AgentConfig
from brain_mets_agent.orchestrator.tools import (
    ViewerTool, ProposalTool, VerifierTool,
)
from brain_mets_agent.orchestrator.tools.verifier import VerifierVerdict
from brain_mets_agent.orchestrator.tools.ranker import rank_candidates


def _make_phantom():
    rng = np.random.RandomState(0)
    shape = (64, 64, 64)
    seg = np.zeros(shape, dtype=np.int32)
    seg[20:30, 20:30, 20:30] = 1     # large "seed"
    seg[40:43, 40:43, 40:43] = 1     # smaller additional lesion
    base = rng.randn(*shape).astype(np.float32) * 0.05
    enhance = (seg > 0).astype(np.float32)
    images = {
        "T1pre":       base.copy(),
        "T1post":      base + enhance,
        "FLAIR":       base + 0.5 * enhance,
        "subtraction": base + enhance,
    }
    affine = np.eye(4)
    spacing = (1.0, 1.0, 1.0)
    return images, seg, affine, spacing


class _ConfirmingVLM:
    def verify(self, panels, ctx):
        return VerifierVerdict(label="confirm", confidence=0.9, rationale="mock confirm")


class _RejectingVLM:
    def verify(self, panels, ctx):
        return VerifierVerdict(label="reject", confidence=0.9, rationale="mock reject")


def _mock_proposal_tool(seg):
    def predict(multi_modal):
        return ndi.gaussian_filter((seg > 0).astype(np.float32), sigma=1.0)
    return ProposalTool(predict)


def _build_agent(seg, vlm):
    return DiscrepancyHunterAgent(
        proposal=_mock_proposal_tool(seg),
        viewer_factory=lambda imgs: ViewerTool(imgs),
        verifier=VerifierTool(vlm),
        modalities=("T1pre", "T1post", "FLAIR", "subtraction"),
        cfg=AgentConfig(threshold=0.3, second_pass_k=8),
    )


def test_lesion_extraction():
    images, seg, affine, spacing = _make_phantom()
    inst = extract_instances(seg, spacing, affine)
    assert len(inst) == 2
    assert inst[0].voxel_count > inst[1].voxel_count   # sorted desc


def test_seed_selection_separates_seed_from_rest():
    images, seg, affine, spacing = _make_phantom()
    inst = extract_instances(seg, spacing, affine)
    seed, others = select_seed(inst, "largest")
    assert seed.voxel_count == 1000      # 10**3
    assert len(others) == 1
    assert others[0].voxel_count == 27   # 3**3


def test_orchestrator_finds_additional_lesion():
    images, seg, affine, spacing = _make_phantom()
    inst = extract_instances(seg, spacing, affine)
    seed, others = select_seed(inst, "largest")

    agent = _build_agent(seg, _ConfirmingVLM())
    state = agent.run(
        study_id="phantom",
        study_images=images,
        seed_mask=seed.mask,
        seed_coord_vox=tuple(int(round(c)) for c in seed.centroid_vox),
    )

    assert len(state.candidates) >= 1, "agent produced zero candidates"
    confirmed = state.confirmed()
    assert confirmed, "no confirmed candidates"
    add_centroid = np.array(others[0].centroid_vox)
    near = [
        np.linalg.norm(np.array(c.coord_vox) - add_centroid)
        for c in confirmed
    ]
    assert any(d < 10 for d in near), f"confirmed candidates too far from additional lesion: {near}"


def test_seed_duplicate_is_rejected():
    images, seg, affine, spacing = _make_phantom()
    inst = extract_instances(seg, spacing, affine)
    seed, _ = select_seed(inst, "largest")

    agent = _build_agent(seg, _ConfirmingVLM())
    state = agent.run(
        study_id="phantom",
        study_images=images,
        seed_mask=seed.mask,
        seed_coord_vox=tuple(int(round(c)) for c in seed.centroid_vox),
    )
    rejects = [c for c in state.candidates
               if c.verdict == "reject" and "duplicate" in c.rationale.lower()]
    assert rejects, "expected at least one candidate rejected as seed-duplicate"


def test_ranker_orders_confirmed_first():
    images, seg, affine, spacing = _make_phantom()
    inst = extract_instances(seg, spacing, affine)
    seed, _ = select_seed(inst, "largest")

    agent = _build_agent(seg, _ConfirmingVLM())
    state = agent.run(
        study_id="phantom",
        study_images=images,
        seed_mask=seed.mask,
        seed_coord_vox=tuple(int(round(c)) for c in seed.centroid_vox),
    )
    ranked = rank_candidates(state.candidates)
    assert ranked, "ranker returned empty list"
    assert ranked[0].verdict in ("confirm", "uncertain")


def test_audit_trace_records_protocol_steps():
    """Rule-based agent now logs the 7-step protocol explicitly."""
    images, seg, affine, spacing = _make_phantom()
    inst = extract_instances(seg, spacing, affine)
    seed, _ = select_seed(inst, "largest")
    agent = _build_agent(seg, _ConfirmingVLM())
    state = agent.run(
        study_id="phantom",
        study_images=images,
        seed_mask=seed.mask,
        seed_coord_vox=tuple(int(round(c)) for c in seed.centroid_vox),
        study_meta={"affine": affine, "spacing_mm": spacing},
    )
    names = [a.name for a in state.actions]
    for expected in (
        "step1.characterize_seed", "step2.proposal_first_pass", "residual.init",
        "step3.residual_update", "step4.lower_threshold_pass",
        "step5.residual_topk", "step6.phenotype_compare", "step7.ready_for_rank",
        "vlm.verify",
    ):
        assert expected in names, f"missing action {expected!r}; got {names}"


# ---------- Centered tile crop ----------

def test_viewer_tile_is_centered():
    from brain_mets_agent.orchestrator.tools import ViewerTool
    images, seg, _, _ = _make_phantom()
    viewer = ViewerTool(images)
    # Tile at the additional lesion centroid (~ 41,41,41), size 16
    panel = viewer.tile((41, 41, 41), "T1post", plane="axial", size=16)
    assert panel.image.shape == (16, 16)
    # Lesion is at [40:43, 40:43, 40:43] in z=41; the centered axial tile around
    # (y=41, x=41) of size 16 gives crop [33:49, 33:49] - lesion region [40:43, 40:43]
    # falls inside, so the centered pixel (8,8 within tile) IS the lesion.
    assert panel.image[8, 8] >= 0.5


# ---------- LLM-driven agent (scripted) ----------

def test_llm_driven_agent_with_scripted_plan():
    """MockLLM walks the canonical workflow: propose -> dup -> verify -> finalize."""
    from brain_mets_agent.orchestrator import LLMDrivenAgent, MockLLM, AgentConfig
    from brain_mets_agent.orchestrator.tools import ViewerTool, VerifierTool

    images, seg, affine, spacing = _make_phantom()
    inst = extract_instances(seg, spacing, affine)
    seed, others = select_seed(inst, "largest")

    script = [
        # 1. propose
        {"stop_reason": "tool_use", "content": [
            {"type": "text", "text": "Proposing lesions."},
            {"type": "tool_use", "id": "t1", "name": "propose_lesions", "input": {"threshold": 0.3}},
        ]},
        # 2. dup-check id 0
        {"stop_reason": "tool_use", "content": [
            {"type": "tool_use", "id": "t2", "name": "check_duplicate_of_seed", "input": {"candidate_id": 0}},
        ]},
        # 3. dup-check id 1
        {"stop_reason": "tool_use", "content": [
            {"type": "tool_use", "id": "t3", "name": "check_duplicate_of_seed", "input": {"candidate_id": 1}},
        ]},
        # 4. verify id 1 (the non-seed lesion)
        {"stop_reason": "tool_use", "content": [
            {"type": "tool_use", "id": "t4", "name": "verify_candidate", "input": {"candidate_id": 1}},
        ]},
        # 5. update_residual + topk in one turn
        {"stop_reason": "tool_use", "content": [
            {"type": "tool_use", "id": "t5", "name": "update_residual", "input": {}},
        ]},
        # 6. finalize
        {"stop_reason": "end_turn", "content": [
            {"type": "tool_use", "id": "t6", "name": "finalize",
             "input": {"ranked_candidate_ids": [1], "summary": "one additional lesion confirmed"}},
        ]},
    ]

    agent = LLMDrivenAgent(
        proposal=_mock_proposal_tool(seg),
        viewer_factory=lambda imgs: ViewerTool(imgs),
        verifier=VerifierTool(_ConfirmingVLM()),
        modalities=("T1pre", "T1post", "FLAIR", "subtraction"),
        llm=MockLLM(script),
        cfg=AgentConfig(threshold=0.3),
    )
    state = agent.run(
        study_id="phantom",
        study_images=images,
        seed_mask=seed.mask,
        seed_coord_vox=tuple(int(round(c)) for c in seed.centroid_vox),
    )

    names = [a.name for a in state.actions]
    assert "proposal.first_pass" in names
    assert "duplicate.check" in names
    assert "vlm.verify" in names
    assert "residual.update" in names
    assert "finalize" in names
    # candidate 1 (the additional lesion) should be confirmed
    confirmed = state.confirmed()
    assert confirmed, "expected at least one confirmed candidate"
    add_centroid = np.array(others[0].centroid_vox)
    near = [np.linalg.norm(np.array(c.coord_vox) - add_centroid) for c in confirmed]
    assert any(d < 10 for d in near), f"confirmed candidates not near additional lesion: {near}"


# ---------- Second-finding protocol: phenotype + evidence + structured verdict ----------

def test_seed_phenotype_basic():
    from brain_mets_agent.data import characterize_seed
    images, seg, affine, spacing = _make_phantom()
    inst = extract_instances(seg, spacing, affine)
    seed, _ = select_seed(inst, "largest")
    p = characterize_seed(seed.mask, images, affine=affine, spacing_mm=spacing)
    assert p.voxel_count == 1000
    assert p.volume_mm3 == pytest.approx(1000.0)
    # T1post = base + (seg>0); inside lesion mean ~ 1.0; T1pre ~ 0
    assert p.enhancement_t1 > 0.5
    # near-spherical 10x10x10 cube -> low eccentricity
    assert p.eccentricity < 0.2


def test_phenotype_similarity_seed_to_self_is_high():
    from brain_mets_agent.data import (
        characterize_seed, candidate_similarity_to_seed,
    )
    images, seg, affine, spacing = _make_phantom()
    inst = extract_instances(seg, spacing, affine)
    seed, _others = select_seed(inst, "largest")
    p = characterize_seed(seed.mask, images, affine=affine, spacing_mm=spacing)
    sim_self = candidate_similarity_to_seed(
        seed.mask, tuple(int(c) for c in seed.centroid_vox),
        seed.voxel_count, images, p,
    )
    assert sim_self > 0.9, f"self-similarity should be ~1.0, got {sim_self}"


def test_evidence_card_composite_dimensions():
    from brain_mets_agent.orchestrator import build_evidence_card
    from brain_mets_agent.orchestrator.tools import ViewerTool
    images, seg, affine, spacing = _make_phantom()
    viewer = ViewerTool(images)
    card = build_evidence_card(
        viewer=viewer,
        candidate_coord_vox=(41, 41, 41),
        candidate_voxel_count=27, detector_prob=0.7,
        seed_coord_vox=(25, 25, 25),
        seed_phenotype={"voxel_count": 1000, "volume_mm3": 1000.0,
                        "diameter_mm": 12.4, "intensity_mean": {"T1post": 1.0},
                        "enhancement_t1": 1.0, "eccentricity": 0.05},
        modalities=("T1pre", "T1post", "FLAIR", "subtraction"),
        tile_size=64,
    )
    # 4 rows × 4 modalities × 64 tiles + borders -> non-trivial PNG
    assert card.composite_png_b64 and len(card.composite_png_b64) > 200
    # 12 candidate panels (3 planes × 4 mods) + 4 seed panels
    assert len(card.panels) == 12
    assert len(card.seed_panels) == 4
    assert "Candidate centroid" in card.metadata_text


def test_structured_verdict_parses_well_formed_json():
    from brain_mets_agent.orchestrator.tools.verifier import _parse_structured
    text = '''Sure, here is my analysis:
{
  "decision": "accept",
  "evidence_for": ["ring enhancement", "FLAIR correlate"],
  "evidence_against": ["small size"],
  "seed_similarity": 0.78,
  "mimic_risk": "low",
  "confidence": 0.83,
  "reason": "Centred enhancing focus with FLAIR correlate matching seed phenotype."
}'''
    v = _parse_structured(text)
    assert v.label == "confirm"
    assert v.confidence == pytest.approx(0.83)
    assert "ring enhancement" in v.evidence_for
    assert "small size" in v.evidence_against
    assert v.seed_similarity == pytest.approx(0.78)
    assert v.mimic_risk == "low"
    assert "FLAIR correlate" in v.rationale


def test_structured_verdict_falls_back_on_garbage():
    from brain_mets_agent.orchestrator.tools.verifier import _parse_structured
    v = _parse_structured("I'm not sure.")
    assert v.label == "uncertain"
    assert v.mimic_risk == "low"
    assert v.evidence_for == []


def test_protocol_agent_records_phenotype_in_state():
    images, seg, affine, spacing = _make_phantom()
    inst = extract_instances(seg, spacing, affine)
    seed, _others = select_seed(inst, "largest")
    agent = _build_agent(seg, _ConfirmingVLM())
    state = agent.run(
        study_id="phantom",
        study_images=images,
        seed_mask=seed.mask,
        seed_coord_vox=tuple(int(round(c)) for c in seed.centroid_vox),
        study_meta={"affine": affine, "spacing_mm": spacing},
    )
    assert state.seed_phenotype is not None
    assert state.seed_phenotype["voxel_count"] == 1000
    # All non-rejected candidates should carry a similarity score
    sims = [c.seed_similarity for c in state.candidates if c.verdict != "reject"]
    assert sims and all(0.0 <= s <= 1.0 for s in sims)


def test_llm_driven_agent_handles_unknown_tool_gracefully():
    """A bad tool name should surface as an error in tool_result, not crash."""
    from brain_mets_agent.orchestrator import LLMDrivenAgent, MockLLM, AgentConfig
    from brain_mets_agent.orchestrator.tools import ViewerTool, VerifierTool

    images, seg, affine, spacing = _make_phantom()
    inst = extract_instances(seg, spacing, affine)
    seed, _ = select_seed(inst, "largest")
    script = [
        {"stop_reason": "tool_use", "content": [
            {"type": "tool_use", "id": "t1", "name": "nonexistent_tool", "input": {}},
        ]},
        {"stop_reason": "end_turn", "content": [
            {"type": "tool_use", "id": "t2", "name": "finalize",
             "input": {"ranked_candidate_ids": []}},
        ]},
    ]
    agent = LLMDrivenAgent(
        proposal=_mock_proposal_tool(seg),
        viewer_factory=lambda imgs: ViewerTool(imgs),
        verifier=VerifierTool(_ConfirmingVLM()),
        modalities=("T1pre", "T1post", "FLAIR", "subtraction"),
        llm=MockLLM(script),
        cfg=AgentConfig(threshold=0.3),
    )
    state = agent.run(
        study_id="phantom",
        study_images=images,
        seed_mask=seed.mask,
        seed_coord_vox=tuple(int(round(c)) for c in seed.centroid_vox),
    )
    assert any(a.name == "finalize" for a in state.actions)
