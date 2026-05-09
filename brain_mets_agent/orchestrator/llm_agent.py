"""LLM-driven discrepancy-hunter (Anthropic tool-use loop).

The orchestrator LLM only does *planning* - it decides which tool to call
next. Vision work is split between the 3D detector (proposal model) and the
VLM verifier (per-candidate evidence-card judgement). The orchestrator LLM
itself never receives images; this keeps its prompt cache hot and per-call
cost low.

The tool surface implements the second-finding protocol:
  characterize_seed -> propose_lesions -> check_duplicate_of_seed ->
  compare_candidates_to_seed_phenotype -> verify_candidate ->
  update_residual -> query_unexplored_peaks -> add_peak_as_candidate ->
  verify_candidate -> finalize
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Sequence
import json

import numpy as np

from .state import SearchState, TrackedCandidate
from .agent import AgentConfig
from .llm import LLMBackend, ToolSpec
from .tools.viewer import ViewerTool
from .tools.proposal_tool import ProposalTool
from .tools.verifier import VerifierTool
from .tools.duplicate import is_duplicate
from .tools.residual import initial_residual, update_residual, topk_unexplored


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="characterize_seed",
        description=(
            "Step 1 of the second-finding protocol. Compute the seed lesion's "
            "phenotype (volume, per-modality intensity stats, T1pre->T1post "
            "enhancement, eccentricity). Returns a dict you should reference "
            "when interpreting later candidates."
        ),
        schema={"type": "object", "properties": {}, "required": []},
    ),
    ToolSpec(
        name="propose_lesions",
        description=(
            "Run the 3D lesion-proposal model on the full multi-modal volume "
            "(first pass) and register every candidate. Returns each candidate's "
            "id, voxel coord (z,y,x), detector probability, and voxel count."
        ),
        schema={
            "type": "object",
            "properties": {
                "threshold": {"type": "number", "description": "Probability threshold for binarisation."},
                "min_voxels": {"type": "integer", "description": "Drop connected components smaller than this."},
            },
            "required": [],
        },
    ),
    ToolSpec(
        name="check_duplicate_of_seed",
        description="Check whether the candidate is the same lesion as the seed.",
        schema={
            "type": "object",
            "properties": {"candidate_id": {"type": "integer"}},
            "required": ["candidate_id"],
        },
    ),
    ToolSpec(
        name="compare_candidates_to_seed_phenotype",
        description=(
            "Compute phenotype similarity (0..1) between each supplied candidate "
            "and the seed (must call characterize_seed first). Free; no budget cost. "
            "Use to prioritise high-similarity candidates for verification."
        ),
        schema={
            "type": "object",
            "properties": {
                "candidate_ids": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["candidate_ids"],
        },
    ),
    ToolSpec(
        name="verify_candidate",
        description=(
            "Build an evidence card (axial/coronal/sagittal x modalities + seed "
            "axial strip) and send it to the VLM. Returns a structured verdict "
            "(confirm/reject/uncertain + evidence_for/against + seed_similarity "
            "+ mimic_risk + reason). Costs one budget action."
        ),
        schema={
            "type": "object",
            "properties": {
                "candidate_id": {"type": "integer"},
                "plane": {"type": "string", "enum": ["axial", "coronal", "sagittal"]},
            },
            "required": ["candidate_id"],
        },
    ),
    ToolSpec(
        name="update_residual",
        description=(
            "Recompute the residual search-space map: suppress confirmed/seed, "
            "downweight rejected, retain uncertain. Call after a batch of verifications."
        ),
        schema={"type": "object", "properties": {}, "required": []},
    ),
    ToolSpec(
        name="query_unexplored_peaks",
        description=(
            "Top-k local maxima of (residual_map x proposal_probmap). "
            "Use these as second-pass investigation candidates."
        ),
        schema={
            "type": "object",
            "properties": {"k": {"type": "integer"}},
            "required": [],
        },
    ),
    ToolSpec(
        name="add_peak_as_candidate",
        description="Promote a peak from query_unexplored_peaks into a tracked candidate.",
        schema={
            "type": "object",
            "properties": {
                "coord_zyx": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 3, "maxItems": 3,
                },
            },
            "required": ["coord_zyx"],
        },
    ),
    ToolSpec(
        name="finalize",
        description=(
            "Submit the final ranked list of candidate_ids representing the "
            "additional lesions (best first). Call this exactly once when done."
        ),
        schema={
            "type": "object",
            "properties": {
                "ranked_candidate_ids": {"type": "array", "items": {"type": "integer"}},
                "summary": {"type": "string"},
            },
            "required": ["ranked_candidate_ids"],
        },
    ),
]


@dataclass
class _Ctx:
    state: SearchState
    viewer: ViewerTool
    multi_modal: np.ndarray
    study_images: dict[str, np.ndarray]
    affine: np.ndarray | None = None
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)
    seed_phenotype: Any | None = None       # SeedPhenotype object
    candidates_by_id: dict[int, TrackedCandidate] = field(default_factory=dict)
    next_id: int = 0
    prob_map: np.ndarray | None = None
    final_ranking: list[int] | None = None
    summary: str = ""


class LLMDrivenAgent:
    def __init__(
        self,
        proposal: ProposalTool,
        viewer_factory,
        verifier: VerifierTool,
        modalities: Sequence[str],
        llm: LLMBackend,
        cfg: AgentConfig | None = None,
        max_outer_steps: int = 64,
    ):
        self.proposal = proposal
        self.viewer_factory = viewer_factory
        self.verifier = verifier
        self.modalities = list(modalities)
        self.llm = llm
        self.cfg = cfg or AgentConfig()
        self.max_outer_steps = int(max_outer_steps)

    def run(
        self,
        *,
        study_id: str,
        study_images: dict[str, np.ndarray],
        seed_mask: np.ndarray | None,
        seed_coord_vox: tuple[int, int, int],
        study_meta: dict | None = None,
    ) -> SearchState:
        viewer = self.viewer_factory(study_images)
        vol_shape = next(iter(study_images.values())).shape
        state = SearchState(
            study_id=study_id,
            seed_coord_vox=tuple(int(c) for c in seed_coord_vox),
            seed_mask=seed_mask,
            volume_shape=vol_shape,
            budget_remaining=self.cfg.budget_actions,
        )
        multi_modal = np.stack([study_images[m] for m in self.modalities], axis=0)
        ctx = _Ctx(
            state=state, viewer=viewer, multi_modal=multi_modal,
            study_images=study_images,
            affine=(study_meta or {}).get("affine"),
            spacing_mm=tuple((study_meta or {}).get("spacing_mm", (1.0, 1.0, 1.0))),
        )

        system = self._system_prompt()
        messages: list[dict[str, Any]] = [{
            "role": "user",
            "content": [{"type": "text", "text": self._initial_user(state)}],
        }]

        # Allow cfg to disable the residual-peak tools (cheap pool-expansion mode).
        active_tool_specs = TOOL_SPECS
        if getattr(self.cfg, "skip_step5", False):
            active_tool_specs = [t for t in TOOL_SPECS
                                 if t.name not in ("query_unexplored_peaks",
                                                    "add_peak_as_candidate")]

        for _ in range(self.max_outer_steps):
            if ctx.final_ranking is not None or state.budget_remaining <= 0:
                break
            resp = self.llm.step(system, messages, active_tool_specs)
            content = resp.get("content", [])
            messages.append({"role": "assistant", "content": content})
            tool_uses = [b for b in content if _block_type(b) == "tool_use"]
            if not tool_uses:
                break
            tool_results = []
            for block in tool_uses:
                name = _block_get(block, "name")
                args = _block_get(block, "input") or {}
                tool_id = _block_get(block, "id")
                try:
                    result = self._dispatch(name, args, ctx)
                except Exception as e:
                    result = {"error": f"{type(e).__name__}: {e}"}
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": json.dumps(result),
                })
            messages.append({"role": "user", "content": tool_results})

        if ctx.final_ranking is not None:
            id_to_cand = ctx.candidates_by_id
            ordered = [id_to_cand[i] for i in ctx.final_ranking if i in id_to_cand]
            # use identity comparison; dataclass __eq__ on np.ndarray fields is ambiguous
            ordered_pyids = {id(c) for c in ordered}
            others = [c for c in state.candidates if id(c) not in ordered_pyids]
            state.candidates = ordered + others

        return state

    def _dispatch(self, name: str, args: dict, ctx: _Ctx) -> dict:
        handler = {
            "characterize_seed": self._tool_characterize_seed,
            "propose_lesions": self._tool_propose,
            "check_duplicate_of_seed": self._tool_dup,
            "compare_candidates_to_seed_phenotype": self._tool_compare_phenotype,
            "verify_candidate": self._tool_verify,
            "update_residual": self._tool_residual,
            "query_unexplored_peaks": self._tool_topk,
            "add_peak_as_candidate": self._tool_add_peak,
            "finalize": self._tool_finalize,
        }.get(name)
        if handler is None:
            return {"error": f"unknown tool {name!r}"}
        return handler(args, ctx)

    def _tool_characterize_seed(self, args: dict, ctx: _Ctx) -> dict:
        if ctx.state.seed_mask is None or not ctx.state.seed_mask.any():
            return {"error": "no seed mask supplied"}
        from ..data.phenotype import characterize_seed
        p = characterize_seed(
            ctx.state.seed_mask, ctx.study_images,
            affine=ctx.affine, spacing_mm=ctx.spacing_mm,
        )
        ctx.seed_phenotype = p
        ctx.state.seed_phenotype = p.to_dict()
        ctx.state.log(
            "characterize_seed", {},
            f"vol={p.volume_mm3:.1f}mm3 enh={p.enhancement_t1:.2f} ecc={p.eccentricity:.2f}",
        )
        return p.to_dict()

    def _tool_compare_phenotype(self, args: dict, ctx: _Ctx) -> dict:
        if ctx.seed_phenotype is None:
            return {"error": "call characterize_seed first"}
        from ..data.phenotype import candidate_similarity_to_seed
        ids = [int(i) for i in args.get("candidate_ids", [])]
        out: dict[str, float] = {}
        for cid in ids:
            tc = ctx.candidates_by_id.get(cid)
            if tc is None:
                continue
            sim = candidate_similarity_to_seed(
                tc.mask, tc.coord_vox, tc.voxel_count,
                ctx.study_images, ctx.seed_phenotype,
            )
            tc.seed_similarity = max(tc.seed_similarity, float(sim))
            out[str(cid)] = round(float(sim), 4)
        ctx.state.log("compare_phenotype", {"n": len(out)}, f"sims={out}")
        return {"similarities": out}

    def _tool_propose(self, args: dict, ctx: _Ctx) -> dict:
        threshold = float(args.get("threshold", self.cfg.threshold))
        min_voxels = int(args.get("min_voxels", self.cfg.min_voxels))
        prob, cands = self.proposal.first_pass(
            ctx.multi_modal, threshold=threshold, min_voxels=min_voxels,
        )
        ctx.prob_map = prob
        out = []
        for c in cands:
            cid = ctx.next_id
            ctx.next_id += 1
            tc = TrackedCandidate(
                coord_vox=c.coord_vox, bbox=c.bbox, mask=c.mask,
                prob=c.prob, voxel_count=c.voxel_count, pass_idx=1,
            )
            ctx.candidates_by_id[cid] = tc
            ctx.state.candidates.append(tc)
            out.append({
                "id": cid,
                "coord_zyx": list(c.coord_vox),
                "prob": round(float(c.prob), 4),
                "voxels": int(c.voxel_count),
            })
        if ctx.state.residual_map is None:
            ctx.state.residual_map = initial_residual(
                ctx.multi_modal.shape[1:], ctx.state.seed_mask,
            )
        ctx.state.log("proposal.first_pass",
                      {"threshold": threshold, "min_voxels": min_voxels},
                      f"{len(out)} candidates")
        return {"n_candidates": len(out), "candidates": out}

    def _tool_dup(self, args: dict, ctx: _Ctx) -> dict:
        cid = int(args["candidate_id"])
        tc = ctx.candidates_by_id.get(cid)
        if tc is None:
            return {"error": f"no candidate {cid}"}
        is_dup = is_duplicate(
            tc.coord_vox, tc.mask,
            [ctx.state.seed_coord_vox], [ctx.state.seed_mask],
            distance_threshold_vox=self.cfg.duplicate_dist_vox,
            iou_threshold=self.cfg.duplicate_iou,
        )
        if is_dup:
            tc.verdict = "reject"
            tc.rationale = "duplicate of seed"
        ctx.state.log("duplicate.check", {"id": cid},
                      "duplicate" if is_dup else "novel")
        return {"is_duplicate": bool(is_dup)}

    def _tool_verify(self, args: dict, ctx: _Ctx) -> dict:
        if ctx.state.budget_remaining <= 0:
            return {"error": "verifier budget exhausted"}
        cid = int(args["candidate_id"])
        tc = ctx.candidates_by_id.get(cid)
        if tc is None:
            return {"error": f"no candidate {cid}"}
        plane = str(args.get("plane", "axial"))

        card = None
        if ctx.seed_phenotype is not None:
            from .evidence import build_evidence_card
            card = build_evidence_card(
                viewer=ctx.viewer,
                candidate_coord_vox=tc.coord_vox,
                candidate_voxel_count=tc.voxel_count,
                detector_prob=tc.prob,
                seed_coord_vox=ctx.state.seed_coord_vox,
                seed_phenotype=ctx.state.seed_phenotype or {},
                modalities=self.modalities,
                candidate_mask=tc.mask,
                seed_mask=ctx.state.seed_mask,
                candidate_id=cid,
            )
            panels_for_compat = [card.panels[("axial", m)] for m in self.modalities]
        else:
            panels = ctx.viewer.panel(
                tc.coord_vox, modalities=self.modalities, plane=plane, tile_size=64,
            )
            panels_for_compat = [p.image for p in panels]

        verdict = self.verifier.verify(
            panels_for_compat,
            modalities=self.modalities,
            candidate_meta={
                "coord": list(tc.coord_vox),
                "prob": float(tc.prob),
                "voxel_count": int(tc.voxel_count),
                "evidence_card": card,
                "seed_phenotype": ctx.state.seed_phenotype,
            },
        )
        tc.verdict = verdict.label
        tc.vlm_conf = float(verdict.confidence)
        tc.rationale = verdict.rationale
        tc.evidence_for = list(verdict.evidence_for)
        tc.evidence_against = list(verdict.evidence_against)
        if verdict.seed_similarity:
            tc.seed_similarity = max(tc.seed_similarity, float(verdict.seed_similarity))
        if verdict.mimic_risk:
            tc.mimic_risk = verdict.mimic_risk
        tc.evidence_panels = (
            [f"{p}/{m}@{tc.coord_vox}" for p in ("axial", "coronal", "sagittal")
             for m in self.modalities]
            if card else [f"{plane}/{m}@{tc.coord_vox}" for m in self.modalities]
        )
        ctx.state.budget_remaining -= 1
        ctx.state.log(
            "vlm.verify",
            {"id": cid, "plane": plane, "card": card is not None},
            f"{verdict.label}/{verdict.confidence:.2f}",
        )
        return {
            "verdict": verdict.label,
            "confidence": round(float(verdict.confidence), 3),
            "evidence_for": verdict.evidence_for,
            "evidence_against": verdict.evidence_against,
            "seed_similarity": round(float(verdict.seed_similarity), 3),
            "mimic_risk": verdict.mimic_risk,
            "reason": verdict.rationale[:200],
            "budget_remaining": ctx.state.budget_remaining,
        }

    def _tool_residual(self, args: dict, ctx: _Ctx) -> dict:
        if ctx.state.residual_map is None:
            ctx.state.residual_map = initial_residual(
                ctx.multi_modal.shape[1:], ctx.state.seed_mask,
            )
        ctx.state.residual_map = update_residual(
            ctx.state.residual_map,
            [c.mask for c in ctx.state.confirmed() if c.mask is not None],
            [c.mask for c in ctx.state.rejected() if c.mask is not None],
            [c.mask for c in ctx.state.uncertain() if c.mask is not None],
        )
        s = float(ctx.state.residual_map.sum())
        ctx.state.log("residual.update", {}, f"sum={s:.0f}")
        return {"residual_sum": s}

    def _tool_topk(self, args: dict, ctx: _Ctx) -> dict:
        if ctx.prob_map is None or ctx.state.residual_map is None:
            return {"error": "call propose_lesions and update_residual first"}
        k = int(args.get("k", 8))
        peaks = topk_unexplored(ctx.state.residual_map, ctx.prob_map, k=k)
        ctx.state.log("residual.topk", {"k": k}, f"{len(peaks)} peaks")
        return {"peaks": [
            {"coord_zyx": list(c), "score": round(float(s), 4)}
            for c, s in peaks
        ]}

    def _tool_add_peak(self, args: dict, ctx: _Ctx) -> dict:
        if ctx.prob_map is None:
            return {"error": "call propose_lesions first"}
        coord = tuple(int(v) for v in args["coord_zyx"])
        if any(c < 0 or c >= s for c, s in zip(coord, ctx.state.volume_shape)):
            return {"error": f"coord {coord} out of bounds {ctx.state.volume_shape}"}
        prob = float(ctx.prob_map[coord])
        cid = ctx.next_id
        ctx.next_id += 1
        bbox = tuple(
            slice(max(0, c - 8), min(s, c + 9))
            for c, s in zip(coord, ctx.state.volume_shape)
        )
        tc = TrackedCandidate(
            coord_vox=coord, bbox=bbox, mask=None,
            prob=prob, voxel_count=0, pass_idx=2,
        )
        ctx.candidates_by_id[cid] = tc
        ctx.state.candidates.append(tc)
        return {"id": cid, "prob": round(prob, 4)}

    def _tool_finalize(self, args: dict, ctx: _Ctx) -> dict:
        ctx.final_ranking = [int(i) for i in args.get("ranked_candidate_ids", [])]
        ctx.summary = str(args.get("summary", ""))
        ctx.state.log(
            "finalize",
            {"n_ranked": len(ctx.final_ranking)},
            f"summary={ctx.summary[:80]!r}",
        )
        return {"ok": True, "n_ranked": len(ctx.final_ranking)}

    def _system_prompt(self) -> str:
        return (
            "You are an auditable second-finding agent for brain MRI metastases.\n"
            "GOAL: maximize discovery of ADDITIONAL metastases beyond the supplied seed lesion.\n"
            "Multifocal brain mets are common - assume there are likely several additional\n"
            "lesions to find. The seed is the single largest known lesion; do not re-report it.\n"
            "BUDGET: a fixed verifier-call budget (each verify_candidate costs one).\n"
            "\n"
            "SEARCH-THOROUGHNESS REQUIREMENT (this is the most important rule):\n"
            "Brain metastases are often small and multifocal; a thorough reader does NOT stop\n"
            "after finding 1-2 lesions. You must keep searching until either (a) you have used\n"
            "at least 75% of your verification budget, OR (b) you have verified every candidate\n"
            "AND query_unexplored_peaks returns no peak with score > 0.05. Do NOT call\n"
            "finalize before one of these conditions is met.\n"
            "\n"
            "SECOND-FINDING PROTOCOL (loop expansively within budget):\n"
            "  Step 1. characterize_seed -> get the seed phenotype.\n"
            "  Step 2. propose_lesions(threshold=0.3) -> first-pass candidates. Use the\n"
            "          PERMISSIVE threshold of 0.3 (NOT 0.5) so smaller/lower-confidence\n"
            "          lesions are surfaced.\n"
            "  Step 3. compare_candidates_to_seed_phenotype on every new candidate id.\n"
            "          Verify highest-similarity first (most likely to be true mets), but\n"
            "          do not skip lower-similarity ones - they are often the small\n"
            "          additional lesions you are looking for.\n"
            "  Step 4. For each candidate:\n"
            "             check_duplicate_of_seed -> if duplicate, skip (no budget cost).\n"
            "             verify_candidate -> structured verdict.\n"
            "  Step 5. update_residual once a batch of verifications is done.\n"
            "  Step 6. query_unexplored_peaks(k=12) at least ONCE; for any peak with\n"
            "          score > 0.05: add_peak_as_candidate, then verify_candidate.\n"
            "  Step 7. ONLY when the search-thoroughness condition above is satisfied,\n"
            "          finalize with ranked candidate_ids you believe are real additional\n"
            "          lesions (best first). Include accept-verdict candidates AND\n"
            "          uncertain-verdict candidates with strong evidence_for.\n"
            "\n"
            "VERIFIER-OUTPUT GUIDANCE:\n"
            "- Each evidence card the verifier sees is a 4 row x N modality grid:\n"
            "    rows 0-2 = candidate axial/coronal/sagittal with RED boundary outlining\n"
            "    the proposal model's predicted lesion extent;\n"
            "    row 3 = seed lesion axial with GREEN boundary for direct comparison.\n"
            "- The verifier returns structured fields (evidence_for / evidence_against /\n"
            "  seed_similarity / mimic_risk / reason). Cite them in your finalize summary.\n"
            "- Treat 'mimic_risk: high' as a strong reject signal unless evidence_for is\n"
            "  unambiguous (e.g., classic ring enhancement + FLAIR correlate).\n"
            "- Do NOT re-verify a candidate already verified; the cost is sunk.\n"
        )

    def _initial_user(self, state: SearchState) -> str:
        return (
            f"Study: {state.study_id}\n"
            f"Volume shape (z,y,x): {state.volume_shape}\n"
            f"Seed lesion centroid (z,y,x): {state.seed_coord_vox}\n"
            f"Verifier budget: {state.budget_remaining} calls.\n"
            f"Reminder: do not finalize before using ~{int(state.budget_remaining * 0.75)} "
            "verifier calls or covering all candidates + unexplored peaks.\n"
            "Begin with Step 1: characterize_seed."
        )


def _block_type(block: Any) -> str:
    if isinstance(block, dict):
        return block.get("type", "")
    return getattr(block, "type", "")


def _block_get(block: Any, key: str):
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)
