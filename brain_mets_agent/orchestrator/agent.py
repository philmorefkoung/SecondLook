"""Discrepancy-hunter orchestrator implementing the second-finding protocol.

Step 1: Characterize the seed lesion (SeedPhenotype).
Step 2: First-pass proposal; verify candidates ordered by phenotype similarity
        to seed (look for "highly similar" lesions first - radiology search bias
        toward visually similar findings).
Step 3: Update the residual map (suppress confirmed/seed, downweight rejected,
        retain uncertain).
Step 4: Lower-threshold pass on the same probmap to catch smaller / lower-
        confidence lesions; skip duplicates of regions already explored.
Step 5: Residual-guided peaks of (residual_map x prob_map). Without an MNI
        atlas, this is the proxy for "anatomically difficult regions": the
        residual map encodes "where the agent has not yet looked." A
        RegionPriorTool can be plugged in later to bias peaks toward known
        difficult anatomy (posterior fossa, deep grey, near vessels).
Step 6: Phenotype comparison is computed inline at every verification call;
        this step just logs the per-study summary distribution.
Step 7: Ranking is performed by orchestrator.tools.ranker.rank_candidates.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence, Optional

import numpy as np

from .state import SearchState, TrackedCandidate
from .tools.viewer import ViewerTool
from .tools.proposal_tool import ProposalTool
from .tools.verifier import VerifierTool
from .tools.duplicate import is_duplicate
from .tools.residual import initial_residual, update_residual, topk_unexplored
from ..data.phenotype import (
    SeedPhenotype, characterize_seed, candidate_similarity_to_seed,
    phenotype_similarity_probmap,
)
from ..models.proposal import probmap_to_candidates


@dataclass
class AgentConfig:
    threshold: float = 0.5
    threshold_low: float = 0.3
    min_voxels: int = 10
    min_voxels_low: int = 5
    second_pass_k: int = 16
    duplicate_dist_vox: float = 5.0
    duplicate_iou: float = 0.3
    budget_actions: int = 32
    use_evidence_card: bool = True
    tile_size: int = 96
    # Pool-expansion knobs: skip the optional second-pass steps so all budget
    # goes to verifying step-2 candidates. Empirically (v3/v4 cached state)
    # both passes contribute zero metric uplift under prob-heavy ranking.
    skip_step4: bool = False
    skip_step5: bool = False
    # Experimental: weight the residual x prob_map heatmap by per-voxel
    # similarity to the seed's multi-modal intensity signature. Default OFF
    # because the empirical result was a negative one: phenotype-similar
    # sub-threshold blobs are dominated by enhancing non-lesion structures
    # (vessels, choroid plexus, pituitary stalk) rather than real missed
    # micrometastases. Enabling regresses Stanford r@5 by ~18 pt vs no-step-5.
    # See PROGRESS.md Finding #18 for the full record.
    step5_use_phenotype: bool = False


ViewerFactory = Callable[[dict], ViewerTool]


class DiscrepancyHunterAgent:
    def __init__(
        self,
        proposal: ProposalTool,
        viewer_factory: ViewerFactory,
        verifier: VerifierTool,
        modalities: Sequence[str],
        cfg: AgentConfig | None = None,
    ):
        self.proposal = proposal
        self.viewer_factory = viewer_factory
        self.verifier = verifier
        self.modalities = list(modalities)
        self.cfg = cfg or AgentConfig()

    def run(
        self,
        *,
        study_id: str,
        study_images: dict[str, np.ndarray],
        seed_mask: Optional[np.ndarray],
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
        affine = (study_meta or {}).get("affine")
        spacing = tuple((study_meta or {}).get("spacing_mm", (1.0, 1.0, 1.0)))
        multi_modal = np.stack([study_images[m] for m in self.modalities], axis=0)

        # ===== Step 1: characterize seed =====
        phenotype = self._step1_characterize(state, seed_mask, study_images, affine, spacing)

        # ===== Step 2: first-pass proposal + similarity-prioritised verify =====
        prob = self._step2_first_pass(state, multi_modal, phenotype, study_images, viewer)

        # ===== Step 3: residual update =====
        self._step3_residual_update(state)

        # ===== Step 4: lower-threshold pass for smaller / lower-confidence =====
        self._step4_lower_threshold(state, prob, phenotype, study_images, viewer)

        # ===== Step 5: residual-guided unexplored peaks =====
        self._step5_residual_peaks(state, prob, phenotype, study_images, viewer, vol_shape)

        # ===== Step 6: phenotype-comparison summary =====
        self._step6_phenotype_summary(state)

        # ===== Step 7: ready for ranking =====
        state.log("step7.ready_for_rank", {}, f"{len(state.candidates)} candidates")
        return state

    # ---------- step implementations ----------

    def _step1_characterize(self, state, seed_mask, study_images, affine, spacing):
        if seed_mask is None or not seed_mask.any():
            state.log("step1.characterize_seed", {}, "no seed mask")
            return None
        phenotype = characterize_seed(seed_mask, study_images, affine=affine, spacing_mm=spacing)
        state.seed_phenotype = phenotype.to_dict()
        state.log(
            "step1.characterize_seed", {},
            f"vol={phenotype.volume_mm3:.1f}mm3 enh={phenotype.enhancement_t1:.2f} "
            f"ecc={phenotype.eccentricity:.2f}",
        )
        return phenotype

    def _step2_first_pass(self, state, multi_modal, phenotype, study_images, viewer):
        prob, cands = self.proposal.first_pass(
            multi_modal, threshold=self.cfg.threshold, min_voxels=self.cfg.min_voxels,
        )
        state.log("step2.proposal_first_pass",
                  {"threshold": self.cfg.threshold, "min_voxels": self.cfg.min_voxels},
                  f"{len(cands)} candidates")
        state.residual_map = initial_residual(multi_modal.shape[1:], state.seed_mask)
        state.log("residual.init", {"seed_present": state.seed_mask is not None},
                  f"sum={float(state.residual_map.sum()):.0f}")
        for c, sim in self._order_by_phenotype(cands, study_images, phenotype):
            if state.budget_remaining <= 0:
                break
            tracked = TrackedCandidate(
                coord_vox=c.coord_vox, bbox=c.bbox, mask=c.mask,
                prob=c.prob, voxel_count=c.voxel_count, pass_idx=2,
                seed_similarity=float(sim),
            )
            self._verify_one(viewer, tracked, state, phenotype, study_images)
            state.candidates.append(tracked)
        return prob

    def _step3_residual_update(self, state):
        state.residual_map = update_residual(
            state.residual_map,
            [c.mask for c in state.confirmed() if c.mask is not None],
            [c.mask for c in state.rejected() if c.mask is not None],
            [c.mask for c in state.uncertain() if c.mask is not None],
        )
        state.log("step3.residual_update", {}, f"sum={float(state.residual_map.sum()):.0f}")

    def _step4_lower_threshold(self, state, prob, phenotype, study_images, viewer):
        if self.cfg.skip_step4:
            state.log("step4.lower_threshold_pass", {"skipped": True}, "skipped via cfg")
            return
        if state.budget_remaining <= 0:
            return
        cands = probmap_to_candidates(
            prob, threshold=self.cfg.threshold_low, min_voxels=self.cfg.min_voxels_low,
        )
        explored_coords = [state.seed_coord_vox] + [c.coord_vox for c in state.candidates]
        explored_masks = [state.seed_mask] + [c.mask for c in state.candidates]
        n_added = 0
        for c, sim in self._order_by_phenotype(cands, study_images, phenotype):
            if state.budget_remaining <= 0:
                break
            if is_duplicate(
                c.coord_vox, c.mask, explored_coords, explored_masks,
                distance_threshold_vox=self.cfg.duplicate_dist_vox,
                iou_threshold=self.cfg.duplicate_iou,
            ):
                continue
            tracked = TrackedCandidate(
                coord_vox=c.coord_vox, bbox=c.bbox, mask=c.mask,
                prob=c.prob, voxel_count=c.voxel_count, pass_idx=4,
                seed_similarity=float(sim),
            )
            self._verify_one(viewer, tracked, state, phenotype, study_images)
            state.candidates.append(tracked)
            explored_coords.append(c.coord_vox)
            explored_masks.append(c.mask)
            n_added += 1
        state.log("step4.lower_threshold_pass",
                  {"threshold": self.cfg.threshold_low}, f"{n_added} added")

    def _step5_residual_peaks(self, state, prob, phenotype, study_images, viewer, vol_shape):
        if self.cfg.skip_step5:
            state.log("step5.residual_topk", {"skipped": True}, "skipped via cfg")
            return
        if state.budget_remaining <= 0:
            return
        # Optionally weight the residual x prob heatmap by seed-phenotype
        # similarity. Surfaces sub-threshold blobs that look like the seed.
        score_prob = prob
        if self.cfg.step5_use_phenotype and phenotype is not None:
            phen_sim = phenotype_similarity_probmap(study_images, phenotype)
            score_prob = (prob * phen_sim).astype(np.float32)
            state.log("step5.phenotype_weighting", {},
                      f"phen_sim mean={float(phen_sim.mean()):.3f} max={float(phen_sim.max()):.3f}")
        peaks = topk_unexplored(state.residual_map, score_prob, k=self.cfg.second_pass_k)
        explored_coords = [state.seed_coord_vox] + [c.coord_vox for c in state.candidates]
        explored_masks = [state.seed_mask] + [c.mask for c in state.candidates]
        n_added = 0
        for coord, score in peaks:
            if state.budget_remaining <= 0:
                break
            if is_duplicate(
                coord, None, explored_coords, explored_masks,
                distance_threshold_vox=self.cfg.duplicate_dist_vox,
                iou_threshold=self.cfg.duplicate_iou,
            ):
                continue
            sim = (
                candidate_similarity_to_seed(None, coord, 0, study_images, phenotype)
                if phenotype is not None else 0.0
            )
            tracked = TrackedCandidate(
                coord_vox=coord,
                bbox=_point_bbox(coord, 8, vol_shape),
                mask=None, prob=float(prob[coord]), voxel_count=0, pass_idx=5,
                seed_similarity=float(sim),
            )
            self._verify_one(viewer, tracked, state, phenotype, study_images)
            state.candidates.append(tracked)
            explored_coords.append(coord)
            explored_masks.append(None)
            n_added += 1
        state.log("step5.residual_topk",
                  {"k": self.cfg.second_pass_k}, f"{n_added} added")

    def _step6_phenotype_summary(self, state):
        sims = [c.seed_similarity for c in state.candidates]
        if not sims:
            state.log("step6.phenotype_compare", {}, "no candidates")
            return
        p10, p50, p90 = (
            float(np.percentile(sims, p)) for p in (10, 50, 90)
        )
        state.log("step6.phenotype_compare",
                  {"n": len(sims)},
                  f"sim p10/p50/p90={p10:.2f}/{p50:.2f}/{p90:.2f}")

    # ---------- helpers ----------

    def _order_by_phenotype(self, cands, study_images, phenotype: SeedPhenotype | None):
        if phenotype is None:
            return [(c, 0.0) for c in cands]
        scored = [
            (c, candidate_similarity_to_seed(
                c.mask, c.coord_vox, c.voxel_count, study_images, phenotype,
            ))
            for c in cands
        ]
        scored.sort(key=lambda kv: -kv[1])
        return scored

    def _verify_one(self, viewer, tracked, state, phenotype, study_images):
        if state.budget_remaining <= 0:
            return
        if state.seed_mask is not None and is_duplicate(
            tracked.coord_vox, tracked.mask,
            [state.seed_coord_vox], [state.seed_mask],
            distance_threshold_vox=self.cfg.duplicate_dist_vox,
            iou_threshold=self.cfg.duplicate_iou,
        ):
            tracked.verdict = "reject"
            tracked.rationale = "duplicate of seed"
            tracked.evidence_against = ["overlaps the seed lesion"]
            state.log("duplicate.check", {"coord": list(tracked.coord_vox)},
                      "duplicate of seed")
            return

        card = None
        if self.cfg.use_evidence_card and phenotype is not None:
            from .evidence import build_evidence_card
            card = build_evidence_card(
                viewer=viewer,
                candidate_coord_vox=tracked.coord_vox,
                candidate_voxel_count=tracked.voxel_count,
                detector_prob=tracked.prob,
                seed_coord_vox=state.seed_coord_vox,
                seed_phenotype=state.seed_phenotype or {},
                modalities=self.modalities,
                candidate_mask=tracked.mask,
                seed_mask=state.seed_mask,
                tile_size=self.cfg.tile_size,
            )
            panels_for_compat = [card.panels[("axial", m)] for m in self.modalities]
        else:
            panels = viewer.panel(
                tracked.coord_vox, modalities=self.modalities,
                plane="axial", tile_size=64,
            )
            panels_for_compat = [p.image for p in panels]

        verdict = self.verifier.verify(
            panels_for_compat,
            modalities=self.modalities,
            candidate_meta={
                "coord": list(tracked.coord_vox),
                "prob": tracked.prob,
                "voxel_count": tracked.voxel_count,
                "evidence_card": card,
                "seed_phenotype": state.seed_phenotype,
            },
        )
        tracked.verdict = verdict.label
        tracked.vlm_conf = verdict.confidence
        tracked.rationale = verdict.rationale
        tracked.evidence_for = list(verdict.evidence_for)
        tracked.evidence_against = list(verdict.evidence_against)
        if verdict.seed_similarity:
            tracked.seed_similarity = max(tracked.seed_similarity, verdict.seed_similarity)
        if verdict.mimic_risk:
            tracked.mimic_risk = verdict.mimic_risk
        tracked.evidence_panels = (
            [f"{p}/{m}@{tracked.coord_vox}"
             for p in ("axial", "coronal", "sagittal") for m in self.modalities]
            if card else
            [f"axial/{m}@{tracked.coord_vox}" for m in self.modalities]
        )
        state.budget_remaining -= 1
        state.log(
            "vlm.verify",
            {"coord": list(tracked.coord_vox), "card": card is not None},
            f"{verdict.label}/{verdict.confidence:.2f}",
        )


def _point_bbox(coord, half: int, shape) -> tuple[slice, slice, slice]:
    return tuple(
        slice(max(0, c - half), min(s, c + half + 1))
        for c, s in zip(coord, shape)
    )
