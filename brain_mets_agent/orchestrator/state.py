"""State carried through one discrepancy-hunting run.

The orchestrator mutates a single SearchState as tools fire. Every tool call
appends a ToolEvent so the trace is fully auditable post-hoc.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional, Any

import numpy as np


Verdict = Literal["confirm", "reject", "uncertain"]
MimicRisk = Literal["low", "medium", "high"]


@dataclass
class TrackedCandidate:
    coord_vox: tuple[int, int, int]
    bbox: tuple[slice, slice, slice]
    mask: Optional[np.ndarray]
    prob: float
    voxel_count: int
    pass_idx: int = 0
    verdict: Optional[Verdict] = None
    vlm_conf: float = 0.0
    rationale: str = ""
    evidence_panels: list[str] = field(default_factory=list)
    # second-finding-protocol fields
    seed_similarity: float = 0.0
    mimic_risk: MimicRisk = "low"
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)


@dataclass
class ToolEvent:
    name: str
    args: dict
    result_summary: str


@dataclass
class SearchState:
    study_id: str
    seed_coord_vox: tuple[int, int, int]
    seed_mask: Optional[np.ndarray]
    volume_shape: tuple[int, int, int]
    candidates: list[TrackedCandidate] = field(default_factory=list)
    residual_map: Optional[np.ndarray] = None
    actions: list[ToolEvent] = field(default_factory=list)
    budget_remaining: int = 32
    seed_phenotype: Optional[dict[str, Any]] = None

    def confirmed(self) -> list[TrackedCandidate]:
        return [c for c in self.candidates if c.verdict == "confirm"]

    def rejected(self) -> list[TrackedCandidate]:
        return [c for c in self.candidates if c.verdict == "reject"]

    def uncertain(self) -> list[TrackedCandidate]:
        return [c for c in self.candidates if c.verdict == "uncertain"]

    def log(self, name: str, args: dict, summary: str) -> None:
        self.actions.append(ToolEvent(name=name, args=args, result_summary=summary))

    def to_summary(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "seed_coord_vox": list(self.seed_coord_vox),
            "seed_phenotype_present": self.seed_phenotype is not None,
            "n_candidates": len(self.candidates),
            "n_confirmed": len(self.confirmed()),
            "n_rejected": len(self.rejected()),
            "n_uncertain": len(self.uncertain()),
            "budget_remaining": self.budget_remaining,
            "actions": [(a.name, a.result_summary) for a in self.actions],
        }
