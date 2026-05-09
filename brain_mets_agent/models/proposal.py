"""Lesion-proposal model: 3D segmentation backbone + connected-component postproc.

Default backbone is MONAI Swin-UNETR. Torch/MONAI imports are deferred so the
rest of the package can be imported without a GPU/PyTorch install (useful for
the orchestrator unit tests).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage as ndi


@dataclass
class Candidate:
    coord_vox: tuple[int, int, int]    # centroid (z, y, x)
    bbox: tuple[slice, slice, slice]
    mask: Optional[np.ndarray]         # bool, full-volume; may be None
    prob: float                        # mean foreground probability
    voxel_count: int


class LesionProposalModel:
    """Swin-UNETR wrapper.

    `predict_probmap(multi_modal)`: (C,H,W,D) -> (H,W,D) ∈ [0, 1]
    `propose(multi_modal)`:        (C,H,W,D) -> list[Candidate]
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 2,
        feature_size: int = 48,
        patch_size: tuple[int, int, int] = (96, 96, 96),
        overlap: float = 0.5,
        threshold: float = 0.5,
        device: str = "cuda",
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size
        self.patch_size = tuple(patch_size)
        self.overlap = overlap
        self.threshold = threshold
        self.device = device
        self._net = None

    def _build(self):
        from monai.networks.nets import SwinUNETR
        import torch
        net = SwinUNETR(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            feature_size=self.feature_size,
        )
        net = net.to(self.device)
        net.eval()
        self._net = net

    def load_state(self, path: str) -> None:
        import torch
        if self._net is None:
            self._build()
        sd = torch.load(path, map_location=self.device)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        self._net.load_state_dict(sd, strict=False)

    def predict_probmap(self, multi_modal: np.ndarray) -> np.ndarray:
        import torch
        from monai.inferers import sliding_window_inference
        if self._net is None:
            self._build()
        if multi_modal.ndim != 4:
            raise ValueError(f"expected (C,H,W,D), got {multi_modal.shape}")
        x = torch.from_numpy(multi_modal[None]).to(self.device).float()
        with torch.no_grad():
            logits = sliding_window_inference(
                inputs=x,
                roi_size=self.patch_size,
                sw_batch_size=2,
                predictor=self._net,
                overlap=self.overlap,
            )
            prob = torch.softmax(logits, dim=1)[0, 1]
        return prob.detach().cpu().numpy().astype(np.float32)

    def propose(self, multi_modal: np.ndarray, min_voxels: int = 10) -> list[Candidate]:
        prob = self.predict_probmap(multi_modal)
        return probmap_to_candidates(prob, threshold=self.threshold, min_voxels=min_voxels)


def probmap_to_candidates(
    prob: np.ndarray,
    threshold: float = 0.5,
    min_voxels: int = 10,
    connectivity: int = 1,
) -> list[Candidate]:
    binary = prob >= threshold
    if not binary.any():
        return []
    labeled, n = ndi.label(binary, structure=ndi.generate_binary_structure(3, connectivity))
    if n == 0:
        return []
    counts = np.bincount(labeled.ravel())

    out: list[Candidate] = []
    for lab in range(1, n + 1):
        cnt = int(counts[lab])
        if cnt < min_voxels:
            continue
        coords = np.argwhere(labeled == lab)
        cz, cy, cx = coords.mean(axis=0)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0) + 1
        bbox = (
            slice(int(mins[0]), int(maxs[0])),
            slice(int(mins[1]), int(maxs[1])),
            slice(int(mins[2]), int(maxs[2])),
        )
        mask = labeled == lab
        mean_prob = float(prob[mask].mean())
        out.append(Candidate(
            coord_vox=(int(round(cz)), int(round(cy)), int(round(cx))),
            bbox=bbox,
            mask=mask,
            prob=mean_prob,
            voxel_count=cnt,
        ))
    out.sort(key=lambda c: c.prob, reverse=True)
    return out
