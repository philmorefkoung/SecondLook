"""Adapter that consumes pre-computed BMSR-paper nnU-Net softmax probmaps.

The official UCSF-BMSR-benchmarks repo ships nnU-Net v1 weights (Rudie et al.,
Radiology: AI 2024). nnU-Net v1 predates Python 3.13 and pulls a heavy
batchgenerators / SimpleITK / older-torch stack that conflicts with our
project env, so inference is run once in a separate v1 conda env via
`scripts/run_nnunet_inference.py`. The output (per-study foreground softmax)
is dumped to `<dir>/<study_id>.npz` (key="softmax", shape (H,W,D), dtype
float32, same voxel grid as the input study). This class then loads those
probmaps on demand inside the agent's regular eval flow.

CAVEAT: the BMSR-paper nnU-Net was trained on the entire 461-study UCSF-BMSR
dataset, which includes our test split. Baseline numbers from this detector
will be inflated by leakage. This is acceptable for an "agent-vs-strong-
baseline" comparison: if the agent still adds value over a leakage-inflated
baseline, the result is more (not less) defensible.

Also exposes `EnsembleProbmapCache` for combining two or more probmap sources
(e.g. nnU-Net + AURORA) via union-of-binary-segs followed by joint size-ranking.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np


class NNUNetProbmapCache:
    """Read-only cache of pre-computed nnU-Net softmax volumes."""

    def __init__(self, probmap_dir: str | Path):
        self.probmap_dir = Path(probmap_dir)
        if not self.probmap_dir.exists():
            raise FileNotFoundError(self.probmap_dir)

    def has(self, study_id: str) -> bool:
        return any((self.probmap_dir / f"{study_id}{ext}").exists()
                   for ext in (".npz", "_softmax.npz", ".nii.gz"))

    def predict_probmap_for(self, study_id: str) -> np.ndarray:
        """Returns foreground probmap, shape (H,W,D), dtype float32, in [0,1].

        Preference order:
          1. `<id>_softmax.npz` (resampled-to-original softmax, if a future
             post-processing step generates it — best calibration)
          2. `<id>.nii.gz` (binary seg in original space; per-component
             prob = 0.3 + 0.7 * (size / max_size_in_study). This gives the
             size-ranked baseline FROC a clean monotonic sweep without
             requiring softmax resampling.)
          3. `<id>.npz` (raw nnUNet internal-space softmax, only useful if
             shape happens to match the study — otherwise downstream
             matching will misalign)
        """
        resampled = self.probmap_dir / f"{study_id}_softmax.npz"
        if resampled.exists():
            with np.load(resampled) as f:
                key = ("softmax" if "softmax" in f.files
                       else "prob" if "prob" in f.files
                       else f.files[0])
                arr = f[key]
                if arr.ndim == 4 and arr.shape[0] in (2, 3):
                    arr = arr[1]
                return arr.astype(np.float32)

        nii = self.probmap_dir / f"{study_id}.nii.gz"
        if nii.exists():
            import nibabel as nib
            from scipy import ndimage as ndi
            seg = (np.asanyarray(nib.load(str(nii)).dataobj) > 0).astype(np.int32)
            if not seg.any():
                return np.zeros(seg.shape, dtype=np.float32)
            labeled, n = ndi.label(seg, structure=ndi.generate_binary_structure(3, 1))
            out = np.zeros(seg.shape, dtype=np.float32)
            sizes = np.bincount(labeled.ravel())
            max_size = float(max(int(sizes[1:].max()), 1))
            for lab in range(1, n + 1):
                w = float(sizes[lab]) / max_size
                out[labeled == lab] = 0.3 + 0.7 * w
            return out

        raw = self.probmap_dir / f"{study_id}.npz"
        if raw.exists():
            with np.load(raw) as f:
                key = ("softmax" if "softmax" in f.files
                       else "prob" if "prob" in f.files
                       else f.files[0])
                arr = f[key]
                if arr.ndim == 4 and arr.shape[0] in (2, 3):
                    arr = arr[1]
                return arr.astype(np.float32)

        raise FileNotFoundError(
            f"No probmap for {study_id} in {self.probmap_dir}")


class EnsembleProbmapCache:
    """Combine 2+ binary-seg probmap sources via union of foreground masks.

    Each source dir is expected to hold per-study `<sid>.nii.gz` segmentation
    files (any positive value treated as foreground). The combined map:
      1. Loads each source's binary mask
      2. Unions them voxelwise (logical OR)
      3. Runs connected-component labeling on the union
      4. Returns a size-ranked probmap on the union (same `0.3 + 0.7 * size_norm`
         scheme as `NNUNetProbmapCache.predict_probmap_for`)

    The union approach captures lesions either detector finds individually
    (the recall-rescue path for the BraTS-Mets-winner ensemble experiment).
    Detectors that disagree spatially produce two separate CCs; detectors
    that agree get merged into one CC at the boundary union.
    """

    def __init__(self, *probmap_dirs):
        self.probmap_dirs = [Path(d) for d in probmap_dirs]
        for d in self.probmap_dirs:
            if not d.exists():
                raise FileNotFoundError(d)

    def has(self, study_id: str) -> bool:
        return all(
            any((d / f"{study_id}{ext}").exists()
                for ext in (".nii.gz", ".npz", "_softmax.npz"))
            for d in self.probmap_dirs
        )

    def predict_probmap_for(self, study_id: str) -> np.ndarray:
        import nibabel as nib
        from scipy import ndimage as ndi

        union: np.ndarray | None = None
        for d in self.probmap_dirs:
            nii = d / f"{study_id}.nii.gz"
            if not nii.exists():
                raise FileNotFoundError(
                    f"Ensemble member missing seg for {study_id} in {d}")
            arr = (np.asanyarray(nib.load(str(nii)).dataobj) > 0).astype(np.int32)
            union = arr if union is None else np.logical_or(union, arr).astype(np.int32)
        assert union is not None
        if not union.any():
            return np.zeros(union.shape, dtype=np.float32)

        labeled, n = ndi.label(union, structure=ndi.generate_binary_structure(3, 1))
        out = np.zeros(union.shape, dtype=np.float32)
        sizes = np.bincount(labeled.ravel())
        max_size = float(max(int(sizes[1:].max()), 1))
        for lab in range(1, n + 1):
            w = float(sizes[lab]) / max_size
            out[labeled == lab] = 0.3 + 0.7 * w
        return out
