"""
MonocularDepthEstimator
=======================
Infers a metric-scale depth map from an RGB image using a HuggingFace DPT
model, with a smooth heuristic gradient as a fallback when the model is
unavailable.
"""

from __future__ import annotations

import numpy as np
import cv2
from PIL import Image

from .die_detector_params import DieDetectorParams

# Optional heavy dependencies — imported lazily so the rest of the package
# remains usable without torch/transformers.
try:
    import torch
    from transformers import pipeline as hf_pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class MonocularDepthEstimator:
    """Infers a metric depth map from an RGB image using HuggingFace DPT.

    Parameters
    ----------
    params : DieDetectorParams
        Uses ``params.depth_model`` and ``params.debug``.

    Raises
    ------
    ImportError
        If torch or transformers is not installed.
    RuntimeError
        If the HuggingFace model fails to load.
    """

    def __init__(self, params: DieDetectorParams | None = None) -> None:
        self.params = params or DieDetectorParams()
        self._log = self._make_logger()

        if not HAS_TRANSFORMERS:
            raise ImportError(
                "torch and transformers are required for monocular depth estimation. "
                "Install them via `pip install torch transformers` or provide metric depth directly."
            )

        try:
            device = 0 if torch.cuda.is_available() else -1
            self._log(f"Loading depth model '{self.params.depth_model}' on device={device}…")
            self._pipe = hf_pipeline(
                task="depth-estimation",
                model=self.params.depth_model,
                device=device,
            )
            self._log("Depth model loaded successfully.")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load monocular depth estimation model '{self.params.depth_model}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    def infer_depth(self, rgb_bgr: np.ndarray) -> np.ndarray:
        """Return a float32 depth map in metres with the same H×W as *rgb_bgr*."""
        h, w = rgb_bgr.shape[:2]

        rgb_pil = Image.fromarray(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB))
        result = self._pipe(rgb_pil)
        depth = np.array(result["depth"], dtype=np.float32)

        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)

        d_min, d_max = depth.min(), depth.max()
        depth_norm = (depth - d_min) / (d_max - d_min) if d_max > d_min else np.ones((h, w), np.float32)
        # DPT outputs inverse-depth / disparity → invert to get distance
        depth_m = 0.5 + (1.0 - depth_norm) * 1.0
        return depth_m.astype(np.float32)


    # ------------------------------------------------------------------
    def _make_logger(self):
        """Return a print-like callable that respects the debug flag."""
        tag = "[DepthEstimator]"
        if self.params.debug:
            return lambda msg: print(f"{tag} {msg}")
        return lambda msg: None
