"""
CNNDieClassifier
================
Lightweight Two-Head MobileNetV3-Small classifier for 3D die orientation estimation.
Predicts Top Face (1-6) and Front Face (1-6, or None if ambiguous/top-down).

Runs on standard CPU in < 3 ms per crop.
"""

from __future__ import annotations

import os
import time
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch


class CNNDieClassifier:
    """CPU-optimised classifier for die orientation prediction from image crops.

    Parameters
    ----------
    model_path : str | None
        Path to the traced TorchScript model (.pt file). If None, defaults to
        PACKAGE_ROOT/weights/die_mobilenet_v3.pt.
    conf_thresh : float
        Confidence threshold for accepting a prominent front face (default 0.60).
    device : str
        Inference device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        model_path: str | None = None,
        conf_thresh: float = 0.60,
        device: str = "cpu",
        num_threads: int | None = 4,
    ) -> None:
        self.conf_thresh = conf_thresh
        self.device = torch.device(device)
        # PyTorch defaults to one thread per core; for a model this small that
        # oversubscribes the CPU (~350-900 ms/crop on 20 threads vs ~5 ms on 4).
        # Note: torch's intra-op thread count is process-global.
        if num_threads and self.device.type == "cpu" and torch.get_num_threads() > num_threads:
            torch.set_num_threads(num_threads)
        self._model = None

        if model_path is None:
            # Default package weights path
            pkg_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            model_path = os.path.join(pkg_root, "weights", "die_mobilenet_v3.pt")

        self.model_path = model_path
        self._load_model()

    def _load_model(self) -> None:
        if os.path.exists(self.model_path):
            try:
                self._model = torch.jit.load(self.model_path, map_location=self.device)
                self._model.eval()
                # Warmup run
                dummy = torch.randn(1, 3, 224, 224, device=self.device)
                with torch.no_grad():
                    _ = self._model(dummy)
            except Exception as e:
                print(f"[CNNDieClassifier] Failed to load TorchScript model '{self.model_path}': {e}")
                self._model = None
        else:
            self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """Preprocess BGR crop into normalized tensor (1, 3, 224, 224)."""
        # Resize to 224x224
        resized = cv2.resize(crop_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb - mean) / std

        # Transpose from (H, W, C) to (1, C, H, W)
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

    def predict(self, crop_bgr: np.ndarray) -> Dict[str, Any]:
        """Predict Top Face and Front Face from die crop.

        Parameters
        ----------
        crop_bgr : (H, W, 3) uint8 image crop of the die.

        Returns
        -------
        dict with:
            top_face: int (1..6)
            top_conf: float (0.0..1.0)
            front_face: int | None (1..6, or None if ambiguous/shaded/top-down)
            front_conf: float (0.0..1.0)
            top_probs: list of 6 floats
            front_probs: list of 7 floats (index 0 is None)
            latency_ms: float
        """
        if self._model is None:
            self._load_model()
            if self._model is None:
                raise RuntimeError(f"CNN model could not be loaded from '{self.model_path}'")

        t0 = time.perf_counter()
        inp = self.preprocess(crop_bgr)

        with torch.no_grad():
            outputs = self._model(inp)
            if isinstance(outputs, (tuple, list)) and len(outputs) >= 4:
                logits_top, logits_front, logits_double, logits_joint = outputs[:4]
                probs_top = torch.softmax(logits_top, dim=1)[0].cpu().numpy()
                probs_front = torch.softmax(logits_front, dim=1)[0].cpu().numpy()
                probs_double = torch.softmax(logits_double, dim=1)[0].cpu().numpy()
                probs_joint = torch.softmax(logits_joint, dim=1)[0].cpu().numpy()
            elif isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
                logits_top, logits_front = outputs[:2]
                probs_top = torch.softmax(logits_top, dim=1)[0].cpu().numpy()
                probs_front = torch.softmax(logits_front, dim=1)[0].cpu().numpy()
                probs_double = None
                probs_joint = None
            else:
                probs_top = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                probs_front = None
                probs_double = None
                probs_joint = None

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Top Face: index 0..5 -> Face 1..6
        top_idx = int(np.argmax(probs_top))
        top_face = top_idx + 1
        top_conf = float(probs_top[top_idx])

        # Front Face: index 0 is None/Single, index 1..6 is Face 1..6
        if probs_front is not None:
            front_idx = int(np.argmax(probs_front))
            front_conf = float(probs_front[front_idx])
            if front_idx == 0 or front_conf < self.conf_thresh:
                front_face = None
            else:
                front_face = front_idx
        else:
            front_face = None
            front_conf = 0.0

        is_double_face = (front_face is not None)
        double_conf = float(probs_double[1]) if probs_double is not None else (front_conf if is_double_face else 0.0)
        double_face_str = f"Top {top_face}, Front {front_face}" if is_double_face else f"Top {top_face} (Single Face)"

        joint_class_id = int(np.argmax(probs_joint)) if probs_joint is not None else -1

        return {
            "top_face": top_face,
            "top_conf": top_conf,
            "front_face": front_face,
            "front_conf": front_conf,
            "is_double_face": is_double_face,
            "double_conf": double_conf,
            "double_face_str": double_face_str,
            "joint_class_id": joint_class_id,
            "top_probs": probs_top.tolist(),
            "front_probs": probs_front.tolist() if probs_front is not None else [],
            "latency_ms": latency_ms,
        }

