"""
silhouette_pipeline.py
======================
Die detection + 6-DOF pose without face / pip segmentation:

  1. YOLOE instance segmentation  -> up to ``max_yoloe_candidates`` die masks
     (if no YOLOE fit passes: one HSV / k-means colour candidate, ``color_fallback``)
  2. convex die outline per candidate (silhouette_pose.die_silhouette)
  3. cube-silhouette fit of (x, y, yaw) on the table plane; the candidate with
     the best IoU wins (trying stops at ``accept_fit_iou``), and the frame is
     rejected if the best IoU < ``min_fit_iou``.
     YOLOE runs with a low confidence (``yoloe_candidate_conf``) because a
     wrong candidate costs one fit and loses on IoU.
  4. CNN top / front face classification on the winning crop

The table plane (camera optical frame, ``n · X + h = 0`` with n pointing up)
and the intrinsics are passed in per frame, so the caller decides where they
come from (TF + table height, a fixed calibration, or a depth RANSAC fit).

The result dict uses the same keys as DieDetectorPipeline where they overlap
(``centroid`` = top-face centre, ``quaternion``, ``die_centroid_tf``, ...), so
the ROS node can publish either.
"""

from __future__ import annotations

import copy
import os
import time

import cv2
import numpy as np

from .cnn_die_classifier import CNNDieClassifier
from .die_detector_params import PACKAGE_ROOT, DieDetectorParams
from .rgb_die_detector import RGBDieDetector
from .silhouette_pose import SilhouettePoseFitter, compose_debug_panel, die_silhouette, draw_die_pose


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class SilhouettePipeline:
    """YOLOE + cube-silhouette pose + CNN orientation."""

    def __init__(self, params: DieDetectorParams | None = None) -> None:
        self.params = params or DieDetectorParams()
        self.device = resolve_device(self.params.device)
        # YOLOE loading (weights resolved inside <pkg>/weights) lives in RGBDieDetector
        yoloe_det = RGBDieDetector(self.params)
        self._yoloe = yoloe_det._get_yoloe_model()
        if self._yoloe is None:
            raise RuntimeError(f"YOLOE model could not be loaded: {yoloe_det.yoloe_error}")
        # Colour-segmentation fallback (classical Steps 1-2 of RGBDieDetector)
        self._color_det = None
        if self.params.color_fallback:
            cp = copy.copy(self.params)
            if cp.detection_mode not in ("hsv", "kmeans"):
                cp.detection_mode = "hsv"
            self._color_det = RGBDieDetector(cp)
        model_path = self.params.cnn_model_path
        if not os.path.isabs(model_path):
            model_path = os.path.join(PACKAGE_ROOT, model_path)
        self._cnn = CNNDieClassifier(model_path=model_path, conf_thresh=self.params.cnn_conf_thresh,
                                     device=self.device)
        self.last_fitter: SilhouettePoseFitter | None = None

    # ──────────────────────────────────────────────────────────────────────
    def _yoloe_candidates(self, rgb_bgr: np.ndarray):
        """[(conf, full-image uint8 mask, bbox)] sorted by confidence."""
        p = self.params
        preds = self._yoloe.predict(rgb_bgr, conf=p.yoloe_candidate_conf, imgsz=p.yoloe_imgsz,
                                    device=self.device, verbose=False)
        if not preds or preds[0].masks is None or preds[0].boxes is None:
            return []
        res = preds[0]
        H, W = rgb_bgr.shape[:2]
        order = np.argsort(-res.boxes.conf.cpu().numpy())
        out = []
        for i in order:
            poly = res.masks.xy[i]           # polygon in original image pixels
            if len(poly) < 3:
                continue
            x1, y1, x2, y2 = res.boxes.xyxy[i].cpu().numpy()
            if (x2 - x1) * (y2 - y1) < 1000:
                continue
            mask = np.zeros((H, W), np.uint8)
            cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 255)
            bbox = (int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1)))
            out.append((float(res.boxes.conf[i]), mask, bbox, "yoloe"))
            if len(out) >= p.max_yoloe_candidates:
                break
        return out

    def _color_candidate(self, rgb_bgr: np.ndarray):
        det = self._color_det.detect(rgb_bgr)
        x, y, w, h = det["bbox"]
        if w <= 10 or h <= 10 or det["contour"] is None:
            return []
        mask = np.zeros(rgb_bgr.shape[:2], np.uint8)
        cv2.drawContours(mask, [det["contour"]], -1, 255, -1)
        return [(0.0, mask, (int(x), int(y), int(w), int(h)), "color")]

    def process(self, rgb_bgr: np.ndarray, K: np.ndarray, plane_n: np.ndarray, plane_h: float) -> dict:
        """Detect the die and estimate its pose in the camera optical frame."""
        p = self.params
        s = p.die_size_m
        timings = {}
        t0 = time.perf_counter()
        cands = self._yoloe_candidates(rgb_bgr)
        timings["yoloe"] = (time.perf_counter() - t0) * 1e3

        fitter = SilhouettePoseFitter(K, plane_n, plane_h, s)
        self.last_fitter = fitter
        best = None
        tried = []

        def try_candidates(candidates):
            nonlocal best
            for conf, mask, bbox, source in candidates:
                contour = die_silhouette(rgb_bgr, mask, bbox)
                if contour is None or len(contour) < 3:
                    continue
                fit = fitter.fit(contour, rgb_bgr.shape)
                tried.append({"conf": conf, "iou": fit["iou"], "bbox": bbox, "source": source})
                if best is None or fit["iou"] > best[0]["iou"]:
                    best = (fit, contour, bbox, conf, source)
                if fit["iou"] >= p.accept_fit_iou:
                    break

        t1 = time.perf_counter()
        try_candidates(cands)
        timings["outline+fit"] = (time.perf_counter() - t1) * 1e3
        if (best is None or best[0]["iou"] < p.min_fit_iou) and self._color_det is not None:
            tc = time.perf_counter()
            try_candidates(self._color_candidate(rgb_bgr))
            timings["color_fallback"] = (time.perf_counter() - tc) * 1e3

        result = {"valid": False, "reason": "", "candidates": tried, "timings_ms": timings,
                  "faces": [], "pip_count": 0, "top_face_str": "None", "front_face_str": "None",
                  "top_face_pips": None, "front_face_pips": None, "cnn_classification": None}
        if best is None:
            result["reason"] = "no die candidate (YOLOE" + (" + colour fallback)" if self._color_det else ")")
        else:
            fit, contour, bbox, conf, source = best
            result.update({"fit": fit, "contour": contour, "bbox": bbox, "yoloe_conf": conf, "source": source,
                           "iou": fit["iou"], "yaw_deg": fit["theta_deg"], "R": fit["R"]})
            if fit["iou"] < p.min_fit_iou:
                result["reason"] = f"best silhouette IoU {fit['iou']:.2f} < min_fit_iou {p.min_fit_iou:.2f}"
            else:
                # CNN on the detector-style crop (bbox + 12 px margin), as in training
                t2 = time.perf_counter()
                H, W = rgb_bgr.shape[:2]
                x, y, w, h = cv2.boundingRect(contour.reshape(-1, 2).astype(np.int32))
                m = 12
                crop = rgb_bgr[max(0, y - m):min(H, y + h + m), max(0, x - m):min(W, x + w + m)]
                cnn = self._cnn.predict(crop)
                timings["cnn"] = (time.perf_counter() - t2) * 1e3

                z = fit["R"][:, 2]
                centre = fit["centroid"]
                top_face = int(cnn["top_face"])
                front_face = cnn["front_face"]
                result.update({
                    "valid": True,
                    "cnn_classification": cnn,
                    "centroid": (centre + z * s / 2).astype(np.float32),        # top-face centre
                    "die_centroid_tf": centre.astype(np.float32),
                    "table_surface_tf": (centre - z * s / 2).astype(np.float32),
                    "quaternion": fit["quat"],
                    "top_face_pips": top_face,
                    "front_face_pips": front_face,
                    "top_face_str": str(top_face),
                    "front_face_str": str(front_face) if front_face is not None else "None",
                })
        timings["total"] = (time.perf_counter() - t0) * 1e3
        return result

    # ──────────────────────────────────────────────────────────────────────
    def build_debug_panels(self, rgb_bgr: np.ndarray, result: dict, zoom_px: int = 400) -> np.ndarray:
        """Full image with the fitted cube / TF axes on top; zoomed die + detection info below."""
        full = rgb_bgr.copy()
        zoom = np.full((zoom_px, zoom_px, 3), 30, np.uint8)
        fit = result.get("fit")
        cnn = result.get("cnn_classification")
        if result.get("valid") and cnn is not None:
            lines = [f"TOP {cnn['top_face']}  FRONT {cnn['front_face'] or '-'}"]
        else:
            lines = ["NO POSE: " + result.get("reason", "")]
        if fit is not None and self.last_fitter is not None:
            contour = result["contour"]
            draw_die_pose(full, self.last_fitter, fit, contour)
            x, y, w, h = cv2.boundingRect(contour.reshape(-1, 2).astype(np.int32))
            side = min(int(max(w, h) * 2.0), min(rgb_bgr.shape[:2]))
            x0 = int(np.clip(x + w / 2 - side / 2, 0, rgb_bgr.shape[1] - side))
            y0 = int(np.clip(y + h / 2 - side / 2, 0, rgb_bgr.shape[0] - side))
            zoom = cv2.resize(rgb_bgr[y0:y0 + side, x0:x0 + side], (zoom_px, zoom_px),
                              interpolation=cv2.INTER_CUBIC)
            draw_die_pose(zoom, self.last_fitter, fit, contour, scale=zoom_px / side, off=(x0, y0), thick=3)
            c = fit["centroid"] * 1000
            lines += [f"xyz (camera) [mm]: {c[0]:.0f}  {c[1]:.0f}  {c[2]:.0f}",
                      f"yaw: {fit['theta_deg']:+.1f} deg",
                      f"silhouette IoU: {fit['iou']:.3f}  ({result.get('source', '')})"]
        if cnn is not None:
            lines.append(f"CNN conf: top {cnn['top_conf']:.2f}, front {cnn['front_conf']:.2f}")
        t = result.get("timings_ms", {})
        if "total" in t:
            lines.append(f"time: {t['total']:.0f} ms")
        return compose_debug_panel(full, zoom, lines, bool(result.get("valid")))
