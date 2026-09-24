"""
face_polygon_pipeline.py
========================
Die detection + pose from the *face polygons* of RGBDieDetector (the original
standard-CV path), on a given table plane:

  1. die detection + segmentation: YOLOE (``detection_mode="yoloe"``, with the
     detector's own HSV fallback) or colour segmentation (``"hsv"`` / ``"kmeans"``)
  2. face / pip segmentation inside the die crop (RGBDieDetector Steps 3-5)
  3. top face = uppermost face polygon; top / front face = their pip counts
  4. pose: PoseEstimator.compute_pose from the top-face polygon edges
  5. quality check: silhouette IoU of the resulting cube against the die
     outline (same measure as SilhouettePipeline); < ``min_face_pose_iou`` is
     rejected (lower than ``min_fit_iou``: these poses are not IoU-optimised,
     so a correct pose scores ~0.7-0.95, a wrong object / truncated die < 0.4)

``pose_method: yoloe_faces`` in the ROS node is this pipeline with YOLOE; with
colour detection it is the classical path (benchmarked in run_bag_pose.py).
Plane, intrinsics, axis convention (x = front-face normal) and result keys
match SilhouettePipeline, so both are interchangeable.
"""

from __future__ import annotations

import copy
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R_sci

from .die_detector_params import DieDetectorParams
from .pose_estimator import PoseEstimator
from .rgb_die_detector import RGBDieDetector
from .silhouette_pipeline import resolve_device
from .silhouette_pose import SilhouettePoseFitter, compose_debug_panel, die_silhouette, draw_die_pose


def _centre(poly):
    M = cv2.moments(poly)
    if M["m00"] > 0:
        return M["m10"] / M["m00"], M["m01"] / M["m00"]
    return float(np.mean(poly[:, 0, 0])), float(np.mean(poly[:, 0, 1]))


class FacePolygonPipeline:
    """Die detection (YOLOE or colour) + face/pip segmentation + top-face-polygon pose."""

    def __init__(self, params: DieDetectorParams | None = None, detection_mode: str = "yoloe") -> None:
        self.params = copy.copy(params or DieDetectorParams())
        self.params.detection_mode = detection_mode
        self.device = resolve_device(self.params.device)
        self._det = RGBDieDetector(self.params)
        self._pose = PoseEstimator(self.params)
        self._yoloe_ms = 0.0
        if detection_mode == "yoloe":
            model = self._det._get_yoloe_model()
            if model is None:
                raise RuntimeError(f"YOLOE model could not be loaded: {self._det.yoloe_error}")
            predict = model.predict

            def timed_predict(*a, **k):       # force the device and time YOLOE separately
                k.setdefault("device", self.device)
                t = time.perf_counter()
                out = predict(*a, **k)
                self._yoloe_ms = (time.perf_counter() - t) * 1e3
                return out
            model.predict = timed_predict
        self.last_fitter: SilhouettePoseFitter | None = None

    def _face_valid(self, f) -> bool:
        if f is None or not f.get("is_trusted", True) or not f.get("aligned_pips", True):
            return False
        return self.params.min_pips <= f.get("num_pips", 0) <= self.params.max_pips

    def process(self, rgb_bgr: np.ndarray, K: np.ndarray, plane_n: np.ndarray, plane_h: float) -> dict:
        p = self.params
        s = p.die_size_m
        timings = {}
        t0 = time.perf_counter()
        self._yoloe_ms = 0.0
        det = self._det.detect(rgb_bgr)
        timings["detector"] = (time.perf_counter() - t0) * 1e3
        if p.detection_mode == "yoloe":
            timings["yoloe"] = self._yoloe_ms

        fitter = SilhouettePoseFitter(K, plane_n, plane_h, s)
        self.last_fitter = fitter
        result = {"valid": False, "reason": "", "timings_ms": timings, "faces": det["faces"],
                  "pip_count": det["total_pips"], "top_face_str": "None", "front_face_str": "None",
                  "top_face_pips": None, "front_face_pips": None, "cnn_classification": None,
                  "source": p.detection_mode}
        x_c, y_c, w_c, h_c = det["bbox"]
        faces = det["faces"]
        if w_c <= 10 or h_c <= 10:
            result["reason"] = "no die detected"
        elif not faces:
            result["reason"] = "no face polygons in the die crop"
        else:
            t1 = time.perf_counter()
            # Top face = uppermost polygon; primary lateral = next uppermost (DieDetectorPipeline)
            cy = lambda f: _centre(f["polygon"])[1]
            top = min(faces, key=cy)
            lats = sorted([f for f in faces if not f.get("is_top_face")], key=cy)
            lat = lats[0] if lats else None
            poly = top["polygon"] + np.array([x_c, y_c])
            lat_2d = _centre(lat["polygon"] + np.array([x_c, y_c])) if lat is not None else None
            centroid_top, _, R_est, _ = self._pose.compute_pose(
                die_points=np.zeros((0, 3), np.float32), plane_normal=fitter.n.astype(np.float32),
                R_plane=np.eye(3, dtype=np.float32), top_face_polygon=poly,
                top_face_centroid_2d=_centre(poly), primary_lat_centroid_2d=lat_2d,
                plane_D=fitter.h, die_size=s, camera_params=(K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
            # Same die frame convention as the silhouette fit: x = front-face normal
            x_ax = R_est[:, 0]
            theta = float(np.arctan2(float(x_ax @ fitter.b), float(x_ax @ fitter.a)))
            contact = centroid_top.astype(np.float64) - fitter.n * s
            R_die, yaw_deg = fitter.front_frame(contact, theta)
            centre = contact + fitter.n * s / 2
            quat = R_sci.from_matrix(R_die).as_quat()
            timings["pose"] = (time.perf_counter() - t1) * 1e3

            # Same quality check as the silhouette fit: IoU of this pose's cube vs the die outline
            outline = die_silhouette(rgb_bgr, det["debug_steps"]["step2_whitest_mask"][:, :, 0], det["bbox"])
            iou = fitter.score_pose(outline, rgb_bgr.shape, contact, theta) if outline is not None else 0.0

            top_pips = int(top["num_pips"]) if self._face_valid(top) else None
            front_pips = int(lat["num_pips"]) if self._face_valid(lat) else None
            fit = {"centroid": centre, "R": R_die, "theta_deg": yaw_deg, "quat": quat, "iou": iou,
                   "corners_2d": fitter.project(fitter.corners_3d(contact, theta))}
            valid = iou >= p.min_face_pose_iou
            if not valid:
                result["reason"] = f"pose silhouette IoU {iou:.2f} < min_face_pose_iou {p.min_face_pose_iou:.2f}"
            result.update({
                "valid": valid, "fit": fit, "contour": poly, "yaw_deg": yaw_deg, "R": R_die, "iou": iou,
                "centroid": (centre + fitter.n * s / 2).astype(np.float32),     # top-face centre
                "die_centroid_tf": centre.astype(np.float32),
                "table_surface_tf": (centre - fitter.n * s / 2).astype(np.float32),
                "quaternion": quat,
                "top_face_pips": top_pips, "front_face_pips": front_pips,
                "top_face_str": str(top_pips) if top_pips is not None else "None",
                "front_face_str": str(front_pips) if front_pips is not None else "None",
                "num_visible_faces": det["num_visible_faces"],
            })
        timings["total"] = (time.perf_counter() - t0) * 1e3
        return result

    def build_debug_panels(self, rgb_bgr: np.ndarray, result: dict, zoom_px: int = 400) -> np.ndarray:
        """Full image with top-face polygon, cube and TF axes on top; zoomed die + detection info below."""
        full = rgb_bgr.copy()
        zoom = np.full((zoom_px, zoom_px, 3), 30, np.uint8)
        fit = result.get("fit")
        if result.get("valid"):
            lines = [f"TOP {result['top_face_str']}  FRONT {result['front_face_str']}"]
        else:
            lines = ["NO POSE: " + result.get("reason", "")]
        if fit is not None and self.last_fitter is not None:
            draw_die_pose(full, self.last_fitter, fit, result["contour"])
            ref = fit["corners_2d"].astype(np.float32)
            x, y, w, h = cv2.boundingRect(ref.reshape(-1, 2).astype(np.int32))
            side = min(int(max(w, h) * 2.0), min(rgb_bgr.shape[:2]))
            x0 = int(np.clip(x + w / 2 - side / 2, 0, rgb_bgr.shape[1] - side))
            y0 = int(np.clip(y + h / 2 - side / 2, 0, rgb_bgr.shape[0] - side))
            zoom = cv2.resize(rgb_bgr[y0:y0 + side, x0:x0 + side], (zoom_px, zoom_px),
                              interpolation=cv2.INTER_CUBIC)
            draw_die_pose(zoom, self.last_fitter, fit, result["contour"], scale=zoom_px / side,
                          off=(x0, y0), thick=3)
            c = fit["centroid"] * 1000
            lines += [f"xyz (camera) [mm]: {c[0]:.0f}  {c[1]:.0f}  {c[2]:.0f}",
                      f"yaw: {fit['theta_deg']:+.1f} deg",
                      f"silhouette IoU: {fit['iou']:.3f}  ({result.get('source', '')})",
                      f"visible faces: {result.get('num_visible_faces', 0)}, pips: {result.get('pip_count', 0)}"]
        t = result.get("timings_ms", {})
        if "total" in t:
            lines.append(f"time: {t['total']:.0f} ms")
        return compose_debug_panel(full, zoom, lines, bool(result.get("valid")))
