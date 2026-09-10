"""
RGBDieDetector
==============
2D die detection using 5-step colour-clustering and polygon segmentation.

Step 1 — K-Means colour quantisation in LAB space.
Step 2 — Identify the cluster matching the target die colour; compute convex hull.
Step 3 — Crop the image to the bounding box of the best blob.
Step 4 — B&W thresholding + contour analysis to segment faces and find pips.
Step 5 — Return visible faces (with pip counts) and debug step images.
"""

from __future__ import annotations

import math
from itertools import combinations

import cv2
import numpy as np

from .die_detector_params import DieDetectorParams


class RGBDieDetector:
    """Detects a die and its pips from a single RGB image.

    Parameters
    ----------
    params : DieDetectorParams
        Uses ``die_color``, ``num_color_clusters``, ``debug``.
    """

    # BGR values for named colours
    _COLOR_MAP_BGR: dict = {
        "white":   (255, 255, 255),
        "black":   (0, 0, 0),
        "red":     (0, 0, 255),
        "green":   (0, 255, 0),
        "blue":    (255, 0, 0),
        "yellow":  (0, 255, 255),
        "orange":  (0, 165, 255),
        "cyan":    (255, 255, 0),
        "magenta": (255, 0, 255),
        "purple":  (128, 0, 128),
        "gray":    (128, 128, 128),
        "grey":    (128, 128, 128),
    }

    def __init__(self, params: DieDetectorParams | None = None) -> None:
        self.params = params or DieDetectorParams()
        self._log = self._make_logger()

    # ──────────────────────────────────────────────────────────────────────
    def detect(
        self,
        rgb_bgr: np.ndarray,
        die_color: str | None = None,
        num_color_clusters: int | None = None,
    ) -> dict:
        """Run the full 5-step detection on *rgb_bgr*.

        Parameters
        ----------
        rgb_bgr : (H, W, 3) uint8 BGR image
        die_color : str, optional — overrides ``params.die_color``
        num_color_clusters : int, optional — overrides ``params.num_color_clusters``

        Returns
        -------
        dict with keys:
            bbox, contour, convex_hull,
            num_visible_faces, faces, total_pips, num_pips,
            debug_steps (dict of step images)
        """
        color = die_color or self.params.die_color
        k = num_color_clusters or self.params.num_color_clusters
        return self._detect_5step(rgb_bgr, num_color_clusters=k, die_color=color)

    # Backward-compatible static alias
    @classmethod
    def detect_die_5step(
        cls,
        rgb_bgr: np.ndarray,
        num_color_clusters: int = 5,
        die_color: str = "white",
    ) -> dict:
        inst = cls()
        return inst._detect_5step(rgb_bgr, num_color_clusters, die_color)

    @classmethod
    def detect_die_2d(cls, rgb_bgr: np.ndarray, die_color: str = "white") -> dict:
        return cls.detect_die_5step(rgb_bgr, die_color=die_color)

    # ──────────────────────────────────────────────────────────────────────
    # Internal implementation
    # ──────────────────────────────────────────────────────────────────────

    def _detect_5step(
        self,
        rgb_bgr: np.ndarray,
        num_color_clusters: int = 5,
        die_color: str = "white",
    ) -> dict:
        h, w = rgb_bgr.shape[:2]

        # ── Step 1: K-Means colour clustering in LAB ──────────────────────
        lab = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2LAB)
        pixels = lab.reshape((-1, 3)).astype(np.float32)
        K = num_color_clusters
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers_bgr = cv2.cvtColor(
            np.uint8(centers).reshape(1, K, 3), cv2.COLOR_LAB2BGR
        ).reshape(K, 3)
        clustered_flat = centers_bgr[labels.flatten()]
        step1_img = clustered_flat.reshape((h, w, 3))
        self._log("Step 1: K-Means clustering done.")

        # ── Step 2: Target cluster → convex hull ──────────────────────────
        target_idx = self._target_cluster(centers, K, die_color)
        labels_2d = labels.reshape((h, w))
        whitest_mask = (labels_2d == target_idx).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        whitest_clean = cv2.morphologyEx(whitest_mask, cv2.MORPH_CLOSE, kernel)
        whitest_clean = cv2.morphologyEx(whitest_clean, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(whitest_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        step2_img = rgb_bgr.copy()
        best_hull = best_cnt = best_bbox = None
        best_score = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 800 < area < 60_000:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = float(bw) / float(bh)
                if 0.4 <= aspect <= 2.2:
                    hull = cv2.convexHull(cnt)
                    crop_gray = cv2.cvtColor(rgb_bgr[y:y+bh, x:x+bw], cv2.COLOR_BGR2GRAY)
                    score = area * crop_gray.std()
                    if score > best_score:
                        best_score = score
                        best_hull = hull
                        best_cnt = cnt
                        best_bbox = (x, y, bw, bh)

        if best_hull is not None:
            cv2.drawContours(step2_img, [best_hull], -1, (0, 255, 0), 3)
            if best_cnt is not None:
                cv2.drawContours(step2_img, [best_cnt], -1, (0, 0, 255), 1)
        self._log(f"Step 2: Best blob score={best_score:.1f}, bbox={best_bbox}.")

        # ── Step 3: Crop ──────────────────────────────────────────────────
        if best_bbox is not None:
            bx, by, bw, bh = best_bbox
            margin = 12
            x1 = max(0, bx - margin)
            y1 = max(0, by - margin)
            x2 = min(w, bx + bw + margin)
            y2 = min(h, by + bh + margin)
            step3_crop = rgb_bgr[y1:y2, x1:x2].copy()
            actual_bbox = (x1, y1, x2 - x1, y2 - y1)
        else:
            x1, y1, x2, y2 = 0, 0, w, h
            actual_bbox = (0, 0, w, h)
            step3_crop = rgb_bgr.copy()

        # ── Step 4: Face & pip segmentation ───────────────────────────────
        crop_h, crop_w = step3_crop.shape[:2]
        crop_gray = cv2.cvtColor(step3_crop, cv2.COLOR_BGR2GRAY)

        hull_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        if best_hull is not None:
            hull_crop = best_hull - np.array([x1, y1])
            cv2.drawContours(hull_mask, [hull_crop], -1, 255, -1)
        else:
            hull_mask.fill(255)

        _, bw_mask = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw_mask = cv2.bitwise_and(bw_mask, hull_mask)

        _, dark_mask = cv2.threshold(crop_gray, 105, 255, cv2.THRESH_BINARY_INV)
        dark_mask = cv2.bitwise_and(dark_mask, hull_mask)

        edges = cv2.Canny(crop_gray, 40, 130)
        edges = cv2.bitwise_and(edges, hull_mask)
        edges_dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        # Pip candidates
        pip_cnts, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_pip_area = crop_w * crop_h * 0.025
        max_pip_radius = min(crop_w, crop_h) * 0.09
        valid_pips = []
        for pc in pip_cnts:
            pa = cv2.contourArea(pc)
            if 6 < pa < max_pip_area:
                (px, py), pr = cv2.minEnclosingCircle(pc)
                if pr <= max_pip_radius:
                    circle_area = math.pi * pr ** 2
                    if circle_area > 0 and (pa / circle_area) > 0.18:
                        bx_p, by_p, bw_p, bh_p = cv2.boundingRect(pc)
                        asp = float(bw_p) / float(bh_p) if bh_p > 0 else 0
                        if 0.3 <= asp <= 3.2:
                            valid_pips.append({
                                "center": (int(px), int(py)),
                                "radius": max(2, int(pr)),
                                "contour": pc,
                                "area": pa,
                            })

        # Fill pip holes to preserve solid face regions
        bw_filled = bw_mask.copy()
        cnts_cc, hier = cv2.findContours(bw_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is not None:
            for i in range(len(cnts_cc)):
                if hier[0][i][3] >= 0:
                    if cv2.contourArea(cnts_cc[i]) < (crop_w * crop_h * 0.08):
                        cv2.drawContours(bw_filled, [cnts_cc[i]], -1, 255, -1)

        face_cnts, _ = cv2.findContours(bw_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_candidates = sorted(
            [fc for fc in face_cnts if cv2.contourArea(fc) > crop_w * crop_h * 0.03],
            key=cv2.contourArea,
            reverse=True,
        )

        candidate_faces = []
        if all_candidates:
            max_area = cv2.contourArea(all_candidates[0])
            for fc in all_candidates:
                area = cv2.contourArea(fc)
                if area >= max(crop_w * crop_h * 0.04, 0.35 * max_area):
                    candidate_faces.append(fc)
                if len(candidate_faces) >= 3:
                    break

        step4_vis = step3_crop.copy()
        face_colors = [(255, 0, 0), (0, 255, 0), (0, 165, 255)]
        visible_faces: list[dict] = []

        for face_idx, fc in enumerate(candidate_faces, start=1):
            poly = self._fit_quadrilateral(fc)
            if poly is None:
                continue
            face_pips = [p for p in valid_pips
                         if cv2.pointPolygonTest(poly, (float(p["center"][0]), float(p["center"][1])), False) >= 0]
            visible_faces.append({
                "face_index": face_idx,
                "contour": fc,
                "polygon": poly,
                "num_pips": len(face_pips),
                "pips": face_pips,
            })
            color = face_colors[(face_idx - 1) % len(face_colors)]
            cv2.polylines(step4_vis, [poly], True, color, 2)

        if not visible_faces:
            poly_crop = np.array([[[0, 0]], [[crop_w, 0]], [[crop_w, crop_h]], [[0, crop_h]]])
            visible_faces.append({
                "face_index": 1,
                "contour": poly_crop,
                "polygon": poly_crop,
                "num_pips": len(valid_pips),
                "pips": valid_pips,
            })

        visible_faces = visible_faces[:3]

        # Mark top face (lowest Y centroid in image = topmost face)
        def _cy(f):
            M = cv2.moments(f["polygon"])
            return M["m01"] / M["m00"] if M["m00"] > 0 else np.mean(f["polygon"][:, 0, 1])

        top_f = min(visible_faces, key=_cy)
        for f in visible_faces:
            f["is_top_face"] = f is top_f

        # Draw pips
        for p in valid_pips:
            cv2.circle(step4_vis, p["center"], p["radius"], (0, 0, 255), 2)
            cv2.circle(step4_vis, p["center"], 2, (0, 255, 0), -1)

        # ── Step 5: Gather results ─────────────────────────────────────────
        total_pips = sum(f["num_pips"] for f in visible_faces)
        self._log(f"Step 5: {len(visible_faces)} visible faces, {total_pips} total pips.")

        return {
            "bbox": actual_bbox,
            "contour": best_cnt if best_cnt is not None else best_hull,
            "convex_hull": best_hull,
            "num_visible_faces": len(visible_faces),
            "faces": visible_faces,
            "total_pips": total_pips,
            "num_pips": total_pips,
            "debug_steps": {
                "step1_color_clustered": step1_img,
                "step2_whitest_convex_hull": step2_img,
                "step2_whitest_mask": cv2.cvtColor(whitest_clean, cv2.COLOR_GRAY2BGR),
                "step3_crop_rgb": step3_crop,
                "step4_bw_mask": cv2.cvtColor(bw_mask, cv2.COLOR_GRAY2BGR),
                "step4_dark_pips_mask": cv2.cvtColor(dark_mask, cv2.COLOR_GRAY2BGR),
                "step4_faces_and_pips": step4_vis,
                "step5_annotated_result": step4_vis.copy(),
            },
        }

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _target_cluster(self, centers: np.ndarray, K: int, die_color: str) -> int:
        """Return the K-Means cluster index that best matches *die_color*."""
        dc = die_color.lower()
        if dc in ("white", "whitest"):
            return int(np.argmax(centers[:, 0]))   # max L*
        if dc in ("black", "darkest"):
            return int(np.argmin(centers[:, 0]))   # min L*
        bgr_target = self._COLOR_MAP_BGR.get(dc)
        if bgr_target is None and isinstance(die_color, (tuple, list, np.ndarray)):
            bgr_target = tuple(die_color)
        if bgr_target is None:
            bgr_target = (255, 255, 255)
        target_lab = cv2.cvtColor(np.uint8([[bgr_target]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
        distances = np.linalg.norm(centers - target_lab, axis=1)
        return int(np.argmin(distances))

    @staticmethod
    def _fit_quadrilateral(fc: np.ndarray) -> np.ndarray | None:
        """Fit a convex quadrilateral to a face contour."""
        hull = cv2.convexHull(fc)
        arc_len = cv2.arcLength(hull, True)
        if arc_len == 0:
            return None

        best_quad = None
        for eps in np.linspace(0.015, 0.15, 40):
            approx = cv2.approxPolyDP(hull, eps * arc_len, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                best_quad = approx
                break

        if best_quad is None:
            approx = cv2.approxPolyDP(hull, 0.02 * arc_len, True)
            pts = approx.reshape(-1, 2)
            if len(pts) >= 4:
                max_area = -1.0
                for idxs in combinations(range(len(pts)), 4):
                    quad = pts[list(idxs)].reshape(-1, 1, 2)
                    if cv2.isContourConvex(quad):
                        area = cv2.contourArea(quad)
                        if area > max_area:
                            max_area = area
                            best_quad = quad

        if best_quad is None:
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            best_quad = np.int32(box).reshape(-1, 1, 2)

        return best_quad

    def _make_logger(self):
        tag = "[RGBDieDetector]"
        if self.params.debug:
            return lambda msg: print(f"{tag} {msg}")
        return lambda msg: None
