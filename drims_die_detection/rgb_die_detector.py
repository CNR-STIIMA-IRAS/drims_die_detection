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


def safe_find_contours(img, mode, method):
    """Find contours compatibly across OpenCV 3.x, 4.x, and 5.x."""
    res = cv2.findContours(img, mode, method)
    if len(res) == 2:
        return res[0], res[1]
    return res[1], res[2]


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
        mode = getattr(self.params, "detection_mode", "hsv").lower()

        # ── Step 1: Color Segmentation & Specular Glare Handling ─────────────
        glare_v_cutoff = getattr(self.params, "glare_v_thresh", 245)

        if mode == "kmeans":
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

            target_idx = self._target_cluster(centers, K, die_color)
            labels_2d = labels.reshape((h, w))
            whitest_mask = (labels_2d == target_idx).astype(np.uint8) * 255
            self._log("Step 1: K-Means clustering done.")

        else: # Default: HSV mode
            hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
            h_min = np.array(getattr(self.params, "hsv_min", [0, 0, 150]), dtype=np.uint8)
            h_max = np.array(getattr(self.params, "hsv_max", [180, 80, 255]), dtype=np.uint8)
            color_mask = cv2.inRange(hsv, h_min, h_max)

            # Detect specular glare highlights (high V, low saturation) to include in die body
            glare_mask = cv2.inRange(hsv, np.array([0, 0, glare_v_cutoff], dtype=np.uint8),
                                     np.array([180, 50, 255], dtype=np.uint8))
            whitest_mask = cv2.bitwise_or(color_mask, glare_mask)
            step1_img = cv2.bitwise_and(rgb_bgr, rgb_bgr, mask=whitest_mask)
            self._log(f"Step 1: HSV thresholding ({h_min.tolist()} - {h_max.tolist()}) with glare inclusion done.")

        # Compute Canny Edge Detection Mask for full image (Step 2 Panel & Candidate Scoring)
        gray_full = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray_full, (5, 5), 0)
        c_low = int(getattr(self.params, "canny_low_thresh", 40))
        c_high = int(getattr(self.params, "canny_high_thresh", 130))
        edges_full = cv2.Canny(gray_blur, c_low, c_high)
        edges_dilated = cv2.dilate(edges_full, np.ones((3, 3), np.uint8))

        # ── Step 2: Target cluster / mask → Candidate contour scoring ────────
        # Fill internal dark pip holes in whitest_mask before morphology,
        # so that thin white borders between outer pips and die edges are not eroded away.
        mask_filled = whitest_mask.copy()
        cnts_holes, hier_holes = safe_find_contours(whitest_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier_holes is not None:
            for i in range(len(cnts_holes)):
                if hier_holes[0][i][3] >= 0:  # Internal hole
                    if cv2.contourArea(cnts_holes[i]) < 2500:
                        cv2.drawContours(mask_filled, [cnts_holes[i]], -1, 255, -1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        whitest_clean = cv2.morphologyEx(mask_filled, cv2.MORPH_CLOSE, kernel)
        whitest_clean = cv2.morphologyEx(whitest_clean, cv2.MORPH_OPEN, kernel)

        contours, _ = safe_find_contours(whitest_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        step2_img = rgb_bgr.copy()
        step2_contours_img = rgb_bgr.copy()
        best_hull = best_cnt = best_bbox = None
        best_score = -1.0

        clip_limit = getattr(self.params, "clahe_clip_limit", 3.0)
        min_circ = getattr(self.params, "pip_min_circularity", 0.45)
        glare_cutoff = getattr(self.params, "glare_v_thresh", 245)

        evaluated_candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 400 < area < 15_000:
                x, y, bw, bh = cv2.boundingRect(cnt)
                # Reject contours that touch the frame boundary (background/table bleed)
                if (x <= 1) or (y <= 1) or (x + bw >= w - 1) or (y + bh >= h - 1):
                    continue
                aspect = float(bw) / float(bh) if bh > 0 else 0
                if 0.4 <= aspect <= 2.2:
                    hull = cv2.convexHull(cnt)
                    arc_len = cv2.arcLength(hull, True)

                    # 1. Test Quadrangular Geometry Fit (convex 4-sided polygon)
                    is_quad = False
                    if arc_len > 0:
                        approx = cv2.approxPolyDP(hull, 0.03 * arc_len, True)
                        if len(approx) == 4 and cv2.isContourConvex(approx):
                            is_quad = True

                    # 2. Edge Boundary Overlap Check (verifies sharp edges along die face border)
                    cnt_boundary = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(cnt_boundary, [cnt], -1, 255, thickness=2)
                    n_cnt_px = np.count_nonzero(cnt_boundary)
                    overlap = cv2.bitwise_and(cnt_boundary, edges_dilated)
                    n_overlap_px = np.count_nonzero(overlap)
                    edge_support = float(n_overlap_px) / float(max(1, n_cnt_px))
                    edge_boost = 0.4 if edge_support < 0.10 else (1.0 + 3.0 * edge_support)

                    # 3. Count interior dark circular spots (pips) strictly inside inner contour margin
                    crop_bgr = rgb_bgr[y:y+bh, x:x+bw]
                    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

                    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                    enhanced = clahe.apply(crop_gray)
                    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

                    ad_block = int(getattr(self.params, "adaptive_thresh_block_size", 15))
                    if ad_block % 2 == 0:
                        ad_block += 1
                    ad_c = int(getattr(self.params, "adaptive_thresh_c", 4))

                    dark = cv2.adaptiveThreshold(
                        blurred, 255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                        ad_block, ad_c
                    )
                    _, bw_cand = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    dark = cv2.bitwise_or(dark, cv2.bitwise_not(bw_cand))

                    # Exclude glare highlights from pip candidate count
                    hsv_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
                    glare = cv2.inRange(hsv_crop, np.array([0, 0, glare_cutoff], dtype=np.uint8),
                                       np.array([180, 50, 255], dtype=np.uint8))
                    dark = cv2.bitwise_and(dark, cv2.bitwise_not(cv2.dilate(glare, np.ones((5,5), np.uint8))))

                    p_cnts, _ = safe_find_contours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    inner_pips_radii = []
                    margin_x = int(bw * 0.08)
                    margin_y = int(bh * 0.08)
                    max_p_area = bw * bh * 0.12

                    for pc in p_cnts:
                        pa = cv2.contourArea(pc)
                        if 6 < pa < max_p_area:
                            (px, py), pr = cv2.minEnclosingCircle(pc)
                            # Inner boundary margin check
                            if margin_x <= px <= (bw - margin_x) and margin_y <= py <= (bh - margin_y):
                                # Must be inside candidate contour
                                if cv2.pointPolygonTest(cnt, (float(x + px), float(y + py)), False) >= 0:
                                    perim = cv2.arcLength(pc, True)
                                    if perim > 0:
                                        circ = (4 * math.pi * pa) / (perim ** 2)
                                        if circ >= min_circ:
                                            inner_pips_radii.append(pr)

                    n_inner = len(inner_pips_radii)

                    # Pip radius uniformity check (real die pips have consistent radii)
                    uniformity_penalty = 1.0
                    if n_inner >= 2:
                        std_r = float(np.std(inner_pips_radii))
                        mean_r = float(np.mean(inner_pips_radii))
                        if mean_r > 0 and (std_r / mean_r) > 0.35:
                            uniformity_penalty = 0.2

                    # Area decay penalty for unrealistically large sheets / table regions (> 12,000 px)
                    area_penalty = (12000.0 / float(area)) ** 2.0 if area > 12000 else 1.0

                    # 4. Calculate Composite Score
                    pip_boost = 1.0 + 1000.0 * min(n_inner, 6) * uniformity_penalty
                    quad_boost = 5.0 if is_quad else 1.0
                    score = area_penalty * float(area) * pip_boost * quad_boost * edge_boost

                    evaluated_candidates.append({
                        "contour": cnt,
                        "hull": hull,
                        "bbox": (x, y, bw, bh),
                        "score": score,
                        "area": area,
                        "n_pips": n_inner,
                        "is_quad": is_quad,
                        "edge_support": edge_support,
                    })

                    if score > best_score:
                        best_score = score
                        best_hull = hull
                        best_cnt = cnt
                        best_bbox = (x, y, bw, bh)

        # Draw all evaluated candidate contours for Panel 2 visualization
        for cand in evaluated_candidates:
            cnt = cand["contour"]
            hull = cand["hull"]
            x, y, bw, bh = cand["bbox"]
            is_best = (cand["bbox"] == best_bbox)

            color = (0, 255, 0) if is_best else (255, 165, 0)  # Green for best die, orange for others
            cv2.drawContours(step2_contours_img, [hull], -1, color, 3 if is_best else 2)
            cv2.rectangle(step2_contours_img, (x, y), (x + bw, y + bh), (0, 255, 255), 1)

            quad_str = "QUAD" if cand["is_quad"] else "NON-QUAD"
            lbl = f"Pips:{cand['n_pips']} | {quad_str} | Score:{int(cand['score'])}"
            cv2.putText(step2_contours_img, lbl, (x, max(15, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        if best_hull is not None:
            cv2.drawContours(step2_img, [best_hull], -1, (0, 255, 0), 3)
            if best_cnt is not None:
                cv2.drawContours(step2_img, [best_cnt], -1, (0, 0, 255), 1)

        self._log(f"Step 2: Best die candidate score={best_score:.1f}, bbox={best_bbox}.")

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

        clip_limit = getattr(self.params, "clahe_clip_limit", 3.0)
        min_circ = getattr(self.params, "pip_min_circularity", 0.45)
        glare_cutoff = getattr(self.params, "glare_v_thresh", 245)

        # 1. Apply CLAHE contrast enhancement for pip segmentation
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(crop_gray)
        blurred_gray = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)

        # 2. Adaptive & Otsu thresholding
        _, bw_mask = cv2.threshold(blurred_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw_mask = cv2.bitwise_and(bw_mask, hull_mask)

        ad_block = int(getattr(self.params, "adaptive_thresh_block_size", 15))
        if ad_block % 2 == 0:
            ad_block += 1
        ad_c = int(getattr(self.params, "adaptive_thresh_c", 4))

        dark_mask = cv2.adaptiveThreshold(
            blurred_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            ad_block, ad_c
        )
        dark_mask = cv2.bitwise_and(dark_mask, hull_mask)

        # Glare mask: exclude blown-out specular highlights from pip candidates
        # Avoid dilating glare into dark pips which shears away pip boundaries
        hsv_crop = cv2.cvtColor(step3_crop, cv2.COLOR_BGR2HSV)
        glare_mask = cv2.inRange(hsv_crop, np.array([0, 0, glare_cutoff], dtype=np.uint8),
                                 np.array([180, 50, 255], dtype=np.uint8))
        dark_mask = cv2.bitwise_and(dark_mask, cv2.bitwise_not(glare_mask))

        # Morphological opening to disconnect pips touching outer drawn border and eliminate stray specks
        morph_ksize = int(getattr(self.params, "morph_open_kernel_size", 3))
        if morph_ksize % 2 == 0:
            morph_ksize += 1
        morph_ksize = min(morph_ksize, 3)  # Gentle opening (3x3 max) to avoid dissolving small pips
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel_open)

        # Pip candidates filtered by circularity and area
        pip_cnts, _ = safe_find_contours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area_ratio = getattr(self.params, "max_pip_area_ratio", 0.08)
        max_radius_ratio = getattr(self.params, "max_pip_radius_ratio", 0.22)
        min_area_ratio = getattr(self.params, "min_pip_area_ratio", 0.001)

        min_pip_area = max(5.0, crop_w * crop_h * min_area_ratio)
        max_pip_area = crop_w * crop_h * max_area_ratio
        max_pip_radius = min(crop_w, crop_h) * max_radius_ratio
        valid_pips = []
        for pc in pip_cnts:
            pa = cv2.contourArea(pc)
            if min_pip_area <= pa <= max_pip_area:
                perim = cv2.arcLength(pc, True)
                if perim == 0:
                    continue
                circularity = (4 * math.pi * pa) / (perim ** 2)
                if circularity >= min_circ:
                    (px, py), pr = cv2.minEnclosingCircle(pc)
                    if pr <= max_pip_radius:
                        bx_p, by_p, bw_p, bh_p = cv2.boundingRect(pc)
                        asp = float(bw_p) / float(bh_p) if bh_p > 0 else 0
                        if 0.25 <= asp <= 4.0:
                            valid_pips.append({
                                "center": (int(round(px)), int(round(py))),
                                "center_float": (float(px), float(py)),
                                "radius": max(2, int(round(pr))),
                                "contour": pc,
                                "area": pa,
                            })

        # Deduplicate overlapping pip candidates (keep larger area)
        filtered_pips = []
        for p in sorted(valid_pips, key=lambda x: x["area"], reverse=True):
            cx, cy = p["center"]
            r = p["radius"]
            overlap = False
            for kept in filtered_pips:
                kcx, kcy = kept["center"]
                dist = math.hypot(cx - kcx, cy - kcy)
                if dist < max(r, kept["radius"]) * 1.2:
                    overlap = True
                    break
            if not overlap:
                filtered_pips.append(p)
        valid_pips = filtered_pips

        # Fill pip holes to preserve solid face regions
        bw_filled = bw_mask.copy()
        cnts_cc, hier = safe_find_contours(bw_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is not None:
            for i in range(len(cnts_cc)):
                if hier[0][i][3] >= 0:
                    if cv2.contourArea(cnts_cc[i]) < (crop_w * crop_h * 0.08):
                        cv2.drawContours(bw_filled, [cnts_cc[i]], -1, 255, -1)

        # Edge-guided face separation: detect internal die ridge edges to split adjacent faces
        c_low = int(getattr(self.params, "canny_low_thresh", 40))
        c_high = int(getattr(self.params, "canny_high_thresh", 130))
        crop_edges = cv2.Canny(blurred_gray, c_low, c_high)
        hull_erode = cv2.erode(hull_mask, np.ones((7, 7), np.uint8))
        internal_edges = cv2.bitwise_and(crop_edges, hull_erode)
        internal_edges_dilated = cv2.dilate(internal_edges, np.ones((3, 3), np.uint8))

        # Protect valid pips from being severed by internal edge lines
        if valid_pips:
            pip_prot = np.zeros_like(bw_filled)
            for p in valid_pips:
                cv2.circle(pip_prot, p["center"], int(round(p["radius"] * 1.5)), 255, -1)
            internal_edges_dilated = cv2.bitwise_and(internal_edges_dilated, cv2.bitwise_not(pip_prot))

        bw_separated = cv2.bitwise_and(bw_filled, cv2.bitwise_not(internal_edges_dilated))

        face_cnts, _ = safe_find_contours(bw_separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_candidates = sorted(
            [fc for fc in face_cnts if cv2.contourArea(fc) > crop_w * crop_h * 0.03],
            key=cv2.contourArea,
            reverse=True,
        )

        # Check if bw_separated over-split a face containing valid pips
        if valid_pips:
            pips_enclosed = 0
            if all_candidates:
                pips_enclosed = sum(
                    1 for p in valid_pips
                    if cv2.pointPolygonTest(all_candidates[0], p.get("center_float", p["center"]), False) >= -2.0
                )
            if pips_enclosed < len(valid_pips):
                # Fallback to non-separated mask if internal edges over-split
                face_cnts, _ = safe_find_contours(bw_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                all_candidates = sorted(
                    [fc for fc in face_cnts if cv2.contourArea(fc) > crop_w * crop_h * 0.03],
                    key=cv2.contourArea,
                    reverse=True,
                )
        elif not all_candidates:
            # Fallback to non-separated mask if internal edges over-split
            face_cnts, _ = safe_find_contours(bw_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
                if area >= max(crop_w * crop_h * 0.04, 0.30 * max_area):
                    candidate_faces.append(fc)
                if len(candidate_faces) >= 2:  # Cap detection to up to 2 faces
                    break

        step4_vis = step3_crop.copy()
        face_colors = [(255, 0, 0), (0, 255, 0)]  # Up to 2 faces (Blue for face 1, Green for face 2)
        visible_faces: list[dict] = []

        for face_idx, fc in enumerate(candidate_faces, start=1):
            poly = self._fit_quadrilateral(fc)
            if poly is None:
                continue
            # Check pip counts strictly within the contour/polygon of this face
            face_pips = []
            for p in valid_pips:
                pt = p.get("center_float", (float(p["center"][0]), float(p["center"][1])))
                in_poly = cv2.pointPolygonTest(poly, pt, True) >= -2.5
                in_cnt = cv2.pointPolygonTest(fc, pt, True) >= -2.5
                if in_poly or in_cnt:
                    face_pips.append(p)

            # Outlier rejection for false positive noise specks
            outlier_ratio = float(getattr(self.params, "pip_area_outlier_ratio", 0.30))
            if len(face_pips) >= 2:
                areas = [p["area"] for p in face_pips]
                med_area = float(np.median(areas))
                radii = [p["radius"] for p in face_pips]
                med_radius = float(np.median(radii))
                clean_face_pips = [
                    p for p in face_pips
                    if (p["area"] >= med_area * outlier_ratio) and (p["radius"] >= med_radius * 0.45)
                ]
                if clean_face_pips:
                    face_pips = clean_face_pips

            # Anchor quad center to pip constellation centroid for jitter-free alignment
            if face_pips:
                pip_cx = float(np.mean([p.get("center_float", p["center"])[0] for p in face_pips]))
                pip_cy = float(np.mean([p.get("center_float", p["center"])[1] for p in face_pips]))
                M = cv2.moments(poly)
                if M["m00"] > 0:
                    quad_cx = M["m10"] / M["m00"]
                    quad_cy = M["m01"] / M["m00"]
                    dx = pip_cx - quad_cx
                    dy = pip_cy - quad_cy
                    if math.hypot(dx, dy) < 8.0:
                        poly = np.round(poly.astype(np.float32) + np.array([[[dx, dy]]], dtype=np.float32)).astype(np.int32)

            n_p = len(face_pips)
            is_trusted = True
            aligned_pips = True

            # If 3 pips detected, verify linearity (diagonal alignment)
            if n_p == 3:
                aligned_pips = self._check_3pips_linearity(face_pips)
                if not aligned_pips:
                    is_trusted = False
                    self._log(f"Face {face_idx}: Detected 3 pips but they are NOT roughly aligned! Marking untrusted.")

            visible_faces.append({
                "face_index": face_idx,
                "contour": fc,
                "polygon": poly,
                "num_pips": n_p,
                "pips": face_pips,
                "is_trusted": is_trusted,
                "aligned_pips": aligned_pips,
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
                "is_trusted": True,
                "aligned_pips": True,
            })

        visible_faces = visible_faces[:2]  # Cap detection to up to 2 faces

        # Mark top face (lowest Y centroid in image = topmost face)
        def _cy(f):
            M = cv2.moments(f["polygon"])
            return M["m01"] / M["m00"] if M["m00"] > 0 else np.mean(f["polygon"][:, 0, 1])

        top_f = min(visible_faces, key=_cy)
        for f in visible_faces:
            f["is_top_face"] = f is top_f

        # Draw pips belonging to visible faces
        for f in visible_faces:
            for p in f["pips"]:
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
                "step2_edges": cv2.cvtColor(edges_full, cv2.COLOR_GRAY2BGR),
                "step2_whitest_convex_hull": step2_img,
                "step2_contours": step2_contours_img,
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
        """Fit an oriented orthogonal quadrilateral (rectangle) to a face contour.

        Uses cv2.minAreaRect to guarantee true 90-degree corners and parallel edges,
        avoiding trapezoidal or diamond artifacts caused by approxPolyDP on chamfers.
        """
        if fc is None or len(fc) < 3:
            return None
        hull = cv2.convexHull(fc)
        if len(hull) < 3:
            return None
        rect = cv2.minAreaRect(hull)
        (cx, cy), (w, h), angle = rect
        if w < 4 or h < 4:
            return None

        box = cv2.boxPoints(rect)  # 4 x 2 float32
        # Order vertices consistently clockwise starting from top-left:
        pts = box[np.argsort(box[:, 1])]
        top = pts[:2]
        top = top[np.argsort(top[:, 0])]  # TL, TR
        bot = pts[2:]
        bot = bot[np.argsort(bot[:, 0])]  # BL, BR

        ordered = np.array([top[0], top[1], bot[1], bot[0]], dtype=np.int32).reshape(-1, 1, 2)
        return ordered

    @staticmethod
    def _check_3pips_linearity(pips: list[dict], threshold_ratio: float = 0.22) -> bool:
        """Return True if exactly 3 pips are roughly collinear (aligned in a diagonal line).

        Parameters
        ----------
        pips : list of 3 pip dicts with key 'center'
        threshold_ratio : float
            Max allowed perpendicular distance ratio (h / d_max). Standard diagonal 3-pips
            have ratio < 0.15. Non-aligned pips (e.g. triangle corners) have ratio > 0.35.
        """
        if len(pips) != 3:
            return True

        pts = [np.array(p["center"], dtype=np.float64) for p in pips]

        max_d = -1.0
        best_pair = (0, 2)
        mid_idx = 1

        for i in range(3):
            for j in range(i + 1, 3):
                d = float(np.linalg.norm(pts[i] - pts[j]))
                if d > max_d:
                    max_d = d
                    best_pair = (i, j)
                    mid_idx = 3 - (i + j)

        if max_d < 1e-5:
            return False

        A = pts[best_pair[0]]
        C = pts[best_pair[1]]
        B = pts[mid_idx]

        h = abs((C[1] - A[1]) * B[0] - (C[0] - A[0]) * B[1] + C[0] * A[1] - C[1] * A[0]) / max_d
        ratio = h / max_d
        return ratio <= threshold_ratio

    def _make_logger(self):
        tag = "[RGBDieDetector]"
        if self.params.debug:
            return lambda msg: print(f"{tag} {msg}")
        return lambda msg: None
