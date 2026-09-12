"""
PipDetector
===========
Counts pips on a die face from a cropped, approximately top-down BGR image.

Two strategies are tried in sequence:
1. Adaptive thresholding + CLAHE + contour circularity filtering.
2. SimpleBlobDetector fallback if strategy 1 returns 0 or >6 pips.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .die_detector_params import DieDetectorParams


class PipDetector:
    """Detects and counts pips on a die face crop.

    Parameters
    ----------
    params : DieDetectorParams
        Uses ``debug``.
    """

    def __init__(self, params: DieDetectorParams | None = None) -> None:
        self.params = params or DieDetectorParams()
        self._log = self._make_logger()

    # ──────────────────────────────────────────────────────────────────────
    def detect_pips(
        self,
        top_down_rgb: np.ndarray,
        raw_crop: np.ndarray | None = None,
    ):
        """Count pips in *top_down_rgb*.

        Parameters
        ----------
        top_down_rgb : BGR image of the die face
        raw_crop : optional un-resized crop (tried as a second candidate)

        Returns
        -------
        pip_count : int
        annotated : BGR image with pip circles drawn
        keypoints : list of cv2.KeyPoint
        """
        images_to_try = [top_down_rgb]
        if raw_crop is not None and raw_crop.size > 0:
            resized_raw = cv2.resize(raw_crop, (top_down_rgb.shape[1], top_down_rgb.shape[0]))
            images_to_try.append(resized_raw)

        best_count = 0
        best_annotated = top_down_rgb.copy()
        best_keypoints: list = []

        for img in images_to_try:
            if img is None or img.size == 0:
                continue
            count, annotated, kps = self._detect_on_image(img)
            self._log(f"Pip detection attempt: {count} pips found.")
            if 1 <= count <= 6:
                return count, annotated, kps
            if count > best_count:
                best_count = count
                best_annotated = annotated
                best_keypoints = kps

        return best_count, best_annotated, best_keypoints

    # Backward-compatible static alias
    @classmethod
    def detect_pips_static(
        cls,
        top_down_rgb: np.ndarray,
        raw_crop: np.ndarray | None = None,
    ):
        return cls().detect_pips(top_down_rgb, raw_crop)

    # ──────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────

    def _detect_on_image(self, img: np.ndarray):
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        clip_limit = getattr(self.params, "clahe_clip_limit", 3.0)
        min_circ = getattr(self.params, "pip_min_circularity", 0.45)
        glare_cutoff = getattr(self.params, "glare_v_thresh", 245)

        # Glare mask: exclude blown-out specular highlights from pip detection
        glare_mask = (gray >= glare_cutoff).astype(np.uint8) * 255
        glare_mask_dilated = cv2.dilate(glare_mask, np.ones((5, 5), np.uint8))

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            15, 4,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # Subtract glare highlights & mask out border region
        cleaned = cv2.bitwise_and(cleaned, cv2.bitwise_not(glare_mask_dilated))

        margin = int(w * 0.08)
        border_mask = np.zeros((h, w), dtype=np.uint8)
        border_mask[margin:h - margin, margin:w - margin] = 255
        cleaned = cv2.bitwise_and(cleaned, border_mask)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        die_area = w * h
        min_pip_area = die_area * 0.003
        max_pip_area = die_area * 0.08

        valid_cnts = []
        keypoints = []
        annotated = img.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_pip_area <= area <= max_pip_area:
                perim = cv2.arcLength(cnt, True)
                if perim == 0:
                    continue
                circularity = 4 * math.pi * area / (perim ** 2)
                if circularity >= min_circ:
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    center = (int(x), int(y))
                    radius = int(radius)
                    valid_cnts.append(cnt)
                    keypoints.append(cv2.KeyPoint(float(x), float(y), float(radius * 2)))
                    cv2.circle(annotated, center, radius, (0, 0, 255), 2)
                    cv2.circle(annotated, center, 2, (0, 255, 0), -1)

        pip_count = len(valid_cnts)

        # Blob detector fallback
        if pip_count == 0 or pip_count > 6:
            params = cv2.SimpleBlobDetector_Params()
            params.filterByArea = True
            params.minArea = min_pip_area
            params.maxArea = max_pip_area
            params.filterByCircularity = True
            params.minCircularity = min_circ
            params.filterByConvexity = True
            params.minConvexity = 0.5
            params.filterByInertia = True
            params.minInertiaRatio = 0.4

            detector = cv2.SimpleBlobDetector_create(params)
            blob_kps = detector.detect(cleaned)
            if 1 <= len(blob_kps) <= 6:
                pip_count = len(blob_kps)
                annotated = cv2.drawKeypoints(
                    img, blob_kps, np.array([]), (0, 0, 255),
                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                )

        return pip_count, annotated, keypoints

    def _make_logger(self):
        tag = "[PipDetector]"
        if self.params.debug:
            return lambda msg: print(f"{tag} {msg}")
        return lambda msg: None
