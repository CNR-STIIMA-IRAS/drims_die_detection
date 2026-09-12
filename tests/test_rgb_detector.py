#!/usr/bin/env python3
"""Tests for RGBDieDetector — 2D five-step detection on synthetic images."""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drims_die_detection.die_detector_params import DieDetectorParams
from drims_die_detection.rgb_die_detector import RGBDieDetector


def _make_synthetic_die(w=320, h=240, die_color=(220, 220, 220), n_pips=3):
    """Create a simple BGR image with a white rectangle (die) and n_pips dark circles."""
    img = np.full((h, w, 3), 80, dtype=np.uint8)   # grey background
    # Draw die face
    x1, y1, x2, y2 = w // 4, h // 4, 3 * w // 4, 3 * h // 4
    cv2.rectangle(img, (x1, y1), (x2, y2), die_color, -1)
    # Draw pips
    cx_die = (x1 + x2) // 2
    cy_die = (y1 + y2) // 2
    pip_r = 8
    offsets = [(-20, -20), (0, 0), (20, 20)][:n_pips]
    for ox, oy in offsets:
        cv2.circle(img, (cx_die + ox, cy_die + oy), pip_r, (20, 20, 20), -1)
    return img


class TestRGBDieDetectorOutput(unittest.TestCase):

    def setUp(self):
        self.params = DieDetectorParams(debug=False, num_color_clusters=4)
        self.det = RGBDieDetector(self.params)

    def test_result_keys_present(self):
        img = _make_synthetic_die()
        res = self.det.detect(img)
        required = {"bbox", "contour", "num_visible_faces", "faces",
                    "total_pips", "num_pips", "debug_steps"}
        for key in required:
            self.assertIn(key, res, msg=f"Missing key: {key}")

    def test_bbox_within_image(self):
        img = _make_synthetic_die()
        h, w = img.shape[:2]
        res = self.det.detect(img)
        x, y, bw, bh = res["bbox"]
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + bw, w)
        self.assertLessEqual(y + bh, h)

    def test_debug_steps_images_not_none(self):
        img = _make_synthetic_die()
        res = self.det.detect(img)
        ds = res["debug_steps"]
        for step_name, step_img in ds.items():
            self.assertIsNotNone(step_img, msg=f"{step_name} is None")
            self.assertEqual(step_img.ndim, 3, msg=f"{step_name} not 3-channel")

    def test_die_color_override(self):
        """detect() should accept a die_color override without raising."""
        img = _make_synthetic_die()
        res = self.det.detect(img, die_color="white")
        self.assertIn("faces", res)

    def test_static_alias_works(self):
        img = _make_synthetic_die()
        res = RGBDieDetector.detect_die_5step(img)
        self.assertIn("total_pips", res)

    def test_all_die_colors_dont_raise(self):
        img = _make_synthetic_die()
        for color in ["white", "black", "red", "blue", "green", "gray"]:
            try:
                self.det.detect(img, die_color=color)
            except Exception as exc:
                self.fail(f"detect() raised {exc!r} for die_color='{color}'")

    def test_faces_list_not_empty(self):
        img = _make_synthetic_die(w=400, h=300)
        res = self.det.detect(img)
        # At least one face should be returned (fallback poly if nothing detected)
        self.assertGreaterEqual(len(res["faces"]), 1)

    def test_top_face_flagged(self):
        img = _make_synthetic_die(w=400, h=300)
        res = self.det.detect(img)
        top_faces = [f for f in res["faces"] if f.get("is_top_face")]
        self.assertEqual(len(top_faces), 1)

    def test_faces_capped_to_max_two(self):
        img = _make_synthetic_die(w=400, h=300)
        res = self.det.detect(img)
        self.assertLessEqual(len(res["faces"]), 2)

    def test_3pips_linearity_check(self):
        # 1. Collinear 3 pips (diagonal)
        collinear_pips = [
            {"center": (10, 10)},
            {"center": (20, 20)},
            {"center": (30, 30)},
        ]
        self.assertTrue(RGBDieDetector._check_3pips_linearity(collinear_pips))

        # 2. Non-aligned 3 pips (triangle formation)
        triangle_pips = [
            {"center": (10, 10)},
            {"center": (10, 30)},
            {"center": (30, 10)},
        ]
        self.assertFalse(RGBDieDetector._check_3pips_linearity(triangle_pips))


if __name__ == "__main__":
    unittest.main()
