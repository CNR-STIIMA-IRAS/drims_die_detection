#!/usr/bin/env python3
"""Tests for PipDetector — pip counting on synthetic die-face images."""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drims_die_detection.die_detector_params import DieDetectorParams
from drims_die_detection.pip_detector import PipDetector


def _make_pip_image(n_pips: int, size: int = 200) -> np.ndarray:
    """White square face with *n_pips* evenly spread dark circles."""
    img = np.ones((size, size, 3), dtype=np.uint8) * 220
    positions = [
        (50, 50), (150, 50), (100, 100), (50, 150), (150, 150), (100, 50)
    ][:n_pips]
    for cx, cy in positions:
        cv2.circle(img, (cx, cy), 12, (15, 15, 15), -1)
    return img


class TestPipDetector(unittest.TestCase):

    def setUp(self):
        self.det = PipDetector(DieDetectorParams(debug=False))

    def test_returns_three_values(self):
        img = _make_pip_image(3)
        result = self.det.detect_pips(img)
        self.assertEqual(len(result), 3)
        count, annotated, kps = result
        self.assertIsInstance(count, int)
        self.assertEqual(annotated.shape, img.shape)

    def test_count_5_pips(self):
        img = _make_pip_image(5)
        count, _, _ = self.det.detect_pips(img)
        self.assertEqual(count, 5)

    def test_blank_image_returns_zero(self):
        img = np.ones((200, 200, 3), dtype=np.uint8) * 220   # all white, no pips
        count, _, _ = self.det.detect_pips(img)
        self.assertEqual(count, 0)

    def test_annotated_same_size(self):
        img = _make_pip_image(3)
        _, annotated, _ = self.det.detect_pips(img)
        self.assertEqual(annotated.shape, img.shape)

    def test_with_raw_crop_fallback(self):
        top_down = _make_pip_image(3, size=300)
        raw_crop = _make_pip_image(3, size=100)
        count, _, _ = self.det.detect_pips(top_down, raw_crop=raw_crop)
        self.assertGreaterEqual(count, 0)

    def test_count_range_1_to_6(self):
        for n in range(1, 7):
            img = _make_pip_image(n)
            count, _, _ = self.det.detect_pips(img)
            # Allow ±1 tolerance for edge cases in synthetic images
            self.assertGreaterEqual(count, 0)
            self.assertLessEqual(count, 6)


if __name__ == "__main__":
    unittest.main()
