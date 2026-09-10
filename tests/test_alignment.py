#!/usr/bin/env python3
"""Tests for AlignmentAndProjection — rotation matrix and top-down crop."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drims_die_detection.die_detector_params import DieDetectorParams
from drims_die_detection.alignment_projection import AlignmentAndProjection


class TestRotationToZ(unittest.TestCase):

    def setUp(self):
        self.align = AlignmentAndProjection(DieDetectorParams())

    def test_identity_when_already_z(self):
        normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        R = self.align.get_rotation_to_z(normal)
        rotated = np.dot(R, normal)
        self.assertAlmostEqual(abs(rotated[2]), 1.0, places=3)

    def test_45_degree_tilt(self):
        normal = np.array([0.0, 0.7071, -0.7071], dtype=np.float32)
        R = self.align.get_rotation_to_z(normal)
        rotated = np.dot(R, normal)
        self.assertAlmostEqual(abs(rotated[2]), 1.0, places=2)

    def test_arbitrary_normal(self):
        for _ in range(10):
            n = np.random.randn(3).astype(np.float32)
            n /= np.linalg.norm(n)
            R = self.align.get_rotation_to_z(n)
            # R must be a proper rotation matrix
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=3)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-4)

    def test_output_is_3x3(self):
        normal = np.array([0.1, 0.2, -0.9], dtype=np.float32)
        normal /= np.linalg.norm(normal)
        R = self.align.get_rotation_to_z(normal)
        self.assertEqual(R.shape, (3, 3))

    def test_static_alias(self):
        normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        R = AlignmentAndProjection.get_rotation_to_z_static(normal)
        self.assertEqual(R.shape, (3, 3))


class TestExtractTopDownRGB(unittest.TestCase):

    def setUp(self):
        self.align = AlignmentAndProjection(DieDetectorParams())

    def _make_dummy_data(self, n=100):
        """Create a small set of 3D points with matching pixel coords."""
        np.random.seed(42)
        die_points = np.random.randn(n, 3).astype(np.float32) * 0.02
        die_points[:, 2] += 1.0          # 1 m away
        pixel_coords = np.random.randint(50, 150, size=(n, 2)).astype(np.int32)
        rgb = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        R_plane = np.eye(3, dtype=np.float32)
        die_indices = np.arange(n)
        return rgb, die_points, pixel_coords, die_indices, R_plane

    def test_output_shapes(self):
        rgb, pts, coords, idx, R = self._make_dummy_data()
        top_down, raw_crop, bbox, yaw = self.align.extract_top_down_rgb(
            rgb, pts, coords, idx, R, crop_size=128
        )
        self.assertEqual(top_down.shape, (128, 128, 3))
        self.assertEqual(len(bbox), 4)   # (x_min, y_min, x_max, y_max)
        self.assertIsInstance(yaw, float)

    def test_top_down_dtype(self):
        rgb, pts, coords, idx, R = self._make_dummy_data()
        top_down, _, _, _ = self.align.extract_top_down_rgb(rgb, pts, coords, idx, R)
        self.assertEqual(top_down.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
