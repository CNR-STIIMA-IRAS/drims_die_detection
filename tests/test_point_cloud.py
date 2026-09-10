#!/usr/bin/env python3
"""Tests for PointCloudProcessor — point cloud creation and RANSAC plane fitting."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drims_die_detection.die_detector_params import DieDetectorParams
from drims_die_detection.point_cloud_processor import PointCloudProcessor


class TestPointCloudCreation(unittest.TestCase):

    def _make_processor(self, fx=500.0, fy=500.0):
        p = DieDetectorParams(fx=fx, fy=fy)
        return PointCloudProcessor(p)

    def test_basic_shape(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.ones((100, 100), dtype=np.float32)   # 1 m everywhere
        proc = self._make_processor()
        pts, cols, coords, (cx, cy) = proc.create_point_cloud(rgb, depth)

        self.assertEqual(pts.shape[1], 3)
        self.assertEqual(cols.shape[1], 3)
        self.assertEqual(coords.shape[1], 2)
        # All 10000 pixels should be valid at depth=1 m
        self.assertEqual(len(pts), 10_000)

    def test_depth_values_correct(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.ones((100, 100), dtype=np.float32) * 1.5
        proc = self._make_processor()
        pts, _, _, _ = proc.create_point_cloud(rgb, depth)
        # Z values should equal depth
        np.testing.assert_allclose(pts[:, 2], 1.5, atol=1e-3)

    def test_invalid_depth_filtered(self):
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        depth = np.zeros((10, 10), dtype=np.float32)    # all invalid
        proc = self._make_processor()
        pts, _, _, _ = proc.create_point_cloud(rgb, depth)
        self.assertEqual(len(pts), 0)

    def test_principal_point_default(self):
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        depth = np.ones((100, 200), dtype=np.float32)
        proc = self._make_processor()
        _, _, _, (cx, cy) = proc.create_point_cloud(rgb, depth)
        self.assertAlmostEqual(cx, 100.0)
        self.assertAlmostEqual(cy, 50.0)

    def test_color_normalisation(self):
        rgb = np.ones((10, 10, 3), dtype=np.uint8) * 255
        depth = np.ones((10, 10), dtype=np.float32)
        proc = self._make_processor()
        _, cols, _, _ = proc.create_point_cloud(rgb, depth)
        np.testing.assert_allclose(cols, 1.0, atol=1e-3)


class TestRANSACPlaneFitting(unittest.TestCase):

    def _make_processor(self):
        p = DieDetectorParams(fx=500.0, fy=500.0)
        return PointCloudProcessor(p)

    def test_flat_plane_inliers(self):
        """All points on a flat horizontal plane should be inliers."""
        xs = np.linspace(-1, 1, 50)
        ys = np.linspace(-1, 1, 50)
        xg, yg = np.meshgrid(xs, ys)
        points = np.stack([xg.ravel(), yg.ravel(), np.ones(2500)], axis=1).astype(np.float32)

        proc = self._make_processor()
        _, inliers, outliers, normal = proc.fit_plane_ransac(points, distance_threshold=0.02)

        self.assertGreater(len(inliers), 2000)
        self.assertEqual(len(normal), 3)
        self.assertAlmostEqual(np.linalg.norm(normal), 1.0, places=3)

    def test_angled_plane(self):
        """RANSAC should recover an angled plane with most points as inliers."""
        y_grid, x_grid = np.indices((50, 50), dtype=np.float32)
        z_grid = 0.5 * (y_grid / 50.0) + 1.0
        x_3d = (x_grid - 25) * z_grid / 500.0
        y_3d = (y_grid - 25) * z_grid / 500.0
        points = np.stack([x_3d.ravel(), y_3d.ravel(), z_grid.ravel()], axis=1)

        proc = self._make_processor()
        _, inliers, _, normal = proc.fit_plane_ransac(points, distance_threshold=0.02)

        self.assertGreater(len(inliers), 2000)
        self.assertAlmostEqual(np.linalg.norm(normal), 1.0, places=3)

    def test_normal_points_toward_camera(self):
        """Normal Z component should be negative (pointing toward camera)."""
        points = np.random.randn(500, 3).astype(np.float32)
        points[:, 2] += 1.0                 # push away from origin

        proc = self._make_processor()
        _, _, _, normal = proc.fit_plane_ransac(points)

        # Normal must point towards camera (z < 0 in camera frame convention)
        self.assertLessEqual(normal[2], 0.0)

    def test_returns_four_values(self):
        points = np.random.randn(100, 3).astype(np.float32) + np.array([0, 0, 1])
        proc = self._make_processor()
        result = proc.fit_plane_ransac(points)
        self.assertEqual(len(result), 4)
        plane_model, inliers, outliers, normal = result
        self.assertEqual(len(plane_model), 4)
        self.assertEqual(len(normal), 3)


if __name__ == "__main__":
    unittest.main()
