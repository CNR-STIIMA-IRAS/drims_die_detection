#!/usr/bin/env python3
"""Tests for PoseEstimator — 6-DOF pose computation."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drims_die_detection.die_detector_params import DieDetectorParams
from drims_die_detection.pose_estimator import PoseEstimator


class TestPoseEstimatorPCA(unittest.TestCase):
    """Tests using the PCA fallback path (no polygon/camera params)."""

    def setUp(self):
        self.est = PoseEstimator(DieDetectorParams(debug=False))

    def _simple_die_points(self):
        return np.array(
            [[0.10, 0.20, 1.00],
             [0.12, 0.22, 1.01],
             [0.08, 0.18, 0.99],
             [0.11, 0.19, 1.00]],
            dtype=np.float32,
        )

    def test_returns_four_values(self):
        pts = self._simple_die_points()
        normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        R_plane = np.eye(3, dtype=np.float32)
        result = self.est.compute_pose(pts, normal, R_plane)
        self.assertEqual(len(result), 4)

    def test_centroid_approximately_correct(self):
        pts = self._simple_die_points()
        normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        R_plane = np.eye(3, dtype=np.float32)
        centroid, _, _, _ = self.est.compute_pose(pts, normal, R_plane)
        self.assertAlmostEqual(centroid[0], np.mean(pts[:, 0]), places=2)
        self.assertAlmostEqual(centroid[1], np.mean(pts[:, 1]), places=2)

    def test_quaternion_unit_norm(self):
        pts = self._simple_die_points()
        normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        R_plane = np.eye(3, dtype=np.float32)
        _, quat, _, _ = self.est.compute_pose(pts, normal, R_plane)
        self.assertEqual(len(quat), 4)
        self.assertAlmostEqual(np.linalg.norm(quat), 1.0, places=3)

    def test_rotation_matrix_orthonormal(self):
        pts = self._simple_die_points()
        normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        R_plane = np.eye(3, dtype=np.float32)
        _, _, R_die, _ = self.est.compute_pose(pts, normal, R_plane)
        self.assertEqual(R_die.shape, (3, 3))
        np.testing.assert_allclose(R_die @ R_die.T, np.eye(3), atol=1e-4)
        self.assertAlmostEqual(np.linalg.det(R_die), 1.0, places=3)

    def test_axes_unit_vectors(self):
        pts = self._simple_die_points()
        normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        R_plane = np.eye(3, dtype=np.float32)
        _, _, _, (x_ax, y_ax, z_ax) = self.est.compute_pose(pts, normal, R_plane)
        for ax in (x_ax, y_ax, z_ax):
            self.assertAlmostEqual(np.linalg.norm(ax), 1.0, places=3)


class TestPoseEstimatorBackprojection(unittest.TestCase):
    """Tests the ray-backprojection path with full camera params."""

    def setUp(self):
        self.est = PoseEstimator(DieDetectorParams(debug=False, die_size_m=0.05))

    def test_backprojection_centroid_is_finite(self):
        pts = np.random.randn(50, 3).astype(np.float32) * 0.02 + np.array([0.0, 0.0, 1.0])
        normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        R_plane = np.eye(3, dtype=np.float32)
        poly = np.array([[[100, 100]], [[200, 100]], [[200, 200]], [[100, 200]]], dtype=np.int32)
        centroid_2d = (150.0, 150.0)
        plane_D = -1.0
        cam = (500.0, 500.0, 320.0, 240.0)

        centroid, quat, _, _ = self.est.compute_pose(
            pts, normal, R_plane,
            top_face_polygon=poly,
            top_face_centroid_2d=centroid_2d,
            plane_D=plane_D,
            camera_params=cam,
        )
        self.assertTrue(np.all(np.isfinite(centroid)), msg="Centroid contains non-finite values")
        self.assertAlmostEqual(np.linalg.norm(quat), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
