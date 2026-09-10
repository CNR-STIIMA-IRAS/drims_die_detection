#!/usr/bin/env python3
"""
Unit tests for Angled Camera Die Detection Pipeline.
"""

import sys
import os
import unittest
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from detect_die import (
    PointCloudProcessor,
    AlignmentAndProjection,
    PipDetector,
    PoseEstimator
)


class TestDieDetector3d(unittest.TestCase):

    def test_point_cloud_creation(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.ones((100, 100), dtype=np.float32) * 1.0  # 1 meter depth
        processor = PointCloudProcessor(fx=500, fy=500)
        pts, colors, coords, (cx, cy) = processor.create_point_cloud(rgb, depth)

        self.assertEqual(len(pts), 10000)
        self.assertEqual(pts.shape[1], 3)
        self.assertAlmostEqual(pts[5000, 2], 1.0, places=3)

    def test_ransac_plane_fitting(self):
        # Create points lying on plane Z = 0.5 * Y + 1.0
        y_grid, x_grid = np.indices((50, 50), dtype=np.float32)
        z_grid = 0.5 * (y_grid / 50.0) + 1.0
        x_3d = (x_grid - 25) * z_grid / 500.0
        y_3d = (y_grid - 25) * z_grid / 500.0

        points = np.stack((x_3d.reshape(-1), y_3d.reshape(-1), z_grid.reshape(-1)), axis=-1)

        processor = PointCloudProcessor(fx=500, fy=500)
        plane_model, inliers, outliers, normal = processor.fit_plane_ransac(points, distance_threshold=0.02)

        self.assertGreater(len(inliers), 2000)
        self.assertEqual(len(normal), 3)
        # Normal should be unit length
        self.assertAlmostEqual(np.linalg.norm(normal), 1.0, places=3)

    def test_alignment_rotation_matrix(self):
        normal = np.array([0.0, 0.7071, -0.7071], dtype=np.float32)  # 45 degree tilt
        R_plane = AlignmentAndProjection.get_rotation_to_z(normal)

        # Rotated normal should align with Z axis [0, 0, 1]
        rotated_normal = np.dot(R_plane, normal)
        self.assertAlmostEqual(abs(rotated_normal[2]), 1.0, places=2)

    def test_pip_detection_synthetic(self):
        # Synthetic top-down image of die with 5 pips (corners + center)
        img = np.ones((200, 200, 3), dtype=np.uint8) * 220  # White face

        pip_centers = [(50, 50), (150, 50), (100, 100), (50, 150), (150, 150)]
        for center in pip_centers:
            cv2.circle(img, center, 12, (20, 20, 20), -1)  # Dark pips

        pip_count, annotated, _ = PipDetector.detect_pips(img)
        self.assertEqual(pip_count, 5)

    def test_pose_estimation(self):
        pts = np.array([[0.1, 0.2, 1.0], [0.12, 0.22, 1.01], [0.08, 0.18, 0.99]], dtype=np.float32)
        normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        R_plane = np.identity(3, dtype=np.float32)

        centroid, quat, R_full, axes = PoseEstimator.compute_pose(pts, normal, R_plane)

        self.assertAlmostEqual(centroid[0], 0.1, places=2)
        self.assertAlmostEqual(centroid[1], 0.2, places=2)
        self.assertAlmostEqual(centroid[2], 1.0, places=2)
        self.assertEqual(len(quat), 4)
        # Quaternion norm should be 1.0
        self.assertAlmostEqual(np.linalg.norm(quat), 1.0, places=3)


if __name__ == '__main__':
    unittest.main()
