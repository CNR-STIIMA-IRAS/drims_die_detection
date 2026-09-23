#!/usr/bin/env python3
"""Tests for PoseStabilizer and temporal pose continuity."""

import math
import os
import sys
import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R_sci

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drims_die_detection.die_detector_params import DieDetectorParams
from drims_die_detection.pose_stabilizer import PoseStabilizer
from drims_die_detection.pose_estimator import PoseEstimator


class TestPoseStabilizer(unittest.TestCase):
    """Unit tests for PoseStabilizer."""

    def setUp(self):
        self.params = DieDetectorParams(
            enable_pose_filter=True,
            pose_filter_alpha_trans=0.25,
            pose_filter_alpha_rot=0.20,
            pose_filter_trans_jump_m=0.08,
            pose_filter_rot_jump_deg=45.0,
            pose_filter_deadband_trans_m=0.0015,
            pose_filter_deadband_rot_deg=0.5,
        )
        self.stabilizer = PoseStabilizer(self.params)

    def test_filter_plane_initial_and_smooth(self):
        n0 = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        d0 = -0.85
        n_out, d_out = self.stabilizer.filter_plane(n0, d0)
        np.testing.assert_allclose(n_out, n0, atol=1e-5)
        self.assertAlmostEqual(d_out, d0, places=5)

        # Small noisy measurement
        n1 = np.array([0.01, 0.0, -0.9999], dtype=np.float32)
        n1 /= np.linalg.norm(n1)
        d1 = -0.852
        n_out2, d_out2 = self.stabilizer.filter_plane(n1, d1)
        # Should be smoothed (between n0 and n1)
        self.assertTrue(0.0 < n_out2[0] < n1[0])
        self.assertTrue(d0 > d_out2 > d1)

    def test_filter_plane_sign_inversion_handled(self):
        n0 = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        d0 = -0.85
        self.stabilizer.filter_plane(n0, d0)

        # Inverted normal (e.g. RANSAC flipped sign: [0, 0, 1] with d=0.85)
        n_inv = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        d_inv = 0.85
        n_out, d_out = self.stabilizer.filter_plane(n_inv, d_inv)
        # Should maintain negative z normal and negative d
        self.assertAlmostEqual(n_out[2], -1.0, places=3)
        self.assertAlmostEqual(d_out, -0.85, places=3)

    def test_filter_plane_large_jump_resets(self):
        n0 = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        d0 = -0.85
        self.stabilizer.filter_plane(n0, d0)

        # Tilted by 45 degrees (> 25 deg jump threshold)
        n_jump = np.array([0.7071, 0.0, -0.7071], dtype=np.float32)
        d_jump = -0.85
        n_out, _ = self.stabilizer.filter_plane(n_jump, d_jump)
        np.testing.assert_allclose(n_out, n_jump, atol=1e-4)

    def test_filter_pose_deadband_translation(self):
        c0 = np.array([0.1, 0.2, 0.8], dtype=np.float32)
        q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        c_f0, q_f0, _, _ = self.stabilizer.filter_pose(c0, q0)

        # Micro-jitter < 1.5 mm (e.g. 0.5 mm)
        c_noisy = c0 + np.array([0.0003, 0.0004, 0.0], dtype=np.float32)
        c_f1, _, _, _ = self.stabilizer.filter_pose(c_noisy, q0)
        # Should remain exactly locked at c0
        np.testing.assert_allclose(c_f1, c0, atol=1e-6)

    def test_filter_pose_ema_translation(self):
        c0 = np.array([0.1, 0.2, 0.8], dtype=np.float32)
        q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.stabilizer.filter_pose(c0, q0)

        # Moderate motion: 10 mm (above 1.5 mm deadband, below 80 mm jump)
        c1 = c0 + np.array([0.010, 0.0, 0.0], dtype=np.float32)
        c_f, _, _, _ = self.stabilizer.filter_pose(c1, q0)
        expected_x = (1.0 - 0.25) * c0[0] + 0.25 * c1[0]
        self.assertAlmostEqual(c_f[0], expected_x, places=5)

    def test_filter_pose_jump_gate_translation(self):
        c0 = np.array([0.1, 0.2, 0.8], dtype=np.float32)
        q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.stabilizer.filter_pose(c0, q0)

        # Big jump: 15 cm (> 8 cm)
        c_jump = c0 + np.array([0.15, 0.0, 0.0], dtype=np.float32)
        c_f, _, _, _ = self.stabilizer.filter_pose(c_jump, q0)
        np.testing.assert_allclose(c_f, c_jump, atol=1e-5)

    def test_filter_pose_quaternion_double_cover_and_slerp(self):
        c0 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        rot0 = R_sci.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True)
        q0 = rot0.as_quat()
        self.stabilizer.filter_pose(c0, q0)

        # Rotate by 5 degrees around Z, but pass negative quaternion (-q)
        rot1 = R_sci.from_euler("xyz", [0.0, 0.0, 5.0], degrees=True)
        q1_neg = -rot1.as_quat()

        _, q_f, R_die, _ = self.stabilizer.filter_pose(c0, q1_neg)
        # The filtered angle should be smooth (between 0 and 5 degrees, around 1 degree for alpha=0.2)
        angle_deg = R_sci.from_quat(q_f).as_euler("xyz", degrees=True)[2]
        self.assertTrue(0.5 < angle_deg < 2.0, f"Expected angle around 1.0 deg, got {angle_deg}")

    def test_filter_pose_quaternion_deadband(self):
        c0 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        rot0 = R_sci.from_euler("xyz", [0.0, 0.0, 10.0], degrees=True)
        q0 = rot0.as_quat()
        self.stabilizer.filter_pose(c0, q0)

        # Jitter by 0.2 degrees (< 0.5 deg deadband)
        rot_jitter = R_sci.from_euler("xyz", [0.0, 0.0, 10.2], degrees=True)
        q_jitter = rot_jitter.as_quat()
        _, q_f, _, _ = self.stabilizer.filter_pose(c0, q_jitter)
        # Should remain exactly locked at q0
        np.testing.assert_allclose(q_f, q0, atol=1e-6)

    def test_filter_pose_quaternion_jump(self):
        c0 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        rot0 = R_sci.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True)
        q0 = rot0.as_quat()
        self.stabilizer.filter_pose(c0, q0)

        # Jump by 90 degrees (> 45 deg jump threshold)
        rot_jump = R_sci.from_euler("xyz", [0.0, 0.0, 90.0], degrees=True)
        q_jump = rot_jump.as_quat()
        _, q_f, _, _ = self.stabilizer.filter_pose(c0, q_jump)
        # Should snap immediately to q_jump
        angle_deg = R_sci.from_quat(q_f).as_euler("xyz", degrees=True)[2]
        self.assertAlmostEqual(angle_deg, 90.0, places=3)


class TestPoseEstimatorContinuity(unittest.TestCase):
    """Test that PoseEstimator eliminates 90-degree axis hopping across frames."""

    def setUp(self):
        self.params = DieDetectorParams(debug=False, die_size_m=0.05, enable_pose_filter=True)
        self.est = PoseEstimator(self.params)

    def test_quad_edge_tracking_avoids_90_deg_flip(self):
        # Top-face quad vertices roughly rotated by 44 degrees
        pts = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        R_plane = np.eye(3, dtype=np.float32)
        cam = (500.0, 500.0, 320.0, 240.0)

        # Frame 1: Quad corners rotated ~45 deg
        poly1 = np.array([[[320, 200]], [[360, 240]], [[320, 280]], [[280, 240]]], dtype=np.int32)
        c1, q1, R1, axes1 = self.est.compute_pose(
            pts, normal, R_plane,
            top_face_polygon=poly1,
            top_face_centroid_2d=(320.0, 240.0),
            plane_D=-1.0,
            camera_params=cam,
        )
        x1 = axes1[0]

        # Frame 2: Slight 1-pixel noise that would previously flip which edge is closest to target_dir
        poly2 = np.array([[[321, 200]], [[360, 241]], [[319, 280]], [[280, 239]]], dtype=np.int32)
        c2, q2, R2, axes2 = self.est.compute_pose(
            pts, normal, R_plane,
            top_face_polygon=poly2,
            top_face_centroid_2d=(320.0, 240.0),
            plane_D=-1.0,
            camera_params=cam,
        )
        x2 = axes2[0]

        # Dot product between x1 and x2 must be close to 1.0 (not ~0.0 which indicates 90 deg hop)
        dot_x = float(np.dot(x1, x2))
        self.assertGreater(dot_x, 0.95, f"Axes hopped! Dot product was {dot_x}")


class TestPipStabilizer(unittest.TestCase):
    """Unit tests for pip temporal consensus voting in PoseStabilizer."""

    def setUp(self):
        self.params = DieDetectorParams(
            enable_pip_filter=True,
            pip_filter_window=5,
        )
        self.stabilizer = PoseStabilizer(self.params)

    def test_filter_pips_glitch_rejection(self):
        # Establish stable 6
        for _ in range(4):
            t, f = self.stabilizer.filter_pips(6, None)
            self.assertEqual(t, 6)

        # Single frame glitch (e.g. temporary occlusion or noise drops to 3)
        t_glitch, _ = self.stabilizer.filter_pips(3, None)
        # Filter should reject single glitch and hold 6
        self.assertEqual(t_glitch, 6)

        # Subsequent frame returns to 6
        t_next, _ = self.stabilizer.filter_pips(6, None)
        self.assertEqual(t_next, 6)

    def test_filter_pips_persistent_transition(self):
        # Stable 6
        for _ in range(5):
            self.stabilizer.filter_pips(6, None)

        # Persistent transition to 2 (die was flipped or rolled)
        self.stabilizer.filter_pips(2, None)
        self.stabilizer.filter_pips(2, None)
        t, _ = self.stabilizer.filter_pips(2, None)
        # After 3 frames of 2 out of 5 in window, majority vote transitions to 2
        self.assertEqual(t, 2)

    def test_filter_pips_reset(self):
        for _ in range(5):
            self.stabilizer.filter_pips(6, None)
        self.stabilizer.reset()
        # Immediately adopts new value upon reset
        t, _ = self.stabilizer.filter_pips(1, None)
        self.assertEqual(t, 1)


if __name__ == "__main__":
    unittest.main()
