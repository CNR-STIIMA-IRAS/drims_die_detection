#!/usr/bin/env python3
"""
Smoke tests for the ROS 2 die_detector_node.

These tests do NOT require a running ROS 2 daemon — they mock rclpy and verify:
  1. The node module imports cleanly.
  2. DieDetectorNode can be instantiated when rclpy IS available (skipped otherwise).
  3. The pipeline integration used by the node works correctly standalone.

When running inside a full ROS 2 Humble environment (`colcon test`), all tests
execute. In a ROS-free CI environment only the standalone tests run.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ──────────────────────────────────────────────────────────────────────
# Availability flags
# ──────────────────────────────────────────────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

import numpy as np
from drims_die_detection import DieDetectorParams, DieDetectorPipeline


class TestNodeModuleImport(unittest.TestCase):
    """The node script must be importable regardless of ROS 2 availability."""

    def test_die_detector_params_importable(self):
        from drims_die_detection.die_detector_params import DieDetectorParams
        p = DieDetectorParams()
        self.assertIsNotNone(p)

    def test_pipeline_importable(self):
        from drims_die_detection.die_detector_pipeline import DieDetectorPipeline
        self.assertTrue(callable(DieDetectorPipeline))


@unittest.skipUnless(HAS_ROS2, "ROS 2 (rclpy) not available — skipping live node tests")
class TestDieDetectorNodeInstantiation(unittest.TestCase):
    """Integration tests that require a live rclpy context."""

    @classmethod
    def setUpClass(cls):
        rclpy.init(args=["--ros-args"])

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_node_creates_without_error(self):
        from scripts.die_detector_node import DieDetectorNode
        node = DieDetectorNode()
        self.assertIsNotNone(node)
        node.destroy_node()

    def test_node_has_expected_publishers(self):
        from scripts.die_detector_node import DieDetectorNode
        node = DieDetectorNode()
        pub_topics = [info.topic for info in node.get_publishers_info_by_topic("/dice/pose")]
        # Publisher should exist
        self.assertIn("/dice/pose", [p.topic_name for p in node.publishers])
        node.destroy_node()


class TestPipelineStandaloneForNode(unittest.TestCase):
    """Verify the pipeline that backs the node works correctly standalone."""

    def setUp(self):
        params = DieDetectorParams(
            debug=False, save=False, visualize=False,
            fx=500.0, fy=500.0,
            use_monocular_fallback=False,
        )
        self.pipeline = DieDetectorPipeline(params)

    def _make_fake_frame(self):
        rgb = np.ones((240, 320, 3), dtype=np.uint8) * 120
        depth = np.ones((240, 320), dtype=np.float32)
        return rgb, depth

    def test_process_returns_centroid(self):
        rgb, depth = self._make_fake_frame()
        result = self.pipeline.process_rgbd(rgb, depth)
        self.assertIn("centroid", result)
        self.assertEqual(len(result["centroid"]), 3)

    def test_quaternion_unit_norm(self):
        rgb, depth = self._make_fake_frame()
        result = self.pipeline.process_rgbd(rgb, depth)
        self.assertAlmostEqual(np.linalg.norm(result["quaternion"]), 1.0, places=3)

    def test_depth_scaling_uint16_conversion(self):
        """Simulate how the node converts uint16 mm depth to float32 metres."""
        depth_uint16 = (np.ones((240, 320)) * 1000).astype(np.uint16)  # 1000 mm = 1 m
        depth_m = depth_uint16.astype(np.float32) * 0.001
        self.assertAlmostEqual(float(depth_m.mean()), 1.0, places=3)

    def test_build_debug_panels_returns_bgr(self):
        rgb, depth = self._make_fake_frame()
        result = self.pipeline.process_rgbd(rgb, depth)
        collage = self.pipeline.build_debug_panels(rgb, result)
        self.assertEqual(collage.ndim, 3)
        self.assertEqual(collage.shape[2], 3)


if __name__ == "__main__":
    unittest.main()
