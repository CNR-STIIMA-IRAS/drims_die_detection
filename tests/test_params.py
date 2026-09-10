#!/usr/bin/env python3
"""Tests for DieDetectorParams — YAML loading and default values."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drims_die_detection.die_detector_params import DieDetectorParams


class TestDieDetectorParamsDefaults(unittest.TestCase):

    def test_default_construction(self):
        p = DieDetectorParams()
        self.assertFalse(p.debug)
        self.assertFalse(p.visualize)
        self.assertTrue(p.save)
        self.assertAlmostEqual(p.fx, 500.0)
        self.assertAlmostEqual(p.fy, 500.0)
        self.assertIsNone(p.cx)
        self.assertIsNone(p.cy)
        self.assertEqual(p.die_color, "white")
        self.assertEqual(p.num_color_clusters, 5)
        self.assertAlmostEqual(p.die_size_m, 0.050)
        self.assertAlmostEqual(p.ransac_distance_threshold, 0.015)
        self.assertTrue(p.use_monocular_fallback)

    def test_from_dict_partial(self):
        p = DieDetectorParams.from_dict({"debug": True, "fx": 640.0, "die_color": "red"})
        self.assertTrue(p.debug)
        self.assertAlmostEqual(p.fx, 640.0)
        self.assertEqual(p.die_color, "red")
        # Unspecified fields retain defaults
        self.assertAlmostEqual(p.fy, 500.0)

    def test_from_dict_ignores_unknown_keys(self):
        """Unknown keys must not raise an error."""
        p = DieDetectorParams.from_dict({"unknown_key": 42, "debug": True})
        self.assertTrue(p.debug)

    def test_from_yaml_flat(self):
        yaml_content = "debug: true\nfx: 620.0\ndie_color: black\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            tmp = f.name
        try:
            p = DieDetectorParams.from_yaml(tmp)
            self.assertTrue(p.debug)
            self.assertAlmostEqual(p.fx, 620.0)
            self.assertEqual(p.die_color, "black")
        finally:
            os.unlink(tmp)

    def test_from_yaml_ros2_style(self):
        yaml_content = (
            "die_detector_node:\n"
            "  ros__parameters:\n"
            "    debug: false\n"
            "    fx: 615.0\n"
            "    die_size_m: 0.040\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            tmp = f.name
        try:
            p = DieDetectorParams.from_yaml(tmp)
            self.assertFalse(p.debug)
            self.assertAlmostEqual(p.fx, 615.0)
            self.assertAlmostEqual(p.die_size_m, 0.040)
        finally:
            os.unlink(tmp)

    def test_ros_topic_defaults(self):
        p = DieDetectorParams()
        self.assertEqual(p.rgb_topic, "/camera/color/image_raw")
        self.assertEqual(p.depth_topic, "/camera/aligned_depth_to_color/image_raw")
        self.assertEqual(p.camera_info_topic, "/camera/color/camera_info")
        self.assertEqual(p.debug_panels_topic, "/dice/debug_panels")


if __name__ == "__main__":
    unittest.main()
