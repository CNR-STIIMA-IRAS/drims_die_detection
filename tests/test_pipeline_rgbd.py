#!/usr/bin/env python3
"""
Integration tests for DieDetectorPipeline using real images from images/rgb/.

These tests run the full pipeline with monocular depth estimation (no real
depth camera required) and assert that the pipeline returns well-formed results.

The depth images in images/depth/ are colourmap PNGs for visualisation only;
the pipeline generates its own metric depth internally via monocular estimation.
"""

import os
import sys
import glob
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drims_die_detection import DieDetectorParams, DieDetectorPipeline
from drims_die_detection.depth_estimator import HAS_TRANSFORMERS, MonocularDepthEstimator


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RGB_DIR = os.path.join(REPO_ROOT, "images", "rgb")


def _get_test_images(n=2):
    """Return up to *n* RGB image paths from images/rgb/."""
    paths = sorted(
        glob.glob(os.path.join(RGB_DIR, "*.jpg")) +
        glob.glob(os.path.join(RGB_DIR, "*.png"))
    )
    return paths[:n]


class TestMonocularDepthEstimatorBehavior(unittest.TestCase):
    """Tests for MonocularDepthEstimator error handling."""

    def test_raises_if_no_transformers(self):
        if not HAS_TRANSFORMERS:
            with self.assertRaises(ImportError):
                MonocularDepthEstimator()


@unittest.skipIf(not HAS_TRANSFORMERS or not os.path.isdir(RGB_DIR) or not _get_test_images(),
                 reason="Requires torch/transformers and RGB test images in images/rgb/")
class TestPipelineWithRealImages(unittest.TestCase):
    """Full pipeline integration test using the first 2 real images."""

    @classmethod
    def setUpClass(cls):
        params = DieDetectorParams(
            debug=False,
            visualize=False,
            save=False,
            # Use image-centre principal point (cx/cy=None → auto)
            fx=615.0, fy=615.0,
            use_monocular_fallback=True,
        )
        cls.pipeline = DieDetectorPipeline(params)
        cls.image_paths = _get_test_images(n=2)

    def _run_pipeline(self, img_path):
        rgb = cv2.imread(img_path)
        self.assertIsNotNone(rgb, f"Failed to load {img_path}")
        return self.pipeline.process_rgb(rgb, image_name=os.path.basename(img_path))

    def test_result_keys_present(self):
        result = self._run_pipeline(self.image_paths[0])
        required = {
            "pip_count", "num_visible_faces", "faces",
            "centroid", "quaternion", "R_die", "axes",
            "plane_model", "normal", "die_points", "points",
            "annotated_rgb", "top_down_crop", "debug_steps",
        }
        for key in required:
            self.assertIn(key, result, msg=f"Missing result key: '{key}'")

    def test_quaternion_unit_norm(self):
        result = self._run_pipeline(self.image_paths[0])
        quat = result["quaternion"]
        self.assertEqual(len(quat), 4)
        self.assertAlmostEqual(np.linalg.norm(quat), 1.0, places=3)

    def test_centroid_finite(self):
        result = self._run_pipeline(self.image_paths[0])
        centroid = result["centroid"]
        self.assertTrue(np.all(np.isfinite(centroid)),
                        msg=f"Centroid is not finite: {centroid}")

    def test_normal_unit_length(self):
        result = self._run_pipeline(self.image_paths[0])
        normal = result["normal"]
        self.assertAlmostEqual(np.linalg.norm(normal), 1.0, places=3)

    def test_point_cloud_non_empty(self):
        result = self._run_pipeline(self.image_paths[0])
        self.assertGreater(len(result["points"]), 0)

    def test_debug_panels_shape(self):
        rgb = cv2.imread(self.image_paths[0])
        result = self._run_pipeline(self.image_paths[0])
        collage = self.pipeline.build_debug_panels(rgb, result)
        self.assertEqual(collage.ndim, 3)
        self.assertEqual(collage.dtype, np.uint8)

    def test_top_down_crop_is_image(self):
        result = self._run_pipeline(self.image_paths[0])
        td = result["top_down_crop"]
        self.assertEqual(td.ndim, 3)
        self.assertEqual(td.dtype, np.uint8)
        self.assertGreater(td.size, 0)

    def test_faces_list_is_list(self):
        result = self._run_pipeline(self.image_paths[0])
        self.assertIsInstance(result["faces"], list)

    def test_pip_count_non_negative(self):
        result = self._run_pipeline(self.image_paths[0])
        self.assertGreaterEqual(result["pip_count"], 0)

    def test_all_images_complete_without_crash(self):
        """Every image in the test set must complete without raising."""
        for path in self.image_paths:
            try:
                self._run_pipeline(path)
            except Exception as exc:
                self.fail(f"Pipeline crashed on {path}: {exc!r}")


class TestPipelineSyntheticDepth(unittest.TestCase):
    """Pipeline tests with fully synthetic RGB+depth — no real images needed."""

    @classmethod
    def setUpClass(cls):
        params = DieDetectorParams(
            debug=False, visualize=False, save=False,
            fx=500.0, fy=500.0,
            use_monocular_fallback=False,   # depth is provided, no fallback needed
        )
        cls.pipeline = DieDetectorPipeline(params)

    def _make_synthetic_rgbd(self, h=240, w=320):
        rgb = np.ones((h, w, 3), dtype=np.uint8) * 150
        # Draw a bright die-like white rectangle
        cv2.rectangle(rgb, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (220, 220, 220), -1)
        depth = np.ones((h, w), dtype=np.float32)   # 1 m everywhere
        return rgb, depth

    def test_process_rgbd_does_not_crash(self):
        rgb, depth = self._make_synthetic_rgbd()
        result = self.pipeline.process_rgbd(rgb, depth, image_name="synthetic.jpg")
        self.assertIn("centroid", result)

    def test_build_debug_panels_returns_image(self):
        rgb, depth = self._make_synthetic_rgbd()
        result = self.pipeline.process_rgbd(rgb, depth)
        collage = self.pipeline.build_debug_panels(rgb, result)
        self.assertEqual(collage.ndim, 3)
        self.assertGreater(collage.size, 0)


if __name__ == "__main__":
    unittest.main()
