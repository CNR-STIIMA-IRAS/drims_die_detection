"""
AlignmentAndProjection
======================
Rotates the 3D point cloud so that the table-plane normal aligns with the
camera Z-axis, and extracts a rectified top-down RGB crop of the die face.
"""

from __future__ import annotations

import cv2
import numpy as np

from .die_detector_params import DieDetectorParams


class AlignmentAndProjection:
    """Plane-normal-to-Z rotation and top-down orthographic extraction.

    Parameters
    ----------
    params : DieDetectorParams
        Uses ``debug``.
    """

    def __init__(self, params: DieDetectorParams | None = None) -> None:
        self.params = params or DieDetectorParams()
        self._log = self._make_logger()

    # ──────────────────────────────────────────────────────────────────────
    def get_rotation_to_z(self, normal: np.ndarray) -> np.ndarray:
        """3×3 rotation matrix R that maps *normal* → [0, 0, 1].

        Parameters
        ----------
        normal : (3,) array — unit plane normal (pointing toward camera)

        Returns
        -------
        R : (3, 3) float32
        """
        return self._rotation_to_z(normal)

    def extract_top_down_rgb(
        self,
        rgb: np.ndarray,
        die_points: np.ndarray,
        pixel_coords: np.ndarray,
        die_indices: np.ndarray,
        R_plane: np.ndarray,
        crop_size: int = 300,
    ):
        """Compute an ortho-rectified top-down crop of the die from the RGB image.

        Returns
        -------
        top_down_crop : (crop_size, crop_size, 3) BGR
        raw_crop : (H', W', 3) BGR — the un-resized bounding-box crop
        (x_min, y_min, x_max, y_max) : pixel extents of the die bounding box
        yaw_angle : float — estimated yaw from minAreaRect (degrees)
        """
        h, w = rgb.shape[:2]
        die_pixels = pixel_coords[die_indices]

        x_min, y_min = die_pixels[:, 0].min(), die_pixels[:, 1].min()
        x_max, y_max = die_pixels[:, 0].max(), die_pixels[:, 1].max()

        margin = 10
        xc0 = max(0, x_min - margin)
        xc1 = min(w, x_max + margin)
        yc0 = max(0, y_min - margin)
        yc1 = min(h, y_max + margin)

        raw_crop = rgb[yc0:yc1, xc0:xc1]

        aligned_pts = np.dot(die_points, R_plane.T)
        xy_pts = aligned_pts[:, :2].astype(np.float32)

        yaw_angle = 0.0
        if len(xy_pts) >= 5:
            rect = cv2.minAreaRect(xy_pts)
            yaw_angle = rect[2]

        top_down_crop = (
            cv2.resize(raw_crop, (crop_size, crop_size))
            if raw_crop.size > 0
            else np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        )
        self._log(f"Top-down crop: raw {raw_crop.shape[:2]}, yaw={yaw_angle:.1f}°.")
        return top_down_crop, raw_crop, (x_min, y_min, x_max, y_max), yaw_angle

    # ──────────────────────────────────────────────────────────────────────
    # Static API (backward compat)
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _rotation_to_z(normal: np.ndarray) -> np.ndarray:
        n = normal / np.linalg.norm(normal)
        target = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        v = np.cross(n, target)
        s = np.linalg.norm(v)
        c = float(np.dot(n, target))

        if s < 1e-6:
            if c < 0:
                return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
            return np.identity(3, dtype=np.float32)

        vx = np.array([
            [0,    -v[2],  v[1]],
            [v[2],  0,    -v[0]],
            [-v[1], v[0],  0],
        ], dtype=np.float32)
        R = np.identity(3, dtype=np.float32) + vx + np.matmul(vx, vx) * ((1.0 - c) / (s ** 2))
        return R

    # Keep static method accessible for legacy callers
    @classmethod
    def get_rotation_to_z_static(cls, normal: np.ndarray) -> np.ndarray:
        return cls._rotation_to_z(normal)

    @classmethod
    def extract_top_down_rgb_static(
        cls,
        rgb: np.ndarray,
        die_points: np.ndarray,
        pixel_coords: np.ndarray,
        die_indices: np.ndarray,
        R_plane: np.ndarray,
        crop_size: int = 300,
    ):
        inst = cls()
        return inst.extract_top_down_rgb(rgb, die_points, pixel_coords, die_indices, R_plane, crop_size)

    def _make_logger(self):
        tag = "[Alignment]"
        if self.params.debug:
            return lambda msg: print(f"{tag} {msg}")
        return lambda msg: None
