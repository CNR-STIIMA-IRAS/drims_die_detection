"""
PointCloudProcessor
===================
Generates a 3D point cloud from an aligned RGB+Depth pair and performs
RANSAC plane fitting to identify the table surface.
"""

from __future__ import annotations

import numpy as np
import cv2

from .die_detector_params import DieDetectorParams

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


class PointCloudProcessor:
    """Converts RGB + metric depth into a point cloud and fits a dominant plane.

    Parameters
    ----------
    params : DieDetectorParams
        Uses ``fx``, ``fy``, ``cx``, ``cy``, ``ransac_distance_threshold``,
        ``ransac_max_iterations``, and ``debug``.
    """

    def __init__(self, params: DieDetectorParams | None = None) -> None:
        self.params = params or DieDetectorParams()
        self._log = self._make_logger()

    # ──────────────────────────────────────────────────────────────────────
    def create_point_cloud(
        self,
        rgb_bgr: np.ndarray,
        depth_map: np.ndarray,
    ):
        """Back-project every pixel into 3D camera space.

        Parameters
        ----------
        rgb_bgr : (H, W, 3) uint8
        depth_map : (H, W) float32 — metric depth in metres

        Returns
        -------
        points : (N, 3) float32
        colors : (N, 3) float32  — RGB normalised to [0, 1]
        pixel_coords : (N, 2) int32
        (cx, cy) : principal point actually used
        """
        h, w = depth_map.shape
        cx = self.params.cx if self.params.cx is not None else w / 2.0
        cy = self.params.cy if self.params.cy is not None else h / 2.0
        fx, fy = self.params.fx, self.params.fy

        y_grid, x_grid = np.indices((h, w), dtype=np.float32)
        z_3d = depth_map.astype(np.float32)
        x_3d = (x_grid - cx) * z_3d / fx
        y_3d = (y_grid - cy) * z_3d / fy

        points = np.stack((x_3d, y_3d, z_3d), axis=-1).reshape(-1, 3)
        colors = rgb_bgr.reshape(-1, 3)[:, ::-1] / 255.0      # BGR → RGB [0,1]
        pixel_coords = np.stack((x_grid, y_grid), axis=-1).reshape(-1, 2).astype(np.int32)

        valid_mask = (z_3d.reshape(-1) > 0.1) & np.isfinite(z_3d.reshape(-1))
        self._log(f"Point cloud: {valid_mask.sum()} valid points out of {len(points)}.")
        return points[valid_mask], colors[valid_mask], pixel_coords[valid_mask], (cx, cy)

    # ──────────────────────────────────────────────────────────────────────
    def fit_plane_ransac(
        self,
        points: np.ndarray,
        distance_threshold: float | None = None,
        max_iterations: int | None = None,
    ):
        """Fit the dominant plane via RANSAC.

        Uses Open3D when available (faster); falls back to a pure-NumPy
        implementation otherwise.

        Parameters
        ----------
        points : (N, 3) float32
        distance_threshold : float, optional — overrides params value
        max_iterations : int, optional — overrides params value

        Returns
        -------
        plane_model : (A, B, C, D) — plane equation coefficients
        inliers : array-like of inlier indices into *points*
        outliers : array-like of outlier indices
        normal : (3,) unit normal vector pointing towards the camera (z < 0)
        """
        dist_thresh = distance_threshold or self.params.ransac_distance_threshold
        max_iter = max_iterations or self.params.ransac_max_iterations

        if HAS_OPEN3D:
            return self._ransac_open3d(points, dist_thresh, max_iter)
        return self._ransac_numpy(points, dist_thresh, max_iter)

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _ransac_open3d(self, points, dist_thresh, max_iter):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=dist_thresh,
            ransac_n=3,
            num_iterations=max_iter,
        )
        # SVD refinement on Open3D inliers
        return self._refine_plane_svd(points, inliers, dist_thresh)

    def _ransac_numpy(self, points, dist_thresh, max_iter):
        num_points = len(points)
        if num_points < 3:
            normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            return (0.0, 0.0, -1.0, 0.0), list(range(num_points)), [], normal

        # Subsample for speed
        sub_idx = np.random.choice(num_points, size=min(10_000, num_points), replace=False)
        sub_pts = points[sub_idx]

        best_cost = 1e18
        best_inliers = []

        sq_thresh = dist_thresh ** 2

        for _ in range(max_iter):
            idx = np.random.choice(len(sub_pts), 3, replace=False)
            p1, p2, p3 = sub_pts[idx]
            v1, v2 = p2 - p1, p3 - p1
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-6:
                continue
            n /= norm_n
            d = -np.dot(n, p1)

            # MSAC scoring (M-estimator cost truncated at dist_thresh^2)
            distances = np.abs(np.dot(sub_pts, n) + d)
            cost = float(np.sum(np.minimum(distances ** 2, sq_thresh)))
            if cost < best_cost:
                best_cost = cost
                best_inliers = np.where(np.abs(np.dot(points, n) + d) < dist_thresh)[0]

        if len(best_inliers) < 3:
            self._log("RANSAC failed; returning horizontal fallback plane.")
            normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            return (0.0, 0.0, -1.0, -1.0), [], list(range(num_points)), normal

        # SVD refinement on MSAC inliers
        return self._refine_plane_svd(points, best_inliers, dist_thresh)

    def _refine_plane_svd(self, points, initial_inliers, dist_thresh, passes=3):
        """Refine plane normal and D using Total Least Squares (SVD) on inlier set."""
        curr_inliers = initial_inliers
        n_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        d_val = 0.0

        for _ in range(passes):
            if len(curr_inliers) < 3:
                break
            inlier_pts = points[curr_inliers]
            centroid = np.mean(inlier_pts, axis=0)
            centered = inlier_pts - centroid
            cov = np.dot(centered.T, centered)
            evals, evecs = np.linalg.eigh(cov)
            n_vec = evecs[:, 0].astype(np.float32)
            norm_val = np.linalg.norm(n_vec)
            if norm_val > 0:
                n_vec /= norm_val
            d_val = float(-np.dot(n_vec, centroid))
            dists = np.abs(np.dot(points, n_vec) + d_val)
            curr_inliers = np.where(dists < dist_thresh)[0]

        if n_vec[2] > 0:
            n_vec = -n_vec
            d_val = -d_val

        dists_final = np.abs(np.dot(points, n_vec) + d_val)
        inliers_final = np.where(dists_final < dist_thresh)[0]
        outliers_final = np.where(dists_final >= dist_thresh)[0]

        self._log(f"RANSAC + SVD Refinement: {len(inliers_final)} inliers, normal={n_vec}.")
        return (float(n_vec[0]), float(n_vec[1]), float(n_vec[2]), d_val), inliers_final, outliers_final, n_vec

    def _make_logger(self):
        tag = "[PointCloudProcessor]"
        if self.params.debug:
            return lambda msg: print(f"{tag} {msg}")
        return lambda msg: None
