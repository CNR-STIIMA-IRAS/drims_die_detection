"""
PoseEstimator
=============
Computes the 6-DOF pose of the die (3D centroid + quaternion orientation) using
ray backprojection onto the top-face plane with a PCA fallback.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R_sci

from .die_detector_params import DieDetectorParams


class PoseEstimator:
    """Computes 6-DOF die pose aligned with the table plane and top-face edges.

    Parameters
    ----------
    params : DieDetectorParams
        Uses ``die_size_m`` and ``debug``.
    """

    def __init__(self, params: DieDetectorParams | None = None) -> None:
        self.params = params or DieDetectorParams()
        self._log = self._make_logger()

    # ──────────────────────────────────────────────────────────────────────
    def compute_pose(
        self,
        die_points: np.ndarray,
        plane_normal: np.ndarray,
        R_plane: np.ndarray,
        top_face_polygon: np.ndarray | None = None,
        top_face_centroid_2d: tuple | None = None,
        primary_lat_centroid_2d: tuple | None = None,
        plane_D: float | None = None,
        die_size: float | None = None,
        camera_params: tuple | None = None,
    ):
        """Compute the 6-DOF die pose.

        Preferred path — ray backprojection:
          Requires *top_face_polygon*, *top_face_centroid_2d*, *plane_D*,
          and *camera_params* (fx, fy, cx, cy).

        Fallback path — PCA on aligned point cloud.

        Parameters
        ----------
        die_points : (N, 3) float32 — 3D points belonging to the die
        plane_normal : (3,) — unit normal of the table plane
        R_plane : (3, 3) — rotation aligning plane normal to Z-axis
        top_face_polygon : (M, 1, 2) int32 or None
        top_face_centroid_2d : (u, v) pixels or None
        primary_lat_centroid_2d : (u, v) pixels or None (center of primary lateral face)
        plane_D : float — D coefficient of the table plane (Ax+By+Cz+D=0)
        die_size : float — physical die side length in metres (overrides params)
        camera_params : (fx, fy, cx, cy) or None

        Returns
        -------
        centroid : (3,) float32 — 3D position of the top-face centre
        quat : (4,) float64 — quaternion [qx, qy, qz, qw]
        R_die : (3, 3) float32 — rotation matrix of the die frame
        axes : (x_axis, y_axis, z_axis) — column unit vectors
        """
        d_size = die_size if die_size is not None else self.params.die_size_m
        z_axis = (plane_normal / np.linalg.norm(plane_normal)).astype(np.float32)

        # Canonical reference direction in table plane: camera +X (1, 0, 0) projected on plane
        cam_x_proj = np.array([1.0, 0.0, 0.0], dtype=np.float32) - float(z_axis[0]) * z_axis
        norm_cam_x = np.linalg.norm(cam_x_proj)
        ref_dir = (cam_x_proj / norm_cam_x).astype(np.float32) if norm_cam_x > 1e-4 else np.array([1.0, 0.0, 0.0], dtype=np.float32)

        x_axis: np.ndarray | None = None
        used_backproject = False

        if (top_face_polygon is not None and top_face_centroid_2d is not None
                and plane_D is not None and camera_params is not None
                and len(top_face_polygon) >= 3):

            fx, fy, cx, cy = camera_params
            u_top, v_top = top_face_centroid_2d
            pts = top_face_polygon.reshape(-1, 2)

            # Ray backprojection onto the top-face plane
            ray_top = np.array([(u_top - cx) / fx, (v_top - cy) / fy, 1.0], dtype=np.float32)
            dot_rn = float(np.dot(ray_top, z_axis))
            t_top = (d_size - plane_D) / dot_rn if abs(dot_rn) > 1e-6 else 1.0
            centroid = (t_top * ray_top).astype(np.float32)

            # Backproject top face quad vertices to 3D points on table plane
            pts_3d = []
            for i in range(len(pts)):
                u_i, v_i = float(pts[i][0]), float(pts[i][1])
                ray_i = np.array([(u_i - cx) / fx, (v_i - cy) / fy, 1.0], dtype=np.float32)
                dot_i = float(np.dot(ray_i, z_axis))
                t_i = (d_size - plane_D) / dot_i if abs(dot_i) > 1e-6 else 1.0
                pts_3d.append(t_i * ray_i)

            # Compute principal quad edge directions in 3D
            candidate_dirs = []
            N = len(pts_3d)
            for i in range(N):
                e = pts_3d[(i + 1) % N] - pts_3d[i]
                e_proj = e - np.dot(e, z_axis) * z_axis
                e_len = np.linalg.norm(e_proj)
                if e_len > 1e-4:
                    candidate_dirs.append(e_proj / e_len)

            if candidate_dirs:
                # Target vector for alignment
                target_dir = ref_dir
                if primary_lat_centroid_2d is not None:
                    u_lat, v_lat = primary_lat_centroid_2d
                    edge_2d = np.array([u_lat - u_top, v_lat - v_top], dtype=np.float32)
                    edge_norm = np.linalg.norm(edge_2d)
                    if edge_norm > 1e-3:
                        d_2d = edge_2d / edge_norm
                        u_e, v_e = u_top + 20.0 * d_2d[0], v_top + 20.0 * d_2d[1]
                        ray_e = np.array([(u_e - cx) / fx, (v_e - cy) / fy, 1.0], dtype=np.float32)
                        t_e = (d_size - plane_D) / float(np.dot(ray_e, z_axis)) if abs(np.dot(ray_e, z_axis)) > 1e-6 else 1.0
                        pt_e = t_e * ray_e
                        v_3d = pt_e - centroid
                        x_raw = v_3d - np.dot(v_3d, z_axis) * z_axis
                        if np.linalg.norm(x_raw) > 1e-4:
                            target_dir = x_raw / np.linalg.norm(x_raw)

                # Pick quad edge vector closest to target_dir
                best_score = -1e9
                for cd in candidate_dirs:
                    for sign in (1.0, -1.0):
                        v_test = (sign * cd).astype(np.float32)
                        score = float(np.dot(v_test, target_dir))
                        if score > best_score:
                            best_score = score
                            x_axis = v_test
                            used_backproject = True

        if not used_backproject or x_axis is None:
            self._log("Using PCA fallback for pose estimation.")
            centroid = (np.mean(die_points, axis=0).astype(np.float32) if len(die_points) > 0
                        else np.array([0.0, 0.0, 1.0], dtype=np.float32))
            aligned = np.dot(die_points - centroid, R_plane.T)
            xy_pts = aligned[:, :2].astype(np.float32)

            if len(xy_pts) >= 5:
                cov = np.cov(xy_pts, rowvar=False)
                _, evecs = np.linalg.eigh(cov)
                v1_2d = evecs[:, 1]
            else:
                v1_2d = np.array([1.0, 0.0], dtype=np.float32)

            R_inv = R_plane.T
            x_axis = np.dot(R_inv, np.array([v1_2d[0], v1_2d[1], 0.0], np.float32))
            x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
            x_axis /= np.linalg.norm(x_axis)
            if np.dot(x_axis, ref_dir) < 0:
                x_axis = -x_axis

        # Ensure orthonormal right-handed frame (z = plane_normal, y = z × x, x = y × z)
        y_axis = np.cross(z_axis, x_axis).astype(np.float32)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis).astype(np.float32)
        x_axis /= np.linalg.norm(x_axis)

        R_die = np.stack((x_axis, y_axis, z_axis), axis=1).astype(np.float32)
        quat = R_sci.from_matrix(R_die).as_quat()   # [qx, qy, qz, qw]

        self._log(f"Pose: centroid={centroid}, quat={quat}.")
        return centroid, quat, R_die, (x_axis, y_axis, z_axis)

    # Backward-compatible static alias
    @classmethod
    def compute_pose_static(cls, *args, **kwargs):
        return cls().compute_pose(*args, **kwargs)

    def _make_logger(self):
        tag = "[PoseEstimator]"
        if self.params.debug:
            return lambda msg: print(f"{tag} {msg}")
        return lambda msg: None
