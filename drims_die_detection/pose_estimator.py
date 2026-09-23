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
        self._prev_x_axis: np.ndarray | None = None

    def reset_tracking(self) -> None:
        """Reset temporal tracking of principal axis."""
        self._prev_x_axis = None

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

        # Reference directions in the die plane:
        # Camera +X points to the RIGHT in the image.
        cam_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ref_right = cam_x - float(np.dot(cam_x, z_axis)) * z_axis
        norm_right = float(np.linalg.norm(ref_right))
        ref_right = (ref_right / norm_right).astype(np.float32) if norm_right > 1e-4 else cam_x

        # In camera optical frame, UP in the image is -Y.
        # Since z_axis points towards camera (z < 0), z_axis × ref_right points UP.
        ref_up = np.cross(z_axis, ref_right).astype(np.float32)
        norm_up = float(np.linalg.norm(ref_up))
        ref_up = (ref_up / norm_up).astype(np.float32) if norm_up > 1e-4 else np.array([0.0, -1.0, 0.0], dtype=np.float32)

        x_axis: np.ndarray | None = None
        y_axis: np.ndarray | None = None
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

            base_e1 = None
            if len(pts_3d) == 4:
                # Average opposite edges of the quad for robust side orientation
                e0 = pts_3d[1] - pts_3d[0]
                e1 = pts_3d[2] - pts_3d[1]
                e2 = pts_3d[3] - pts_3d[2]
                e3 = pts_3d[0] - pts_3d[3]
                v1 = 0.5 * (e0 - e2)
                v2 = 0.5 * (e1 - e3)
                v1_proj = v1 - np.dot(v1, z_axis) * z_axis
                v2_proj = v2 - np.dot(v2, z_axis) * z_axis
                n1 = float(np.linalg.norm(v1_proj))
                n2 = float(np.linalg.norm(v2_proj))
                if n1 > 1e-4:
                    u1 = v1_proj / n1
                    if n2 > 1e-4:
                        u2 = v2_proj / n2
                        u1_perp = np.cross(z_axis, u1)
                        if np.dot(u2, u1_perp) < 0:
                            u2 = -u2
                        side_vec = u1 + np.cross(u2, z_axis)
                        norm_s = float(np.linalg.norm(side_vec))
                        base_e1 = (side_vec / norm_s).astype(np.float32) if norm_s > 1e-4 else u1.astype(np.float32)
                    else:
                        base_e1 = u1.astype(np.float32)
                    used_backproject = True

            if not used_backproject:
                candidate_dirs = []
                N = len(pts_3d)
                for i in range(N):
                    e = pts_3d[(i + 1) % N] - pts_3d[i]
                    e_proj = e - np.dot(e, z_axis) * z_axis
                    e_len = float(np.linalg.norm(e_proj))
                    if e_len > 1e-4:
                        candidate_dirs.append(e_proj / e_len)
                if candidate_dirs:
                    base_e1 = candidate_dirs[0].astype(np.float32)
                    used_backproject = True

            if used_backproject and base_e1 is not None:
                base_e2 = np.cross(z_axis, base_e1).astype(np.float32)
                base_e2 /= np.linalg.norm(base_e2)

                # The 4 rotational symmetries of the square top face
                # All candidates have axes parallel to edges (perpendicular to die sides)
                candidates = [
                    (base_e1, base_e2),
                    (base_e2, -base_e1),
                    (-base_e1, -base_e2),
                    (-base_e2, base_e1),
                ]
                best_score = -1e9
                best_pair = (base_e1, base_e2)
                for x_c, y_c in candidates:
                    # Select the candidate where x points RIGHT and y points UP
                    score = float(np.dot(x_c, ref_right) + np.dot(y_c, ref_up))
                    if self._prev_x_axis is not None and getattr(self.params, "enable_pose_filter", True):
                        score += 0.20 * float(np.dot(x_c, self._prev_x_axis))
                    if score > best_score:
                        best_score = score
                        best_pair = (x_c, y_c)
                x_axis, y_axis = best_pair

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
            e1 = np.dot(R_inv, np.array([v1_2d[0], v1_2d[1], 0.0], np.float32))
            e1 = e1 - np.dot(e1, z_axis) * z_axis
            norm_e1 = float(np.linalg.norm(e1))
            e1 = (e1 / norm_e1).astype(np.float32) if norm_e1 > 1e-4 else ref_right.copy()
            e2 = np.cross(z_axis, e1).astype(np.float32)
            e2 /= np.linalg.norm(e2)

            candidates = [
                (e1, e2),
                (e2, -e1),
                (-e1, -e2),
                (-e2, e1),
            ]
            best_score = -1e9
            best_pair = (e1, e2)
            for x_c, y_c in candidates:
                score = float(np.dot(x_c, ref_right) + np.dot(y_c, ref_up))
                if self._prev_x_axis is not None and getattr(self.params, "enable_pose_filter", True):
                    score += 0.20 * float(np.dot(x_c, self._prev_x_axis))
                if score > best_score:
                    best_score = score
                    best_pair = (x_c, y_c)
            x_axis, y_axis = best_pair

        # Ensure orthonormal right-handed frame (z = plane_normal, y = z × x, x = y × z)
        y_axis = np.cross(z_axis, x_axis).astype(np.float32)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis).astype(np.float32)
        x_axis /= np.linalg.norm(x_axis)

        # Retain x_axis for temporal continuity across consecutive frames
        self._prev_x_axis = x_axis.copy()

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
