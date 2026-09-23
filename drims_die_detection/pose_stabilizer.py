"""
pose_stabilizer.py
==================
Temporal filtering and stabilization for 6-DOF die pose estimation.

Applies:
  1. Exponential Moving Average (EMA) with motion-gating and deadband for 3D translation.
  2. Spherical Linear Interpolation (SLERP) with double-cover sign handling and deadband
     for quaternion orientation.
  3. Temporal smoothing of the RANSAC table plane normal and offset to eliminate surface vibration.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R_sci

from collections import deque, Counter

from .die_detector_params import DieDetectorParams


class PoseStabilizer:
    """Temporal filter for smoothing 6-DOF die poses, fitted table planes, and face pip counts."""

    def __init__(self, params: Optional[DieDetectorParams] = None) -> None:
        self.params = params or DieDetectorParams()
        self._prev_centroid: Optional[np.ndarray] = None
        self._prev_quat: Optional[np.ndarray] = None
        self._prev_plane_normal: Optional[np.ndarray] = None
        self._prev_plane_D: Optional[float] = None
        self._frames_tracked: int = 0
        self._top_pips_history: deque[int] = deque()
        self._front_pips_history: deque[int] = deque()

    def reset(self) -> None:
        """Clear all historical filter state."""
        self._prev_centroid = None
        self._prev_quat = None
        self._prev_plane_normal = None
        self._prev_plane_D = None
        self._frames_tracked = 0
        self._top_pips_history.clear()
        self._front_pips_history.clear()

    # ──────────────────────────────────────────────────────────────────────────
    # Pip Count Temporal Voting Filter
    # ──────────────────────────────────────────────────────────────────────────
    def filter_pips(
        self,
        top_pips: Optional[int],
        front_pips: Optional[int],
    ) -> tuple[Optional[int], Optional[int]]:
        """Filters top and front face pip counts across frames using rolling majority vote."""
        if not getattr(self.params, "enable_pip_filter", True):
            return top_pips, front_pips

        w = int(getattr(self.params, "pip_filter_window", 5))
        if w <= 1:
            return top_pips, front_pips

        def _vote(val: Optional[int], history: deque) -> Optional[int]:
            if val is not None:
                history.append(val)
                while len(history) > w:
                    history.popleft()
            elif history:
                history.popleft()

            if not history:
                return val

            counts = Counter(history)
            most_common_val, most_common_count = counts.most_common(1)[0]
            if most_common_count >= math.ceil(len(history) * 0.4):
                return most_common_val
            return val

        f_top = _vote(top_pips, self._top_pips_history)
        f_front = _vote(front_pips, self._front_pips_history)
        return f_top, f_front

    # ──────────────────────────────────────────────────────────────────────────
    # Table Plane Normal Smoothing
    # ──────────────────────────────────────────────────────────────────────────
    def filter_plane(self, normal: np.ndarray, D: float) -> tuple[np.ndarray, float]:
        """Smooths RANSAC plane normal and distance across frames."""
        if not getattr(self.params, "enable_pose_filter", True):
            return normal, D

        n_raw = (normal / np.linalg.norm(normal)).astype(np.float32)

        if self._prev_plane_normal is None:
            self._prev_plane_normal = n_raw.copy()
            self._prev_plane_D = float(D)
            return n_raw, float(D)

        # Handle normal sign ambiguity
        if float(np.dot(n_raw, self._prev_plane_normal)) < 0.0:
            n_raw = -n_raw
            D = -D

        alpha = float(getattr(self.params, "pose_filter_alpha_trans", 0.25))

        # Check for abrupt plane change (> 25 degrees or > 10 cm)
        angle_diff = math.acos(float(np.clip(np.dot(n_raw, self._prev_plane_normal), -1.0, 1.0)))
        dist_diff = abs(D - self._prev_plane_D)

        if angle_diff > math.radians(25.0) or dist_diff > 0.10:
            self._prev_plane_normal = n_raw.copy()
            self._prev_plane_D = float(D)
            return n_raw, float(D)

        n_filt = (1.0 - alpha) * self._prev_plane_normal + alpha * n_raw
        n_filt = (n_filt / np.linalg.norm(n_filt)).astype(np.float32)

        D_filt = float((1.0 - alpha) * self._prev_plane_D + alpha * D)

        self._prev_plane_normal = n_filt.copy()
        self._prev_plane_D = D_filt
        return n_filt, D_filt

    # ──────────────────────────────────────────────────────────────────────────
    # 6-DOF Pose (Translation & Quaternion) Smoothing
    # ──────────────────────────────────────────────────────────────────────────
    def filter_pose(
        self,
        centroid: np.ndarray,
        quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Filters 3D centroid and orientation quaternion.

        Parameters
        ----------
        centroid : (3,) float32
            Raw computed 3D translation vector.
        quat : (4,) float64 [qx, qy, qz, qw]
            Raw computed orientation quaternion.

        Returns
        -------
        f_centroid : (3,) float32
        f_quat : (4,) float64 [qx, qy, qz, qw]
        f_R : (3, 3) float32
        f_axes : tuple of (x_axis, y_axis, z_axis)
        """
        if not getattr(self.params, "enable_pose_filter", True):
            R_die = R_sci.from_quat(quat).as_matrix().astype(np.float32)
            axes = (R_die[:, 0], R_die[:, 1], R_die[:, 2])
            return centroid, quat, R_die, axes

        c_raw = np.array(centroid, dtype=np.float32)
        q_raw = np.array(quat, dtype=np.float64)
        q_norm = np.linalg.norm(q_raw)
        if q_norm > 1e-6:
            q_raw /= q_norm
        else:
            q_raw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        if self._prev_centroid is None or self._prev_quat is None:
            self._prev_centroid = c_raw.copy()
            self._prev_quat = q_raw.copy()
            self._frames_tracked = 1
            R_die = R_sci.from_quat(q_raw).as_matrix().astype(np.float32)
            axes = (R_die[:, 0], R_die[:, 1], R_die[:, 2])
            return c_raw, q_raw, R_die, axes

        # ── 1. Translation Filtering (EMA + Deadband + Jump Gate) ─────────────
        alpha_t = float(getattr(self.params, "pose_filter_alpha_trans", 0.25))
        jump_t = float(getattr(self.params, "pose_filter_trans_jump_m", 0.08))
        deadband_t = float(getattr(self.params, "pose_filter_deadband_trans_m", 0.0015))

        trans_dist = float(np.linalg.norm(c_raw - self._prev_centroid))

        if trans_dist > jump_t:
            # Sudden large movement (die was picked up or slid quickly) -> reset filter immediately
            c_filt = c_raw.copy()
            self._top_pips_history.clear()
            self._front_pips_history.clear()
        elif trans_dist < deadband_t:
            # Sub-millimeter noise when die is stationary -> hold steady
            c_filt = self._prev_centroid.copy()
        else:
            c_filt = (1.0 - alpha_t) * self._prev_centroid + alpha_t * c_raw

        # ── 2. Quaternion Filtering (SLERP + Double-Cover + Deadband) ─────────
        alpha_r = float(getattr(self.params, "pose_filter_alpha_rot", 0.20))
        jump_r_deg = float(getattr(self.params, "pose_filter_rot_jump_deg", 45.0))
        deadband_r_deg = float(getattr(self.params, "pose_filter_deadband_rot_deg", 0.5))

        # Enforce shortest path double-cover (q and -q represent same rotation)
        dot_q = float(np.dot(q_raw, self._prev_quat))
        if dot_q < 0.0:
            q_raw = -q_raw
            dot_q = -dot_q

        dot_q_clipped = float(np.clip(dot_q, -1.0, 1.0))
        rot_angle_deg = math.degrees(2.0 * math.acos(dot_q_clipped))

        if rot_angle_deg > jump_r_deg:
            # Physical flip or genuine 90+ deg rotation -> snap immediately
            q_filt = q_raw.copy()
            self._top_pips_history.clear()
            self._front_pips_history.clear()
        elif rot_angle_deg < deadband_r_deg:
            # Below jitter threshold -> keep locked
            q_filt = self._prev_quat.copy()
        else:
            # Spherical linear interpolation between prev_quat and q_raw
            try:
                rot_stack = R_sci.from_quat([self._prev_quat, q_raw])
                slerp_fn = R_sci.from_quat([self._prev_quat, q_raw])
                # Direct analytical SLERP for efficiency and precision
                sin_theta = math.sqrt(max(0.0, 1.0 - dot_q_clipped * dot_q_clipped))
                if sin_theta < 1e-4:
                    q_filt = (1.0 - alpha_r) * self._prev_quat + alpha_r * q_raw
                else:
                    theta = math.acos(dot_q_clipped)
                    w1 = math.sin((1.0 - alpha_r) * theta) / sin_theta
                    w2 = math.sin(alpha_r * theta) / sin_theta
                    q_filt = w1 * self._prev_quat + w2 * q_raw
                q_filt /= np.linalg.norm(q_filt)
            except Exception:
                q_filt = q_raw.copy()

        # Update historical state
        self._prev_centroid = c_filt.copy()
        self._prev_quat = q_filt.copy()
        self._frames_tracked += 1

        R_die = R_sci.from_quat(q_filt).as_matrix().astype(np.float32)
        axes = (R_die[:, 0], R_die[:, 1], R_die[:, 2])
        return c_filt.astype(np.float32), q_filt.astype(np.float64), R_die, axes

