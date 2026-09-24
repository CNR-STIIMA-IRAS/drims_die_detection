"""
silhouette_pose.py
==================
Die pose from its segmentation mask, with no face or edge fitting.

The die is modelled as a cube of known side ``s`` resting on a known table
plane, so its pose has only three unknowns: the contact point (x, y) on the
plane and the yaw θ about the plane normal. They are found by maximising the
IoU between the YOLOE mask and the convex hull of the projected cube corners
(the silhouette of a convex body is the hull of its projected vertices).

Because a cube looks identical every 90°, the silhouette only fixes θ modulo
90°. The die frame is then chosen as (an arbitrary, repeatable convention):
    z  table normal (up)
    x  outward normal of the *front face* — the side face turned most towards
       the camera (the face the CNN's "front" head names)
    y  z × x

Conventions (camera optical frame: x right, y down, z forward):
    plane : n · X + h = 0, with ``n`` the unit table normal pointing *up*
            (towards the camera side) and ``h`` the camera height above it.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R_sci

# 8 unit-cube corners (x, y in [-0.5, 0.5], height in {0, 1}) and its 12 edges
_CUBE = np.array([[i, j, k] for i in (-0.5, 0.5) for j in (-0.5, 0.5) for k in (0.0, 1.0)])
CUBE_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8)
              if np.sum(np.abs(_CUBE[a] - _CUBE[b]) > 1e-9) == 1]


def die_silhouette(image_bgr: np.ndarray, seg_mask: np.ndarray, bbox, grow: float = 0.35,
                   max_area_gain: float = 1.8) -> np.ndarray | None:
    """Convex die outline (full-image contour) from a segmentation mask.

    The raw YOLOE mask has notches where pips / dark edge lines touch the
    outline and can drop a shaded side face. The cube silhouette is convex, so
    take the hull; and since every die face is much brighter than the dark
    table, add bright (Otsu) pixels near the box that touch the mask. Falls
    back to the plain mask hull if that grows the area implausibly (e.g. a
    hand touching the die).
    """
    x, y, w, h = bbox
    H, W = seg_mask.shape[:2]
    gx, gy = int(w * grow), int(h * grow)
    x0, y0, x1, y1 = max(0, x - gx), max(0, y - gy), min(W, x + w + gx), min(H, y + h + gy)
    seg = (seg_mask[y0:y1, x0:x1] > 0).astype(np.uint8)
    if not seg.any():
        return None

    gray = cv2.cvtColor(image_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n_lab, lab = cv2.connectedComponents(bright | seg)
    keep = np.isin(lab, np.unique(lab[seg > 0]))
    keep &= lab > 0

    def hull_of(m):
        pts = cv2.findNonZero(m.astype(np.uint8))
        return cv2.convexHull(pts) + np.array([x0, y0]) if pts is not None else None

    hull_seg = hull_of(seg)
    hull_all = hull_of(keep)
    if hull_all is None or cv2.contourArea(hull_all) > max_area_gain * cv2.contourArea(hull_seg):
        return hull_seg
    return hull_all


def plane_from_angles(tilt_rad: float, roll_rad: float, height: float):
    """Table plane in the camera frame for a camera looking down by *tilt*
    (angle between optical axis and table) with *roll* about the optical axis."""
    n = np.array([0.0, -np.cos(tilt_rad), -np.sin(tilt_rad)])
    c, s = np.cos(roll_rad), np.sin(roll_rad)
    n = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ n
    return n / np.linalg.norm(n), float(height)


class SilhouettePoseFitter:
    """Fit (x, y, θ) of a cube on a plane to a binary die mask."""

    def __init__(self, K: np.ndarray, plane_n: np.ndarray, plane_h: float,
                 die_size: float = 0.041, theta_step_deg: float = 2.0, pad_px: int = 30):
        self.K = np.asarray(K, dtype=np.float64)
        self.n = np.asarray(plane_n, dtype=np.float64) / np.linalg.norm(plane_n)
        self.h = float(plane_h)
        self.s = float(die_size)
        self.theta_grid = np.radians(np.arange(0.0, 90.0, theta_step_deg))
        self.pad = pad_px

        # In-plane basis (a, b): a = camera +X projected on the plane
        a = np.array([1.0, 0.0, 0.0]) - self.n[0] * self.n
        self.a = a / np.linalg.norm(a)
        self.b = np.cross(self.n, self.a)

    # ── geometry ──────────────────────────────────────────────────────────
    def _ray_plane(self, u: float, v: float, lift: float = 0.0) -> np.ndarray:
        """Intersect the pixel ray with the plane raised by *lift* along n."""
        ray = np.linalg.solve(self.K, np.array([u, v, 1.0]))
        t = (lift - self.h) / float(self.n @ ray)
        return t * ray

    def corners_3d(self, contact: np.ndarray, theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        ex = c * self.a + s * self.b          # cube edge direction 1
        ey = -s * self.a + c * self.b         # cube edge direction 2
        return (contact + self.s * (_CUBE[:, 0:1] * ex + _CUBE[:, 1:2] * ey + _CUBE[:, 2:3] * self.n))

    def project(self, X: np.ndarray) -> np.ndarray:
        uv = (self.K @ X.T).T
        return uv[:, :2] / uv[:, 2:3]

    def _contact(self, p: np.ndarray) -> np.ndarray:
        return self.origin + p[0] * self.a + p[1] * self.b

    def front_frame(self, contact: np.ndarray, theta: float):
        """Die frame for a cube at yaw *theta* (any of its 4 equivalent values):
        x = outward normal of the side face turned most towards the camera,
        z = table normal, y = z × x. Returns (R, yaw_of_x_deg in (-180, 180])."""
        centroid = contact + self.n * self.s / 2
        to_cam = -centroid - float(-centroid @ self.n) * self.n   # camera sits at the origin
        yaws = theta + np.arange(4) * np.pi / 2
        dirs = np.cos(yaws)[:, None] * self.a + np.sin(yaws)[:, None] * self.b
        k = int(np.argmax(dirs @ to_cam))
        x_axis = dirs[k]
        R_die = np.stack([x_axis, np.cross(self.n, x_axis), self.n], axis=1)
        yaw = float(np.degrees(np.arctan2(np.sin(yaws[k]), np.cos(yaws[k]))))
        return R_die, yaw

    # ── fitting ───────────────────────────────────────────────────────────
    def _iou(self, p) -> float:
        uv = self.project(self.corners_3d(self._contact(p), p[2])) - self.off
        hull = cv2.convexHull(np.round(uv).astype(np.int32))
        self._canvas[:] = 0
        cv2.fillConvexPoly(self._canvas, hull, 1)
        inter = np.count_nonzero(self._canvas & self._mask)
        union = self._mask_area + np.count_nonzero(self._canvas) - inter
        return inter / union if union else 0.0

    def _prepare(self, contour: np.ndarray, image_shape: tuple) -> np.ndarray:
        """Rasterise the die contour into the padded crop used by _iou."""
        cnt = contour.reshape(-1, 2)
        x0, y0, w, h = cv2.boundingRect(cnt)
        H, W = image_shape[:2]
        x0p, y0p = max(0, x0 - self.pad), max(0, y0 - self.pad)
        x1p, y1p = min(W, x0 + w + self.pad), min(H, y0 + h + self.pad)
        self.off = np.array([x0p, y0p], dtype=np.float64)
        self._mask = np.zeros((y1p - y0p, x1p - x0p), np.uint8)
        cv2.drawContours(self._mask, [(cnt - self.off).astype(np.int32)], -1, 1, -1)
        self._mask_area = int(np.count_nonzero(self._mask))
        self._canvas = np.zeros_like(self._mask)
        return cnt

    def score_pose(self, contour: np.ndarray, image_shape: tuple, contact: np.ndarray, theta: float) -> float:
        """Silhouette IoU of a given cube pose (e.g. from another estimator) against *contour*."""
        self._prepare(contour, image_shape)
        self.origin = np.asarray(contact, dtype=np.float64)
        return self._iou((0.0, 0.0, theta))

    def fit(self, contour: np.ndarray, image_shape: tuple, refine_top_k: int = 2) -> dict:
        """Fit the cube to a die contour (full-image pixel coordinates)."""
        t0 = time.perf_counter()
        cnt = self._prepare(contour, image_shape)
        x0, y0, w, h = cv2.boundingRect(cnt)

        # Initial contact point: ray through the mask centroid hits the die
        # centre at height s/2; drop it onto the plane.
        m = cv2.moments(cnt.astype(np.int32))
        u_c, v_c = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] > 0 else (x0 + w / 2, y0 + h / 2)
        self.origin = self._ray_plane(u_c, v_c, lift=self.s / 2) - self.n * self.s / 2

        n_evals = len(self.theta_grid)
        grid = [(self._iou((0.0, 0.0, th)), th) for th in self.theta_grid]
        best_p, best_iou = None, -1.0
        for _, th in sorted(grid, reverse=True)[:refine_top_k]:
            res = minimize(lambda p: -self._iou(p), [0.0, 0.0, th], method="Nelder-Mead",
                           options=dict(xatol=2e-4, fatol=1e-4, maxfev=200,
                                        initial_simplex=np.array([[0, 0, th], [0.004, 0, th],
                                                                  [0, 0.004, th], [0, 0, th + 0.05]])))
            n_evals += res.nfev
            if -res.fun > best_iou:
                best_iou, best_p = -res.fun, res.x

        contact = self._contact(best_p)
        theta = float(best_p[2])
        R_die, yaw_deg = self.front_frame(contact, theta)
        centroid = contact + self.n * self.s / 2

        return {
            "centroid": centroid,                           # die centre, camera frame [m]
            "contact": contact,                             # bottom-face centre on the table
            "theta_deg": yaw_deg,                           # yaw of x (front-face normal) w.r.t. camera +X
            "R": R_die,
            "quat": R_sci.from_matrix(R_die).as_quat(),     # [qx, qy, qz, qw]
            "iou": float(best_iou),
            "corners_2d": self.project(self.corners_3d(contact, theta)),
            "n_evals": n_evals,
            "time_ms": (time.perf_counter() - t0) * 1e3,
        }


def calibrate_plane_from_masks(contours, image_shape, fx_init, cx, cy, die_size,
                               tilt_init_deg=50.0, h_init=None, fit_fx=True, log=print):
    """Recover the table plane (and optionally fx) of a static camera from die
    masks at several table positions: the plane that lets a cube of the known
    size explain every silhouette best (mean IoU)."""
    def build(p):
        fx = p[3] if fit_fx else fx_init
        K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1.0]])
        n, h = plane_from_angles(np.radians(p[0]), np.radians(p[1]), p[2])
        return SilhouettePoseFitter(K, n, h, die_size, theta_step_deg=6.0)

    def cost(p):
        if not (10 < p[0] < 85 and p[2] > 0.05 and (not fit_fx or p[3] > 200)):
            return 1.0
        f = build(p)
        return -float(np.mean([f.fit(c, image_shape, refine_top_k=1)["iou"] for c in contours]))

    if h_init is None:
        # Die apparent size -> distance -> camera height above the table
        sizes = [max(cv2.boundingRect(c.reshape(-1, 2).astype(np.int32))[2:]) for c in contours]
        z = fx_init * die_size * 1.4 / float(np.median(sizes))
        h_init = z * np.sin(np.radians(tilt_init_deg))
    p0 = [tilt_init_deg, 0.0, h_init] + ([fx_init] if fit_fx else [])
    step = [8.0, 3.0, 0.3 * h_init] + ([0.15 * fx_init] if fit_fx else [])
    res = minimize(cost, p0, method="Nelder-Mead",
                   options=dict(maxfev=400, xatol=1e-3, fatol=1e-4,
                                initial_simplex=np.array([p0] + [np.array(p0) + np.eye(len(p0))[i] * step[i]
                                                                  for i in range(len(p0))])))
    p = res.x
    fx = float(p[3]) if fit_fx else float(fx_init)
    n, h = plane_from_angles(np.radians(p[0]), np.radians(p[1]), p[2])
    log(f"Plane calibration: tilt={p[0]:.1f}°, roll={p[1]:.1f}°, height={p[2]*1000:.0f} mm, "
        f"fx={fx:.0f}px, mean IoU={-res.fun:.3f} ({res.nfev} evals)")
    return {"fx": fx, "fy": fx, "cx": cx, "cy": cy, "plane_n": n.tolist(), "plane_h": h,
            "tilt_deg": float(p[0]), "roll_deg": float(p[1]), "mean_iou": float(-res.fun)}


# ── Visualisation ─────────────────────────────────────────────────────────
AXIS_COLORS = [(0, 0, 255), (0, 200, 0), (255, 80, 0)]   # x red, y green, z blue (BGR)


def draw_die_pose(img, fitter: SilhouettePoseFitter, pose: dict, contour=None,
                  scale: float = 1.0, off=(0.0, 0.0), thick: int = 2) -> None:
    """Draw the die outline (yellow), fitted cube wireframe (magenta) and TF
    axes (x red, y green, z blue) in place. Pixel coordinates are mapped by
    (p - off) * scale, so the same call can draw into a zoomed crop."""
    tf = lambda uv: tuple(np.round((np.asarray(uv) - off) * scale).astype(int))
    if contour is not None:
        cnt = np.round((np.asarray(contour).reshape(-1, 2) - off) * scale).astype(np.int32)
        cv2.polylines(img, [cnt], True, (0, 255, 255), max(1, thick - 1), cv2.LINE_AA)
    c2d = pose["corners_2d"]
    for a, b in CUBE_EDGES:
        cv2.line(img, tf(c2d[a]), tf(c2d[b]), (255, 0, 255), thick, cv2.LINE_AA)
    o = np.asarray(pose["centroid"], dtype=np.float64)
    o2d = fitter.project(o[None])[0]
    for k in range(3):
        tip = fitter.project((o + pose["R"][:, k] * fitter.s * 1.3)[None])[0]
        cv2.arrowedLine(img, tf(o2d), tf(tip), AXIS_COLORS[k], thick + 1, cv2.LINE_AA, tipLength=0.2)


def compose_debug_panel(full: np.ndarray, zoom: np.ndarray, lines: list, ok: bool) -> np.ndarray:
    """Debug image: full camera image on top; below it the zoomed die panel
    with the detection info written to its right (large font)."""
    W = full.shape[1]
    zh = zoom.shape[0]
    bottom = np.full((zh, W, 3), 30, np.uint8)
    zw = min(zoom.shape[1], W)
    bottom[:, :zw] = zoom[:, :zw]
    colour = (255, 255, 255) if ok else (80, 80, 255)
    x0, y = zw + 24, 46
    for i, text in enumerate(lines):
        scale, thick = (1.1, 3) if i == 0 else (0.9, 2)
        cv2.putText(bottom, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.putText(bottom, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    colour if i == 0 else (230, 230, 230), thick, cv2.LINE_AA)
        y += 52 if i == 0 else 42
    return np.vstack([full, bottom])
