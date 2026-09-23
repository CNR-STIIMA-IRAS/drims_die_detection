"""
DieDetectorPipeline
===================
Full 3D die detection pipeline orchestrator — works without ROS.

Usage
-----
::

    from drims_die_detection import DieDetectorPipeline, DieDetectorParams

    params = DieDetectorParams.from_yaml("config/die_detector_params.yaml")
    pipeline = DieDetectorPipeline(params)

    # With real depth (metres, float32):
    result = pipeline.process_rgbd(rgb_bgr, depth_m)

    # With only an RGB image (monocular fallback):
    result = pipeline.process_rgb(rgb_bgr)

    # Build the 6-panel debug collage:
    collage = pipeline.build_debug_panels(rgb_bgr, result)

Result dict keys
----------------
pip_count, num_visible_faces, faces,
centroid, quaternion, R_die, axes,
top_face_tf, die_centroid_tf,
plane_model, normal,
die_points, points, colors,
debug_steps (from RGBDieDetector),
top_down_annotated (pip-annotated top-down crop)
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from .die_detector_params import DieDetectorParams
from .depth_estimator import MonocularDepthEstimator
from .point_cloud_processor import PointCloudProcessor
from .rgb_die_detector import RGBDieDetector
from .alignment_projection import AlignmentAndProjection
from .pip_detector import PipDetector
from .pose_estimator import PoseEstimator
from .pose_stabilizer import PoseStabilizer
from .plotly_visualizer import PlotlyVisualizer
from .die_orientation import resolve_die_orientation


class DieDetectorPipeline:
    """Orchestrates the full 3D die detection pipeline.

    Parameters
    ----------
    params : DieDetectorParams | None
        Pipeline configuration. If None, defaults are used.
    """

    def __init__(self, params: DieDetectorParams | None = None) -> None:
        self.params = params or DieDetectorParams()
        self._log = self._make_logger()

        self._depth_est: MonocularDepthEstimator | None = None
        self._pc_proc = PointCloudProcessor(self.params)
        self._rgb_det = RGBDieDetector(self.params)
        self._align = AlignmentAndProjection(self.params)
        self._pip_det = PipDetector(self.params)
        self._pose_est = PoseEstimator(self.params)
        self._stabilizer = PoseStabilizer(self.params)
        self._plotly = PlotlyVisualizer(self.params)

        if self.params.debug:
            self.params.log()

    def reset_tracking(self) -> None:
        """Reset temporal pose filtering and tracking history."""
        self._stabilizer.reset()
        self._pose_est.reset_tracking()

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def process_rgb(self, rgb_bgr: np.ndarray, image_name: str = "image") -> dict:
        """Run the full pipeline using monocular depth estimation."""
        self._log(f"Running monocular depth estimation for '{image_name}'.")
        if self._depth_est is None:
            self._depth_est = MonocularDepthEstimator(self.params)
        depth_m = self._depth_est.infer_depth(rgb_bgr)
        return self.process_rgbd(rgb_bgr, depth_m, image_name=image_name)

    def process_rgbd(
        self,
        rgb_bgr: np.ndarray,
        depth_m: np.ndarray,
        image_name: str = "image",
    ) -> dict:
        """Run the full pipeline with a provided metric depth map.

        Parameters
        ----------
        rgb_bgr : (H, W, 3) uint8 BGR
        depth_m : (H, W) float32 — metric depth in metres
        image_name : str — used when saving output files

        Returns
        -------
        dict with detection results (see module docstring).
        """
        if depth_m.ndim == 3:
            depth_m = depth_m[:, :, 0]
        h, w = rgb_bgr.shape[:2]
        self._log(f"process_rgbd: image={image_name} ({w}×{h})")

        # Monocular fallback if depth is all zeros or missing
        if self.params.use_monocular_fallback:
            valid_depth = np.isfinite(depth_m) & (depth_m > 0.1)
            if valid_depth.sum() < 100:
                self._log("Depth map has too few valid pixels — invoking monocular fallback.")
                depth_m = self._depth_est.infer_depth(rgb_bgr)

        # ── Step 1: Point Cloud ───────────────────────────────────────────
        points, colors, pixel_coords, (cx, cy) = self._pc_proc.create_point_cloud(rgb_bgr, depth_m)
        self._log(f"Step 1: {len(points)} valid 3D points.")

        # ── Step 2: 2D RGB Detection ──────────────────────────────────────
        det = self._rgb_det.detect(rgb_bgr)
        x_c, y_c, w_c, h_c = det["bbox"]
        faces = det["faces"]
        num_visible_faces = det["num_visible_faces"]
        total_pips = det["total_pips"]
        self._log(f"Step 2: {num_visible_faces} faces, {total_pips} pips (2D).")

        # ── Step 3: RANSAC Plane on Table Surface (Excluding Die Points) ──
        px = pixel_coords[:, 0]
        py = pixel_coords[:, 1]
        is_die_2d = (px >= x_c) & (px < x_c + w_c) & (py >= y_c) & (py < y_c + h_c)
        table_points = points[~is_die_2d]
        if len(table_points) < 50:
            table_points = points

        plane_model, inliers, outliers, normal = self._pc_proc.fit_plane_ransac(table_points)
        A_p, B_p, C_p, D_p = plane_model

        # Temporal plane smoothing
        normal, D_p = self._stabilizer.filter_plane(normal, D_p)
        plane_model = (float(normal[0]), float(normal[1]), float(normal[2]), float(D_p))
        A_p, B_p, C_p, D_p = plane_model
        self._log(f"Step 3: Plane normal={normal}, D={D_p:.4f} (fit on {len(table_points)} background points).")

        # ── Step 4: Die Point Segmentation ────────────────────────────────
        heights = np.dot(points, normal) + D_p
        inside = (
            is_die_2d &
            (heights >= self.params.min_die_height_m) &
            (heights <= self.params.max_die_height_m)
        )
        die_indices = np.where(inside)[0]

        if len(die_indices) < 10:
            self._log("Height-filtered blob too small — using full 2D bbox fallback.")
            die_indices = np.where(is_die_2d)[0]

        die_points = points[die_indices] if len(die_indices) > 0 else points
        self._log(f"Step 4: {len(die_points)} die blob points.")

        # ── Step 5: Top Face ──────────────────────────────────────────────
        top_face = None
        min_y_cen = 1e9
        for f in faces:
            M = cv2.moments(f["polygon"])
            cy_poly = M["m01"] / M["m00"] if M["m00"] > 0 else np.mean(f["polygon"][:, 0, 1])
            if cy_poly < min_y_cen:
                min_y_cen = cy_poly
                top_face = f

        die_size = self.params.die_size_m
        u_top: float = cx
        v_top: float = cy

        if top_face is not None:
            poly_full = top_face["polygon"] + np.array([x_c, y_c])
            M_top = cv2.moments(poly_full)
            if M_top["m00"] > 0:
                u_top = M_top["m10"] / M_top["m00"]
                v_top = M_top["m01"] / M_top["m00"]
            else:
                u_top = float(np.mean(poly_full[:, 0, 0]))
                v_top = float(np.mean(poly_full[:, 0, 1]))

        # ── Step 6: 3D Alignment ──────────────────────────────────────────
        R_plane = self._align.get_rotation_to_z(normal)
        self._log("Step 6: Computed R_plane.")

        # ── Step 7: Top-Down Crop ─────────────────────────────────────────
        top_down_crop, raw_crop, bbox_3d, yaw_angle = self._align.extract_top_down_rgb(
            rgb_bgr, die_points, pixel_coords, die_indices, R_plane, crop_size=300
        )

        # ── Step 8: Pose Estimation & Orientation Resolution ──────────────
        top_poly_full = (top_face["polygon"] + np.array([x_c, y_c])) if top_face is not None else None
        cam_params = (self.params.fx, self.params.fy, cx, cy)

        def _cy_face(f):
            M = cv2.moments(f["polygon"])
            return M["m01"] / M["m00"] if M["m00"] > 0 else float(np.mean(f["polygon"][:, 0, 1]))

        lat_faces = [f for f in faces if not f.get("is_top_face")]
        lat_faces = sorted(lat_faces, key=_cy_face)
        primary_lat = lat_faces[0] if len(lat_faces) > 0 else None
        primary_lat_2d = None
        if primary_lat is not None:
            poly_lat_full = primary_lat["polygon"] + np.array([x_c, y_c])
            M_lat = cv2.moments(poly_lat_full)
            if M_lat["m00"] > 0:
                u_lat = M_lat["m10"] / M_lat["m00"]
                v_lat = M_lat["m01"] / M_lat["m00"]
            else:
                u_lat = float(np.mean(poly_lat_full[:, 0, 0]))
                v_lat = float(np.mean(poly_lat_full[:, 0, 1]))
            primary_lat_2d = (u_lat, v_lat)

        centroid, quat, R_die, (x_ax, y_ax, z_ax) = self._pose_est.compute_pose(
            die_points=die_points,
            plane_normal=normal,
            R_plane=R_plane,
            top_face_polygon=top_poly_full,
            top_face_centroid_2d=(u_top, v_top),
            primary_lat_centroid_2d=primary_lat_2d,
            plane_D=D_p,
            die_size=die_size,
            camera_params=cam_params,
        )

        # Apply temporal pose stabilization (filtering centroid & orientation)
        centroid, quat, R_die, (x_ax, y_ax, z_ax) = self._stabilizer.filter_pose(centroid, quat)

        top_pips = top_face["num_pips"] if (top_face is not None and top_face.get("is_trusted", True)) else total_pips
        x_pos_pips = primary_lat["num_pips"] if (primary_lat is not None and primary_lat.get("is_trusted", True)) else None
        y_pos_pips = lat_faces[1]["num_pips"] if (len(lat_faces) >= 2 and lat_faces[1].get("is_trusted", True)) else None
        die_orient = resolve_die_orientation(top_pips=top_pips, x_pos_pips=x_pos_pips, y_pos_pips=y_pos_pips)

        top_face_tf = centroid.copy()
        die_centroid_tf = centroid - (die_size / 2.0) * z_ax
        table_surface_tf = centroid - die_size * z_ax

        # ── Step 9: Annotated RGB ─────────────────────────────────────────
        ann_rgb = rgb_bgr.copy()

        def _proj(pt):
            px_ = int(pt[0] * self.params.fx / pt[2] + cx)
            py_ = int(pt[1] * self.params.fy / pt[2] + cy)
            return (px_, py_)

        c_2d = (int(u_top), int(v_top))
        arm = 0.10
        cv2.line(ann_rgb, c_2d, _proj(centroid + arm * x_ax), (0, 0, 255), 5)
        cv2.line(ann_rgb, c_2d, _proj(centroid + arm * y_ax), (0, 255, 0), 5)
        cv2.line(ann_rgb, c_2d, _proj(centroid + arm * z_ax), (255, 0, 0), 5)
        cv2.circle(ann_rgb, c_2d, 8, (0, 0, 0), -1)
        cv2.circle(ann_rgb, c_2d, 5, (255, 255, 255), -1)

        # Validate Top Face and Front Face according to min_pips and max_pips parameters
        min_p = getattr(self.params, "min_pips", 1)
        max_p = getattr(self.params, "max_pips", 6)

        def _is_valid(f):
            if f is None:
                return False
            if not f.get("is_trusted", True) or not f.get("aligned_pips", True):
                return False
            n = f.get("num_pips", 0)
            return min_p <= n <= max_p

        top_valid = _is_valid(top_face)
        front_valid = _is_valid(primary_lat)

        top_pips = top_face.get("num_pips", 0) if top_valid else None
        front_pips = primary_lat.get("num_pips", 0) if front_valid else None

        # Apply pip temporal consensus filtering
        top_pips, front_pips = self._stabilizer.filter_pips(top_pips, front_pips)

        top_str = str(top_pips) if top_pips is not None else "None"
        front_str = str(front_pips) if front_pips is not None else "None"

        badge_text = f"TOP FACE: {top_str} | FRONT FACE: {front_str}"

        # Render larger, high-contrast badge horizontally centered wrt image/panel center
        h_full, w_full = rgb_bgr.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.1
        thickness = 3

        (tw, th), baseline = cv2.getTextSize(badge_text, font, font_scale, thickness)
        max_b_w = w_full - 30
        if tw > max_b_w:
            font_scale = font_scale * (max_b_w / float(tw))
            (tw, th), baseline = cv2.getTextSize(badge_text, font, font_scale, thickness)

        pad_x = 24
        pad_y = 16
        box_w = tw + 2 * pad_x
        box_h = th + 2 * pad_y

        center_x = w_full // 2
        box_x1 = max(5, center_x - box_w // 2)
        box_x2 = min(w_full - 5, center_x + box_w // 2)

        box_y2 = h_full - 25
        box_y1 = max(5, box_y2 - box_h)

        # High contrast background rectangle & border
        cv2.rectangle(ann_rgb, (box_x1, box_y1), (box_x2, box_y2), (15, 15, 15), -1)
        cv2.rectangle(ann_rgb, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 255), 3)

        # Centered crisp text
        text_x = box_x1 + (box_w - tw) // 2
        text_y = box_y2 - pad_y
        cv2.putText(ann_rgb, badge_text, (text_x, text_y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

        # Arrow from top center of badge box pointing to die top-face 2D centroid
        arrow_start = (center_x, box_y1)
        cv2.arrowedLine(ann_rgb, arrow_start, c_2d, (0, 0, 0), 5, tipLength=0.03, line_type=cv2.LINE_AA)
        cv2.arrowedLine(ann_rgb, arrow_start, c_2d, (0, 255, 255), 3, tipLength=0.03, line_type=cv2.LINE_AA)

        self._log(
            f"Step 8: Top-face TF = ({top_face_tf[0]:.3f}, {top_face_tf[1]:.3f}, {top_face_tf[2]:.3f}) m"
        )

        result = {
            # Identification & validation
            "top_face_str": top_str,
            "front_face_str": front_str,
            "top_face_pips": top_pips,
            "front_face_pips": front_pips,
            "top_face_valid": top_valid,
            "front_face_valid": front_valid,
            # Detection
            "image_name": image_name,
            "pip_count": total_pips,
            "num_visible_faces": num_visible_faces,
            "faces": faces,
            "top_face": top_face,
            "die_orientation": die_orient,
            # Pose
            "centroid": centroid,
            "quaternion": quat,
            "R_die": R_die,
            "axes": (x_ax, y_ax, z_ax),
            "top_face_tf": top_face_tf,
            "die_centroid_tf": die_centroid_tf,
            "table_surface_tf": table_surface_tf,
            # 3D
            "plane_model": plane_model,
            "normal": normal,
            "die_points": die_points,
            "points": points,
            "colors": colors,
            # Images
            "annotated_rgb": ann_rgb,
            "top_down_crop": top_down_crop,
            "debug_steps": det["debug_steps"],
            "bbox": det["bbox"],
            "depth_m": depth_m,
        }
        return result

    # ──────────────────────────────────────────────────────────────────────
    def build_debug_panels(
        self,
        rgb_bgr: np.ndarray,
        result: dict,
    ) -> np.ndarray:
        """Build a 3×2 six-panel debug collage image.

        Parameters
        ----------
        rgb_bgr : original full BGR image
        result : dict returned by process_rgbd / process_rgb

        Returns
        -------
        collage : (H, W, 3) uint8 BGR — suitable for cv2.imshow or ROS publish
        """
        h, w = rgb_bgr.shape[:2]
        ds = result["debug_steps"]
        ann_rgb = result["annotated_rgb"]

        step1_img = ds["step1_color_clustered"]
        step2_edges = ds.get("step2_edges", ds.get("step2_whitest_mask"))
        step2_hull = ds["step2_whitest_convex_hull"]
        bw_mask = ds["step4_bw_mask"]
        dark_mask = ds["step4_dark_pips_mask"]
        faces_pips = ds["step4_faces_and_pips"]

        # Fixed high-definition panel canvas resolution for 100% consistent text sizing
        pw = 640
        ph = 480

        font = cv2.FONT_HERSHEY_SIMPLEX
        title_color = (0, 255, 0)

        def _panel(img, label):
            p = cv2.resize(img, (pw, ph))
            # Auto-fit title text so it never overflows panel boundary
            scale = 0.85
            thick = 2
            max_w = pw - 30
            (tw, _), _ = cv2.getTextSize(label, font, scale, thick)
            if tw > max_w:
                scale = scale * (max_w / float(tw))
            cv2.putText(p, label, (15, 40), font, scale, title_color, thick, cv2.LINE_AA)
            return p

        mode_str = getattr(self.params, "detection_mode", "hsv").upper()

        p1 = _panel(step1_img, f"1. Color Segment ({mode_str})")
        p2 = _panel(step2_edges, "2. Edge Detection")
        p3 = _panel(step2_hull, "3. Convex Hull BBox")

        p4_bw = cv2.resize(bw_mask, (pw // 2, ph))
        p4_dark = cv2.resize(dark_mask, (pw - pw // 2, ph))
        p4_img = np.hstack((p4_bw, p4_dark))
        p4 = _panel(p4_img, "4. Face & Pip Masks")

        p5 = _panel(faces_pips, "5. Detected Faces & Pips")
        p6 = _panel(ann_rgb, "6. 3D Pose Frame")

        row1 = np.hstack((p1, p2, p3))
        row2 = np.hstack((p4, p5, p6))
        collage = np.vstack((row1, row2))
        return collage

    # ──────────────────────────────────────────────────────────────────────
    def save_outputs(
        self,
        result: dict,
        collage: np.ndarray,
        depth_m: np.ndarray | None = None,
    ) -> None:
        """Save the 6-panel collage, the depth colourmap, and the Plotly HTML."""
        out_dir = self.params.resolved_output_dir
        os.makedirs(out_dir, exist_ok=True)
        img_name = result.get("image_name", "image")
        stem = os.path.splitext(img_name)[0]

        # 6-panel collage
        collage_path = os.path.join(out_dir, f"detected_{img_name}")
        cv2.imwrite(collage_path, collage)
        self._log(f"Saved collage → {collage_path}")

        # Depth files (raw .npy + colourmap PNG)
        if depth_m is not None:
            depth_npy_path = os.path.join(out_dir, f"depth_{stem}.npy")
            np.save(depth_npy_path, depth_m)
            self._log(f"Saved raw depth → {depth_npy_path}")

            depth_vis = cv2.normalize(depth_m, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)
            depth_png_path = os.path.join(out_dir, f"depth_{stem}.png")
            cv2.imwrite(depth_png_path, depth_color)
            self._log(f"Saved depth colourmap → {depth_png_path}")

        # Plotly HTML
        plotly_path = os.path.join(out_dir, f"plotly_3d_{stem}.html")
        x_ax, y_ax, z_ax = result["axes"]
        self._plotly.visualize(
            points=result["points"],
            colors=result["colors"],
            die_points=result["die_points"],
            plane_model=result["plane_model"],
            centroid=result["top_face_tf"],
            die_centroid=result["die_centroid_tf"],
            die_axes=(x_ax, y_ax, z_ax),
            die_size=self.params.die_size_m,
            title=f"3D Point Cloud & Die Transforms ({img_name})",
            output_html_path=plotly_path,
        )

    # ──────────────────────────────────────────────────────────────────────
    def process_image_file(self, image_path: str) -> dict:
        """Convenience wrapper: read RGB file, infer depth, process, and save."""
        rgb = cv2.imread(image_path)
        if rgb is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        img_name = os.path.basename(image_path)
        depth_m = self._depth_est.infer_depth(rgb)
        result = self.process_rgbd(rgb, depth_m, image_name=img_name)
        if self.params.save:
            collage = self.build_debug_panels(rgb, result)
            self.save_outputs(result, collage, depth_m)
        if self.params.visualize:
            collage = self.build_debug_panels(rgb, result)
            cv2.imshow("Die Detection Pipeline", collage)
            cv2.waitKey(1)
        return result

    # ──────────────────────────────────────────────────────────────────────
    def _make_logger(self):
        tag = "[DieDetectorPipeline]"
        if self.params.debug:
            return lambda msg: print(f"{tag} {msg}")
        return lambda msg: None
