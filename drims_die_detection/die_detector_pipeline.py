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
        self._plotly = PlotlyVisualizer(self.params)

        if self.params.debug:
            self.params.log()

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

        # ── Step 2: RANSAC Plane ──────────────────────────────────────────
        plane_model, inliers, outliers, normal = self._pc_proc.fit_plane_ransac(points)
        A_p, B_p, C_p, D_p = plane_model
        self._log(f"Step 2: Plane normal={normal}, D={D_p:.4f}.")

        # ── Step 3: 2D RGB Detection ──────────────────────────────────────
        det = self._rgb_det.detect(rgb_bgr)
        x_c, y_c, w_c, h_c = det["bbox"]
        faces = det["faces"]
        num_visible_faces = det["num_visible_faces"]
        total_pips = det["total_pips"]
        self._log(f"Step 3: {num_visible_faces} faces, {total_pips} pips (2D).")

        # ── Step 4: Die Point Segmentation ────────────────────────────────
        heights = np.dot(points, normal) + D_p
        px = pixel_coords[:, 0]
        py = pixel_coords[:, 1]
        inside = (
            (px >= x_c) & (px < x_c + w_c) &
            (py >= y_c) & (py < y_c + h_c) &
            (heights >= self.params.min_die_height_m) &
            (heights <= self.params.max_die_height_m)
        )
        die_indices = np.where(inside)[0]

        if len(die_indices) < 10:
            self._log("Height-filtered blob too small — using full 2D bbox fallback.")
            inside = (px >= x_c) & (px < x_c + w_c) & (py >= y_c) & (py < y_c + h_c)
            die_indices = np.where(inside)[0]

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

        lat_faces = [f for f in faces if not f.get("is_top_face")]
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

        top_pips = top_face["num_pips"] if top_face is not None else total_pips
        x_pos_pips = primary_lat["num_pips"] if primary_lat is not None else None
        y_pos_pips = lat_faces[1]["num_pips"] if len(lat_faces) >= 2 else None
        die_orient = resolve_die_orientation(top_pips=top_pips, x_pos_pips=x_pos_pips, y_pos_pips=y_pos_pips)

        top_face_tf = centroid.copy()
        die_centroid_tf = centroid - (die_size / 2.0) * z_ax

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

        # Free-area arrow and "TOP FACE" badge for Panel 6 (with larger font)
        h_full, w_full = rgb_bgr.shape[:2]
        free_x = int(w_full * 0.82) if u_top < w_full / 2.0 else int(w_full * 0.18)
        free_y = int(h_full * 0.82) if v_top < h_full / 2.0 else int(h_full * 0.18)
        label_pos = (free_x, free_y)

        # Draw high-contrast arrow from free area towards die top face
        cv2.arrowedLine(ann_rgb, label_pos, c_2d, (0, 0, 0), 5, tipLength=0.03, line_type=cv2.LINE_AA)
        cv2.arrowedLine(ann_rgb, label_pos, c_2d, (0, 255, 255), 3, tipLength=0.03, line_type=cv2.LINE_AA)

        # Draw larger "TOP FACE" text badge at label_pos
        text = "TOP FACE"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

        box_x1 = max(5, free_x - tw // 2 - 14)
        box_y1 = max(5, free_y - th // 2 - 10)
        box_x2 = min(w_full - 5, free_x + tw // 2 + 14)
        box_y2 = min(h_full - 5, free_y + th // 2 + 10)

        cv2.rectangle(ann_rgb, (box_x1, box_y1), (box_x2, box_y2), (15, 15, 15), -1)
        cv2.rectangle(ann_rgb, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 255), 2)
        cv2.putText(ann_rgb, text, (box_x1 + 14, box_y2 - 10), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

        self._log(
            f"Step 8: Top-face TF = ({top_face_tf[0]:.3f}, {top_face_tf[1]:.3f}, {top_face_tf[2]:.3f}) m"
        )

        result = {
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
        step2_mask = ds["step2_whitest_mask"]
        step2_hull = ds["step2_whitest_convex_hull"]
        bw_mask = ds["step4_bw_mask"]
        dark_mask = ds["step4_dark_pips_mask"]
        faces_pips = ds["step4_faces_and_pips"]

        pw = w // 3
        ph = h // 2

        def _panel(img, label, color=(255, 255, 255)):
            p = cv2.resize(img, (pw, ph))
            cv2.putText(p, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            return p

        p1 = _panel(step1_img, "1. Color Clusters (K-Means)")
        p2 = _panel(step2_mask, "2. Whitest Cluster", (0, 255, 255))
        p3 = _panel(step2_hull, "3. Whitest Convex Hull", (0, 255, 0))

        bw_dark = np.hstack((
            cv2.resize(bw_mask,   (pw // 2, ph)),
            cv2.resize(dark_mask, (pw - pw // 2, ph)),
        ))
        cv2.putText(bw_dark, "4. B&W & Pips Masks", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        p5_canvas = np.zeros((ph, pw, 3), dtype=np.uint8)
        fp_resized = cv2.resize(faces_pips, (min(pw, faces_pips.shape[1] * 2),
                                             min(ph, faces_pips.shape[0] * 2)))
        sh, sw = fp_resized.shape[:2]
        p5_canvas[:sh, :sw] = fp_resized
        cv2.putText(p5_canvas, "5. Detected Faces & Pips", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        p6 = _panel(ann_rgb, "6. 3D Pose Frame")

        row1 = np.hstack((p1, p2, p3))
        row2 = np.hstack((bw_dark, p5_canvas, p6))
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
        os.makedirs(self.params.output_dir, exist_ok=True)
        img_name = result.get("image_name", "image")
        stem = os.path.splitext(img_name)[0]

        # 6-panel collage
        collage_path = os.path.join(self.params.output_dir, f"detected_{img_name}")
        cv2.imwrite(collage_path, collage)
        self._log(f"Saved collage → {collage_path}")

        # Depth files (raw .npy + colourmap PNG)
        if depth_m is not None:
            depth_npy_path = os.path.join(self.params.output_dir, f"depth_{stem}.npy")
            np.save(depth_npy_path, depth_m)
            self._log(f"Saved raw depth → {depth_npy_path}")

            depth_vis = cv2.normalize(depth_m, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)
            depth_png_path = os.path.join(self.params.output_dir, f"depth_{stem}.png")
            cv2.imwrite(depth_png_path, depth_color)
            self._log(f"Saved depth colourmap → {depth_png_path}")

        # Plotly HTML
        plotly_path = os.path.join(self.params.output_dir, f"plotly_3d_{stem}.html")
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
