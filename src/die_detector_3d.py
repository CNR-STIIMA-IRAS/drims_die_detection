#!/usr/bin/env python3
"""
Angled Camera Die Detection Pipeline (ROS-Free Python Prototype)

This script processes RGB images of a scene containing a die resting on a flat surface,
captured by an angled (non-vertical) camera.

Pipeline Steps:
1. Depth Inference: Generates relative/metric depth map using monocular depth estimation.
2. 3D Point Cloud Generation: Converts RGB + Depth into a 3D point cloud via camera intrinsics.
3. RANSAC Plane Fitting: Identifies the table/surface plane and extracts normal vector n.
4. Segmentation & DBSCAN Clustering: Filters out table inliers and isolates points resting above plane (die).
5. 3D Rotation Alignment: Computes rotation matrix R aligning plane normal to Z-axis [0, 0, 1].
6. Top-Down Orthographic Projection: Warps angled RGB view of die face into an undistorted top-down crop.
7. Pip Detection: Uses adaptive thresholding and contour/blob filtering to count die pips.
8. 6DOF Pose Estimation: Computes 3D translation (centroid) and rotation quaternion (normal + yaw).
"""

import os
import glob
import math
import numpy as np
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation as R_sci

# Optional dependencies check
try:
    import torch
    from transformers import pipeline as hf_pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class MonocularDepthEstimator:
    """Infers depth map from RGB image using HuggingFace DPT or heuristic fallback."""
    def __init__(self, model_name="Intel/dpt-hybrid-midas"):
        self.has_model = False
        if HAS_TRANSFORMERS:
            try:
                device = 0 if torch.cuda.is_available() else -1
                print(f"[DepthEstimator] Loading model {model_name} on device={device}...")
                self.pipe = hf_pipeline(task="depth-estimation", model=model_name, device=device)
                self.has_model = True
                print("[DepthEstimator] Model loaded successfully.")
            except Exception as e:
                print(f"[DepthEstimator] Could not load {model_name}: {e}. Using fallback depth estimator.")
        else:
            print("[DepthEstimator] Transformers/Torch not found. Using fallback depth estimator.")

    def infer_depth(self, rgb_bgr: np.ndarray) -> np.ndarray:
        """
        Generates relative depth map from RGB image.
        Returns float32 numpy array normalized to simulated distance in meters (e.g. 0.5m - 2.0m).
        """
        h, w = rgb_bgr.shape[:2]
        if self.has_model:
            rgb_pil = Image.fromarray(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB))
            depth_result = self.pipe(rgb_pil)
            depth_map = np.array(depth_result["depth"], dtype=np.float32)
            # Resize if necessary
            if depth_map.shape[:2] != (h, w):
                depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_CUBIC)
            # Invert if model outputs disparity/inverse depth (DPT outputs relative depth/disparity)
            depth_min, depth_max = depth_map.min(), depth_map.max()
            if depth_max > depth_min:
                depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
            else:
                depth_norm = np.ones((h, w), dtype=np.float32)
            # Map normalized disparity to plausible distance range (e.g., 0.5 to 1.5 meters)
            depth_meters = 0.5 + (1.0 - depth_norm) * 1.0
            return depth_meters.astype(np.float32)
        else:
            # Fallback heuristic depth generator (smooth gradient simulating angled view)
            print("[DepthEstimator] Generating synthetic angled depth map...")
            y_indices, x_indices = np.indices((h, w), dtype=np.float32)
            # Simulating angled view where bottom of image is closer (0.6m) and top is farther (1.2m)
            depth_meters = 0.6 + (y_indices / h) * 0.6
            return depth_meters.astype(np.float32)


class PointCloudProcessor:
    """Generates 3D point cloud from RGB+Depth and performs RANSAC plane fitting & clustering."""
    def __init__(self, fx=500.0, fy=500.0, cx=None, cy=None):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def create_point_cloud(self, rgb_bgr: np.ndarray, depth_map: np.ndarray):
        """Converts RGB and Depth map to 3D point cloud arrays."""
        h, w = depth_map.shape
        cx = self.cx if self.cx is not None else w / 2.0
        cy = self.cy if self.cy is not None else h / 2.0

        y_grid, x_grid = np.indices((h, w), dtype=np.float32)
        z_3d = depth_map.astype(np.float32)
        x_3d = (x_grid - cx) * z_3d / self.fx
        y_3d = (y_grid - cy) * z_3d / self.fy

        points = np.stack((x_3d, y_3d, z_3d), axis=-1).reshape(-1, 3)
        colors = rgb_bgr.reshape(-1, 3)[:, ::-1] / 255.0  # BGR to RGB [0, 1]
        pixel_coords = np.stack((x_grid, y_grid), axis=-1).reshape(-1, 2).astype(np.int32)

        # Filter out invalid depth points
        valid_mask = (z_3d.reshape(-1) > 0.1) & np.isfinite(z_3d.reshape(-1))
        return points[valid_mask], colors[valid_mask], pixel_coords[valid_mask], (cx, cy)

    def fit_plane_ransac(self, points: np.ndarray, distance_threshold=0.015, max_iterations=1000):
        """
        RANSAC plane fitting: finds plane Ax + By + Cz + D = 0.
        Returns (A, B, C, D), inlier_indices, outlier_indices, normal_vector.
        """
        if HAS_OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                                     ransac_n=3,
                                                     num_iterations=max_iterations)
            A, B, C, D = plane_model
            all_indices = set(range(len(points)))
            outliers = list(all_indices - set(inliers))
            normal = np.array([A, B, C], dtype=np.float32)
            norm_val = np.linalg.norm(normal)
            if norm_val > 0:
                normal /= norm_val
                D /= norm_val
            # Ensure normal points towards camera (negative Z component in camera frame)
            if normal[2] > 0:
                normal = -normal
                D = -D
            return (normal[0], normal[1], normal[2], D), inliers, outliers, normal

        # Pure NumPy/SciPy RANSAC fallback
        num_points = len(points)
        best_inliers = []
        best_plane = None

        # Downsample points for speed if cloud is large
        sample_indices = np.random.choice(num_points, size=min(10000, num_points), replace=False)
        sub_points = points[sample_indices]

        for _ in range(max_iterations):
            # Sample 3 random non-collinear points
            idx = np.random.choice(len(sub_points), 3, replace=False)
            p1, p2, p3 = sub_points[idx]
            v1, v2 = p2 - p1, p3 - p1
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-6:
                continue
            n /= norm_n
            d = -np.dot(n, p1)

            # Compute distances for all points
            distances = np.abs(np.dot(sub_points, n) + d)
            inliers = np.where(distances < distance_threshold)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane = (n[0], n[1], n[2], d)

        if best_plane is None:
            # Fallback horizontal plane
            return (0.0, 0.0, 1.0, -1.0), [], list(range(num_points)), np.array([0.0, 0.0, 1.0])

        n_vec = np.array(best_plane[:3], dtype=np.float32)
        d_val = best_plane[3]

        # Re-evaluate on full set of points
        distances_all = np.abs(np.dot(points, n_vec) + d_val)
        inliers_all = np.where(distances_all < distance_threshold)[0]
        outliers_all = np.where(distances_all >= distance_threshold)[0]

        if n_vec[2] > 0:
            n_vec = -n_vec
            d_val = -d_val

        return (n_vec[0], n_vec[1], n_vec[2], d_val), inliers_all, outliers_all, n_vec

class RGBDieDetector:
    """
    Detects a white die with black borders and pips using a 5-step color clustering & polygon segmentation approach:
    Step 1: Cluster the scene based on colors (K-Means color quantization in LAB space).
    Step 2: Detect the whitest cluster and compute its Convex Hull.
    Step 3: Crop the image to the bounding box surrounding the blob.
    Step 4: Cluster in black & white colors and perform contour/edge detection to segment faces and pips.
    Step 5: Return how many faces are visible and which ones (with pip count per face).
    """

    @staticmethod
    def fit_quadrilateral(fc):
        """Fits a 4-sided convex quadrilateral polygon to a face region contour."""
        hull = cv2.convexHull(fc)
        arc_len = cv2.arcLength(hull, True)
        if arc_len == 0:
            return None

        # 1. Try varying epsilon in approxPolyDP to find an approximation with exactly 4 convex vertices
        best_quad = None
        for eps_factor in np.linspace(0.015, 0.15, 40):
            approx = cv2.approxPolyDP(hull, eps_factor * arc_len, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                best_quad = approx
                break

        # 2. If no single epsilon yielded exactly 4 convex vertices, pick 4 strongest corners from N-gon
        if best_quad is None:
            approx = cv2.approxPolyDP(hull, 0.02 * arc_len, True)
            pts = approx.reshape(-1, 2)
            if len(pts) >= 4:
                max_area = -1.0
                for idxs in combinations(range(len(pts)), 4):
                    quad = pts[list(idxs)].reshape(-1, 1, 2)
                    if cv2.isContourConvex(quad):
                        area = cv2.contourArea(quad)
                        if area > max_area:
                            max_area = area
                            best_quad = quad

        # 3. Fallback: Minimum area bounding box corners (always 4-sided quadrilateral)
        if best_quad is None:
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            best_quad = np.int32(box).reshape(-1, 1, 2)

        return best_quad

    @classmethod
    def detect_die_5step(cls, rgb_bgr: np.ndarray, num_color_clusters=5, die_color='white'):
        h, w = rgb_bgr.shape[:2]

        # ----------------------------------------------------
        # Step 1: Cluster the scene based on colors (K-Means)
        # ----------------------------------------------------
        lab = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2LAB)
        pixels = lab.reshape((-1, 3)).astype(np.float32)

        K = num_color_clusters
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers_bgr = cv2.cvtColor(np.uint8(centers).reshape(1, K, 3), cv2.COLOR_LAB2BGR).reshape(K, 3)
        clustered_flat = centers_bgr[labels.flatten()]
        step1_color_clustered = clustered_flat.reshape((h, w, 3))

        # ----------------------------------------------------
        # Step 2: Detect the target color cluster matching die_color, find its convex hull
        # ----------------------------------------------------
        COLOR_MAP_BGR = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'orange': (0, 165, 255),
            'cyan': (255, 255, 0),
            'magenta': (255, 0, 255),
            'purple': (128, 0, 128),
            'gray': (128, 128, 128),
            'grey': (128, 128, 128),
        }

        if isinstance(die_color, str) and die_color.lower() in ['white', 'whitest']:
            lightness_values = centers[:, 0]  # L channel in LAB
            target_cluster_idx = int(np.argmax(lightness_values))
        elif isinstance(die_color, str) and die_color.lower() in ['black', 'darkest']:
            lightness_values = centers[:, 0]
            target_cluster_idx = int(np.argmin(lightness_values))
        else:
            if isinstance(die_color, str):
                bgr_target = COLOR_MAP_BGR.get(die_color.lower(), (255, 255, 255))
            elif isinstance(die_color, (tuple, list, np.ndarray)) and len(die_color) == 3:
                bgr_target = tuple(die_color)
            else:
                bgr_target = (255, 255, 255)

            target_lab = cv2.cvtColor(np.uint8([[bgr_target]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
            distances = np.linalg.norm(centers - target_lab, axis=1)
            target_cluster_idx = int(np.argmin(distances))

        labels_2d = labels.reshape((h, w))
        whitest_mask = (labels_2d == target_cluster_idx).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        whitest_mask_clean = cv2.morphologyEx(whitest_mask, cv2.MORPH_CLOSE, kernel)
        whitest_mask_clean = cv2.morphologyEx(whitest_mask_clean, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(whitest_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        step2_hull_img = rgb_bgr.copy()
        best_hull = None
        best_cnt = None
        best_bbox = None
        best_score = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 800 < area < 60000:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = float(bw) / float(bh)
                if 0.4 <= aspect <= 2.2:
                    hull = cv2.convexHull(cnt)
                    crop_gray = cv2.cvtColor(rgb_bgr[y:y+bh, x:x+bw], cv2.COLOR_BGR2GRAY)
                    std_dev = crop_gray.std()
                    score = area * std_dev
                    if score > best_score:
                        best_score = score
                        best_hull = hull
                        best_cnt = cnt
                        best_bbox = (x, y, bw, bh)

        if best_hull is not None:
            cv2.drawContours(step2_hull_img, [best_hull], -1, (0, 255, 0), 3)
            if best_cnt is not None:
                cv2.drawContours(step2_hull_img, [best_cnt], -1, (0, 0, 255), 1)

        # ----------------------------------------------------
        # Step 3: Crop image to bounding box surrounding the blob
        # ----------------------------------------------------
        if best_bbox is not None:
            bx, by, bw, bh = best_bbox
            margin = 12
            x1 = max(0, bx - margin)
            y1 = max(0, by - margin)
            x2 = min(w, bx + bw + margin)
            y2 = min(h, by + bh + margin)
            step3_crop_rgb = rgb_bgr[y1:y2, x1:x2].copy()
            actual_bbox = (x1, y1, x2 - x1, y2 - y1)
        else:
            x1, y1, x2, y2 = 0, 0, w, h
            actual_bbox = (0, 0, w, h)
            step3_crop_rgb = rgb_bgr.copy()

        # ----------------------------------------------------
        # Step 4: Cluster in Black & White / contour & edge detection for faces and pips
        # ----------------------------------------------------
        crop_h, crop_w = step3_crop_rgb.shape[:2]
        crop_gray = cv2.cvtColor(step3_crop_rgb, cv2.COLOR_BGR2GRAY)

        # Create binary mask for convex hull region inside the crop
        hull_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        if best_hull is not None:
            best_hull_crop = best_hull - np.array([x1, y1])
            cv2.drawContours(hull_mask, [best_hull_crop], -1, 255, -1)
        else:
            hull_mask.fill(255)

        # Black and white thresholding (Otsu) - applied ONLY to convex hull region (outside is always False/0)
        _, bw_mask = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw_mask = cv2.bitwise_and(bw_mask, hull_mask)

        # Pip mask (dark circular regions inside white body) - applied ONLY to convex hull region
        _, dark_mask = cv2.threshold(crop_gray, 105, 255, cv2.THRESH_BINARY_INV)
        dark_mask = cv2.bitwise_and(dark_mask, hull_mask)

        # Detect edge lines separating faces (restricted to convex hull region)
        edges = cv2.Canny(crop_gray, 40, 130)
        edges = cv2.bitwise_and(edges, hull_mask)
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel_line)

        # Dark pips contour extraction (with strict area, radius, and shape constraints to avoid big circle false positives)
        pip_contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_pips = []
        max_pip_area = crop_w * crop_h * 0.025  # Pip area threshold (max 2.5% of crop area)
        max_pip_radius = min(crop_w, crop_h) * 0.09  # Pip radius threshold (max 9% of crop dimension)

        for pc in pip_contours:
            pa = cv2.contourArea(pc)
            if 6 < pa < max_pip_area:
                (px, py), pr = cv2.minEnclosingCircle(pc)
                if pr <= max_pip_radius:
                    circle_area = math.pi * (pr ** 2)
                    if circle_area > 0 and (pa / circle_area) > 0.18:
                        bx_p, by_p, bw_p, bh_p = cv2.boundingRect(pc)
                        aspect_p = float(bw_p) / float(bh_p) if bh_p > 0 else 0
                        if 0.3 <= aspect_p <= 3.2:
                            valid_pips.append({
                                'center': (int(px), int(py)),
                                'radius': max(2, int(pr)),
                                'contour': pc,
                                'area': pa
                            })

        # Face segmentation: Fill internal pip holes inside bw_mask to preserve solid face region areas
        bw_mask_filled = bw_mask.copy()
        cnts_ccomp, hierarchy = cv2.findContours(bw_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            for i in range(len(cnts_ccomp)):
                # Fill internal holes (parent >= 0) with area < 8% of crop area (pip holes inside faces)
                if hierarchy[0][i][3] >= 0:
                    if cv2.contourArea(cnts_ccomp[i]) < (crop_w * crop_h * 0.08):
                        cv2.drawContours(bw_mask_filled, [cnts_ccomp[i]], -1, 255, -1)

        face_cnts, _ = cv2.findContours(bw_mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort candidate faces by area descending
        all_candidate_faces = [fc for fc in face_cnts if cv2.contourArea(fc) > (crop_w * crop_h * 0.03)]
        all_candidate_faces = sorted(all_candidate_faces, key=cv2.contourArea, reverse=True)

        # Filter out partial sliver/half faces cut off at the edge (area must be >= 35% of the largest face)
        candidate_faces = []
        if all_candidate_faces:
            max_face_area = cv2.contourArea(all_candidate_faces[0])
            for fc in all_candidate_faces:
                area = cv2.contourArea(fc)
                if area >= max(crop_w * crop_h * 0.04, 0.35 * max_face_area):
                    candidate_faces.append(fc)
                if len(candidate_faces) >= 3:
                    break

        step4_face_vis = step3_crop_rgb.copy()
        visible_faces = []
        face_colors = [(255, 0, 0), (0, 255, 0), (0, 165, 255)]  # Max 3 distinct face colors

        face_idx = 1
        for fc in candidate_faces:
            poly = cls.fit_quadrilateral(fc)
            if poly is None:
                continue

            # Count pips inside this quadrilateral face polygon
            face_pips = []
            for p in valid_pips:
                cx, cy = p['center']
                if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
                    face_pips.append(p)

            visible_faces.append({
                'face_index': face_idx,
                'contour': fc,
                'polygon': poly,
                'num_pips': len(face_pips),
                'pips': face_pips
            })

            color = face_colors[(face_idx - 1) % len(face_colors)]
            cv2.polylines(step4_face_vis, [poly], True, color, 2)
            face_idx += 1

        if not visible_faces:
            # Fallback if no sub-faces partitioned
            poly_crop = np.array([[[0, 0]], [[crop_w, 0]], [[crop_w, crop_h]], [[0, crop_h]]])
            visible_faces.append({
                'face_index': 1,
                'contour': poly_crop,
                'polygon': poly_crop,
                'num_pips': len(valid_pips),
                'pips': valid_pips
            })

        # Ensure at most 3 visible faces
        visible_faces = visible_faces[:3]

        # Identify TOP face by minimum 2D Y centroid (topmost face in cropped image space)
        top_f = min(visible_faces, key=lambda f: cv2.moments(f['polygon'])['m01'] / cv2.moments(f['polygon'])['m00'] if cv2.moments(f['polygon'])['m00'] > 0 else np.mean(f['polygon'][:, 0, 1]))
        for f in visible_faces:
            f['is_top_face'] = (f is top_f)

        # Draw detected face polygons
        for f in visible_faces:
            poly = f['polygon']
            color = face_colors[(f['face_index'] - 1) % len(face_colors)]
            thickness = 2
            cv2.polylines(step4_face_vis, [poly], True, color, thickness)

        # Draw detected pips on step4 visualization first
        for p in valid_pips:
            cv2.circle(step4_face_vis, p['center'], p['radius'], (0, 0, 255), 2)
            cv2.circle(step4_face_vis, p['center'], 2, (0, 255, 0), -1)

        # Draw black arrow pointing to top face centroid and label "TOP" as TOPMOST layer
        if top_f:
            poly = top_f['polygon']
            M_f = cv2.moments(poly)
            if M_f['m00'] > 0:
                cx_f = int(M_f['m10'] / M_f['m00'])
                cy_f = int(M_f['m01'] / M_f['m00'])
            else:
                cx_f = int(np.mean(poly[:, 0, 0]))
                cy_f = int(np.mean(poly[:, 0, 1]))

            pt_end = (cx_f, cy_f)
            # Arrow start point in South-East (SE) direction pointing to North-West (NW) top face centroid
            start_x = min(crop_w - 5, cx_f + 20)
            start_y = min(crop_h - 5, cy_f + 20)
            pt_start = (start_x, start_y)

            # Draw black arrow (thickness 2) and label "TOP" (thickness 2)
            cv2.arrowedLine(step4_face_vis, pt_start, pt_end, (0, 0, 0), 2, tipLength=0.2)
            label_pos = (max(2, start_x - 10), min(crop_h - 2, start_y + 14))
            cv2.putText(step4_face_vis, "TOP", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2)

        # ----------------------------------------------------
        # Step 5: Return visible faces and debug steps
        # ----------------------------------------------------
        total_pips = sum([f['num_pips'] for f in visible_faces])
        step5_annotated_crop = step4_face_vis.copy()

        # Prepare binary debug masks for plotting
        whitest_mask_bgr = cv2.cvtColor(whitest_mask_clean, cv2.COLOR_GRAY2BGR)
        bw_mask_bgr = cv2.cvtColor(bw_mask, cv2.COLOR_GRAY2BGR)
        dark_mask_bgr = cv2.cvtColor(dark_mask, cv2.COLOR_GRAY2BGR)

        return {
            'bbox': actual_bbox,
            'contour': best_cnt if best_cnt is not None else best_hull,
            'convex_hull': best_hull,
            'num_visible_faces': len(visible_faces),
            'faces': visible_faces,
            'total_pips': total_pips,
            'num_pips': total_pips,
            'debug_steps': {
                'step1_color_clustered': step1_color_clustered,
                'step2_whitest_convex_hull': step2_hull_img,
                'step2_whitest_mask': whitest_mask_bgr,
                'step3_crop_rgb': step3_crop_rgb,
                'step4_bw_mask': bw_mask_bgr,
                'step4_dark_pips_mask': dark_mask_bgr,
                'step4_faces_and_pips': step4_face_vis,
                'step5_annotated_result': step5_annotated_crop
            }
        }

    # Backward-compatible alias
    @classmethod
    def detect_die_2d(cls, rgb_bgr: np.ndarray, die_color='white'):
        return cls.detect_die_5step(rgb_bgr, die_color=die_color)


class AlignmentAndProjection:
    """Rotates 3D cloud to align plane normal to Z-axis and performs top-down orthographic extraction."""
    @staticmethod
    def get_rotation_to_z(normal: np.ndarray) -> np.ndarray:
        """
        Computes 3x3 rotation matrix R aligning plane normal to Z-axis [0, 0, 1].
        """
        n = normal / np.linalg.norm(normal)
        target = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        v = np.cross(n, target)
        s = np.linalg.norm(v)
        c = np.dot(n, target)

        if s < 1e-6:
            if c < 0:
                return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
            else:
                return np.identity(3, dtype=np.float32)

        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ], dtype=np.float32)

        R_mat = np.identity(3, dtype=np.float32) + vx + np.matmul(vx, vx) * ((1.0 - c) / (s ** 2))
        return R_mat

    @classmethod
    def extract_top_down_rgb(cls, rgb: np.ndarray, die_points: np.ndarray,
                             pixel_coords: np.ndarray, die_indices: np.ndarray,
                             R_plane: np.ndarray, crop_size=300):
        """
        Generates an unwarped, flat top-down RGB crop of the top face of the die.
        """
        h, w = rgb.shape[:2]
        die_pixels = pixel_coords[die_indices]

        x_min, y_min = die_pixels[:, 0].min(), die_pixels[:, 1].min()
        x_max, y_max = die_pixels[:, 0].max(), die_pixels[:, 1].max()

        margin = 10
        x_min_crop = max(0, x_min - margin)
        x_max_crop = min(w, x_max + margin)
        y_min_crop = max(0, y_min - margin)
        y_max_crop = min(h, y_max + margin)

        raw_crop = rgb[y_min_crop:y_max_crop, x_min_crop:x_max_crop]

        aligned_pts = np.dot(die_points, R_plane.T)
        xy_pts = aligned_pts[:, :2].astype(np.float32)

        yaw_angle = 0.0
        if len(xy_pts) >= 5:
            rect = cv2.minAreaRect(xy_pts)
            yaw_angle = rect[2]

        top_down_crop = cv2.resize(raw_crop, (crop_size, crop_size)) if raw_crop.size > 0 else np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        return top_down_crop, raw_crop, (x_min, y_min, x_max, y_max), yaw_angle


class PipDetector:
    """Detects and counts pips on the top face of a die using adaptive thresholding and blob analysis."""
    @staticmethod
    def detect_pips(top_down_rgb: np.ndarray, raw_crop: np.ndarray = None):
        """
        Processes cropped top-down face image to count pips.
        Returns: pip_count, annotated_top_down_image, keypoints.
        """
        images_to_try = [top_down_rgb]
        if raw_crop is not None and raw_crop.size > 0:
            resized_raw = cv2.resize(raw_crop, (top_down_rgb.shape[1], top_down_rgb.shape[0]))
            images_to_try.append(resized_raw)

        best_pip_count = 0
        best_annotated = top_down_rgb.copy()
        best_keypoints = []

        for target_img in images_to_try:
            if target_img is None or target_img.size == 0:
                continue

            h, w = target_img.shape[:2]
            gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 15, 4
            )

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

            margin = int(w * 0.08)
            border_mask = np.zeros((h, w), dtype=np.uint8)
            border_mask[margin:h-margin, margin:w-margin] = 255
            cleaned = cv2.bitwise_and(cleaned, cleaned, mask=border_mask)

            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            valid_pip_contours = []
            keypoints = []
            annotated = target_img.copy()

            die_area = w * h
            min_pip_area = die_area * 0.003
            max_pip_area = die_area * 0.08

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if min_pip_area <= area <= max_pip_area:
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter == 0:
                        continue
                    circularity = 4 * math.pi * area / (perimeter ** 2)

                    if circularity >= 0.45:
                        (x, y), radius = cv2.minEnclosingCircle(cnt)
                        center = (int(x), int(y))
                        radius = int(radius)
                        valid_pip_contours.append(cnt)
                        keypoints.append(cv2.KeyPoint(float(x), float(y), radius * 2.0))

                        cv2.circle(annotated, center, radius, (0, 0, 255), 2)
                        cv2.circle(annotated, center, 2, (0, 255, 0), -1)

            pip_count = len(valid_pip_contours)

            if pip_count == 0 or pip_count > 6:
                params = cv2.SimpleBlobDetector_Params()
                params.filterByArea = True
                params.minArea = min_pip_area
                params.maxArea = max_pip_area
                params.filterByCircularity = True
                params.minCircularity = 0.4
                params.filterByConvexity = True
                params.minConvexity = 0.5
                params.filterByInertia = True
                params.minInertiaRatio = 0.4

                detector = cv2.SimpleBlobDetector_create(params)
                blob_keypoints = detector.detect(cleaned)
                if 1 <= len(blob_keypoints) <= 6:
                    pip_count = len(blob_keypoints)
                    annotated = cv2.drawKeypoints(target_img, blob_keypoints, np.array([]), (0, 0, 255),
                                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            if 1 <= pip_count <= 6:
                best_pip_count = pip_count
                best_annotated = annotated
                best_keypoints = keypoints
                break
            elif pip_count > best_pip_count:
                best_pip_count = pip_count
                best_annotated = annotated
                best_keypoints = keypoints

        return best_pip_count, best_annotated, best_keypoints


class PoseEstimator:
    """Computes 6DOF pose (3D Centroid + Reference Frame aligned with table normal and upper face edges)."""
    @staticmethod
    def compute_pose(die_points: np.ndarray, plane_normal: np.ndarray, R_plane: np.ndarray,
                     top_face_polygon: np.ndarray = None,
                     top_face_centroid_2d: tuple = None,
                     plane_D: float = None,
                     die_size: float = 0.05,
                     camera_params: tuple = None):
        """
        Calculates exact 3D centroid of top face and 3D reference frame aligned with top face edges:
        - Z_die: Aligned with table plane normal (pointing towards camera)
        - X_die: Aligned with top face edge projected in 3D (starting from top face 2D center along edge direction)
        - Y_die: Orthogonal to X_die and Z_die (forming right-handed coordinate frame)
        """
        z_axis = plane_normal / np.linalg.norm(plane_normal)

        used_top_face_edge = False
        if (top_face_polygon is not None and top_face_centroid_2d is not None and
            plane_D is not None and camera_params is not None and len(top_face_polygon) >= 3):

            fx, fy, cx, cy = camera_params
            u_top, v_top = top_face_centroid_2d
            pts = top_face_polygon.reshape(-1, 2)

            # Compute 3D Centroid of Top Face via Ray Backprojection onto top-face plane (die_size offset)
            ray_top = np.array([(u_top - cx) / fx, (v_top - cy) / fy, 1.0], dtype=np.float32)
            dot_rn = np.dot(ray_top, z_axis)
            t_top = (die_size - plane_D) / dot_rn if abs(dot_rn) > 1e-6 else 1.0
            centroid = t_top * ray_top

            # Find direction of one edge of upper face (longest edge of 2D polygon)
            edges = [pts[(i + 1) % len(pts)] - pts[i] for i in range(len(pts))]
            edge_lengths = [np.linalg.norm(e) for e in edges]
            best_idx = np.argmax(edge_lengths)
            edge_vec_2d = edges[best_idx].astype(np.float32)
            edge_norm = np.linalg.norm(edge_vec_2d)

            if edge_norm > 1e-3:
                d_2d = edge_vec_2d / edge_norm

                # Select another point in 2D image starting from upper face centroid in direction of upper face edge
                u_edge = u_top + 20.0 * d_2d[0]
                v_edge = v_top + 20.0 * d_2d[1]

                # Backproject 2D edge point onto 3D top-face plane
                ray_edge = np.array([(u_edge - cx) / fx, (v_edge - cy) / fy, 1.0], dtype=np.float32)
                dot_edge = np.dot(ray_edge, z_axis)
                t_edge = (die_size - plane_D) / dot_edge if abs(dot_edge) > 1e-6 else 1.0
                pt_edge_3d = t_edge * ray_edge

                # 3D vector for X-axis along upper face edge
                v_3d = pt_edge_3d - centroid
                x_raw = v_3d - np.dot(v_3d, z_axis) * z_axis
                x_norm = np.linalg.norm(x_raw)

                if x_norm > 1e-4:
                    x_axis = x_raw / x_norm
                    y_axis = np.cross(z_axis, x_axis)
                    y_axis /= np.linalg.norm(y_axis)
                    used_top_face_edge = True

        if not used_top_face_edge:
            # Fallback: 3D Centroid of point cloud & PCA
            centroid = np.mean(die_points, axis=0) if len(die_points) > 0 else np.array([0, 0, 1], dtype=np.float32)
            aligned_pts = np.dot(die_points - centroid, R_plane.T)
            xy_pts = aligned_pts[:, :2].astype(np.float32)

            if len(xy_pts) >= 5:
                cov = np.cov(xy_pts, rowvar=False)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                v1_2d = eigenvectors[:, 1]
                v2_2d = eigenvectors[:, 0]
            else:
                v1_2d = np.array([1.0, 0.0], dtype=np.float32)
                v2_2d = np.array([0.0, 1.0], dtype=np.float32)

            v1_3d_aligned = np.array([v1_2d[0], v1_2d[1], 0.0], dtype=np.float32)
            v2_3d_aligned = np.array([v2_2d[0], v2_2d[1], 0.0], dtype=np.float32)

            R_inv = R_plane.T
            x_axis = np.dot(R_inv, v1_3d_aligned)
            y_axis = np.dot(R_inv, v2_3d_aligned)

            x_axis /= np.linalg.norm(x_axis)
            y_axis /= np.linalg.norm(y_axis)

            z_calc = np.cross(x_axis, y_axis)
            if np.dot(z_calc, z_axis) < 0:
                y_axis = -y_axis

        # 3x3 Rotation matrix of the die frame
        R_die = np.stack((x_axis, y_axis, z_axis), axis=1)

        rot_obj = R_sci.from_matrix(R_die)
        quat = rot_obj.as_quat()  # [qx, qy, qz, qw]

        return centroid, quat, R_die, (x_axis, y_axis, z_axis)


class PlotlyVisualizer:
    """Generates interactive 3D point cloud plots saved as HTML files using Plotly."""
    @staticmethod
    def visualize_point_cloud_3d(points: np.ndarray, colors: np.ndarray,
                                die_points: np.ndarray = None,
                                plane_model: tuple = None,
                                centroid: np.ndarray = None,
                                die_centroid: np.ndarray = None,
                                die_axes: tuple = None,
                                die_size: float = 0.05,
                                title: str = "3D Point Cloud & Segmented Die Pose",
                                output_html_path: str = None,
                                max_points: int = 15000):
        if not HAS_PLOTLY:
            print("[Plotly] Plotly is not installed. Skipping interactive 3D plot.")
            return None

        num_points = len(points)
        if num_points > max_points:
            sub_indices = np.random.choice(num_points, max_points, replace=False)
            pts_sub = points[sub_indices]
            cols_sub = colors[sub_indices]
        else:
            pts_sub = points
            cols_sub = colors

        rgb_strings = [f"rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})" for c in cols_sub]

        fig = go.Figure()

        # Trace 1: Full Scene Point Cloud
        fig.add_trace(go.Scatter3d(
            x=pts_sub[:, 0],
            y=pts_sub[:, 1],
            z=pts_sub[:, 2],
            mode='markers',
            marker=dict(size=2, color=rgb_strings, opacity=0.5),
            name='Scene Point Cloud'
        ))

        # Trace 2: Segmented Die 3D Point Blob (Highlighted in Red)
        if die_points is not None and len(die_points) > 0:
            die_sub = die_points
            if len(die_points) > 3000:
                die_sub = die_points[np.random.choice(len(die_points), 3000, replace=False)]

            fig.add_trace(go.Scatter3d(
                x=die_sub[:, 0],
                y=die_sub[:, 1],
                z=die_sub[:, 2],
                mode='markers',
                marker=dict(size=5, color='red', symbol='diamond', opacity=0.9),
                name='Segmented Die Blob'
            ))

        # Trace 3: Camera Origin Versors at (0,0,0)
        versor_len_cam = 0.15
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers+text',
            marker=dict(size=7, color='black', symbol='circle'),
            text=["Camera Origin (0,0,0)"],
            textposition="top center",
            name="Camera Origin"
        ))
        fig.add_trace(go.Scatter3d(
            x=[0, versor_len_cam], y=[0, 0], z=[0, 0],
            mode='lines+text', line=dict(color='red', width=6),
            text=["", "Cam X"], name="Cam X (Red)"
        ))
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, versor_len_cam], z=[0, 0],
            mode='lines+text', line=dict(color='green', width=6),
            text=["", "Cam Y"], name="Cam Y (Green)"
        ))
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[0, versor_len_cam],
            mode='lines+text', line=dict(color='blue', width=6),
            text=["", "Cam Z"], name="Cam Z (Blue)"
        ))

        # Trace 4: Segmented Die Transforms (Top Face TF & Die Centroid TF)
        if centroid is not None and die_axes is not None:
            x_ax, y_ax, z_ax = die_axes
            arm = 0.16  # 16cm frame arms

            if die_centroid is None:
                die_centroid = centroid - (die_size / 2.0) * z_ax

            # --- 1. TOP FACE TF ---
            fig.add_trace(go.Scatter3d(
                x=[centroid[0]], y=[centroid[1]], z=[centroid[2]],
                mode='markers+text',
                marker=dict(size=14, color='white', line=dict(color='black', width=3), symbol='circle'),
                text=["Top Face TF"],
                textposition="top center",
                name='Top Face TF Origin'
            ))

            fig.add_trace(go.Scatter3d(
                x=[centroid[0], centroid[0] + arm * x_ax[0]],
                y=[centroid[1], centroid[1] + arm * x_ax[1]],
                z=[centroid[2], centroid[2] + arm * x_ax[2]],
                mode='lines', line=dict(color='red', width=10),
                name="Top Face X-axis"
            ))

            fig.add_trace(go.Scatter3d(
                x=[centroid[0], centroid[0] + arm * y_ax[0]],
                y=[centroid[1], centroid[1] + arm * y_ax[1]],
                z=[centroid[2], centroid[2] + arm * y_ax[2]],
                mode='lines', line=dict(color='green', width=10),
                name="Top Face Y-axis"
            ))

            fig.add_trace(go.Scatter3d(
                x=[centroid[0], centroid[0] + arm * z_ax[0]],
                y=[centroid[1], centroid[1] + arm * z_ax[1]],
                z=[centroid[2], centroid[2] + arm * z_ax[2]],
                mode='lines', line=dict(color='blue', width=10),
                name="Top Face Z-axis"
            ))

            # --- 2. DIE CENTROID TF ---
            fig.add_trace(go.Scatter3d(
                x=[die_centroid[0]], y=[die_centroid[1]], z=[die_centroid[2]],
                mode='markers+text',
                marker=dict(size=14, color='yellow', line=dict(color='black', width=3), symbol='diamond'),
                text=["Die Centroid TF"],
                textposition="bottom center",
                name='Die Centroid TF Origin'
            ))

            fig.add_trace(go.Scatter3d(
                x=[die_centroid[0], die_centroid[0] + arm * x_ax[0]],
                y=[die_centroid[1], die_centroid[1] + arm * x_ax[1]],
                z=[die_centroid[2], die_centroid[2] + arm * x_ax[2]],
                mode='lines', line=dict(color='darkred', width=8),
                name="Die Centroid X-axis"
            ))

            fig.add_trace(go.Scatter3d(
                x=[die_centroid[0], die_centroid[0] + arm * y_ax[0]],
                y=[die_centroid[1], die_centroid[1] + arm * y_ax[1]],
                z=[die_centroid[2], die_centroid[2] + arm * y_ax[2]],
                mode='lines', line=dict(color='darkgreen', width=8),
                name="Die Centroid Y-axis"
            ))

            fig.add_trace(go.Scatter3d(
                x=[die_centroid[0], die_centroid[0] + arm * z_ax[0]],
                y=[die_centroid[1], die_centroid[1] + arm * z_ax[1]],
                z=[die_centroid[2], die_centroid[2] + arm * z_ax[2]],
                mode='lines', line=dict(color='darkblue', width=8),
                name="Die Centroid Z-axis"
            ))

        # Layout settings with Z inverted (autorange='reversed': depth increases top to bottom)
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (meters)',
                yaxis_title='Y (meters)',
                zaxis_title='Z (depth meters: Top -> Bottom)',
                zaxis=dict(autorange='reversed'),
                aspectmode='data',
                camera=dict(
                    eye=dict(x=0.5, y=-1.5, z=-1.5)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        if output_html_path:
            fig.write_html(output_html_path)
            print(f"[Plotly] Interactive 3D Point Cloud saved to: {output_html_path}")

        return fig


class DieDetectorPipeline:
    """Complete ROS-free pipeline orchestration for processing RGB images."""
    def __init__(self, fx=500.0, fy=500.0):
        self.depth_estimator = MonocularDepthEstimator()
        self.processor = PointCloudProcessor(fx=fx, fy=fy)
        self.fx = fx
        self.fy = fy

    def process_image(self, image_path: str, output_dir: str = "output", die_size: float = 0.05, die_color: str = 'white'):
        """Processes a single RGB image file and saves diagnostic outputs."""
        rgb = cv2.imread(image_path)
        if rgb is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")

        h, w = rgb.shape[:2]
        img_name = os.path.basename(image_path)
        print(f"\n==========================================")
        print(f"Processing: {img_name} ({w}x{h}) [Target Die Color: {die_color}]")

        # 1. Infer Depth Map on Full Uncropped Image
        depth = self.depth_estimator.infer_depth(rgb)

        # Save depth visualization to images/depth/
        depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)
        depth_output_path = os.path.join("images", "depth", f"depth_{img_name}")
        cv2.imwrite(depth_output_path, depth_colormap)
        print(f"[Step 1] Depth map generated for full image ({w}x{h}) and saved to {depth_output_path}")

        # 2. Generate 3D Point Cloud
        points, colors, pixel_coords, (cx, cy) = self.processor.create_point_cloud(rgb, depth)
        print(f"[Step 2] Created point cloud with {len(points)} valid 3D points.")

        # 3. Fit Table Surface Plane via RANSAC
        plane_model, inliers, outliers, normal = self.processor.fit_plane_ransac(points)
        print(f"[Step 3] Plane model (Ax+By+Cz+D=0): {plane_model}")
        print(f"         Extracted Plane Normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")

        # 4. 2D Whitest Region Detection & 3D Point Cloud Segmentation of Die
        detection_res = RGBDieDetector.detect_die_5step(rgb, die_color=die_color)
        A_p, B_p, C_p, D_p = plane_model
        heights = np.dot(points, normal) + D_p  # Perpendicular height above table

        x_c, y_c, w_c, h_c = detection_res['bbox']
        num_visible_faces = detection_res['num_visible_faces']
        pip_count = detection_res['total_pips']
        faces = detection_res['faces']

        print(f"[Step 4] RGB 5-Step Detection: DETECTED {num_visible_faces} VISIBLE FACES with {pip_count} total pips.")
        for f in faces:
            print(f"         - Face {f['face_index']}: {f['num_pips']} pips")

        pixel_x = pixel_coords[:, 0]
        pixel_y = pixel_coords[:, 1]

        # Segment points resting 0.3cm to 6.5cm above table plane inside 2D white die region
        inside_mask = (pixel_x >= x_c) & (pixel_x < x_c + w_c) & \
                      (pixel_y >= y_c) & (pixel_y < y_c + h_c) & \
                      (heights >= 0.003) & (heights <= 0.065)
        orig_die_indices = np.where(inside_mask)[0]

        if len(orig_die_indices) < 10:
            # Fallback to points in 2D bounding box
            inside_mask = (pixel_x >= x_c) & (pixel_x < x_c + w_c) & (pixel_y >= y_c) & (pixel_y < y_c + h_c)
            orig_die_indices = np.where(inside_mask)[0]

        die_points = points[orig_die_indices] if len(orig_die_indices) > 0 else points
        print(f"         Segmented 5cm die blob with {len(die_points)} 3D points.")

        # Identify Top Face (face positioned highest up in 2D image space, min Y centroid)
        top_face = None
        min_y_centroid = 1e9

        for f in faces:
            poly = f['polygon']
            M = cv2.moments(poly)
            cy_poly = (M['m01'] / M['m00']) if M['m00'] > 0 else np.mean(poly[:, 0, 1])
            if cy_poly < min_y_centroid:
                min_y_centroid = cy_poly
                top_face = f

        die_size = 0.05  # 5cm die size
        u_top, v_top = cx, cy

        if top_face is not None:
            # 2D centroid of Top Face in full image coordinates
            poly_full = top_face['polygon'] + np.array([x_c, y_c])
            M_top = cv2.moments(poly_full)
            u_top = (M_top['m10'] / M_top['m00']) if M_top['m00'] > 0 else np.mean(poly_full[:, 0, 0])
            v_top = (M_top['m01'] / M_top['m00']) if M_top['m01'] > 0 else np.mean(poly_full[:, 0, 1])

            # Ray backprojection to 3D top-face plane (die_size taller than table plane)
            ray = np.array([(u_top - cx) / self.fx, (v_top - cy) / self.fy, 1.0], dtype=np.float32)
            dot_rn = np.dot(ray, normal)
            t = (die_size - D_p) / dot_rn if abs(dot_rn) > 1e-6 else 1.0
            top_centroid = t * ray
        else:
            top_centroid = np.mean(die_points, axis=0)

        # 5. 3D Alignment Matrix to Z-Axis
        R_plane = AlignmentAndProjection.get_rotation_to_z(normal)
        print(f"[Step 5] Computed 3D alignment rotation matrix R_plane to Z-axis.")

        # 6. Top-Down Orthographic Face Extraction
        top_down_crop, raw_crop, (x1, y1, x2, y2), yaw_angle = AlignmentAndProjection.extract_top_down_rgb(
            rgb, die_points, pixel_coords, orig_die_indices, R_plane, crop_size=300
        )

        # 7. Compute XYZ Top-Face Centroid & 6DOF Table-Aligned Frame
        top_poly_full = (top_face['polygon'] + np.array([x_c, y_c])) if top_face is not None else None
        camera_params = (self.fx, self.fy, cx, cy)

        centroid, quat, R_die, (x_ax, y_ax, z_ax) = PoseEstimator.compute_pose(
            die_points, normal, R_plane,
            top_face_polygon=top_poly_full,
            top_face_centroid_2d=(u_top, v_top),
            plane_D=D_p,
            die_size=die_size,
            camera_params=camera_params
        )

        top_face_tf_origin = centroid
        die_centroid_tf_origin = centroid - (die_size / 2.0) * z_ax

        if top_face is not None:
            print(f"[Step 7] Identified TOP FACE: Face {top_face['face_index']} ({top_face['num_pips']} pips) [2D Center: ({u_top:.1f}, {v_top:.1f})px].")
        print(f"         Computed 6DOF Transforms (die_size = {die_size*100:.1f}cm):")
        print(f"         - Top Face TF Translation:     X={top_face_tf_origin[0]:.3f}m, Y={top_face_tf_origin[1]:.3f}m, Z={top_face_tf_origin[2]:.3f}m")
        print(f"         - Die Centroid TF Translation: X={die_centroid_tf_origin[0]:.3f}m, Y={die_centroid_tf_origin[1]:.3f}m, Z={die_centroid_tf_origin[2]:.3f}m")
        print(f"         Orientation Quaternion:        [qx={quat[0]:.3f}, qy={quat[1]:.3f}, qz={quat[2]:.3f}, qw={quat[3]:.3f}]")

        # 8. Build 6-Panel Debug Collage Plotting Each Processing Step and Binary Masks
        debug_steps = detection_res['debug_steps']
        step1_img = debug_steps['step1_color_clustered']
        step2_img = debug_steps['step2_whitest_convex_hull']
        whitest_mask_img = debug_steps['step2_whitest_mask']
        bw_mask_img = debug_steps['step4_bw_mask']
        dark_pips_mask_img = debug_steps['step4_dark_pips_mask']
        step34_img = debug_steps['step4_faces_and_pips']

        # Annotated RGB full scene
        annotated_rgb = rgb.copy()
    
        # Project 3D Centroid and Versor tips onto 2D image
        def project_pt(pt_3d):
            px = int((pt_3d[0] * self.fx / pt_3d[2]) + cx)
            py = int((pt_3d[1] * self.fy / pt_3d[2]) + cy)
            return (px, py)

        c_2d = (int(u_top), int(v_top)) if top_face is not None else project_pt(centroid)
        arm_2d = 0.10  # 10cm versors (longer 3D/2D frame axes)
        tip_x_2d = project_pt(centroid + arm_2d * x_ax)
        tip_y_2d = project_pt(centroid + arm_2d * y_ax)
        tip_z_2d = project_pt(centroid + arm_2d * z_ax)

        # Draw 3D coordinate frame at top face centroid in 2D panel (Red X, Green Y, Blue Z) - thicker lines
        cv2.line(annotated_rgb, c_2d, tip_x_2d, (0, 0, 255), 5)  # Red X (thicker)
        cv2.line(annotated_rgb, c_2d, tip_y_2d, (0, 255, 0), 5)  # Green Y (thicker)
        cv2.line(annotated_rgb, c_2d, tip_z_2d, (255, 0, 0), 5)  # Blue Z (thicker)
        # Centroid: white dot with black border
        cv2.circle(annotated_rgb, c_2d, 8, (0, 0, 0), -1)       # Black border
        cv2.circle(annotated_rgb, c_2d, 5, (255, 255, 255), -1) # Inner white dot

        # Resize debug panels for 3x2 collage (3 columns, 2 rows)
        h_third = h // 2
        w_third = w // 3

        p1 = cv2.resize(step1_img, (w_third, h_third))
        cv2.putText(p1, "1. Color Clusters (K-Means)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        p2 = cv2.resize(whitest_mask_img, (w_third, h_third))
        cv2.putText(p2, "2. Whitest Cluster", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        p3 = cv2.resize(step2_img, (w_third, h_third))
        cv2.putText(p3, "3. Whitest Convex Hull", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Combine B&W and dark pips binary masks into panel 4
        bw_dark_combined = np.hstack((cv2.resize(bw_mask_img, (w_third // 2, h_third)),
                                       cv2.resize(dark_pips_mask_img, (w_third - w_third // 2, h_third))))
        cv2.putText(bw_dark_combined, "4. B&W & Pips Masks", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        p5_canvas = np.zeros((h_third, w_third, 3), dtype=np.uint8)
        step34_resized = cv2.resize(step34_img, (min(w_third, step34_img.shape[1] * 2), min(h_third, step34_img.shape[0] * 2)))
        sh, sw = step34_resized.shape[:2]
        p5_canvas[:sh, :sw] = step34_resized
        cv2.putText(p5_canvas, "5. Detected Faces & Pips", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        p6 = cv2.resize(annotated_rgb, (w_third, h_third))
        cv2.putText(p6, "6. 3D Pose Frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        row1 = np.hstack((p1, p2, p3))
        row2 = np.hstack((bw_dark_combined, p5_canvas, p6))
        collage = np.vstack((row1, row2))

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"detected_{img_name}")
        cv2.imwrite(out_path, collage)
        print(f"[Output] Diagnostic 6-panel collage saved to: {out_path}")

        # 9. Generate Interactive Plotly 3D Point Cloud Plot (.html) with Dual Transforms
        plotly_html_path = os.path.join(output_dir, f"plotly_3d_{os.path.splitext(img_name)[0]}.html")
        PlotlyVisualizer.visualize_point_cloud_3d(
            points=points,
            colors=colors,
            die_points=die_points,
            plane_model=plane_model,
            centroid=top_face_tf_origin,
            die_centroid=die_centroid_tf_origin,
            die_axes=(x_ax, y_ax, z_ax),
            die_size=die_size,
            title=f"3D Point Cloud & Die Transforms ({img_name})",
            output_html_path=plotly_html_path
        )

        return {
            "image_name": img_name,
            "pip_count": pip_count,
            "num_faces": num_visible_faces,
            "faces": faces,
            "centroid": centroid,
            "quaternion": quat,
            "output_path": out_path,
            "plotly_html_path": plotly_html_path
        }


def main():
    rgb_dir = os.path.join("images", "rgb")
    output_dir = "output"

    image_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")) + glob.glob(os.path.join(rgb_dir, "*.png")))
    if not image_paths:
        print(f"No RGB images found in {rgb_dir}.")
        return

    print(f"Found {len(image_paths)} images to process in {rgb_dir}.")
    pipeline = DieDetectorPipeline()

    results = []
    for path in image_paths:
        res = pipeline.process_image(path, output_dir)
        results.append(res)

    print("\n" + "=" * 75)
    print("SUMMARY OF DETECTION RESULTS:")
    print("=" * 75)
    for r in results:
        pos = r['centroid']
        q = r['quaternion']
        top_face_idx = next((f['face_index'] for f in r.get('faces', []) if f.get('is_top_face', False)), 1)
        faces_str = f"{r.get('num_faces', 0)} faces (" + ", ".join([f"{'TOP ' if f['face_index']==top_face_idx else ''}Face {f['face_index']}: {f['num_pips']}" for f in r.get('faces', [])]) + ")"
        print(f"Image: {r['image_name']:<25} | Faces: {faces_str:<32} | Total Pips: {r['pip_count']} | Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})m | Quat: [{q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}, {q[3]:.2f}]")


if __name__ == "__main__":
    main()

