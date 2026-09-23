#!/usr/bin/env python3
"""
detection_optimization.py
=========================
Interactive calibration and parameter optimization tool for die detection.

Workflow:
  1. Sequentially prompts the user to place die faces 1 through 6 under the camera.
  2. Captures synchronized RGB (+ optional depth) frames on each [ENTER] / [SPACE].
  3. Runs an optimization grid search to determine parameters (HSV bounds, glare
     cutoff, CLAHE, pip circularity, pip area/radius ratios) that achieve 6/6 correct
     face classifications.
  4. Displays a 6-face verification collage and automatically saves optimal
     parameters to config/die_detector_params.yaml.

Usage:
------
  # Live camera interactive calibration:
  ros2 run drims_die_detection detection_optimization.py

  # Or direct python:
  python3 scripts/detection_optimization.py

  # Re-optimize on existing captured dataset without camera:
  python3 scripts/detection_optimization.py --offline
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import copy
from collections import Counter
from typing import Optional, Any
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import yaml


def get_package_root() -> str:
    """Finds the root directory of drims_die_detection (prioritizing source workspace)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Direct parent if running from <pkg>/scripts/
    cand_src = os.path.abspath(os.path.join(script_dir, ".."))
    if os.path.isfile(os.path.join(cand_src, "package.xml")):
        return cand_src

    # 2. Known workspace source directories
    for ws_base in ["/home/galileo/projects/NDR_ws", os.getcwd()]:
        cand = os.path.join(ws_base, "src", "drims_die_detection")
        if os.path.isfile(os.path.join(cand, "package.xml")):
            return cand

    # 3. ament_index share directory
    try:
        from ament_index_python.packages import get_package_share_directory
        share_dir = get_package_share_directory("drims_die_detection")
        if os.path.isdir(share_dir):
            return share_dir
    except Exception:
        pass

    return cand_src


# Allow importing package from repo root
_REPO_ROOT = get_package_root()
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from drims_die_detection.die_detector_params import DieDetectorParams
from drims_die_detection.rgb_die_detector import RGBDieDetector

# ROS 2 imports (optional fallback if running standalone offline)
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge
    import message_filters
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False


# ──────────────────────────────────────────────────────────────────────────────
# 1. ROS 2 Frame Grabber
# ──────────────────────────────────────────────────────────────────────────────

class CameraFrameGrabber:
    """Subscribes to ROS 2 camera topics and captures synchronized RGB/Depth frames."""

    def __init__(self, rgb_topic: str, depth_topic: str):
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.bridge = CvBridge()
        self._node: Optional[Node] = None
        self._burst_collecting: bool = False
        self._burst_frames: list[tuple[np.ndarray, Optional[np.ndarray]]] = []

        if HAS_ROS2:
            if not rclpy.ok():
                rclpy.init()
            self._node = rclpy.create_node("die_detection_optimizer_grabber")
            qos = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=5,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
            self._rgb_sub = message_filters.Subscriber(self._node, Image, rgb_topic, qos_profile=qos)
            self._depth_sub = message_filters.Subscriber(self._node, Image, depth_topic, qos_profile=qos)
            self._sync = message_filters.ApproximateTimeSynchronizer(
                [self._rgb_sub, self._depth_sub], queue_size=10, slop=0.08
            )
            self._sync.registerCallback(self._sync_cb)
            # Standalone RGB fallback in case depth is not publishing
            self._node.create_subscription(Image, rgb_topic, self._rgb_only_cb, qos)

    def _sync_cb(self, rgb_msg: Image, depth_msg: Image):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            if depth_msg.encoding == "16UC1":
                d = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
                self.latest_depth = d.astype(np.float32) * 0.001
            elif depth_msg.encoding == "32FC1":
                self.latest_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")

            if self._burst_collecting and self.latest_rgb is not None:
                self._burst_frames.append((
                    self.latest_rgb.copy(),
                    self.latest_depth.copy() if self.latest_depth is not None else None
                ))
        except Exception as e:
            print(f"[Grabber] Error converting synced frame: {e}")

    def _rgb_only_cb(self, rgb_msg: Image):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            if self._burst_collecting and self.latest_rgb is not None:
                self._burst_frames.append((self.latest_rgb.copy(), None))
        except Exception:
            pass

    def spin_once(self, timeout_sec: float = 0.05):
        if self._node and rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=timeout_sec)

    def capture_burst(
        self,
        num_frames: int = 12,
        timeout_sec: float = 3.5,
        progress_cb=None,
    ) -> list[tuple[np.ndarray, Optional[np.ndarray]]]:
        """Collects a sequence of distinct synchronized frames for video stability analysis."""
        self._burst_frames = []
        self._burst_collecting = True
        t0 = time.time()
        last_count = 0

        while len(self._burst_frames) < num_frames and (time.time() - t0 < timeout_sec):
            self.spin_once(0.02)
            cur_count = len(self._burst_frames)
            if cur_count != last_count and progress_cb:
                progress_cb(cur_count, num_frames)
                last_count = cur_count

        self._burst_collecting = False

        if not self._burst_frames and self.latest_rgb is not None:
            self._burst_frames.append((
                self.latest_rgb.copy(),
                self.latest_depth.copy() if self.latest_depth is not None else None
            ))

        # If camera framerate is slower than timeout, pad with latest available frame
        while len(self._burst_frames) < num_frames and self._burst_frames:
            last_pair = self._burst_frames[-1]
            self._burst_frames.append((last_pair[0].copy(), last_pair[1].copy() if last_pair[1] is not None else None))

        return self._burst_frames[:num_frames]

    def destroy(self):
        if self._node:
            self._node.destroy_node()


# ──────────────────────────────────────────────────────────────────────────────
# 2. Interactive Face Acquisition (5 Workspace Positions: Center + 4 Corners)
# ──────────────────────────────────────────────────────────────────────────────

WORKSPACE_POSITIONS = [
    {
        "id": "center",
        "name": "Center of Workspace",
        "rel_x": 0.50,
        "rel_y": 0.50,
    },
    {
        "id": "top_left",
        "name": "Top-Left Corner",
        "rel_x": 0.25,
        "rel_y": 0.25,
    },
    {
        "id": "top_right",
        "name": "Top-Right Corner",
        "rel_x": 0.75,
        "rel_y": 0.25,
    },
    {
        "id": "bottom_right",
        "name": "Bottom-Right Corner",
        "rel_x": 0.75,
        "rel_y": 0.75,
    },
    {
        "id": "bottom_left",
        "name": "Bottom-Left Corner",
        "rel_x": 0.25,
        "rel_y": 0.75,
    },
]


def capture_dataset_interactive(
    grabber: CameraFrameGrabber,
    output_dir: str,
    target_faces: list[int] = [1, 2, 3, 4, 5, 6],
    burst_frames: int = 8,
    positions: list[dict] | None = None,
) -> dict[int, dict[str, list[np.ndarray]]]:
    """Guides user through faces 1 to 6 across selected workspace positions."""
    os.makedirs(output_dir, exist_ok=True)
    captured_dataset: dict[int, dict[str, list[np.ndarray]]] = {}
    active_positions = positions if positions is not None else WORKSPACE_POSITIONS

    win_name = "Die Detection Calibration — Live Preview"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 720)

    print("\n" + "=" * 76)
    print("  DIE DETECTION OPTIMIZATION — 5-POSITION WORKSPACE CAPTURE")
    print("=" * 76)
    print("For each face (1 through 6), you will capture 5 workspace positions:")
    print("  1. Center of Workspace")
    print("  2. Top-Left Corner")
    print("  3. Top-Right Corner")
    print("  4. Bottom-Right Corner")
    print("  5. Bottom-Left Corner")
    print("\nControls:")
    print("  [SPACE] / [ENTER]  : Record burst of frames at current position")
    print("  [S]                : Skip current position")
    print("  [Q] / [ESC]        : Abort calibration")
    print("=" * 76 + "\n")

    for face in target_faces:
        captured_dataset[face] = {}
        burst_dir = os.path.join(output_dir, f"face_{face}_burst")
        os.makedirs(burst_dir, exist_ok=True)

        for pos_idx, pos in enumerate(active_positions, 1):
            pos_id = pos["id"]
            pos_name = pos["name"]

            print(f"\n>>> [FACE {face}/6 | POS {pos_idx}/{len(active_positions)}] Place FACE {face} UP in {pos_name.upper()}.")
            print(f"    Press [ENTER] in terminal (or SPACE in window) to record {burst_frames} frames ([S] to skip)...")

            captured = False
            skipped = False
            last_frame = None

            while True:
                grabber.spin_once(0.03)
                frame = grabber.latest_rgb
                if frame is not None:
                    last_frame = frame.copy()
                    disp = frame.copy()
                    h, w = disp.shape[:2]

                    # Target reticle positioned for this specific workspace area
                    tx = int(w * pos["rel_x"])
                    ty = int(h * pos["rel_y"])
                    bw, bh = 140, 140
                    cv2.drawMarker(disp, (tx, ty), (0, 255, 255), cv2.MARKER_CROSS, 30, 2)
                    cv2.rectangle(disp, (tx - bw // 2, ty - bh // 2), (tx + bw // 2, ty + bh // 2), (0, 255, 255), 2)
                    # Label above targeting box
                    cv2.putText(disp, pos_name.upper(), (tx - bw // 2, max(20, ty - bh // 2 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

                    # Top status banner
                    cv2.rectangle(disp, (0, 0), (w, 65), (20, 20, 20), -1)
                    cv2.rectangle(disp, (0, 63), (w, 65), (0, 255, 0), 2)
                    title_msg = f"FACE {face}/6 | POS {pos_idx}/{len(active_positions)}: {pos_name.upper()}"
                    sub_msg = "Press SPACE/ENTER to Record Burst  |  [S] Skip Position  |  [Q] Quit"
                    cv2.putText(disp, title_msg, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(disp, sub_msg, (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

                    cv2.imshow(win_name, disp)

                key = cv2.waitKey(20) & 0xFF
                if key == ord(' '):
                    captured = True
                    break
                elif key in (ord('s'), ord('S')):
                    skipped = True
                    break
                elif key == ord('q') or key == 27:
                    print("\n[INFO] Calibration aborted by user.")
                    cv2.destroyAllWindows()
                    sys.exit(0)

                import select
                if select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline().strip().lower()
                    if line in ("s", "skip"):
                        skipped = True
                    else:
                        captured = True
                    break

            if skipped:
                print(f"    [INFO] Skipped position {pos_name}.")
                continue

            if captured:
                print(f"    [INFO] Recording video burst of {burst_frames} frames for Face {face} ({pos_name})...")

                def _burst_progress(cur, tot):
                    frame_now = grabber.latest_rgb if grabber.latest_rgb is not None else last_frame
                    if frame_now is None:
                        return
                    disp = frame_now.copy()
                    h, w = disp.shape[:2]
                    cx, cy = w // 2, h // 2

                    cv2.rectangle(disp, (0, 0), (w, 65), (0, 140, 255), -1)
                    prog_msg = f"RECORDING {pos_name.upper()}: {cur}/{tot} FRAMES"
                    cv2.putText(disp, prog_msg, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

                    pw = 360
                    px1, px2 = cx - pw // 2, cx + pw // 2
                    py1, py2 = cy + 120, cy + 145
                    cv2.rectangle(disp, (px1, py1), (px2, py2), (30, 30, 30), -1)
                    frac = min(1.0, float(cur) / float(tot)) if tot > 0 else 0.0
                    cv2.rectangle(disp, (px1, py1), (int(px1 + pw * frac), py2), (0, 255, 0), -1)
                    cv2.rectangle(disp, (px1, py1), (px2, py2), (255, 255, 255), 1)
                    cv2.imshow(win_name, disp)
                    cv2.waitKey(1)

                burst = grabber.capture_burst(num_frames=burst_frames, timeout_sec=4.0, progress_cb=_burst_progress)

                if burst:
                    rgb_frames = [b[0] for b in burst]
                    mid_idx = len(rgb_frames) // 2
                    rep_frame = rgb_frames[mid_idx]
                    rep_depth = burst[mid_idx][1]

                    # Save position image
                    pos_img_path = os.path.join(output_dir, f"face_{face}_{pos_id}.png")
                    cv2.imwrite(pos_img_path, rep_frame)

                    # Backward compatibility: center position also writes face_{face}.png
                    if pos_id == "center" or len(captured_dataset[face]) == 0:
                        cv2.imwrite(os.path.join(output_dir, f"face_{face}.png"), rep_frame)
                        if rep_depth is not None:
                            np.save(os.path.join(output_dir, f"face_{face}_depth.npy"), rep_depth)

                    # Save burst frames with position prefix
                    for k, (b_rgb, b_depth) in enumerate(burst):
                        cv2.imwrite(os.path.join(burst_dir, f"pos_{pos_id}_frame_{k:02d}.png"), b_rgb)
                        if b_depth is not None:
                            np.save(os.path.join(burst_dir, f"pos_{pos_id}_frame_{k:02d}_depth.npy"), b_depth)
                        # Also save legacy frame_*.png for center
                        if pos_id == "center":
                            cv2.imwrite(os.path.join(burst_dir, f"frame_{k:02d}.png"), b_rgb)

                    captured_dataset[face][pos_id] = rgb_frames

                    # Green flash confirmation
                    flash_img = rep_frame.copy()
                    cv2.rectangle(flash_img, (0, 0), (flash_img.shape[1], 65), (0, 180, 0), -1)
                    succ_msg = f"FACE {face} [{pos_name.upper()}]: {len(rgb_frames)} FRAMES SAVED!"
                    cv2.putText(flash_img, succ_msg, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow(win_name, flash_img)
                    cv2.waitKey(400)
                    print(f"    [OK] Captured Face {face} ({pos_name}: {len(rgb_frames)} frames)")
                else:
                    print(f"    [WARN] No frames captured for Face {face} at {pos_name}!")

    cv2.destroyAllWindows()
    return captured_dataset


def load_dataset_from_dir(
    dataset_dir: str,
    positions: list[dict] | None = None,
    augment_single: bool = True,
) -> dict[int, dict[str, list[np.ndarray]]]:
    """Loads captured face sequences from dataset_dir across specified workspace positions.

    Returns:
        dict mapping face (1..6) -> dict mapping pos_id -> list of frames (np.ndarray)
    """
    dataset: dict[int, dict[str, list[np.ndarray]]] = {}
    active_positions = positions if positions is not None else WORKSPACE_POSITIONS

    for face in range(1, 7):
        burst_dir = os.path.join(dataset_dir, f"face_{face}_burst")
        face_dict: dict[str, list[np.ndarray]] = {}

        for pos in active_positions:
            pos_id = pos["id"]
            frames = []

            # 1. Check position-specific burst frames in burst_dir (pos_{pos_id}_frame_*.png)
            if os.path.isdir(burst_dir):
                prefix = f"pos_{pos_id}_frame_"
                pos_files = sorted([
                    f for f in os.listdir(burst_dir)
                    if f.startswith(prefix) and f.endswith((".png", ".jpg", ".jpeg"))
                ])
                for fname in pos_files:
                    fpath = os.path.join(burst_dir, fname)
                    im = cv2.imread(fpath)
                    if im is not None:
                        frames.append(im)

            # 2. Check position-specific still image (face_{face}_{pos_id}.png)
            if not frames:
                cand = os.path.join(dataset_dir, f"face_{face}_{pos_id}.png")
                if os.path.exists(cand):
                    im = cv2.imread(cand)
                    if im is not None:
                        frames.append(im)

            # 3. Fallback for center position: legacy frame_*.png or face_{face}.png
            if not frames and pos_id == "center":
                if os.path.isdir(burst_dir):
                    legacy_files = sorted([
                        f for f in os.listdir(burst_dir)
                        if f.startswith("frame_") and f.endswith((".png", ".jpg", ".jpeg"))
                    ])
                    for fname in legacy_files:
                        im = cv2.imread(os.path.join(burst_dir, fname))
                        if im is not None:
                            frames.append(im)

                if not frames:
                    for ext in (".png", ".jpg", ".jpeg"):
                        cand = os.path.join(dataset_dir, f"face_{face}{ext}")
                        if os.path.exists(cand):
                            im = cv2.imread(cand)
                            if im is not None:
                                frames.append(im)
                                break

                if not frames:
                    pkg_root = get_package_root()
                    legacy_dir = os.path.join(pkg_root, "output", "calibration_dataset")
                    if os.path.isdir(legacy_dir):
                        for ext in (".png", ".jpg", ".jpeg"):
                            cand = os.path.join(legacy_dir, f"face_{face}{ext}")
                            if os.path.exists(cand):
                                im = cv2.imread(cand)
                                if im is not None:
                                    frames.append(im)
                                    os.makedirs(dataset_dir, exist_ok=True)
                                    cv2.imwrite(os.path.join(dataset_dir, f"face_{face}.png"), im)
                                    break

            if frames:
                if len(frames) == 1 and augment_single:
                    base = frames[0]
                    rng = np.random.RandomState(42 + face)
                    noise_f = np.clip(base.astype(np.float32) + rng.normal(0, 2.5, base.shape), 0, 255).astype(np.uint8)
                    f_p3 = np.clip(base.astype(np.int16) + 3, 0, 255).astype(np.uint8)
                    f_m3 = np.clip(base.astype(np.int16) - 3, 0, 255).astype(np.uint8)
                    f_p6 = np.clip(base.astype(np.int16) + 6, 0, 255).astype(np.uint8)
                    f_m6 = np.clip(base.astype(np.int16) - 6, 0, 255).astype(np.uint8)
                    face_dict[pos_id] = [base, f_p3, f_m3, noise_f, f_p6, f_m6]
                else:
                    face_dict[pos_id] = frames

        if face_dict:
            dataset[face] = face_dict

    return dataset


# ──────────────────────────────────────────────────────────────────────────────
# 3. Parameter Evaluation & Optimization Engine (Parallel Top-K Beam Search)
# ──────────────────────────────────────────────────────────────────────────────

_WORKER_PRIMARY_FRAMES: list[tuple[int, np.ndarray]] = []
_WORKER_BURST_DATASET: dict[int, list[np.ndarray]] = {}


def _init_eval_worker(primary_frames: list[tuple[int, np.ndarray]], burst_dataset: dict[int, list[np.ndarray]]) -> None:
    """Stores dataset references in worker process memory to avoid inter-process pickling overhead."""
    global _WORKER_PRIMARY_FRAMES, _WORKER_BURST_DATASET
    _WORKER_PRIMARY_FRAMES = primary_frames
    _WORKER_BURST_DATASET = burst_dataset


def _eval_frame_core(detector: RGBDieDetector, img: np.ndarray):
    """Core single-frame evaluation extracting faces, top face pips, alignment, and circularity."""
    det = detector.detect(img)
    faces = det.get("faces", [])
    top_face = None
    if faces:
        min_y = 1e9
        for f in faces:
            pts = f["polygon"]
            cy = np.mean(pts[:, 0, 1])
            if cy < min_y:
                min_y = cy
                top_face = f

    detected_pips = top_face.get("num_pips", 0) if top_face is not None else 0
    is_trusted = top_face.get("is_trusted", True) if top_face is not None else False
    aligned = top_face.get("aligned_pips", True) if top_face is not None else False

    pips_list = top_face.get("pips", []) if top_face is not None else []
    circs = []
    for p in pips_list:
        c = p.get("contour")
        if c is not None:
            a = cv2.contourArea(c)
            pm = cv2.arcLength(c, True)
            if pm > 0:
                circs.append((4 * np.pi * a) / (pm ** 2))
    avg_circ = float(np.mean(circs)) if circs else 0.0

    return det, top_face, detected_pips, is_trusted, aligned, avg_circ, len(faces)


def _eval_fast_worker(params_dict: dict) -> tuple[int, float]:
    """Fast candidate evaluation across representative reference frames covering all positions."""
    global _WORKER_PRIMARY_FRAMES
    try:
        p = DieDetectorParams(**params_dict)
        detector = RGBDieDetector(p)
        correct_by_face: dict[int, int] = {}
        total_by_face: dict[int, int] = {}
        score = 0.0

        for item in _WORKER_PRIMARY_FRAMES:
            expected_face = item[0]
            img = item[-1]
            _, _, pips, trusted, aligned, circ, _ = _eval_frame_core(detector, img)
            total_by_face[expected_face] = total_by_face.get(expected_face, 0) + 1
            if (pips == expected_face) and trusted and aligned:
                correct_by_face[expected_face] = correct_by_face.get(expected_face, 0) + 1
                score += 100.0 + 10.0 * circ
            else:
                err = abs(pips - expected_face)
                score -= 20.0 * max(1, err) - (15.0 if not trusted else 0.0)

        # Accuracy is count of faces (1..6) that succeed across all tested positions
        acc = sum(1 for f, tot in total_by_face.items() if correct_by_face.get(f, 0) == tot)
        return acc, score
    except Exception:
        return 0, -999.0


def _eval_stability_worker(params_dict: dict) -> tuple[int, float, dict[int, dict]]:
    """Evaluates candidates across the full burst frame dataset with jitter and stability penalties."""
    global _WORKER_BURST_DATASET
    try:
        p = DieDetectorParams(**params_dict)
        accuracy, score, details = evaluate_parameters(p, _WORKER_BURST_DATASET, fast_prune=True)
        return accuracy, score, details
    except Exception as e:
        return 0, -999.0, {"error": str(e)}


def evaluate_parameters(
    params: DieDetectorParams,
    dataset: dict[int, Any],
    fast_prune: bool = True,
) -> tuple[int, float, dict[int, dict]]:
    """Evaluates parameters across frame sequences with temporal stability & jitter metrics per position.

    Returns:
        accuracy: count of reliably classified faces (all positions reliable) (0 to 6)
        score: total numerical score (accuracy + stability bonus - jitter penalty)
        details: per-face result dict containing 'positions' breakdown
    """
    detector = RGBDieDetector(params)
    accuracy = 0
    score = 0.0
    details: dict[int, dict] = {}

    for expected_face, pos_data in dataset.items():
        if not pos_data:
            continue

        if isinstance(pos_data, dict):
            pos_dict = pos_data
        elif isinstance(pos_data, list):
            pos_dict = {"center": pos_data}
        else:
            continue

        face_positions: dict[str, dict] = {}
        all_positions_reliable = True
        face_stabilities = []
        face_circs = []

        for pos_id, frames in pos_dict.items():
            if not frames:
                continue

            try:
                # Step 1: Evaluate primary reference frame (frame 0)
                det0, top0, pips0, trusted0, aligned0, circ0, num_faces0 = _eval_frame_core(detector, frames[0])
                is_correct0 = (pips0 == expected_face) and trusted0 and aligned0

                # Fast pruning on primary frame if requested
                if fast_prune and not is_correct0:
                    err = abs(pips0 - expected_face)
                    pos_score = -20.0 * max(1, err) - (15.0 if not trusted0 else 0.0)
                    score += pos_score
                    all_positions_reliable = False
                    face_positions[pos_id] = {
                        "expected": expected_face,
                        "detected": pips0,
                        "detected_summary": str(pips0),
                        "is_correct": False,
                        "stability_ratio": 0.0,
                        "stability_percent": 0.0,
                        "pip_std": 0.0,
                        "avg_circ": circ0,
                        "num_faces": num_faces0,
                        "det": det0,
                        "frame": frames[0],
                    }
                    continue

                # Step 2: Full sequence evaluation across burst frames
                all_pips = [pips0]
                all_circs = [circ0] if circ0 > 0 else []
                correct_frames = 1 if is_correct0 else 0

                for frame in frames[1:]:
                    _, _, p_i, tr_i, al_i, c_i, _ = _eval_frame_core(detector, frame)
                    all_pips.append(p_i)
                    if c_i > 0:
                        all_circs.append(c_i)
                    if (p_i == expected_face) and tr_i and al_i:
                        correct_frames += 1

                total_frames = len(frames)
                stability_ratio = float(correct_frames) / float(total_frames) if total_frames > 0 else 0.0
                pip_std = float(np.std(all_pips))
                mean_circ = float(np.mean(all_circs)) if all_circs else circ0

                counts = Counter(all_pips)
                modal_pip, _ = counts.most_common(1)[0]
                unique_pips = sorted(list(set(all_pips)))
                if len(unique_pips) == 1:
                    det_summary = str(unique_pips[0])
                else:
                    det_summary = ", ".join(str(x) for x in unique_pips) + " (Jitter)"

                # Scoring per position
                pos_score = 100.0 * stability_ratio + 10.0 * mean_circ
                if stability_ratio == 1.0 and pip_std == 0.0:
                    pos_score += 20.0
                pos_score -= 40.0 * pip_std
                if stability_ratio < 0.85:
                    pos_score -= 30.0 * (1.0 - stability_ratio)

                is_reliable = (stability_ratio >= 0.85) and (modal_pip == expected_face)
                if not is_reliable:
                    all_positions_reliable = False

                score += pos_score
                face_stabilities.append(stability_ratio * 100.0)
                if mean_circ > 0:
                    face_circs.append(mean_circ)

                face_positions[pos_id] = {
                    "expected": expected_face,
                    "detected": modal_pip,
                    "detected_summary": det_summary,
                    "is_correct": is_reliable,
                    "stability_ratio": stability_ratio,
                    "stability_percent": round(stability_ratio * 100.0, 1),
                    "pip_std": round(pip_std, 2),
                    "avg_circ": mean_circ,
                    "num_faces": num_faces0,
                    "det": det0,
                    "frame": frames[len(frames) // 2],
                }

            except Exception as e:
                score -= 100.0
                all_positions_reliable = False
                face_positions[pos_id] = {
                    "expected": expected_face,
                    "detected": -1,
                    "detected_summary": "Error",
                    "is_correct": False,
                    "stability_ratio": 0.0,
                    "stability_percent": 0.0,
                    "pip_std": 0.0,
                    "avg_circ": 0.0,
                    "num_faces": 0,
                    "det": {},
                    "frame": frames[0] if frames else None,
                    "error": str(e),
                }

        # Aggregate face-level stats
        if all_positions_reliable and len(face_positions) > 0:
            accuracy += 1

        avg_face_stab = float(np.mean(face_stabilities)) if face_stabilities else 0.0
        avg_face_circ = float(np.mean(face_circs)) if face_circs else 0.0

        # Modal pip / summary across positions (prefer center position if available)
        center_pos = face_positions.get("center")
        if center_pos is not None:
            face_detected = center_pos["detected"]
            face_summary = center_pos["detected_summary"]
            rep_det = center_pos["det"]
            rep_frame = center_pos["frame"]
        elif face_positions:
            first_pos = list(face_positions.values())[0]
            face_detected = first_pos["detected"]
            face_summary = first_pos["detected_summary"]
            rep_det = first_pos["det"]
            rep_frame = first_pos["frame"]
        else:
            face_detected = -1
            face_summary = "N/A"
            rep_det = {}
            rep_frame = None

        details[expected_face] = {
            "expected": expected_face,
            "detected": face_detected,
            "detected_summary": face_summary,
            "is_correct": all_positions_reliable,
            "stability_percent": round(avg_face_stab, 1),
            "avg_circ": avg_face_circ,
            "positions": face_positions,
            "det": rep_det,
            "frame": rep_frame,
        }

    return accuracy, score, details


def analyze_scene_statistics(primary_items: list[tuple] | dict[int, np.ndarray]) -> dict:
    """Extracts empirical contrast, brightness, and color statistics from calibration images across positions."""
    if isinstance(primary_items, dict):
        items = list(primary_items.items())
    else:
        items = primary_items

    v_die_vals = []
    s_die_vals = []
    v_table_vals = []

    for item in items:
        img = item[-1]
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu threshold as an initial table vs foreground separator
        thresh_val, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find die candidate contours anywhere across the frame (center or corners)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        die_mask = np.zeros((h, w), dtype=bool)
        for c in cnts:
            a = cv2.contourArea(c)
            if 300 < a < 25000:
                bx, by, bw, bh = cv2.boundingRect(c)
                if bx > 2 and by > 2 and (bx + bw < w - 2) and (by + bh < h - 2):
                    c_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(c_mask, [c], -1, 255, -1)
                    die_mask |= (c_mask > 0)

        table_mask = (gray <= thresh_val) & ~die_mask
        # Exclude outer frame border
        table_mask[:10, :] = False
        table_mask[-10:, :] = False
        table_mask[:, :10] = False
        table_mask[:, -10:] = False

        if np.count_nonzero(die_mask) > 100:
            v_die_vals.append(float(np.median(hsv[die_mask, 2])))
            s_die_vals.append(float(np.median(hsv[die_mask, 1])))
        if np.count_nonzero(table_mask) > 100:
            v_table_vals.append(float(np.median(hsv[table_mask, 2])))

    v_table_med = float(np.median(v_table_vals)) if v_table_vals else 125.0
    v_die_med = float(np.median(v_die_vals)) if v_die_vals else 230.0
    s_die_med = float(np.median(s_die_vals)) if s_die_vals else 75.0

    # Clean bounds centered around physical separation point
    v_min_center = int(round((v_table_med + v_die_med) / 2.0))
    v_min_candidates = sorted(list(set([
        int(np.clip(v_table_med + 15, 110, 185)),
        int(np.clip(v_table_med + 25, 110, 185)),
        int(np.clip(v_min_center, 120, 185)),
        160, 170  # Include proven empirical values
    ])))

    # For white dice, saturation is low (close to 0). S_max should be generously high
    # to avoid clipping subtle warm lighting or colored reflections on die edges.
    s_max_candidates = sorted(list(set([
        95, 110, 130, 160,
        int(np.clip(s_die_med + 35, 95, 180))
    ])))

    glare_candidates = [215, 225, 240]

    return {
        "v_table": v_table_med,
        "v_die": v_die_med,
        "s_die": s_die_med,
        "v_min_candidates": v_min_candidates,
        "s_max_candidates": s_max_candidates,
        "glare_candidates": glare_candidates,
    }


def select_top_diverse(
    candidates_with_scores: list[tuple[DieDetectorParams, int, float]],
    top_k: int,
    param_keys: list[str],
) -> list[DieDetectorParams]:
    """Selects top K diverse candidates to seed the next optimization stage."""
    sorted_cands = sorted(candidates_with_scores, key=lambda x: (x[1], x[2]), reverse=True)
    selected: list[DieDetectorParams] = []
    seen_combos = set()

    for cand, acc, score in sorted_cands:
        key_tuple = tuple(getattr(cand, k) for k in param_keys)
        # Convert lists to tuples for hashing
        key_tuple = tuple(tuple(x) if isinstance(x, list) else x for x in key_tuple)
        if key_tuple not in seen_combos:
            seen_combos.add(key_tuple)
            selected.append(cand)
            if len(selected) >= top_k:
                break

    # If diversity filter picked fewer than top_k, fill with remaining highest scoring
    if len(selected) < top_k:
        for cand, _, _ in sorted_cands:
            if cand not in selected:
                selected.append(cand)
                if len(selected) >= top_k:
                    break

    return selected


def run_parameter_optimization(
    dataset: dict[int, list[np.ndarray]],
    base_params: DieDetectorParams,
    quick: bool = False,
) -> tuple[DieDetectorParams, int, float, dict[int, dict]]:
    """Runs a high-speed, parallel Top-K beam search optimization over key detection parameters."""
    print("\n" + "=" * 70)
    print("  STARTING HIGH-SPEED PARALLEL BEAM SEARCH OPTIMIZATION")
    print("=" * 70)
    print(f"Dataset: {len(dataset)} faces loaded ({sorted(list(dataset.keys()))})")

    # Sample representative frames across captured positions for fast screening
    primary_eval_items: list[tuple[int, str, np.ndarray]] = []
    for face, pos_data in dataset.items():
        if isinstance(pos_data, dict):
            for pos_id, frames in pos_data.items():
                if frames:
                    mid_idx = len(frames) // 2
                    primary_eval_items.append((face, pos_id, frames[mid_idx]))
        elif isinstance(pos_data, list):
            n = len(pos_data)
            indices = list(range(min(n, 5)))
            for idx in indices:
                primary_eval_items.append((face, f"pos_{idx}", pos_data[idx]))

    num_cpus = os.cpu_count() or 4
    num_workers = min(num_cpus, 12 if not quick else 8)
    print(f"[INFO] Initializing parallel worker pool with {num_workers} worker processes...")
    print(f"[INFO] Evaluating fast screening across {len(primary_eval_items)} representative position frames...")

    # Data-driven scene analysis across all captured frames
    scene_stats = analyze_scene_statistics(primary_eval_items)
    print(f"[Scene Analysis] Table V: {scene_stats['v_table']:.1f} | Die V: {scene_stats['v_die']:.1f} | Die S: {scene_stats['s_die']:.1f}")
    print(f"                 Anchored V_min search: {scene_stats['v_min_candidates']}")
    print(f"                 Anchored S_max search: {scene_stats['s_max_candidates']}")

    # Baseline evaluation
    base_acc, base_score, best_details = evaluate_parameters(base_params, dataset, fast_prune=False)
    avg_stab = float(np.mean([d.get("stability_percent", 0.0) for d in best_details.values()])) if best_details else 0.0
    print(f"[Initial Baseline] Accuracy: {base_acc}/6 faces reliable | Avg Stability: {avg_stab:.1f}% (Score: {base_score:.1f})")

    t_start = time.time()

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_eval_worker,
        initargs=(primary_eval_items, dataset),
    ) as pool:

        # ── Stage 1: Die Body Segmentation (HSV & Glare Sweep) ───────────────
        print("\n--- [Stage 1/4] Die Body Color & Illumination Sweep (Parallel) ---")
        t0 = time.time()
        stage1_cands = [copy.deepcopy(base_params)]

        v_mins = scene_stats["v_min_candidates"] if not quick else scene_stats["v_min_candidates"][:3]
        s_maxs = scene_stats["s_max_candidates"] if not quick else scene_stats["s_max_candidates"][:3]
        glares = scene_stats["glare_candidates"]

        for v_min in v_mins:
            for s_max in s_maxs:
                for glare in glares:
                    c = copy.deepcopy(base_params)
                    c.hsv_min = [0, 0, v_min]
                    c.hsv_max = [180, s_max, 255]
                    c.glare_v_thresh = glare
                    stage1_cands.append(c)

        dicts1 = [c.to_dict() for c in stage1_cands]
        results1 = list(pool.map(_eval_fast_worker, dicts1))
        cands_with_scores1 = [(stage1_cands[i], results1[i][0], results1[i][1]) for i in range(len(stage1_cands))]

        top_k1 = 6 if not quick else 4
        beam_seeds1 = select_top_diverse(cands_with_scores1, top_k1, ["hsv_min", "hsv_max"])
        best_s1 = max(cands_with_scores1, key=lambda x: (x[1], x[2]))
        print(f"Stage 1 evaluated {len(stage1_cands)} candidates in {time.time() - t0:.2f}s. Top accuracy: {best_s1[1]}/6 (Score: {best_s1[2]:.1f})")

        # ── Stage 2: Pip Contrast & Adaptive Thresholding Sweep ──────────────
        print("\n--- [Stage 2/4] Pip Contrast & Adaptive Thresholding Sweep (Parallel) ---")
        t0 = time.time()
        stage2_cands = [copy.deepcopy(base_params)]

        clahe_list = [1.5, 2.5, 3.5] if not quick else [1.5, 2.5]
        block_list = [15, 21, 27] if not quick else [21, 27]
        c_list = [4, 6]
        morph_list = [3, 5] if not quick else [3]

        for seed in beam_seeds1:
            stage2_cands.append(copy.deepcopy(seed))
            for clahe in clahe_list:
                for b_size in block_list:
                    for c_val in c_list:
                        for m_ksize in morph_list:
                            c = copy.deepcopy(seed)
                            c.clahe_clip_limit = clahe
                            c.adaptive_thresh_block_size = b_size
                            c.adaptive_thresh_c = c_val
                            c.morph_open_kernel_size = m_ksize
                            stage2_cands.append(c)

        dicts2 = [c.to_dict() for c in stage2_cands]
        results2 = list(pool.map(_eval_fast_worker, dicts2))
        cands_with_scores2 = [(stage2_cands[i], results2[i][0], results2[i][1]) for i in range(len(stage2_cands))]

        top_k2 = 8 if not quick else 6
        # Crucial: diversify on adaptive_thresh_c and block_size so both c=4 and c=6 survive
        beam_seeds2 = select_top_diverse(cands_with_scores2, top_k2, ["adaptive_thresh_c", "adaptive_thresh_block_size", "clahe_clip_limit"])
        best_s2 = max(cands_with_scores2, key=lambda x: (x[1], x[2]))
        print(f"Stage 2 evaluated {len(stage2_cands)} candidates in {time.time() - t0:.2f}s. Top accuracy: {best_s2[1]}/6 (Score: {best_s2[2]:.1f})")

        # ── Stage 3: Pip Geometry & Perspective Circularity Sweep ────────────
        print("\n--- [Stage 3/4] Pip Geometry & Filtering Sweep (Parallel) ---")
        t0 = time.time()
        stage3_cands = [copy.deepcopy(base_params)]

        circ_list = [0.18, 0.22, 0.28] if not quick else [0.20, 0.22, 0.25]
        min_pip_area_list = [0.003, 0.005, 0.008]
        pip_area_list = [0.05, 0.08]
        pip_radius_list = [0.18, 0.22]
        outlier_list = [0.25, 0.35]

        for seed in beam_seeds2:
            stage3_cands.append(copy.deepcopy(seed))
            for circ in circ_list:
                for min_area in min_pip_area_list:
                    for max_area in pip_area_list:
                        for max_rad in pip_radius_list:
                            for outl in outlier_list:
                                c = copy.deepcopy(seed)
                                c.pip_min_circularity = circ
                                c.min_pip_area_ratio = min_area
                                c.max_pip_area_ratio = max_area
                                c.max_pip_radius_ratio = max_rad
                                c.pip_area_outlier_ratio = outl
                                stage3_cands.append(c)

        dicts3 = [c.to_dict() for c in stage3_cands]
        results3 = list(pool.map(_eval_fast_worker, dicts3))
        cands_with_scores3 = [(stage3_cands[i], results3[i][0], results3[i][1]) for i in range(len(stage3_cands))]

        top_k3 = 10 if not quick else 8
        finalists = select_top_diverse(cands_with_scores3, top_k3, ["adaptive_thresh_c", "pip_min_circularity", "min_pip_area_ratio"])
        
        # Always retain baseline if baseline had 6/6
        if base_acc == 6 and not any(f.to_dict() == base_params.to_dict() for f in finalists):
            finalists.append(copy.deepcopy(base_params))

        best_s3 = max(cands_with_scores3, key=lambda x: (x[1], x[2]))
        print(f"Stage 3 evaluated {len(stage3_cands)} candidates in {time.time() - t0:.2f}s. Top accuracy: {best_s3[1]}/6 (Score: {best_s3[2]:.1f})")

        # ── Stage 4: Multi-Frame Burst Temporal Stability Verification ───────
        print("\n--- [Stage 4/4] Multi-Frame Burst Temporal Stability Verification ---")
        t0 = time.time()
        finalist_dicts = [f.to_dict() for f in finalists]
        stability_results = list(pool.map(_eval_stability_worker, finalist_dicts))

        # Select overall winner: highest accuracy, highest stability %, lowest jitter (std), highest score
        best_idx = 0
        best_key = (-1, -1.0, 999.0, -9999.0)
        for i, (acc, score, details) in enumerate(stability_results):
            stabs = [d.get("stability_percent", 0.0) for d in details.values()]
            avg_stab_i = float(np.mean(stabs)) if stabs else 0.0
            stds = []
            for d in details.values():
                for pinfo in d.get("positions", {}).values():
                    stds.append(pinfo.get("pip_std", 0.0))
                if "pip_std" in d:
                    stds.append(d["pip_std"])
            mean_std_i = float(np.mean(stds)) if stds else 0.0
            key = (acc, avg_stab_i, -mean_std_i, score)
            if key > best_key:
                best_key = key
                best_idx = i

        best_params = finalists[best_idx]
        best_acc, best_score, best_details = stability_results[best_idx]
        print(f"Stage 4 verified {len(finalists)} finalists in {time.time() - t0:.2f}s.")

    total_time = time.time() - t_start
    avg_stab = float(np.mean([d.get("stability_percent", 0.0) for d in best_details.values()])) if best_details else 0.0
    print(f"\n[OPTIMIZATION COMPLETE] Total run-time: {total_time:.1f}s | Final Accuracy: {best_acc}/6 ({best_acc/6.0*100:.1f}%) | Avg Stability: {avg_stab:.1f}%")

    return best_params, best_acc, best_score, best_details



# ──────────────────────────────────────────────────────────────────────────────
# 4. Result Presentation & Verification Collage
# ──────────────────────────────────────────────────────────────────────────────

def create_summary_collage(
    dataset: dict[int, Any],
    params: DieDetectorParams,
    details: dict[int, dict],
) -> np.ndarray:
    """Renders a 2x3 collage showing the detection output and stability metrics for all 6 faces."""
    detector = RGBDieDetector(params)
    panels = []
    pw, ph = 480, 360

    for face in range(1, 7):
        item = dataset.get(face)
        if item is None:
            # Blank placeholder
            canvas = np.zeros((ph, pw, 3), dtype=np.uint8)
            cv2.putText(canvas, f"Face {face}: Missing", (50, ph // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            panels.append(canvas)
            continue

        if isinstance(item, dict):
            frames = item.get("center") or (list(item.values())[0] if item else [])
            img = frames[len(frames) // 2] if frames else None
        elif isinstance(item, (list, tuple)):
            img = item[0] if len(item) > 0 else None
        else:
            img = item

        if img is None:
            canvas = np.zeros((ph, pw, 3), dtype=np.uint8)
            panels.append(canvas)
            continue

        det = detector.detect(img)
        faces_pips = det.get("debug_steps", {}).get("step4_faces_and_pips", img)
        panel = cv2.resize(faces_pips, (pw, ph))

        det_info = details.get(face, {})
        detected = det_info.get("detected", 0)
        is_ok = det_info.get("is_correct", False)
        stab_pct = det_info.get("stability_percent", 100.0)

        # Status banner at top
        banner_color = (0, 160, 0) if is_ok else ((0, 140, 255) if stab_pct > 0 else (0, 0, 200))
        cv2.rectangle(panel, (0, 0), (pw, 45), (20, 20, 20), -1)
        cv2.rectangle(panel, (0, 43), (pw, 45), banner_color, 2)

        status_str = "PASS" if is_ok else ("JITTER" if stab_pct > 0 else "FAIL")
        label = f"Face {face}: {detected} pips | Stab:{stab_pct:.0f}% [{status_str}]"
        cv2.putText(panel, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                    (0, 255, 0) if is_ok else (0, 165, 255) if stab_pct > 0 else (0, 100, 255), 2, cv2.LINE_AA)

        # Sub-banner showing status of each position if extended test
        pos_dict = det_info.get("positions", {})
        if len(pos_dict) > 1:
            cv2.rectangle(panel, (0, ph - 28), (pw, ph), (18, 18, 18), -1)
            short_codes = {"center": "C", "top_left": "TL", "top_right": "TR", "bottom_right": "BR", "bottom_left": "BL"}
            x_pos = 12
            for pid, pdata in pos_dict.items():
                code = short_codes.get(pid, pid[:2].upper())
                p_ok = pdata.get("is_correct", False)
                p_text = f"{code}:{'PASS' if p_ok else 'FAIL'}"
                p_col = (0, 220, 0) if p_ok else (0, 60, 255)
                cv2.putText(panel, p_text, (x_pos, ph - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.44, p_col, 1, cv2.LINE_AA)
                x_pos += 88

        panels.append(panel)

    row1 = np.hstack(panels[0:3])
    row2 = np.hstack(panels[3:6])
    collage = np.vstack([row1, row2])
    return collage


def create_multi_position_summary_collage(
    dataset: dict[int, Any],
    params: DieDetectorParams,
    details: dict[int, dict],
) -> np.ndarray:
    """Renders a 6 (faces) x 5 (positions) matrix collage showing every position individually."""
    detector = RGBDieDetector(params)
    pw, ph = 280, 210
    pos_ids = [p["id"] for p in WORKSPACE_POSITIONS]
    pos_names = {p["id"]: p["name"] for p in WORKSPACE_POSITIONS}

    face_rows = []
    for face in range(1, 7):
        pos_dict = dataset.get(face, {})
        face_det = details.get(face, {})
        pos_details = face_det.get("positions", {})
        row_cells = []

        for pos_id in pos_ids:
            frames = pos_dict.get(pos_id, []) if isinstance(pos_dict, dict) else []
            if not frames:
                cell = np.zeros((ph, pw, 3), dtype=np.uint8)
                cv2.putText(cell, f"F{face}: {pos_id}", (20, ph // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
                cv2.putText(cell, "Not Captured", (20, ph // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
                row_cells.append(cell)
                continue

            img = frames[len(frames) // 2]
            det = detector.detect(img)
            debug_img = det.get("debug_steps", {}).get("step4_faces_and_pips", img)
            cell = cv2.resize(debug_img, (pw, ph))

            pinfo = pos_details.get(pos_id, {})
            det_pips = pinfo.get("detected", -1)
            is_ok = pinfo.get("is_correct", False)
            stab = pinfo.get("stability_percent", 0.0)

            # Top bar
            b_color = (0, 160, 0) if is_ok else ((0, 140, 255) if stab > 0 else (0, 0, 200))
            cv2.rectangle(cell, (0, 0), (pw, 30), (20, 20, 20), -1)
            cv2.rectangle(cell, (0, 28), (pw, 30), b_color, 2)
            lbl = f"F{face} - {pos_names[pos_id].split()[0]}: {det_pips} pips"
            cv2.putText(cell, lbl, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

            # Bottom bar
            status_txt = "PASS" if is_ok else "FAIL"
            cv2.rectangle(cell, (0, ph - 24), (pw, ph), (20, 20, 20), -1)
            stat_lbl = f"Stab: {stab:.0f}% [{status_txt}]"
            cv2.putText(cell, stat_lbl, (8, ph - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.48, b_color, 1, cv2.LINE_AA)

            row_cells.append(cell)

        face_rows.append(np.hstack(row_cells))

    matrix_collage = np.vstack(face_rows)
    return matrix_collage


def print_results_table(details: dict[int, dict], best_params: DieDetectorParams, test_mode: str = "extended"):
    print("\n" + "=" * 92)
    title = f"OPTIMIZATION FINAL RESULTS ({test_mode.upper()} TEST)"
    print(f"{title:^92}")
    print("=" * 92)
    print(f" {'Face':<6} | {'Position':<18} | {'Expected':<10} | {'Detected':<18} | {'Circularity':<11} | {'Stability':<10} | {'Status':<8}")
    print("-" * 92)

    total_faces_pass = 0
    total_positions_pass = 0
    total_positions_count = 0
    all_stabilities = []

    pos_name_map = {p["id"]: p["name"] for p in WORKSPACE_POSITIONS}

    for f in range(1, 7):
        d = details.get(f, {})
        exp = d.get("expected", f)
        positions = d.get("positions", {})

        face_pos_pass = 0
        face_pos_total = len(positions)

        if not positions:
            # Fallback if no position breakdown
            det_sum = d.get("detected_summary", str(d.get("detected", "N/A")))
            circ = f"{d.get('avg_circ', 0.0):.2f}" if d.get("avg_circ") else "N/A"
            stab_pct = float(d.get("stability_percent", 0.0))
            all_stabilities.append(stab_pct)
            ok = d.get("is_correct", False)
            if ok:
                total_faces_pass += 1
            status = "PASS" if ok else ("FAIL (Jitter)" if stab_pct > 0 else "FAIL")
            print(f" {f:<6} | {'Center':<18} | {exp:<10} | {det_sum:<18} | {circ:<11} | {stab_pct:>6.1f}%   | {status:<8}")
        else:
            first_row = True
            for pos_id, pinfo in positions.items():
                pos_display = pos_name_map.get(pos_id, pos_id.replace("_", " ").title())
                p_exp = pinfo.get("expected", exp)
                p_det = pinfo.get("detected_summary", str(pinfo.get("detected", "N/A")))
                p_circ = f"{pinfo.get('avg_circ', 0.0):.2f}" if pinfo.get("avg_circ") else "N/A"
                p_stab = float(pinfo.get("stability_percent", 0.0))
                all_stabilities.append(p_stab)
                p_ok = pinfo.get("is_correct", False)
                total_positions_count += 1
                if p_ok:
                    face_pos_pass += 1
                    total_positions_pass += 1
                p_status = "PASS" if p_ok else ("FAIL (Jitter)" if p_stab > 0 else "FAIL")

                face_col = str(f) if first_row else ""
                print(f" {face_col:<6} | {pos_display:<18} | {p_exp:<10} | {p_det:<18} | {p_circ:<11} | {p_stab:>6.1f}%   | {p_status:<8}")
                first_row = False

            face_ok = d.get("is_correct", False)
            if face_ok:
                total_faces_pass += 1
            face_stab_avg = d.get("stability_percent", 0.0)
            face_status_str = "PASS" if face_ok else "FAIL"

            if face_pos_total > 1:
                summary_line = f"  ===> Face {f} Overall: {face_pos_pass}/{face_pos_total} positions passed (Avg Stability: {face_stab_avg:.1f}%) [{face_status_str}]"
                print(f"{summary_line:<92}")
                print("-" * 92)
            else:
                print("-" * 92)

    print("=" * 92)
    overall_avg_stab = float(np.mean(all_stabilities)) if all_stabilities else 0.0
    print(" SUMMARY:")
    print(f"   Faces Reliable (All Positions):  {total_faces_pass}/6 ({total_faces_pass / 6.0 * 100:.1f}%)")
    if total_positions_count > 6:
        print(f"   Workspace Positions Passed:      {total_positions_pass}/{total_positions_count} ({total_positions_pass / float(total_positions_count) * 100:.1f}%)")
    print(f"   Average Sequence Stability:      {overall_avg_stab:.1f}%")
    print("=" * 92 + "\n")

    print("Winning Parameters:")
    print(f"  hsv_min:                    {best_params.hsv_min}")
    print(f"  hsv_max:                    {best_params.hsv_max}")
    print(f"  glare_v_thresh:             {best_params.glare_v_thresh}")
    print(f"  clahe_clip_limit:           {best_params.clahe_clip_limit}")
    print(f"  adaptive_thresh_block_size: {best_params.adaptive_thresh_block_size}")
    print(f"  adaptive_thresh_c:          {best_params.adaptive_thresh_c}")
    print(f"  morph_open_kernel_size:     {best_params.morph_open_kernel_size}")
    print(f"  canny_low_thresh:           {best_params.canny_low_thresh}")
    print(f"  canny_high_thresh:          {best_params.canny_high_thresh}")
    print(f"  pip_min_circularity:        {best_params.pip_min_circularity}")
    print(f"  min_pip_area_ratio:         {best_params.min_pip_area_ratio}")
    print(f"  max_pip_area_ratio:         {best_params.max_pip_area_ratio}")
    print(f"  max_pip_radius_ratio:       {best_params.max_pip_radius_ratio}")
    print(f"  pip_area_outlier_ratio:     {best_params.pip_area_outlier_ratio}")
    print("=" * 92 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Main CLI Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def save_calibration_results(
    output_dir: str,
    best_params: DieDetectorParams,
    best_acc: int,
    best_score: float,
    details: dict[int, dict],
    test_mode: str = "extended",
) -> str:
    """Saves comprehensive calibration results and optimal parameters to YAML."""
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "calibration_results.yaml")

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    per_face_summary = {}
    stabs = []
    for f in range(1, 7):
        d = details.get(f, {})
        stab_pct = float(round(float(d.get("stability_percent", 0.0)), 1))
        stabs.append(stab_pct)

        positions_summary = {}
        for pos_id, pinfo in d.get("positions", {}).items():
            positions_summary[pos_id] = {
                "expected_pips": int(pinfo.get("expected", f)),
                "detected_pips": int(pinfo.get("detected", -1)),
                "detected_summary": str(pinfo.get("detected_summary", str(pinfo.get("detected", -1)))),
                "stability_percent": float(pinfo.get("stability_percent", 0.0)),
                "circularity": float(round(float(pinfo.get("avg_circ", 0.0)), 4)),
                "is_correct": bool(pinfo.get("is_correct", False)),
            }

        face_info = {
            "expected_pips": int(d.get("expected", f)),
            "detected_pips": int(d.get("detected", -1)),
            "detected_summary": str(d.get("detected_summary", str(d.get("detected", -1)))),
            "stability_percent": stab_pct,
            "circularity": float(round(float(d.get("avg_circ", 0.0)), 4)),
            "num_faces_detected": int(d.get("num_faces", 0)),
            "is_correct": bool(d.get("is_correct", False)),
            "positions": positions_summary,
        }
        if "error" in d:
            face_info["error"] = str(d["error"])
        per_face_summary[f"face_{f}"] = face_info

    params_dict = best_params.to_dict()
    params_dict.pop("camera_frame_id", None)

    for k, v in list(params_dict.items()):
        if isinstance(v, (np.floating, float)):
            params_dict[k] = float(v)
        elif isinstance(v, (np.integer, int)):
            params_dict[k] = int(v)
        elif isinstance(v, (list, tuple)):
            params_dict[k] = [int(x) if isinstance(x, (np.integer, int)) else float(x) if isinstance(x, (np.floating, float)) else x for x in v]

    data = {
        "calibration_info": {
            "timestamp": timestamp,
            "test_mode": test_mode,
            "accuracy": f"{int(best_acc)}/6",
            "accuracy_percent": round(float(best_acc) / 6.0 * 100.0, 2),
            "average_stability_percent": round(float(np.mean(stabs)), 2) if stabs else 0.0,
            "score": round(float(best_score), 2),
        },
        "per_face_results": per_face_summary,
        "optimal_parameters": {
            "hsv_min": [int(x) for x in best_params.hsv_min],
            "hsv_max": [int(x) for x in best_params.hsv_max],
            "glare_v_thresh": int(best_params.glare_v_thresh),
            "clahe_clip_limit": float(best_params.clahe_clip_limit),
            "adaptive_thresh_block_size": int(best_params.adaptive_thresh_block_size),
            "adaptive_thresh_c": int(best_params.adaptive_thresh_c),
            "morph_open_kernel_size": int(best_params.morph_open_kernel_size),
            "canny_low_thresh": int(best_params.canny_low_thresh),
            "canny_high_thresh": int(best_params.canny_high_thresh),
            "pip_min_circularity": float(best_params.pip_min_circularity),
            "min_pip_area_ratio": float(best_params.min_pip_area_ratio),
            "max_pip_area_ratio": float(best_params.max_pip_area_ratio),
            "max_pip_radius_ratio": float(best_params.max_pip_radius_ratio),
            "pip_area_outlier_ratio": float(best_params.pip_area_outlier_ratio),
        },
        "full_detector_parameters": params_dict,
    }

    with open(results_path, "w") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)

    return results_path


def find_config_yaml(user_path: Optional[str] = None) -> str:
    """Finds the die_detector_params.yaml configuration file reliably."""
    if user_path and os.path.isfile(user_path):
        return os.path.abspath(user_path)

    pkg_root = get_package_root()
    cand_pkg = os.path.join(pkg_root, "config", "die_detector_params.yaml")
    if os.path.isfile(cand_pkg):
        return cand_pkg

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Source workspace path: src/drims_die_detection/config/die_detector_params.yaml
    cand_src = os.path.abspath(os.path.join(script_dir, "..", "config", "die_detector_params.yaml"))
    if os.path.isfile(cand_src):
        return cand_src

    # 2. Installed share path: ../../share/drims_die_detection/config/die_detector_params.yaml
    cand_install = os.path.abspath(os.path.join(script_dir, "..", "..", "share", "drims_die_detection", "config", "die_detector_params.yaml"))
    if os.path.isfile(cand_install):
        return cand_install

    # 3. ament_index_python share directory
    try:
        from ament_index_python.packages import get_package_share_directory
        share_dir = get_package_share_directory("drims_die_detection")
        cand_share = os.path.join(share_dir, "config", "die_detector_params.yaml")
        if os.path.isfile(cand_share):
            return cand_share
    except Exception:
        pass

    # 4. Fallback search in known workspaces
    for ws_base in ["/home/galileo/projects/NDR_ws", os.getcwd()]:
        cand = os.path.join(ws_base, "src", "drims_die_detection", "config", "die_detector_params.yaml")
        if os.path.isfile(cand):
            return cand

    return cand_src


def main():
    parser = argparse.ArgumentParser(
        description="Interactive die detection parameter calibration and optimization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to die_detector_params.yaml (auto-detected if omitted).")
    parser.add_argument("--dataset-dir", type=str, default=None,
                        help="Directory to save / load face images (defaults to calibration_data in package).")
    parser.add_argument("--rgb-topic", type=str, default=None,
                        help="Optional override for ROS 2 color topic (defaults strictly to config file).")
    parser.add_argument("--depth-topic", type=str, default=None,
                        help="Optional override for ROS 2 depth topic (defaults strictly to config file).")
    parser.add_argument("--offline", action="store_true",
                        help="Skip camera acquisition and use existing images in --dataset-dir.")
    parser.add_argument("--auto-save", action="store_true",
                        help="Automatically save winning parameters to YAML without asking.")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick coarse grid search.")
    parser.add_argument(
        "--test-mode", "--mode",
        dest="test_mode",
        choices=["simple", "extended"],
        default="extended",
        help="Optimization test mode: 'simple' (1 position: center) or 'extended' (5 positions: center + 4 corners). Default: extended.",
    )
    parser.add_argument(
        "--simple",
        dest="test_mode",
        action="store_const",
        const="simple",
        help="Shortcut to run a simple test (1 position: center of workspace).",
    )
    parser.add_argument(
        "--extended",
        dest="test_mode",
        action="store_const",
        const="extended",
        help="Shortcut to run an extended test (5 positions: center + 4 corners).",
    )
    parser.add_argument(
        "--single-pos",
        dest="test_mode",
        action="store_const",
        const="simple",
        help="Alias for --simple (1 position: center of workspace).",
    )
    parser.add_argument("--burst-frames", type=int, default=8,
                        help="Number of consecutive frames to record per position during live calibration (default: 8).")
    args = parser.parse_args()

    # Load initial parameters from config file
    config_path = find_config_yaml(args.config)
    if os.path.isfile(config_path):
        base_params = DieDetectorParams.from_yaml(config_path)
        print(f"[INFO] Using configuration file: {config_path}")
    else:
        base_params = DieDetectorParams()
        print(f"[WARN] Config file not found at {config_path}; using defaults.")

    # Strict: Use topics specified in the config file unless explicitly overridden via CLI
    rgb_topic = args.rgb_topic if args.rgb_topic is not None else base_params.rgb_topic
    depth_topic = args.depth_topic if args.depth_topic is not None else base_params.depth_topic

    print(f"[INFO] Active Camera Topics (from config file):")
    print(f"       RGB:   {rgb_topic}")
    print(f"       Depth: {depth_topic}")

    test_mode = args.test_mode
    selected_positions = [WORKSPACE_POSITIONS[0]] if test_mode == "simple" else WORKSPACE_POSITIONS
    pos_desc = "center only (simple test)" if test_mode == "simple" else f"{len(WORKSPACE_POSITIONS)} workspace positions (extended test)"
    print(f"[INFO] Test Mode: {test_mode.upper()} ({pos_desc})")

    pkg_root = get_package_root()
    if args.dataset_dir:
        dataset_dir = os.path.abspath(args.dataset_dir)
    else:
        dataset_dir = os.path.abspath(os.path.join(pkg_root, "calibration_data"))
    os.makedirs(dataset_dir, exist_ok=True)

    dataset: dict[int, dict[str, list[np.ndarray]]] = {}

    # Check if user requested offline or if images already exist
    existing_images = load_dataset_from_dir(dataset_dir, positions=selected_positions)
    if args.offline:
        if len(existing_images) == 0:
            print(f"[ERROR] Offline mode selected but no valid images found in {dataset_dir}!")
            sys.exit(1)
        dataset = existing_images
        num_pos = sum(len(p) for p in dataset.values())
        print(f"[INFO] Offline mode: loaded {len(dataset)} faces ({num_pos} position sets) from {dataset_dir}")
    else:
        if len(existing_images) == 6:
            num_pos = sum(len(p) for p in existing_images.values())
            print(f"[INFO] Found 6 previously captured face datasets ({num_pos} position sets) in {dataset_dir}.")
            choice = input("Do you want to [1] Capture new frames from camera, or [2] Use existing images? [1/2] (default: 1): ").strip().lower()
            if choice in ("2", "existing", "e", "yes", "y"):
                dataset = existing_images
                print(f"[INFO] Retrieved {len(dataset)} existing face datasets ({num_pos} position sets) from {dataset_dir}")

        if not dataset:
            if not HAS_ROS2:
                print("[ERROR] ROS 2 is required for live camera capture.")
                print("Please source your ROS 2 Humble workspace or run with --offline.")
                sys.exit(1)

            print(f"[INFO] Connecting to camera topics:")
            print(f"       RGB:   {rgb_topic}")
            print(f"       Depth: {depth_topic}")
            grabber = CameraFrameGrabber(rgb_topic, depth_topic)

            # Wait briefly for first frame
            print("[INFO] Waiting for camera stream...")
            t_wait = time.time()
            while grabber.latest_rgb is None and (time.time() - t_wait < 5.0):
                grabber.spin_once(0.1)

            if grabber.latest_rgb is None:
                print(f"[ERROR] No frames received on {rgb_topic} after 5 seconds!")
                print(f"Is the camera publishing? Check 'ros2 topic list' and 'ros2 topic hz {rgb_topic}'.")
                grabber.destroy()
                sys.exit(1)

            print(f"[INFO] Camera stream active! Starting interactive capture ({args.burst_frames} frames per position, {pos_desc})...")
            dataset = capture_dataset_interactive(
                grabber, dataset_dir, burst_frames=args.burst_frames, positions=selected_positions
            )
            grabber.destroy()

    if len(dataset) == 0:
        print("[ERROR] No face images available for optimization!")
        sys.exit(1)

    # Run optimization
    best_params, best_acc, best_score, details = run_parameter_optimization(dataset, base_params, quick=args.quick)

    # Print results table showing each position
    print_results_table(details, best_params, test_mode=test_mode)

    # Build and save verification collages
    collage = create_summary_collage(dataset, best_params, details)
    summary_path = os.path.join(dataset_dir, "optimization_summary.png")
    cv2.imwrite(summary_path, collage)
    print(f"[INFO] Saved visual summary collage -> {summary_path}")

    if test_mode == "extended":
        multi_collage = create_multi_position_summary_collage(dataset, best_params, details)
        multi_path = os.path.join(dataset_dir, "optimization_summary_positions.png")
        cv2.imwrite(multi_path, multi_collage)
        print(f"[INFO] Saved 5-position matrix collage -> {multi_path}")

    # Save calibration results to YAML
    results_path = save_calibration_results(dataset_dir, best_params, best_acc, best_score, details, test_mode=test_mode)
    print(f"[INFO] Saved calibration results -> {results_path}")

    # Display results collage window BEFORE asking to overwrite
    window_name = "Die Detection Optimization — Final Results"
    display_active = False
    if os.environ.get("DISPLAY"):
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, collage)
            cv2.waitKey(200)  # Render initial window contents
            display_active = True
            print(f"[INFO] Showing results picture in window: '{window_name}'")
        except Exception as e:
            print(f"[WARN] Could not display results window: {e}")

    # Prompt to save to YAML
    save_yaml = args.auto_save
    if not save_yaml:
        prompt = f"\nDo you want to save these optimal parameters to {config_path}? [Y/n] (Press in terminal or window): "
        print(prompt, end="", flush=True)

        user_response = None
        import select
        while user_response is None:
            if display_active:
                key = cv2.waitKey(30) & 0xFF
                if key in (ord('y'), ord('Y')):
                    user_response = "y"
                    print("y")
                    break
                elif key in (ord('n'), ord('N'), ord('q'), ord('Q'), 27):
                    user_response = "n"
                    print("n")
                    break
                elif key in (10, 13):  # Enter key
                    user_response = "y"
                    print("y")
                    break
            else:
                time.sleep(0.05)

            # Non-blocking check for terminal keyboard input
            try:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if rlist:
                    line = sys.stdin.readline().strip().lower()
                    user_response = line if line else "y"
                    break
            except Exception:
                # Fallback to standard input if select is not supported
                line = input().strip().lower()
                user_response = line if line else "y"
                break

        if user_response in ("", "y", "yes"):
            save_yaml = True

    if save_yaml:
        best_params.save_to_yaml(config_path)
        print(f"[SUCCESS] Updated {config_path} with optimal parameters!")
    else:
        print("[INFO] Parameters were not saved to YAML.")

    # Clean up display window
    if display_active:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

