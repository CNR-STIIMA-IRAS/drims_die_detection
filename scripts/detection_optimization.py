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
  python3 scripts/detection_optimization.py --offline --dataset-dir output/calibration_dataset
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import copy
from typing import Optional

import cv2
import numpy as np
import yaml

# Allow importing package from repo root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
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
        except Exception as e:
            print(f"[Grabber] Error converting synced frame: {e}")

    def _rgb_only_cb(self, rgb_msg: Image):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        except Exception:
            pass

    def spin_once(self, timeout_sec: float = 0.05):
        if self._node and rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=timeout_sec)

    def destroy(self):
        if self._node:
            self._node.destroy_node()


# ──────────────────────────────────────────────────────────────────────────────
# 2. Interactive Face Acquisition (Faces 1 to 6)
# ──────────────────────────────────────────────────────────────────────────────

def capture_dataset_interactive(
    grabber: CameraFrameGrabber,
    output_dir: str,
    target_faces: list[int] = [1, 2, 3, 4, 5, 6],
) -> dict[int, np.ndarray]:
    """Guides user sequentially from face 1 to 6, displaying live preview and capturing frames."""
    os.makedirs(output_dir, exist_ok=True)
    captured_images: dict[int, np.ndarray] = {}

    win_name = "Die Detection Calibration — Live Preview"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 720)

    print("\n" + "=" * 70)
    print("  DIE DETECTION OPTIMIZATION — INTERACTIVE CAPTURE")
    print("=" * 70)
    print("For each face (1 through 6):")
    print("  1. Place the die with the requested face facing UP under the camera.")
    print("  2. Press [ENTER] in the terminal OR [SPACE] in the preview window.")
    print("  3. Press [Q] at any time to cancel.")
    print("=" * 70 + "\n")

    for face in target_faces:
        print(f"\n>>> [STEP {face}/6] Please place FACE {face} facing UP under the camera.")
        print("    Press [ENTER] in terminal (or SPACE in window) when ready...")

        captured = False
        last_frame = None

        # Flush any pending keypresses
        while True:
            grabber.spin_once(0.03)
            frame = grabber.latest_rgb
            if frame is not None:
                last_frame = frame.copy()
                disp = frame.copy()
                h, w = disp.shape[:2]

                # Draw targeting reticle in center
                cx, cy = w // 2, h // 2
                cv2.drawMarker(disp, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 40, 2)
                cv2.rectangle(disp, (cx - 100, cy - 100), (cx + 100, cy + 100), (0, 255, 255), 1)

                # Draw top status banner
                cv2.rectangle(disp, (0, 0), (w, 60), (20, 20, 20), -1)
                cv2.rectangle(disp, (0, 58), (w, 60), (0, 255, 0), 2)
                msg = f"STEP {face}/6: Place FACE {face} UP  |  Press SPACE / ENTER to Capture"
                cv2.putText(disp, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow(win_name, disp)

            key = cv2.waitKey(20) & 0xFF
            if key == ord(' '):
                captured = True
                break
            elif key == ord('q') or key == 27:
                print("\n[INFO] Calibration aborted by user.")
                cv2.destroyAllWindows()
                sys.exit(0)

            # Check if user pressed ENTER in terminal non-blockingly or via sys.stdin
            # Using select on Linux stdin
            import select
            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                captured = True
                break

        if captured and last_frame is not None:
            # Save frame to output directory
            img_path = os.path.join(output_dir, f"face_{face}.png")
            cv2.imwrite(img_path, last_frame)
            if grabber.latest_depth is not None:
                depth_path = os.path.join(output_dir, f"face_{face}_depth.npy")
                np.save(depth_path, grabber.latest_depth)

            captured_images[face] = last_frame

            # Show green success flash
            flash_img = last_frame.copy()
            cv2.rectangle(flash_img, (0, 0), (flash_img.shape[1], 60), (0, 180, 0), -1)
            succ_msg = f"FACE {face} CAPTURED SUCCESSFULLY!"
            cv2.putText(flash_img, succ_msg, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(win_name, flash_img)
            cv2.waitKey(600)
            print(f"    [OK] Captured and saved Face {face} -> {img_path}")
        else:
            print(f"    [WARN] No camera frame received for Face {face}!")

    cv2.destroyAllWindows()
    return captured_images


def load_dataset_from_dir(dataset_dir: str) -> dict[int, np.ndarray]:
    """Loads existing captured face images face_1.png ... face_6.png from disk."""
    images = {}
    for face in range(1, 7):
        cand = os.path.join(dataset_dir, f"face_{face}.png")
        if not os.path.exists(cand):
            cand = os.path.join(dataset_dir, f"face_{face}.jpg")
        if os.path.exists(cand):
            img = cv2.imread(cand)
            if img is not None:
                images[face] = img
        else:
            # Check test_images fallback
            cand_test = os.path.join(_REPO_ROOT, "test_images", "calibration", f"face_{face}.png")
            if os.path.exists(cand_test):
                images[face] = cv2.imread(cand_test)

    return images


# ──────────────────────────────────────────────────────────────────────────────
# 3. Parameter Evaluation & Optimization Engine
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_parameters(
    params: DieDetectorParams,
    dataset: dict[int, np.ndarray],
) -> tuple[int, float, dict[int, dict]]:
    """Evaluates a parameter set on all 6 faces.

    Returns:
        accuracy: count of correctly classified faces (0 to 6)
        score: total numerical score (accuracy + circularity confidence margin)
        results: per-face result dict
    """
    detector = RGBDieDetector(params)
    accuracy = 0
    score = 0.0
    details = {}

    for expected_face, img in dataset.items():
        try:
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
            avg_circ = 0.0
            if pips_list:
                circs = []
                for p in pips_list:
                    c = p.get("contour")
                    if c is not None:
                        a = cv2.contourArea(c)
                        pm = cv2.arcLength(c, True)
                        if pm > 0:
                            circs.append((4 * np.pi * a) / (pm ** 2))
                if circs:
                    avg_circ = float(np.mean(circs))

            is_correct = (detected_pips == expected_face) and is_trusted and aligned

            if is_correct:
                accuracy += 1
                face_score = 100.0 + 10.0 * avg_circ
            else:
                err = abs(detected_pips - expected_face)
                face_score = -20.0 * max(1, err)
                if not is_trusted:
                    face_score -= 15.0

            score += face_score
            details[expected_face] = {
                "expected": expected_face,
                "detected": detected_pips,
                "is_correct": is_correct,
                "avg_circ": avg_circ,
                "num_faces": len(faces),
                "det": det,
            }

        except Exception as e:
            score -= 100.0
            details[expected_face] = {
                "expected": expected_face,
                "detected": -1,
                "is_correct": False,
                "avg_circ": 0.0,
                "error": str(e),
            }

    return accuracy, score, details


def run_parameter_optimization(
    dataset: dict[int, np.ndarray],
    base_params: DieDetectorParams,
    quick: bool = False,
) -> tuple[DieDetectorParams, int, dict[int, dict]]:
    """Runs a multi-stage grid search over key computer vision parameters.

    Returns:
        best_params: DieDetectorParams that maximizes classification accuracy
        best_accuracy: number of correct faces (up to 6)
        best_details: evaluation details on each face
    """
    print("\n" + "=" * 70)
    print("  STARTING PARAMETER OPTIMIZATION GRID SEARCH")
    print("=" * 70)
    print(f"Dataset: {len(dataset)} faces loaded ({sorted(list(dataset.keys()))})")

    # Parameter search spaces
    if quick:
        s_max_list = [120, 150, 180]
        v_min_list = [145, 160, 175]
        glare_list = [225, 240]
        clahe_list = [2.5, 3.5]
        circ_list = [0.24, 0.30, 0.36]
        min_pip_area_list = [0.002, 0.004]
        pip_area_list = [0.05, 0.07]
        pip_radius_list = [0.18, 0.22]
    else:
        s_max_list = [110, 130, 150, 170]
        v_min_list = [140, 155, 165, 175]
        glare_list = [220, 235, 245, 255]
        clahe_list = [2.5, 3.0, 3.5]
        circ_list = [0.22, 0.28, 0.34, 0.40]
        min_pip_area_list = [0.002, 0.003, 0.005]
        pip_area_list = [0.05, 0.07, 0.09]
        pip_radius_list = [0.18, 0.22]

    # Baseline evaluation
    best_params = copy.deepcopy(base_params)
    base_acc, base_score, best_details = evaluate_parameters(best_params, dataset)
    print(f"[Initial Baseline] Accuracy: {base_acc}/6 faces correct (Score: {base_score:.1f})")

    if base_acc == 6:
        print("[INFO] Baseline parameters already achieve 6/6 correct detection! Running fine-tuning...")

    best_score = base_score
    best_acc = base_acc

    # ── Stage 1: Color Segmentation & Glare Sweep ─────────────────────────────
    print("\n--- [Stage 1/2] Optimizing Color Segmentation & Glare Filtering ---")
    total_stage1 = len(s_max_list) * len(v_min_list) * len(glare_list)
    idx = 0
    t0 = time.time()

    for s_max in s_max_list:
        for v_min in v_min_list:
            for glare in glare_list:
                idx += 1
                cand = copy.deepcopy(best_params)
                cand.hsv_min = [0, 0, v_min]
                cand.hsv_max = [180, s_max, 255]
                cand.glare_v_thresh = glare

                acc, score, details = evaluate_parameters(cand, dataset)
                if acc > best_acc or (acc == best_acc and score > best_score):
                    best_acc = acc
                    best_score = score
                    best_params = cand
                    best_details = details
                    print(f"  [Progress] New Best: {acc}/6 correct | Score: {score:.1f} "
                          f"(HSV_min=[0,0,{v_min}], HSV_max=[180,{s_max},255], Glare={glare})")

    print(f"Stage 1 finished in {time.time() - t0:.1f}s. Best accuracy so far: {best_acc}/6")

    # ── Stage 2: Pip Segmentation & Circularity Sweep ────────────────────────
    print("\n--- [Stage 2/2] Optimizing Pip Detection, Circularity & Sizing ---")
    t0 = time.time()

    for clahe in clahe_list:
        for circ in circ_list:
            for p_min_area in min_pip_area_list:
                for p_max_area in pip_area_list:
                    for p_rad in pip_radius_list:
                        cand = copy.deepcopy(best_params)
                        cand.clahe_clip_limit = clahe
                        cand.pip_min_circularity = circ
                        cand.min_pip_area_ratio = p_min_area
                        cand.max_pip_area_ratio = p_max_area
                        cand.max_pip_radius_ratio = p_rad

                        acc, score, details = evaluate_parameters(cand, dataset)
                        if acc > best_acc or (acc == best_acc and score > best_score):
                            best_acc = acc
                            best_score = score
                            best_params = cand
                            best_details = details
                            print(f"  [Progress] New Best: {acc}/6 correct | Score: {score:.1f} "
                                  f"(Circ={circ:.2f}, CLAHE={clahe:.1f}, MinArea={p_min_area:.3f}, MaxArea={p_max_area:.2f})")
                            if acc == 6 and not quick:
                                pass

    print(f"Stage 2 finished in {time.time() - t0:.1f}s. Final Best accuracy: {best_acc}/6")
    return best_params, best_acc, best_details


# ──────────────────────────────────────────────────────────────────────────────
# 4. Result Presentation & Verification Collage
# ──────────────────────────────────────────────────────────────────────────────

def create_summary_collage(
    dataset: dict[int, np.ndarray],
    params: DieDetectorParams,
    details: dict[int, dict],
) -> np.ndarray:
    """Renders a 2x3 collage showing the detection output for all 6 faces."""
    detector = RGBDieDetector(params)
    panels = []
    pw, ph = 480, 360

    for face in range(1, 7):
        img = dataset.get(face)
        if img is None:
            # Blank placeholder
            canvas = np.zeros((ph, pw, 3), dtype=np.uint8)
            cv2.putText(canvas, f"Face {face}: Missing", (50, ph // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            panels.append(canvas)
            continue

        det = detector.detect(img)
        faces_pips = det.get("debug_steps", {}).get("step4_faces_and_pips", img)
        panel = cv2.resize(faces_pips, (pw, ph))

        det_info = details.get(face, {})
        detected = det_info.get("detected", 0)
        is_ok = det_info.get("is_correct", False)

        # Status banner at top
        banner_color = (0, 160, 0) if is_ok else (0, 0, 200)
        cv2.rectangle(panel, (0, 0), (pw, 45), (20, 20, 20), -1)
        cv2.rectangle(panel, (0, 43), (pw, 45), banner_color, 2)

        status_str = "PASS" if is_ok else "FAIL"
        label = f"Face {face}: Detected {detected} pips [{status_str}]"
        cv2.putText(panel, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if is_ok else (0, 100, 255), 2, cv2.LINE_AA)

        panels.append(panel)

    row1 = np.hstack(panels[0:3])
    row2 = np.hstack(panels[3:6])
    collage = np.vstack([row1, row2])
    return collage


def print_results_table(details: dict[int, dict], best_params: DieDetectorParams):
    print("\n" + "=" * 70)
    print("                    OPTIMIZATION FINAL RESULTS")
    print("=" * 70)
    print(f" {'Face':<6} | {'Expected':<10} | {'Detected':<10} | {'Circularity':<12} | {'Status':<8}")
    print("-" * 70)
    pass_count = 0
    for f in range(1, 7):
        d = details.get(f, {})
        exp = d.get("expected", f)
        det = d.get("detected", "N/A")
        circ = f"{d.get('avg_circ', 0.0):.2f}" if d.get("avg_circ") else "N/A"
        ok = d.get("is_correct", False)
        if ok:
            pass_count += 1
        status = "PASS" if ok else "FAIL"
        print(f" {f:<6} | {exp:<10} | {det:<10} | {circ:<12} | {status:<8}")
    print("-" * 70)
    print(f" Total Score: {pass_count}/6 faces correctly detected ({pass_count/6.0*100:.1f}%)")
    print("=" * 70 + "\n")

    print("Winning Parameters:")
    print(f"  hsv_min:              {best_params.hsv_min}")
    print(f"  hsv_max:              {best_params.hsv_max}")
    print(f"  glare_v_thresh:       {best_params.glare_v_thresh}")
    print(f"  clahe_clip_limit:     {best_params.clahe_clip_limit}")
    print(f"  pip_min_circularity:  {best_params.pip_min_circularity}")
    print(f"  min_pip_area_ratio:   {best_params.min_pip_area_ratio}")
    print(f"  max_pip_area_ratio:   {best_params.max_pip_area_ratio}")
    print(f"  max_pip_radius_ratio: {best_params.max_pip_radius_ratio}")
    print("=" * 70 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Main CLI Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def find_config_yaml(user_path: Optional[str] = None) -> str:
    """Finds the die_detector_params.yaml configuration file reliably."""
    if user_path and os.path.isfile(user_path):
        return os.path.abspath(user_path)

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
                        help="Directory to save / load face images (defaults to output/calibration_dataset in package).")
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

    if args.dataset_dir:
        dataset_dir = os.path.abspath(args.dataset_dir)
    else:
        dataset_dir = os.path.abspath(os.path.join(os.path.dirname(config_path), "..", "output", "calibration_dataset"))

    dataset: dict[int, np.ndarray] = {}

    # Check if user requested offline or if images already exist
    existing_images = load_dataset_from_dir(dataset_dir)
    if args.offline:
        if len(existing_images) == 0:
            print(f"[ERROR] Offline mode selected but no face_*.png images found in {dataset_dir}!")
            sys.exit(1)
        dataset = existing_images
        print(f"[INFO] Offline mode: loaded {len(dataset)} images from {dataset_dir}")
    else:
        if len(existing_images) == 6:
            print(f"[INFO] Found 6 previously captured images in {dataset_dir}.")
            choice = input("Do you want to [1] Capture new frames from camera, or [2] Use existing images? [1/2] (default: 1): ").strip()
            if choice == "2":
                dataset = existing_images

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
                print("Is the camera publishing? Check 'ros2 topic list' and 'ros2 topic hz {rgb_topic}'.")
                grabber.destroy()
                sys.exit(1)

            print("[INFO] Camera stream active! Starting interactive capture...")
            dataset = capture_dataset_interactive(grabber, dataset_dir)
            grabber.destroy()

    if len(dataset) == 0:
        print("[ERROR] No face images available for optimization!")
        sys.exit(1)

    # Run optimization
    best_params, best_acc, details = run_parameter_optimization(dataset, base_params, quick=args.quick)

    # Print results table
    print_results_table(details, best_params)

    # Build and save verification collage
    collage = create_summary_collage(dataset, best_params, details)
    summary_path = os.path.join(dataset_dir, "optimization_summary.png")
    cv2.imwrite(summary_path, collage)
    print(f"[INFO] Saved visual summary collage -> {summary_path}")

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

