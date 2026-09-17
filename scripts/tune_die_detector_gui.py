#!/usr/bin/env python3
"""
tune_die_detector_gui.py
========================
User-friendly parameter tuning GUI with sliders, live side-by-side preview,
and parameter explanations.

Supports BOTH Tkinter (if python3-tk is installed) and OpenCV Trackbars
(zero-dependency fallback if tkinter is not installed).

Features:
  * Sliders for Detection Mode, HSV Bounds, Glare Cutoff, CLAHE, and Pip Circularity
  * Live side-by-side image preview (Detection & 3D Pose | Binary Mask)
  * Parameter explanations for each slider
  * Image selector (cycles through test_images/rgb/)
  * One-click 💾 'Save to YAML' (config/die_detector_params.yaml)
  * One-click 📡 'Sync to ROS 2 Node' (ros2 param set /die_detector_node ...)

Usage:
------
    python scripts/tune_die_detector_gui.py
    python scripts/tune_die_detector_gui.py --image test_images/rgb/5834717044920750589.jpg
    python scripts/tune_die_detector_gui.py --cv  # Force OpenCV GUI mode
"""

from __future__ import annotations

import os
import sys
import glob
import argparse
import subprocess
import yaml
import numpy as np
import cv2

# Check Tkinter availability
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import Image, ImageTk
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False

# Check ROS 2 Live camera subscription availability
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image as SensorImage
    from cv_bridge import CvBridge
    HAS_ROS2_LIVE = True
except ImportError:
    HAS_ROS2_LIVE = False

# Allow importing package from repo root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _REPO_ROOT)

from drims_die_detection import DieDetectorParams, RGBDieDetector


class LiveRosImageSubscriber:
    """ROS 2 Node helper for live camera topic tuning."""

    def __init__(self, topic: str) -> None:
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("die_detector_tuner_live_sub")
        self.bridge = CvBridge()
        self.topic = topic
        self.latest_frame: np.ndarray | None = None
        self.frame_count = 0
        self.sub = self.node.create_subscription(
            SensorImage, self.topic, self._cb, qos_profile_sensor_data
        )
        print(f"[Live ROS] 📡 Subscribed to topic: '{self.topic}' (using sensor_data QoS)")


    def _cb(self, msg: SensorImage) -> None:
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.frame_count += 1
        except Exception as e:
            print(f"[Live ROS] Error converting image: {e}")

    def spin_once(self, timeout_sec: float = 0.005) -> None:
        if rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=timeout_sec)

    def shutdown(self) -> None:
        if rclpy.ok():
            self.node.destroy_node()



# ──────────────────────────────────────────────────────────────────────────────
# 1. OPENCV FALLBACK GUI (Zero external dependencies)
# ──────────────────────────────────────────────────────────────────────────────

class OpenCVTunerGUI:
    """OpenCV Trackbars GUI with HUD explanations and hotkey controls."""

    WIN_CONTROLS = "Die Detector — Trackbar Controls"
    WIN_PREVIEW = "Die Detector — Live Detection Preview"

    def __init__(self, config_path: str, initial_image_path: str | None = None, live_topic: str | None = None) -> None:
        self.config_path = config_path
        self.live_topic = live_topic
        if os.path.isfile(self.config_path):
            self.params = DieDetectorParams.from_yaml(self.config_path)
            print(f"[GUI] Loaded config from {self.config_path}")
        else:
            self.params = DieDetectorParams()

        self.detector = RGBDieDetector(self.params)
        self.image_paths = self._find_test_images(initial_image_path)
        self.current_idx = 0
        self.live_sub: LiveRosImageSubscriber | None = None

        if self.live_topic:
            if HAS_ROS2_LIVE:
                self.live_sub = LiveRosImageSubscriber(self.live_topic)
            else:
                print(f"[ERROR] Cannot use live mode: ROS 2 / cv_bridge packages not found in Python path.")

    def _find_test_images(self, preferred_path: str | None) -> list[str]:
        paths = []
        if preferred_path and os.path.isfile(preferred_path):
            paths.append(preferred_path)

        rgb_dir = os.path.join(_REPO_ROOT, "test_images", "rgb")
        if os.path.isdir(rgb_dir):
            found = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")) + glob.glob(os.path.join(rgb_dir, "*.png")))
            for p in found:
                if p not in paths:
                    paths.append(p)
        return paths

    def run(self) -> None:
        if not self.live_sub and not self.image_paths:
            print("[ERROR] No sample images found in test_images/rgb/.")
            return

        cv2.namedWindow(self.WIN_CONTROLS, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WIN_CONTROLS, 550, 650)
        cv2.namedWindow(self.WIN_PREVIEW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WIN_PREVIEW, 1200, 650)
        cv2.waitKey(1)


        def _nop(x): pass

        # Create Trackbars
        cv2.createTrackbar("Mode (0:HSV 1:KM)", self.WIN_CONTROLS, 0 if self.params.detection_mode=="hsv" else 1, 1, _nop)
        cv2.createTrackbar("HSV H Min (0-180)", self.WIN_CONTROLS, self.params.hsv_min[0], 180, _nop)
        cv2.createTrackbar("HSV H Max (0-180)", self.WIN_CONTROLS, self.params.hsv_max[0], 180, _nop)
        cv2.createTrackbar("HSV S Min (0-255)", self.WIN_CONTROLS, self.params.hsv_min[1], 255, _nop)
        cv2.createTrackbar("HSV S Max (0-255)", self.WIN_CONTROLS, self.params.hsv_max[1], 255, _nop)
        cv2.createTrackbar("HSV V Min (0-255)", self.WIN_CONTROLS, self.params.hsv_min[2], 255, _nop)
        cv2.createTrackbar("HSV V Max (0-255)", self.WIN_CONTROLS, self.params.hsv_max[2], 255, _nop)

        cv2.createTrackbar("Glare V Cutoff", self.WIN_CONTROLS, self.params.glare_v_thresh, 255, _nop)
        cv2.createTrackbar("CLAHE Limit x10", self.WIN_CONTROLS, int(self.params.clahe_clip_limit * 10), 100, _nop)
        cv2.createTrackbar("Pip Min Circ x100", self.WIN_CONTROLS, int(self.params.pip_min_circularity * 100), 100, _nop)
        cv2.createTrackbar("Canny Low Thresh", self.WIN_CONTROLS, int(getattr(self.params, "canny_low_thresh", 40)), 255, _nop)
        cv2.createTrackbar("Canny High Thresh", self.WIN_CONTROLS, int(getattr(self.params, "canny_high_thresh", 130)), 255, _nop)

        print("\n" + "=" * 70)
        print("DIE DETECTOR OpenCV GUI TUNER")
        print("=" * 70)
        print("Controls & Hotkeys:")
        print("  [S]  Save current sliders to config/die_detector_params.yaml")
        print("  [R]  Sync current parameters live to ROS 2 node (/die_detector_node)")
        if not self.live_sub:
            print("  [N]  Load next test image")
        print("  [Q / ESC]  Quit tuner")
        print("=" * 70 + "\n")

        while True:
            # Read sliders
            m_idx = cv2.getTrackbarPos("Mode (0:HSV 1:KM)", self.WIN_CONTROLS)
            self.params.detection_mode = "hsv" if m_idx == 0 else "kmeans"

            h_min = cv2.getTrackbarPos("HSV H Min (0-180)", self.WIN_CONTROLS)
            h_max = cv2.getTrackbarPos("HSV H Max (0-180)", self.WIN_CONTROLS)
            s_min = cv2.getTrackbarPos("HSV S Min (0-255)", self.WIN_CONTROLS)
            s_max = cv2.getTrackbarPos("HSV S Max (0-255)", self.WIN_CONTROLS)
            v_min = cv2.getTrackbarPos("HSV V Min (0-255)", self.WIN_CONTROLS)
            v_max = cv2.getTrackbarPos("HSV V Max (0-255)", self.WIN_CONTROLS)

            self.params.hsv_min = [h_min, s_min, v_min]
            self.params.hsv_max = [h_max, s_max, v_max]
            self.params.glare_v_thresh = cv2.getTrackbarPos("Glare V Cutoff", self.WIN_CONTROLS)
            self.params.clahe_clip_limit = max(1.0, cv2.getTrackbarPos("CLAHE Limit x10", self.WIN_CONTROLS) / 10.0)
            self.params.pip_min_circularity = max(0.1, cv2.getTrackbarPos("Pip Min Circ x100", self.WIN_CONTROLS) / 100.0)
            self.params.canny_low_thresh = cv2.getTrackbarPos("Canny Low Thresh", self.WIN_CONTROLS)
            self.params.canny_high_thresh = cv2.getTrackbarPos("Canny High Thresh", self.WIN_CONTROLS)

            # Acquire frame
            if self.live_sub is not None:
                self.live_sub.spin_once(0.01)
                bgr = self.live_sub.latest_frame
                if bgr is None:
                    blank = np.zeros((480, 720, 3), dtype=np.uint8)
                    cv2.putText(blank, f"Waiting for live ROS 2 topic: '{self.live_topic}'...", (20, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.imshow(self.WIN_PREVIEW, blank)
                    key = cv2.waitKey(40) & 0xFF
                    if key in (27, ord('q'), ord('Q')):
                        break
                    continue
                img_name = f"LIVE ROS 2: {self.live_topic} (Frame #{self.live_sub.frame_count})"
            else:
                img_path = self.image_paths[self.current_idx % len(self.image_paths)]
                bgr = cv2.imread(img_path)
                if bgr is None:
                    continue
                img_name = os.path.basename(img_path)

            res = self.detector.detect(bgr)
            debug_steps = res.get("debug_steps", {})

            mask_vis = debug_steps.get("step1_color_clustered", bgr)
            annotated = debug_steps.get("step5_annotated_result", bgr)

            # Resize to fit side-by-side
            target_w = 640
            h_a, w_a = annotated.shape[:2]
            scale = target_w / float(w_a)
            target_h = int(h_a * scale)

            ann_resized = cv2.resize(annotated, (target_w, target_h))
            mask_resized = cv2.resize(mask_vis, (target_w, target_h))

            # Draw HUD explanation banner
            mode_str = self.params.detection_mode.upper()
            hud_txt = f"[{img_name}] Mode: {mode_str} | Faces: {res['num_visible_faces']} | Pips: {res['total_pips']}"
            cv2.putText(ann_resized, hud_txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(mask_resized, f"Color Mask ({mode_str})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # Bottom help HUD on preview
            help_bar = np.zeros((40, target_w * 2, 3), dtype=np.uint8)
            cv2.putText(help_bar, "[S] Save YAML  |  [R] Sync ROS 2 Node  |  [N] Next Image  |  [Q] Quit",
                        (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            combined = np.hstack([ann_resized, mask_resized])
            final_view = np.vstack([combined, help_bar])

            cv2.imshow(self.WIN_PREVIEW, final_view)

            key = cv2.waitKey(40) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            elif key in (ord('s'), ord('S')):
                self._save_yaml()
            elif key in (ord('r'), ord('R')):
                self._sync_ros2()
            elif key in (ord('n'), ord('N')) and not self.live_sub:
                self.current_idx += 1
                print(f"[GUI] Switched to image: {self.image_paths[self.current_idx % len(self.image_paths)]}")

        if self.live_sub:
            self.live_sub.shutdown()
        cv2.destroyAllWindows()


    def _save_yaml(self) -> None:
        try:
            raw_yaml = {}
            if os.path.isfile(self.config_path):
                with open(self.config_path, "r") as f:
                    raw_yaml = yaml.safe_load(f) or {}

            params_dict = {
                "detection_mode": self.params.detection_mode,
                "hsv_min": [int(x) for x in self.params.hsv_min],
                "hsv_max": [int(x) for x in self.params.hsv_max],
                "glare_v_thresh": int(self.params.glare_v_thresh),
                "clahe_clip_limit": float(self.params.clahe_clip_limit),
                "pip_min_circularity": float(self.params.pip_min_circularity),
                "canny_low_thresh": int(getattr(self.params, "canny_low_thresh", 40)),
                "canny_high_thresh": int(getattr(self.params, "canny_high_thresh", 130)),
            }

            if "die_detector_node" in raw_yaml and "ros__parameters" in raw_yaml["die_detector_node"]:
                raw_yaml["die_detector_node"]["ros__parameters"].update(params_dict)
            else:
                raw_yaml.update(params_dict)

            with open(self.config_path, "w") as f:
                yaml.dump(raw_yaml, f, default_flow_style=False)

            print(f"[GUI] ✅ Saved tuned parameters to {self.config_path}")
        except Exception as e:
            print(f"[GUI] ❌ Failed to save YAML config: {e}")

    def _sync_ros2(self) -> None:
        node_name = "/die_detector_node"
        params_to_sync = [
            ("detection_mode", self.params.detection_mode),
            ("hsv_min", [int(x) for x in self.params.hsv_min]),
            ("hsv_max", [int(x) for x in self.params.hsv_max]),
            ("glare_v_thresh", int(self.params.glare_v_thresh)),
            ("clahe_clip_limit", float(self.params.clahe_clip_limit)),
            ("pip_min_circularity", float(self.params.pip_min_circularity)),
            ("canny_low_thresh", int(getattr(self.params, "canny_low_thresh", 40))),
            ("canny_high_thresh", int(getattr(self.params, "canny_high_thresh", 130))),
        ]

        successes = 0
        for name, val in params_to_sync:
            val_str = ("[" + ",".join(map(str, val)) + "]") if isinstance(val, list) else (f"'{val}'" if isinstance(val, str) else str(val))
            cmd = ["ros2", "param", "set", node_name, name, val_str]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
                if res.returncode == 0:
                    successes += 1
            except Exception:
                pass

        if successes > 0:
            print(f"[GUI] 📡 Successfully synced {successes} parameters to {node_name} live!")
        else:
            print(f"[GUI] ⚠️ Could not reach {node_name}. Ensure 'ros2 launch drims_die_detection die_detector.launch.py' is running.")


# ──────────────────────────────────────────────────────────────────────────────
# 2. TKINTER GUI (If python3-tk is installed)
# ──────────────────────────────────────────────────────────────────────────────

if HAS_TKINTER:
    class DieDetectorTunerGUI:
        """Tkinter-based Parameter Tuning GUI with explanations and live ROS 2 sync."""

        def __init__(self, root: tk.Tk, initial_config_path: str, initial_image_path: str | None = None, live_topic: str | None = None) -> None:
            self.root = root
            self.root.title("Die Detector 3D — Parameter Tuning & Calibration GUI")
            self.root.geometry("1400x900")
            self.root.minsize(1100, 750)

            self.config_path = initial_config_path
            self.live_topic = live_topic

            if os.path.isfile(self.config_path):
                self.params = DieDetectorParams.from_yaml(self.config_path)
                print(f"[GUI] Loaded config from {self.config_path}")
            else:
                self.params = DieDetectorParams()

            self.detector = RGBDieDetector(self.params)

            self.image_paths = self._find_test_images(initial_image_path)
            self.current_img_idx = 0
            self.current_bgr: np.ndarray | None = None
            self.live_sub: LiveRosImageSubscriber | None = None

            if self.live_topic:
                if HAS_ROS2_LIVE:
                    self.live_sub = LiveRosImageSubscriber(self.live_topic)
                else:
                    print(f"[ERROR] Cannot use live mode: ROS 2 / cv_bridge packages not found in Python path.")

            self._create_styles()
            self._build_ui()

            if self.live_sub:
                self._poll_live_ros()
            elif self.image_paths:
                self._load_image(self.image_paths[0])
                self._update_pipeline()

        def _poll_live_ros(self) -> None:
            if self.live_sub:
                self.live_sub.spin_once(0.0)
                frame = self.live_sub.latest_frame
                if frame is not None:
                    self.current_bgr = frame
                    self._update_pipeline()
                self.root.after(30, self._poll_live_ros)


        def _find_test_images(self, preferred_path: str | None) -> list[str]:
            paths = []
            if preferred_path and os.path.isfile(preferred_path):
                paths.append(preferred_path)

            rgb_dir = os.path.join(_REPO_ROOT, "test_images", "rgb")
            if os.path.isdir(rgb_dir):
                found = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")) + glob.glob(os.path.join(rgb_dir, "*.png")))
                for p in found:
                    if p not in paths:
                        paths.append(p)
            return paths

        def _create_styles(self) -> None:
            style = ttk.Style()
            style.theme_use("clam")
            style.configure("Header.TLabel", font=("Helvetica", 12, "bold"))
            style.configure("SubHeader.TLabel", font=("Helvetica", 10, "bold"), foreground="#2c3e50")
            style.configure("Help.TLabel", font=("Helvetica", 8), foreground="#555555", wraplength=380)
            style.configure("Val.TLabel", font=("Helvetica", 9, "bold"), foreground="#2980b9")
            style.configure("Action.TButton", font=("Helvetica", 10, "bold"), padding=6)

        def _build_ui(self) -> None:
            paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
            paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            left_frame = ttk.Frame(paned, width=440)
            paned.add(left_frame, weight=0)

            canvas = tk.Canvas(left_frame, borderwidth=0, highlightthickness=0)
            scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=canvas.yview)
            scroll_content = ttk.Frame(canvas)

            scroll_content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas.create_window((0, 0), window=scroll_content, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            # Card 1: Strategy
            card_mode = ttk.LabelFrame(scroll_content, text=" 1. Detection Strategy ", padding=8)
            card_mode.pack(fill=tk.X, padx=5, pady=5)

            self.var_mode = tk.StringVar(value=self.params.detection_mode if self.params.detection_mode in ("hsv", "kmeans") else "hsv")
            rb_hsv = ttk.Radiobutton(card_mode, text="HSV Color Windowing (Recommended)", value="hsv", variable=self.var_mode, command=self._on_slider_change)
            rb_hsv.pack(anchor=tk.W, pady=2)
            ttk.Label(card_mode, text="Filters pixels by Hue, Saturation, and Brightness ranges. Best for white/colored dice under non-uniform light.", style="Help.TLabel").pack(anchor=tk.W, padx=20, pady=(0, 5))

            rb_km = ttk.Radiobutton(card_mode, text="K-Means Clustering (Color-based)", value="kmeans", variable=self.var_mode, command=self._on_slider_change)
            rb_km.pack(anchor=tk.W, pady=2)
            ttk.Label(card_mode, text="Groups image colors into K clusters and picks the target color cluster.", style="Help.TLabel").pack(anchor=tk.W, padx=20)

            # Card 2: HSV Bounds
            card_hsv = ttk.LabelFrame(scroll_content, text=" 2. HSV Color Filters (for Die Body) ", padding=8)
            card_hsv.pack(fill=tk.X, padx=5, pady=5)

            self.slider_h_min = self._add_slider(card_hsv, "Hue Min", 0, 180, self.params.hsv_min[0], "Color tint lower bound. Keep 0 to 180 for white/neutral dice.")
            self.slider_h_max = self._add_slider(card_hsv, "Hue Max", 0, 180, self.params.hsv_max[0], "Color tint upper bound. Keep 0 to 180 for white/neutral dice.")
            self.slider_s_min = self._add_slider(card_hsv, "Saturation Min", 0, 255, self.params.hsv_min[1], "Color purity lower bound.")
            self.slider_s_max = self._add_slider(card_hsv, "Saturation Max", 0, 255, self.params.hsv_max[1], "Color purity upper bound. Lower max (0–40) isolates white/grey objects from vibrant backgrounds.")
            self.slider_v_min = self._add_slider(card_hsv, "Value (Brightness) Min", 0, 255, self.params.hsv_min[2], "Minimum brightness level. Increase (e.g. 100–180) to ignore dark shadows and surfaces.")
            self.slider_v_max = self._add_slider(card_hsv, "Value (Brightness) Max", 0, 255, self.params.hsv_max[2], "Maximum brightness level.")

            # Card 3: Glare
            card_glare = ttk.LabelFrame(scroll_content, text=" 3. Glare Mitigation & Contrast ", padding=8)
            card_glare.pack(fill=tk.X, padx=5, pady=5)

            self.slider_glare = self._add_slider(card_glare, "Glare V Cutoff", 100, 255, self.params.glare_v_thresh, "Brightness threshold above which bright specular glare on the die top face is included in the die body mask.")
            self.slider_clahe = self._add_slider(card_glare, "CLAHE Contrast Limit (x10)", 10, 100, int(self.params.clahe_clip_limit * 10), "Local adaptive contrast enhancement strength for uncovering pips near reflections/shadows.")

            # Card 4: Pip & Edge Thresholds
            card_pip = ttk.LabelFrame(scroll_content, text=" 4. Pip & Edge Thresholds ", padding=8)
            card_pip.pack(fill=tk.X, padx=5, pady=5)

            self.slider_circ = self._add_slider(card_pip, "Min Pip Circularity (x100)", 10, 100, int(self.params.pip_min_circularity * 100), "Minimum roundness factor (4*pi*Area/Perimeter^2). Higher values (0.45–0.60) reject non-circular specks.")
            self.slider_canny_low = self._add_slider(card_pip, "Canny Low Thresh", 0, 255, int(getattr(self.params, "canny_low_thresh", 40)), "Lower hysteresis threshold for Canny edge detection.")
            self.slider_canny_high = self._add_slider(card_pip, "Canny High Thresh", 0, 255, int(getattr(self.params, "canny_high_thresh", 130)), "Upper hysteresis threshold for Canny edge detection.")

            # Card 5: Actions
            card_act = ttk.LabelFrame(scroll_content, text=" Actions & Presets ", padding=8)
            card_act.pack(fill=tk.X, padx=5, pady=5)

            lbl_img = ttk.Label(card_act, text="Active Sample Image:", style="SubHeader.TLabel")
            lbl_img.pack(anchor=tk.W, pady=(0, 2))

            self.img_var = tk.StringVar()
            img_names = [os.path.basename(p) for p in self.image_paths] if self.image_paths else ["No images found"]
            if img_names:
                self.img_var.set(img_names[0])

            cb_img = ttk.Combobox(card_act, textvariable=self.img_var, values=img_names, state="readonly")
            cb_img.pack(fill=tk.X, pady=(0, 8))
            cb_img.bind("<<ComboboxSelected>>", self._on_image_selected)

            btn_browse = ttk.Button(card_act, text="📁 Browse Custom Image...", command=self._browse_custom_image)
            btn_browse.pack(fill=tk.X, pady=(0, 8))

            btn_save = ttk.Button(card_act, text="💾 Save to YAML (die_detector_params.yaml)", style="Action.TButton", command=self._save_yaml)
            btn_save.pack(fill=tk.X, pady=4)

            btn_sync = ttk.Button(card_act, text="📡 Sync Live to ROS 2 Node (/die_detector_node)", style="Action.TButton", command=self._sync_ros2)
            btn_sync.pack(fill=tk.X, pady=4)

            right_frame = ttk.Frame(paned)
            paned.add(right_frame, weight=1)

            self.lbl_info = ttk.Label(right_frame, text="Live Detection Preview", font=("Helvetica", 11, "bold"), foreground="#2980b9")
            self.lbl_info.pack(anchor=tk.W, padx=10, pady=5)

            self.canvas_img = tk.Label(right_frame, background="#111111")
            self.canvas_img.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        def _add_slider(self, parent, label_text: str, min_val: int, max_val: int, default_val: int, help_text: str) -> tk.Scale:
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=4)

            top_row = ttk.Frame(frame)
            top_row.pack(fill=tk.X)

            lbl_name = ttk.Label(top_row, text=label_text, style="SubHeader.TLabel")
            lbl_name.pack(side=tk.LEFT)

            lbl_val = ttk.Label(top_row, text=str(default_val), style="Val.TLabel")
            lbl_val.pack(side=tk.RIGHT)

            slider = tk.Scale(
                frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, showvalue=False,
                command=lambda val, l=lbl_val: self._on_slider_event(val, l)
            )
            slider.set(default_val)
            slider.pack(fill=tk.X, pady=2)

            lbl_help = ttk.Label(frame, text=help_text, style="Help.TLabel")
            lbl_help.pack(anchor=tk.W)

            return slider

        def _on_slider_event(self, val, label_widget: ttk.Label) -> None:
            label_widget.config(text=str(val))
            self._on_slider_change()

        def _on_slider_change(self) -> None:
            if self.current_bgr is None:
                return

            self.params.detection_mode = self.var_mode.get()
            self.params.hsv_min = [self.slider_h_min.get(), self.slider_s_min.get(), self.slider_v_min.get()]
            self.params.hsv_max = [self.slider_h_max.get(), self.slider_s_max.get(), self.slider_v_max.get()]
            self.params.glare_v_thresh = self.slider_glare.get()
            self.params.clahe_clip_limit = float(self.slider_clahe.get()) / 10.0
            self.params.pip_min_circularity = float(self.slider_circ.get()) / 100.0
            self.params.canny_low_thresh = self.slider_canny_low.get()
            self.params.canny_high_thresh = self.slider_canny_high.get()

            self._update_pipeline()

        def _load_image(self, path: str) -> None:
            bgr = cv2.imread(path)
            if bgr is not None:
                self.current_bgr = bgr

        def _on_image_selected(self, event=None) -> None:
            selected_name = self.img_var.get()
            for p in self.image_paths:
                if os.path.basename(p) == selected_name:
                    self._load_image(p)
                    self._update_pipeline()
                    break

        def _browse_custom_image(self) -> None:
            filename = filedialog.askopenfilename(
                title="Select RGB Image File",
                filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")]
            )
            if filename and os.path.isfile(filename):
                if filename not in self.image_paths:
                    self.image_paths.insert(0, filename)
                    self.img_var.set(os.path.basename(filename))
                self._load_image(filename)
                self._update_pipeline()

        def _update_pipeline(self) -> None:
            if self.current_bgr is None:
                return

            res = self.detector.detect(self.current_bgr)
            debug_steps = res.get("debug_steps", {})

            mask_vis = debug_steps.get("step1_color_clustered", self.current_bgr)
            annotated = debug_steps.get("step5_annotated_result", self.current_bgr)

            mode_str = self.params.detection_mode.upper()
            faces = res.get("faces", [])
            face_pips = [f"{f.get('num_pips', 0)} pips" for f in faces]
            detail_str = ", ".join([f"Face {i+1}: {p}" for i, p in enumerate(face_pips)]) if faces else "No faces"

            info_str = (f"Mode: {mode_str}  |  Visible Faces: {res['num_visible_faces']} [{detail_str}]  |  "
                        f"Total Pips: {res['total_pips']}")
            self.lbl_info.config(text=info_str)

            cw = max(400, self.canvas_img.winfo_width() // 2 - 10)
            ch = max(300, self.canvas_img.winfo_height() - 20)

            h_a, w_a = annotated.shape[:2]
            scale_a = min(cw / float(w_a), ch / float(h_a))
            nw_a, nh_a = max(1, int(w_a * scale_a)), max(1, int(h_a * scale_a))

            h_m, w_m = mask_vis.shape[:2]
            scale_m = min(cw / float(w_m), ch / float(h_m))
            nw_m, nh_m = max(1, int(w_m * scale_m)), max(1, int(h_m * scale_m))

            annotated_resized = cv2.resize(annotated, (nw_a, nh_a))
            mask_resized = cv2.resize(mask_vis, (nw_m, nh_m))

            cv2.putText(annotated_resized, "Annotated Detection & Pose", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(mask_resized, f"Color Mask ({mode_str})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            canvas_h = max(nh_a, nh_m)
            canvas_w = nw_a + nw_m + 10
            combined = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            combined[:nh_a, :nw_a] = annotated_resized
            combined[:nh_m, nw_a + 10:nw_a + 10 + nw_m] = mask_resized

            rgb_combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(rgb_combined)
            img_tk = ImageTk.PhotoImage(image=im_pil)

            self.canvas_img.img_tk = img_tk
            self.canvas_img.config(image=img_tk)

        def _save_yaml(self) -> None:
            save_path = self.config_path
            try:
                raw_yaml = {}
                if os.path.isfile(save_path):
                    with open(save_path, "r") as f:
                        raw_yaml = yaml.safe_load(f) or {}

                params_dict = {
                    "detection_mode": self.params.detection_mode,
                    "hsv_min": [int(x) for x in self.params.hsv_min],
                    "hsv_max": [int(x) for x in self.params.hsv_max],
                    "glare_v_thresh": int(self.params.glare_v_thresh),
                    "clahe_clip_limit": float(self.params.clahe_clip_limit),
                    "pip_min_circularity": float(self.params.pip_min_circularity),
                }

                if "die_detector_node" in raw_yaml and "ros__parameters" in raw_yaml["die_detector_node"]:
                    raw_yaml["die_detector_node"]["ros__parameters"].update(params_dict)
                else:
                    raw_yaml.update(params_dict)

                with open(save_path, "w") as f:
                    yaml.dump(raw_yaml, f, default_flow_style=False)

                messagebox.showinfo("Saved", f"Successfully saved tuned parameters to:\n{save_path}")
                print(f"[GUI] ✅ Saved parameters to {save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save YAML config:\n{e}")

        def _sync_ros2(self) -> None:
            node_name = "/die_detector_node"
            params_to_sync = [
                ("detection_mode", self.params.detection_mode),
                ("hsv_min", [int(x) for x in self.params.hsv_min]),
                ("hsv_max", [int(x) for x in self.params.hsv_max]),
                ("glare_v_thresh", int(self.params.glare_v_thresh)),
                ("clahe_clip_limit", float(self.params.clahe_clip_limit)),
                ("pip_min_circularity", float(self.params.pip_min_circularity)),
            ]

            successes = 0
            failures = 0
            for name, val in params_to_sync:
                val_str = ("[" + ",".join(map(str, val)) + "]") if isinstance(val, list) else (f"'{val}'" if isinstance(val, str) else str(val))
                cmd = ["ros2", "param", "set", node_name, name, val_str]
                try:
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
                    if res.returncode == 0:
                        successes += 1
                    else:
                        failures += 1
                except Exception:
                    failures += 1

            if successes > 0 and failures == 0:
                messagebox.showinfo("ROS 2 Sync", f"Successfully updated all {successes} parameters on {node_name} live!")
            elif successes > 0:
                messagebox.showwarning("ROS 2 Sync", f"Updated {successes} parameters on {node_name} ({failures} failed).")
            else:
                messagebox.showerror(
                    "ROS 2 Node Not Reachable",
                    f"Could not connect to {node_name}.\nMake sure 'ros2 launch drims_die_detection die_detector.launch.py' is running!"
                )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Interactive Die Detector Parameter Tuning GUI")
    parser.add_argument("--image", default=None, help="Path to initial sample RGB image")
    parser.add_argument("--config", default="config/die_detector_params.yaml", help="Path to YAML config file")
    parser.add_argument("--live", action="store_true", help="Subscribe to live ROS 2 camera image topic")
    parser.add_argument("--topic", default=None, help="ROS 2 camera image topic (e.g. /head_front_camera/color/image_raw)")
    parser.add_argument("--cv", action="store_true", help="Force OpenCV Trackbar GUI mode (skips Tkinter)")
    args = parser.parse_args()

    config_path = args.config if os.path.isfile(args.config) else os.path.join(_REPO_ROOT, "config", "die_detector_params.yaml")

    live_topic = None
    if args.live or args.topic is not None:
        if args.topic:
            live_topic = args.topic
        else:
            live_topic = "/head_front_camera/color/image_raw"
            if os.path.isfile(config_path):
                try:
                    with open(config_path, "r") as f:
                        cfg_data = yaml.safe_load(f) or {}
                    params_sec = cfg_data.get("die_detector_node", {}).get("ros__parameters", {})
                    live_topic = params_sec.get("rgb_topic", live_topic)
                except Exception:
                    pass

    if HAS_TKINTER and not args.cv:
        mode_desc = f"LIVE ROS Topic '{live_topic}'" if live_topic else "Offline Saved Images"
        print(f"[GUI] Launching Tkinter GUI... Mode: {mode_desc}")
        root = tk.Tk()
        app = DieDetectorTunerGUI(root, config_path, args.image, live_topic=live_topic)
        root.mainloop()
        if app.live_sub:
            app.live_sub.shutdown()
    else:
        mode_desc = f"LIVE ROS Topic '{live_topic}'" if live_topic else "Offline Saved Images"
        print(f"[GUI] Launching OpenCV Trackbar GUI... Mode: {mode_desc}")
        app = OpenCVTunerGUI(config_path, args.image, live_topic=live_topic)
        app.run()


if __name__ == "__main__":
    main()

