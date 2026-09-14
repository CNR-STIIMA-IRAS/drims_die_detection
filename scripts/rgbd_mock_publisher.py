#!/usr/bin/env python3
"""
RGB-D Mock Publisher Node
=========================
Publishes RGB, Depth (16UC1 mm), CameraInfo, and PointCloud2 topics from real
images stored in the `test_images` folder.

NO synthetic fallback is included — frames are only published if test images are
successfully loaded from disk.

Topics Published:
  * /camera/color/camera_info                (sensor_msgs/CameraInfo)
  * /camera/color/image_raw                  (sensor_msgs/Image, bgr8)
  * /camera/aligned_depth_to_color/image_raw  (sensor_msgs/Image, 16UC1, depth in mm)
  * /camera/depth/color/points               (sensor_msgs/PointCloud2)

Usage:
------
    ros2 run drims_die_detection rgbd_mock_publisher.py
    ros2 run drims_die_detection rgbd_mock_publisher.py --ros-args -p image_index:=1
    ros2 run drims_die_detection rgbd_mock_publisher.py --ros-args -p image_name:=5834717044920750595.png
"""

import os
import sys
import glob
import re
import numpy as np
import cv2

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
    from rcl_interfaces.msg import SetParametersResult
    from cv_bridge import CvBridge
    from ament_index_python.packages import get_package_share_directory
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False


def _load_yaml_defaults(node_name: str) -> dict:
    """Load default parameter values from die_detector_params.yaml if available."""
    try:
        import yaml
        yaml_path = None
        try:
            share_dir = get_package_share_directory("drims_die_detection")
            cand = os.path.join(share_dir, "config", "die_detector_params.yaml")
            if os.path.isfile(cand):
                yaml_path = cand
        except Exception:
            pass

        if yaml_path is None:
            src_cand = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "die_detector_params.yaml"))
            if os.path.isfile(src_cand):
                yaml_path = src_cand

        if yaml_path and os.path.isfile(yaml_path):
            with open(yaml_path, "r") as f:
                raw = yaml.safe_load(f) or {}
            if node_name in raw and "ros__parameters" in raw[node_name]:
                return raw[node_name]["ros__parameters"]
            elif "/**" in raw and "ros__parameters" in raw["/**"]:
                return raw["/**"]["ros__parameters"]
    except Exception:
        pass
    return {}


class RGBDMockPublisher(Node):
    """ROS 2 node that publishes user-selected RGB-D images directly from test_images folder."""

    def __init__(self) -> None:
        super().__init__("rgbd_mock_publisher")
        self.get_logger().info("Initialising RGB-D Mock Publisher…")

        yaml_defaults = _load_yaml_defaults("rgbd_mock_publisher")

        def _get_default(param_name: str, fallback):
            return yaml_defaults.get(param_name, fallback)

        self.declare_parameter("publish_rate", float(_get_default("publish_rate", 2.0)))
        self.declare_parameter("frame_id", str(_get_default("frame_id", "camera_color_optical_frame")))
        self.declare_parameter("rgb_topic", str(_get_default("rgb_topic", "/camera/color/image_raw")))
        self.declare_parameter("depth_topic", str(_get_default("depth_topic", "/camera/aligned_depth_to_color/image_raw")))
        self.declare_parameter("camera_info_topic", str(_get_default("camera_info_topic", "/camera/color/camera_info")))
        self.declare_parameter("points_topic", str(_get_default("points_topic", "/camera/depth/color/points")))
        self.declare_parameter("test_images_dir", "")

        # Image selection parameters
        self.declare_parameter("image_index", 0)
        self.declare_parameter("image_name", "")

        self.declare_parameter("fx", 615.0)
        self.declare_parameter("fy", 615.0)
        self.declare_parameter("cx", 0.0)  # 0.0 means auto (width/2)
        self.declare_parameter("cy", 0.0)  # 0.0 means auto (height/2)

        # Retrieve parameter values
        self._rate = float(self.get_parameter("publish_rate").value)
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._rgb_topic = str(self.get_parameter("rgb_topic").value)
        self._depth_topic = str(self.get_parameter("depth_topic").value)
        self._camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self._points_topic = str(self.get_parameter("points_topic").value)

        self._image_index = int(self.get_parameter("image_index").value)
        self._image_name = str(self.get_parameter("image_name").value)

        self._fx = float(self.get_parameter("fx").value)
        self._fy = float(self.get_parameter("fy").value)
        self._cx_param = float(self.get_parameter("cx").value)
        self._cy_param = float(self.get_parameter("cy").value)

        # ── Locate test_images ──────────────────────────────────────────
        user_dir = str(self.get_parameter("test_images_dir").value)
        self._rgb_files = []

        def _natural_key(filepath: str):
            filename = os.path.basename(filepath)
            return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', filename)]

        search_dirs = []
        if user_dir:
            search_dirs.append(user_dir)

        # 1. Traverse parent directories from current file (prioritize source repository)
        script_file = os.path.realpath(__file__)
        curr_dir = os.path.dirname(script_file)
        for _ in range(5):
            search_dirs.append(os.path.join(curr_dir, "test_images"))
            search_dirs.append(os.path.join(curr_dir, "src", "drims_die_detection", "test_images"))
            parent = os.path.dirname(curr_dir)
            if parent == curr_dir:
                break
            curr_dir = parent

        # 2. Standard workspace / container paths
        search_dirs.append("/home/ws/src/drims_die_detection/test_images")
        search_dirs.append("/home/ws/src/drims_cells/test_images")

        # 3. Package share directory (fallback)
        try:
            share_dir = get_package_share_directory("drims_die_detection")
            search_dirs.append(os.path.join(share_dir, "test_images"))
        except Exception:
            pass

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        found_dir = None
        for s_dir in search_dirs:
            rgb_path = os.path.join(s_dir, "rgb")
            if os.path.isdir(rgb_path):
                files = [
                    os.path.join(rgb_path, fname)
                    for fname in os.listdir(rgb_path)
                    if os.path.splitext(fname)[1].lower() in valid_exts
                ]
                if files:
                    files.sort(key=_natural_key)
                    self._rgb_files = files
                    found_dir = rgb_path
                    self.get_logger().info(f"Loaded {len(files)} test images from: {rgb_path}")
                    for idx, f in enumerate(files):
                        self.get_logger().info(f"  [{idx}] {os.path.basename(f)}")
                    break

        if not self._rgb_files:
            self.get_logger().error(
                f"[ERROR] Could not find any test images in test_images/rgb!\n"
                f"Searched directories:\n  " + "\n  ".join(search_dirs)
            )

        # Dynamic parameter update callback
        self.add_on_set_parameters_callback(self._on_params_changed)

        # ── ROS Infrastructure ──────────────────────────────────────────
        self._bridge = CvBridge()
        self._rgb_pub = self.create_publisher(Image, self._rgb_topic, 10)
        self._depth_pub = self.create_publisher(Image, self._depth_topic, 10)
        self._info_pub = self.create_publisher(CameraInfo, self._camera_info_topic, 10)
        self._points_pub = self.create_publisher(PointCloud2, self._points_topic, 10)

        # Timer loop
        timer_period = 1.0 / max(self._rate, 0.1)
        self._timer = self.create_timer(timer_period, self._publish_frame)

        self.get_logger().info(
            f"Publishing mock RGB-D stream at {self._rate:.1f} Hz\n"
            f"  Image Selection: index={self._image_index}, name='{self._image_name}'\n"
            f"  RGB: {self._rgb_topic}\n"
            f"  Depth: {self._depth_topic} (16UC1)\n"
            f"  Points: {self._points_topic} (PointCloud2)\n"
            f"  CameraInfo: {self._camera_info_topic}"
        )

    def _on_params_changed(self, params):
        """Handle live parameter updates (e.g. via ros2 param set)."""
        for param in params:
            if param.name == "image_index":
                self._image_index = int(param.value)
                self.get_logger().info(f"Selected image_index updated to: {self._image_index}")
            elif param.name == "image_name":
                self._image_name = str(param.value)
                self.get_logger().info(f"Selected image_name updated to: '{self._image_name}'")
        return SetParametersResult(successful=True)

    def _build_camera_info(self, width: int, height: int) -> CameraInfo:
        """Construct CameraInfo message for current frame dimensions."""
        msg = CameraInfo()
        msg.header.frame_id = self._frame_id
        msg.width = width
        msg.height = height
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        cx = self._cx_param if self._cx_param > 0.0 else width / 2.0
        cy = self._cy_param if self._cy_param > 0.0 else height / 2.0

        msg.k = [self._fx, 0.0, cx, 0.0, self._fy, cy, 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [self._fx, 0.0, cx, 0.0, 0.0, self._fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return msg

    def _build_point_cloud2(self, stamp, rgb_bgr: np.ndarray, depth_raw: np.uint16) -> PointCloud2:
        """Construct a PointCloud2 message from RGB and uint16 depth image."""
        h, w = rgb_bgr.shape[:2]
        step = 2  # Downsample for fast publishing & RViz rendering
        rgb_s = rgb_bgr[::step, ::step]
        depth_s = depth_raw[::step, ::step]

        u, v = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
        z = depth_s.astype(np.float32) / 1000.0  # meters

        valid = (z > 0.1) & (z < 3.5)
        if not np.any(valid):
            return None

        cx = self._cx_param if self._cx_param > 0.0 else w / 2.0
        cy = self._cy_param if self._cy_param > 0.0 else h / 2.0

        x_pts = (u[valid] - cx) * z[valid] / self._fx
        y_pts = (v[valid] - cy) * z[valid] / self._fy
        z_pts = z[valid]

        b = rgb_s[:, :, 0][valid]
        g = rgb_s[:, :, 1][valid]
        r = rgb_s[:, :, 2][valid]

        # Pack RGB into uint32 / float32
        rgb_packed = (r.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | b.astype(np.uint32)

        n_pts = len(x_pts)
        buffer = np.zeros((n_pts, 4), dtype=np.float32)
        buffer[:, 0] = x_pts
        buffer[:, 1] = y_pts
        buffer[:, 2] = z_pts
        buffer[:, 3] = rgb_packed.view(np.float32)

        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id
        msg.height = 1
        msg.width = n_pts
        msg.is_dense = True
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * n_pts

        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.data = buffer.tobytes()
        return msg

    def _load_selected_frame(self):
        """Load selected RGB and matching Depth image pair from disk."""
        if not self._rgb_files:
            self.get_logger().error("No test images available in test_images/rgb!")
            return None, None

        target_file = None
        if self._image_name:
            for f in self._rgb_files:
                b_name = os.path.basename(f)
                if self._image_name in b_name or b_name.startswith(self._image_name):
                    target_file = f
                    break
            if not target_file:
                self.get_logger().error(
                    f"Image '{self._image_name}' not found among {len(self._rgb_files)} test images."
                )
                return None, None
        else:
            if 0 <= self._image_index < len(self._rgb_files):
                target_file = self._rgb_files[self._image_index]
            else:
                self.get_logger().error(
                    f"Image index {self._image_index} is out of range (valid range: 0 .. {len(self._rgb_files) - 1})."
                )
                return None, None

        rgb = cv2.imread(target_file)
        if rgb is None:
            self.get_logger().error(f"Failed to read RGB image from disk: {target_file}")
            return None, None

        h, w = rgb.shape[:2]
        base_name = os.path.basename(target_file)
        depth_dir = os.path.join(os.path.dirname(os.path.dirname(target_file)), "depth")

        depth_candidates = [
            os.path.join(depth_dir, f"depth_{base_name}"),
            os.path.join(depth_dir, base_name),
            os.path.join(depth_dir, f"depth_{os.path.splitext(base_name)[0]}.png"),
            os.path.join(depth_dir, f"depth_{os.path.splitext(base_name)[0]}.jpg"),
            os.path.join(depth_dir, f"{os.path.splitext(base_name)[0]}.npy"),
        ]

        depth_raw = None
        for cand in depth_candidates:
            if os.path.isfile(cand):
                if cand.endswith(".npy"):
                    depth_m = np.load(cand)
                    depth_raw = (depth_m * 1000.0).astype(np.uint16)
                else:
                    dep_img = cv2.imread(cand, cv2.IMREAD_UNCHANGED)
                    if dep_img is not None:
                        if dep_img.ndim == 3:
                            dep_img = cv2.cvtColor(dep_img, cv2.COLOR_BGR2GRAY)
                        if dep_img.dtype == np.uint8:
                            depth_raw = (300 + dep_img.astype(np.float32) * 3.5).astype(np.uint16)
                        else:
                            depth_raw = dep_img.astype(np.uint16)
                break

        if depth_raw is None:
            depth_raw = np.full((h, w), 700, dtype=np.uint16)

        return rgb, depth_raw

    def _publish_frame(self) -> None:
        """Publish synchronized CameraInfo, RGB, Depth, and PointCloud2 frames if loaded correctly."""
        rgb_img, depth_img = self._load_selected_frame()
        if rgb_img is None or depth_img is None:
            return  # NO fallback: do not publish if image loading failed

        stamp = self.get_clock().now().to_msg()
        h, w = rgb_img.shape[:2]
        info_msg = self._build_camera_info(w, h)
        info_msg.header.stamp = stamp

        rgb_msg = self._bridge.cv2_to_imgmsg(rgb_img, encoding="bgr8")
        rgb_msg.header.stamp = stamp
        rgb_msg.header.frame_id = self._frame_id

        depth_msg = self._bridge.cv2_to_imgmsg(depth_img, encoding="16UC1")
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = self._frame_id

        pc_msg = self._build_point_cloud2(stamp, rgb_img, depth_img)

        self._info_pub.publish(info_msg)
        self._rgb_pub.publish(rgb_msg)
        self._depth_pub.publish(depth_msg)
        if pc_msg is not None:
            self._points_pub.publish(pc_msg)


def main(args=None) -> None:
    if not HAS_ROS2:
        print("[ERROR] rclpy / cv_bridge not available. Source ROS 2 Humble.")
        sys.exit(1)

    rclpy.init(args=args)
    node = RGBDMockPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
