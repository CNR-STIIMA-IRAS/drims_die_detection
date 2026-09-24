#!/usr/bin/env python3
"""
ROS 2 Die Detector Node
=======================
Two pose methods (parameter ``pose_method``):

  * ``silhouette`` — SilhouettePipeline: YOLOE die mask -> cube-silhouette fit
    on the table plane + CNN top/front face. The table plane comes from
    ``plane_source``: ``tf`` (table_frame -> camera TF + ``table_height_m``),
    ``fixed`` (``plane_normal_cam`` / ``plane_height_m``) or ``depth`` (RANSAC).
    With ``tf`` / ``fixed`` only the RGB topic is needed.
  * ``yoloe_faces`` — FacePolygonPipeline: YOLOE die mask -> the original
    face / pip segmentation + top-face-polygon pose, on the same plane sources.
  * ``classical``  — DieDetectorPipeline: RANSAC depth plane + face/pip
    segmentation (needs synchronised RGB + aligned depth).

Publishes:

  * /dice/pose          — geometry_msgs/PoseStamped
  * /dice/debug_panels  — sensor_msgs/Image  (6-panel collage)
  * TF2 transforms      — die_top_face & die_centroid frames

Parameters are loaded from the ROS 2 parameter server (set via the YAML config
in config/die_detector_params.yaml through the launch file).

If no valid depth is received, the node falls back to monocular depth
estimation (controlled by the `use_monocular_fallback` parameter).
"""

from __future__ import annotations

import sys
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
    from sensor_msgs.msg import Image, CameraInfo
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from visualization_msgs.msg import Marker
    from rcl_interfaces.msg import SetParametersResult
    from tf2_ros import TransformBroadcaster
    from tf2_ros.buffer import Buffer
    from tf2_ros.transform_listener import TransformListener
    from rclpy.duration import Duration
    from rclpy.time import Time
    from cv_bridge import CvBridge
    import message_filters
    from drims_homework_interfaces.srv import DieIdentification3D
    HAS_ROS2 = True
    HAS_EASY_MOTION_MSGS = True
except ImportError:
    HAS_ROS2 = False
    HAS_EASY_MOTION_MSGS = False
    DieIdentification3D = None

# The die detection library is always importable regardless of ROS
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from drims_die_detection import DieDetectorParams, DieDetectorPipeline
from drims_die_detection.point_cloud_processor import PointCloudProcessor
from scipy.spatial.transform import Rotation as R_sci


def _load_yaml_defaults(node_name: str) -> dict:
    """Load default parameter values from die_detector_params.yaml if available."""
    try:
        import yaml
        yaml_path = None
        try:
            from ament_index_python.packages import get_package_share_directory
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


def _declare_and_get(node: "Node", name: str, default):
    """Declare a ROS 2 parameter and return its value."""
    node.declare_parameter(name, default)
    return node.get_parameter(name).value


class DieDetectorNode(Node):
    """ROS 2 Humble node for real-time 3D die detection."""

    def __init__(self) -> None:
        super().__init__("die_detector_node")
        self.get_logger().info("Initialising DieDetectorNode…")

        yaml_defaults = _load_yaml_defaults("die_detector_node")

        def _g(name: str, fallback):
            return yaml_defaults.get(name, fallback)

        # ── Read ROS 2 parameters ──────────────────────────────────────
        p = DieDetectorParams(
            debug=_declare_and_get(self, "debug", _g("debug", False)),
            visualize=_declare_and_get(self, "visualize", _g("visualize", False)),
            save=_declare_and_get(self, "save", _g("save", False)),
            fx=_declare_and_get(self, "fx", _g("fx", 615.0)),
            fy=_declare_and_get(self, "fy", _g("fy", 615.0)),
            cx=_declare_and_get(self, "cx", float(_g("cx", -1.0) or -1.0)),
            cy=_declare_and_get(self, "cy", float(_g("cy", -1.0) or -1.0)),
            depth_model=_declare_and_get(self, "depth_model", _g("depth_model", "Intel/dpt-hybrid-midas")),
            depth_scale=_declare_and_get(self, "depth_scale", _g("depth_scale", 0.001)),
            use_monocular_fallback=_declare_and_get(self, "use_monocular_fallback", _g("use_monocular_fallback", True)),
            ransac_distance_threshold=_declare_and_get(self, "ransac_distance_threshold", _g("ransac_distance_threshold", 0.015)),
            ransac_max_iterations=_declare_and_get(self, "ransac_max_iterations", _g("ransac_max_iterations", 1000)),
            die_color=_declare_and_get(self, "die_color", _g("die_color", "white")),
            num_color_clusters=_declare_and_get(self, "num_color_clusters", _g("num_color_clusters", 5)),
            die_size_m=_declare_and_get(self, "die_size_m", _g("die_size_m", 0.050)),
            detection_mode=_declare_and_get(self, "detection_mode", _g("detection_mode", "hsv")),
            hsv_min=_declare_and_get(self, "hsv_min", _g("hsv_min", [0, 0, 150])),
            hsv_max=_declare_and_get(self, "hsv_max", _g("hsv_max", [180, 80, 255])),
            glare_v_thresh=_declare_and_get(self, "glare_v_thresh", _g("glare_v_thresh", 245)),
            clahe_clip_limit=_declare_and_get(self, "clahe_clip_limit", _g("clahe_clip_limit", 3.0)),
            pip_min_circularity=_declare_and_get(self, "pip_min_circularity", _g("pip_min_circularity", 0.45)),
            canny_low_thresh=_declare_and_get(self, "canny_low_thresh", _g("canny_low_thresh", 40)),
            canny_high_thresh=_declare_and_get(self, "canny_high_thresh", _g("canny_high_thresh", 130)),
            min_pips=_declare_and_get(self, "min_pips", _g("min_pips", 1)),
            max_pips=_declare_and_get(self, "max_pips", _g("max_pips", 6)),
            min_die_height_m=_declare_and_get(self, "min_die_height_m", _g("min_die_height_m", 0.003)),
            max_die_height_m=_declare_and_get(self, "max_die_height_m", _g("max_die_height_m", 0.065)),
            output_dir=_declare_and_get(self, "output_dir", _g("output_dir", "output")),
            rgb_topic=_declare_and_get(self, "rgb_topic", _g("rgb_topic", "/camera/color/image_raw")),
            depth_topic=_declare_and_get(self, "depth_topic",
                                         _g("depth_topic", "/camera/aligned_depth_to_color/image_raw")),
            camera_info_topic=_declare_and_get(self, "camera_info_topic",
                                               _g("camera_info_topic", "/camera/color/camera_info")),
            pose_topic=_declare_and_get(self, "pose_topic", _g("pose_topic", "/dice/pose")),
            debug_panels_topic=_declare_and_get(self, "debug_panels_topic", _g("debug_panels_topic", "/dice/debug_panels")),
            top_down_topic=_declare_and_get(self, "top_down_topic", _g("top_down_topic", "/dice/top_down")),
            service_name=_declare_and_get(self, "service_name", _g("service_name", "die_identification")),
            # ── silhouette pose method ──
            pose_method=_declare_and_get(self, "pose_method", _g("pose_method", "silhouette")),
            device=_declare_and_get(self, "device", _g("device", "auto")),
            cnn_model_path=_declare_and_get(self, "cnn_model_path", _g("cnn_model_path", "weights/die_mobilenet_v3.pt")),
            cnn_conf_thresh=_declare_and_get(self, "cnn_conf_thresh", _g("cnn_conf_thresh", 0.60)),
            min_fit_iou=_declare_and_get(self, "min_fit_iou", _g("min_fit_iou", 0.85)),
            accept_fit_iou=_declare_and_get(self, "accept_fit_iou", _g("accept_fit_iou", 0.90)),
            min_face_pose_iou=_declare_and_get(self, "min_face_pose_iou", _g("min_face_pose_iou", 0.60)),
            max_yoloe_candidates=_declare_and_get(self, "max_yoloe_candidates", _g("max_yoloe_candidates", 3)),
            yoloe_candidate_conf=_declare_and_get(self, "yoloe_candidate_conf", _g("yoloe_candidate_conf", 0.02)),
            color_fallback=_declare_and_get(self, "color_fallback", _g("color_fallback", True)),
            plane_source=_declare_and_get(self, "plane_source", _g("plane_source", "tf")),
            table_frame=_declare_and_get(self, "table_frame", _g("table_frame", "base_footprint")),
            table_height_m=_declare_and_get(self, "table_height_m", _g("table_height_m", 0.55)),
            plane_normal_cam=_declare_and_get(self, "plane_normal_cam", _g("plane_normal_cam", [0.0, -0.6, -0.8])),
            plane_height_m=_declare_and_get(self, "plane_height_m", _g("plane_height_m", 0.62)),
        )
        # cx / cy < 0 (ROS parameters cannot be None) = image centre
        p.cx = p.cx if p.cx is not None and p.cx >= 0 else None
        p.cy = p.cy if p.cy is not None and p.cy >= 0 else None
        self._params = p
        p.log(self.get_logger())

        # Parameter change callback
        self.add_on_set_parameters_callback(self._on_param_change)

        # ── Pipeline ───────────────────────────────────────────────────
        # silhouette / yoloe_faces take the table plane from plane_source (tf | fixed | depth)
        self._plane_based = p.pose_method in ("silhouette", "yoloe_faces")
        if p.pose_method == "silhouette":
            from drims_die_detection import SilhouettePipeline
            self._pipeline = SilhouettePipeline(p)
            self.get_logger().info(f"Pose method: silhouette — YOLOE + cube fit + CNN "
                                   f"(device={self._pipeline.device}, plane_source={p.plane_source})")
        elif p.pose_method == "yoloe_faces":
            from drims_die_detection import FacePolygonPipeline
            self._pipeline = FacePolygonPipeline(p, "yoloe")
            self.get_logger().info(f"Pose method: yoloe_faces — YOLOE + face/pip polygons + top-face pose "
                                   f"(device={self._pipeline.device}, plane_source={p.plane_source})")
        elif p.pose_method != "classical":
            raise ValueError(f"Unknown pose_method '{p.pose_method}' (silhouette | yoloe_faces | classical)")
        else:
            self._pipeline = DieDetectorPipeline(p)
            self.get_logger().info("Pose method: classical (RANSAC depth plane + face/pip polygons)")
        self._pc_proc = PointCloudProcessor(p) if (self._plane_based and p.plane_source == "depth") else None
        self._last_result: dict | None = None
        self._warned_no_info = False
        self._stats = {"received": 0, "processed": 0, "poses": 0}
        self.create_timer(10.0, self._log_stats)

        # ── ROS infrastructure ─────────────────────────────────────────
        self._bridge = CvBridge()
        self._tf_broadcaster = TransformBroadcaster(self)
        if self._plane_based and p.plane_source == "tf":
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)

        # Intrinsics: updated from CameraInfo (start with param defaults)
        self._fx = p.fx
        self._fy = p.fy
        self._cx: float | None = p.cx
        self._cy: float | None = p.cy
        self._intrinsics_received = False

        # Publishers
        self._pose_pub = self.create_publisher(PoseStamped, p.pose_topic, 10)
        self._panels_pub = self.create_publisher(Image, p.debug_panels_topic, 10)
        self._plane_marker_pub = self.create_publisher(Marker, "/dice/fitted_plane_marker", 10)

        # Service
        if HAS_EASY_MOTION_MSGS and DieIdentification3D is not None:
            self._service = self.create_service(
                DieIdentification3D, p.service_name, self._die_identification_cb
            )
            self.get_logger().info(f"Die identification service '{p.service_name}' ready.")
        else:
            self.get_logger().error(f"Die identification service '{p.service_name}' not available.")
            self._service = None

        # QoS
        qos_camera = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST,
                                depth=10, 
                                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                durability=QoSDurabilityPolicy.VOLATILE)

        # CameraInfo subscriber (latched once)
        self._info_sub = self.create_subscription(
            CameraInfo, p.camera_info_topic, self._camera_info_cb, qos_camera
        )

        needs_depth = (not self._plane_based) or p.plane_source == "depth"
        if needs_depth:
            # Synchronised RGB + Depth
            self._rgb_sub = message_filters.Subscriber(self, Image, p.rgb_topic, qos_profile=qos_camera)
            self._depth_sub = message_filters.Subscriber(self, Image, p.depth_topic, qos_profile=qos_camera)
            self._sync = message_filters.ApproximateTimeSynchronizer(
                [self._rgb_sub, self._depth_sub], queue_size=10, slop=0.05
            )
            self._sync.registerCallback(self._rgbd_cb)
        else:
            # RGB only (keep just the latest frame: the pipeline is slower than the camera)
            qos_latest = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                                    reliability=QoSReliabilityPolicy.BEST_EFFORT,   # camera drivers publish best-effort
                                    durability=QoSDurabilityPolicy.VOLATILE)
            self._rgb_only_sub = self.create_subscription(Image, p.rgb_topic, self._rgb_cb, qos_latest)

        self.get_logger().info(
            f"Listening — RGB: {p.rgb_topic} | Depth: {p.depth_topic if needs_depth else '(not used)'} | "
            f"Info: {p.camera_info_topic}"
        )

    # ──────────────────────────────────────────────────────────────────
    def _camera_info_cb(self, msg: CameraInfo) -> None:
        """Extract and store camera intrinsics from CameraInfo."""
        self._fx = msg.k[0]
        self._fy = msg.k[4]
        self._cx = msg.k[2]
        self._cy = msg.k[5]
        # Update params so the pipeline uses live intrinsics
        self._params.fx = self._fx
        self._params.fy = self._fy
        self._params.cx = self._cx
        self._params.cy = self._cy
        if not self._intrinsics_received:
            self._intrinsics_received = True
            self.get_logger().info(
                f"Camera intrinsics received: fx={self._fx:.1f} fy={self._fy:.1f} "
                f"cx={self._cx:.1f} cy={self._cy:.1f}"
            )

    # ──────────────────────────────────────────────────────────────────
    def _rgbd_cb(self, rgb_msg: Image, depth_msg: Image) -> None:
        """Callback for synchronised RGB+depth frame pairs."""
        # Convert images
        try:
            rgb = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth_raw = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().error(f"Image conversion failed: {exc}")
            return

        # Convert depth to float32 metres
        if depth_raw.dtype == np.uint16:
            depth_m = depth_raw.astype(np.float32) * self._params.depth_scale
        else:
            depth_m = depth_raw.astype(np.float32)

        if self._plane_based:
            self._process_frame(rgb, rgb_msg, depth_m)
            return

        # Run pipeline
        try:
            result = self._pipeline.process_rgbd(rgb, depth_m)
            result["frame_id"] = rgb_msg.header.frame_id or self._params.camera_frame_id
            self._last_result = result
        except Exception as exc:
            import traceback
            self.get_logger().error(f"Pipeline error: {exc}\n{traceback.format_exc()}")
            return

        stamp = rgb_msg.header.stamp
        frame_id = rgb_msg.header.frame_id or self._params.camera_frame_id

        # Publish PoseStamped
        centroid = result["centroid"]
        quat = result["quaternion"]
        self._publish_pose(centroid, quat, stamp, frame_id)

        # Broadcast TF2 transforms
        self._broadcast_tf("die_top_face", centroid, quat, stamp, frame_id)
        die_c = result["die_centroid_tf"]
        self._broadcast_tf("die_centroid", die_c, quat, stamp, frame_id)

        # Publish transparent fitted plane marker on table surface for RViz 3D visualization
        table_tf = result.get("table_surface_tf", centroid)
        self._publish_plane_marker(table_tf, quat, stamp, frame_id)

        # Publish 6-panel debug collage
        collage = self._pipeline.build_debug_panels(rgb, result)
        self._panels_pub.publish(
            self._to_img_msg(collage, stamp, frame_id)
        )

        # Detailed logging of pips per face, 3D position, and orientation quaternion
        faces = result.get("faces", [])
        face_pips = [f"{f.get('num_pips', 0)} pips" for f in faces]
        faces_detail_str = ", ".join([f"Face {i+1}: {p}" for i, p in enumerate(face_pips)]) if faces else "No faces"

        self.get_logger().info(
            f"Detected {len(faces)} face(s) [{faces_detail_str}] | Total Pips: {result['pip_count']}\n"
            f"  Ref Frame       : {frame_id}\n"
            f"  Position (m)    : x={centroid[0]:.3f}, y={centroid[1]:.3f}, z={centroid[2]:.3f}\n"
            f"  Orientation (q) : x={quat[0]:.3f}, y={quat[1]:.3f}, z={quat[2]:.3f}, w={quat[3]:.3f}"
        )

    # ──────────────────────────────────────────────────────────────────
    def _log_stats(self) -> None:
        s = self._stats
        if s["received"]:
            self.get_logger().info(f"Last 10 s: received {s['received']} frames, processed {s['processed']}, "
                                   f"published {s['poses']} poses ({s['poses'] / 10.0:.1f} Hz)")
        else:
            p = self._params
            needs_depth = (not self._plane_based) or p.plane_source == "depth"
            topics = [p.rgb_topic] + ([p.depth_topic] if needs_depth else [])
            status = ", ".join(f"{t} ({self.count_publishers(t)} publishers)" for t in topics)
            self.get_logger().warn(f"No frames in the last 10 s. Waiting for: {status}"
                                   + (" [RGB + depth must both arrive, time-synchronised]" if needs_depth else ""))
        self._stats = {k: 0 for k in s}

    def _rgb_cb(self, rgb_msg: Image) -> None:
        """RGB-only callback (plane-based methods with a TF / fixed table plane)."""
        try:
            rgb = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Image conversion failed: {exc}")
            return
        self._process_frame(rgb, rgb_msg, None)

    def _intrinsics(self, rgb):
        if not self._intrinsics_received and not self._warned_no_info:
            self.get_logger().warn(f"No CameraInfo on {self._params.camera_info_topic} yet — using the "
                                   f"fx/fy/cx/cy parameters (cx/cy default to the image centre).")
            self._warned_no_info = True
        h, w = rgb.shape[:2]
        cx = self._cx if self._cx is not None else w / 2.0
        cy = self._cy if self._cy is not None else h / 2.0
        return np.array([[self._fx, 0.0, cx], [0.0, self._fy, cy], [0.0, 0.0, 1.0]])

    def _table_plane(self, rgb, depth_m, K, rgb_msg, frame_id):
        """Table plane in the camera optical frame as (n, h): n·X + h = 0, n up."""
        p = self._params
        if p.plane_source == "fixed":
            n = np.asarray(p.plane_normal_cam, dtype=np.float64)
            return n / np.linalg.norm(n), float(p.plane_height_m)
        if p.plane_source == "tf":
            try:
                tf = self._tf_buffer.lookup_transform(frame_id, p.table_frame, rgb_msg.header.stamp,
                                                      timeout=Duration(seconds=0.05))
            except Exception:
                try:   # fall back to the latest available transform
                    tf = self._tf_buffer.lookup_transform(frame_id, p.table_frame, Time())
                except Exception as exc:
                    self.get_logger().warn(f"TF {p.table_frame} -> {frame_id} unavailable: {exc}",
                                           throttle_duration_sec=2.0)
                    return None
            q = tf.transform.rotation
            t = tf.transform.translation
            R = R_sci.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            n = R @ np.array([0.0, 0.0, 1.0])
            p_table = R @ np.array([0.0, 0.0, p.table_height_m]) + np.array([t.x, t.y, t.z])
            return n, float(-n @ p_table)
        # depth: RANSAC on the aligned depth image
        points, _, _, _ = self._pc_proc.create_point_cloud(rgb, depth_m)
        if len(points) < 100:
            self.get_logger().warn("Too few valid depth points for the table plane.", throttle_duration_sec=2.0)
            return None
        (a, b, c, d), _, _, _ = self._pc_proc.fit_plane_ransac(points)
        n = np.array([a, b, c], dtype=np.float64)
        scale = np.linalg.norm(n)
        n, d = n / scale, d / scale
        if d < 0:              # orient n towards the camera (camera height d > 0)
            n, d = -n, -d
        return n, float(d)

    def _process_frame(self, rgb, rgb_msg, depth_m) -> None:
        self._stats["received"] += 1
        stamp = rgb_msg.header.stamp
        frame_id = rgb_msg.header.frame_id or self._params.camera_frame_id
        K = self._intrinsics(rgb)
        plane = self._table_plane(rgb, depth_m, K, rgb_msg, frame_id)
        if plane is None:
            reason = {"tf": f"no TF {self._params.table_frame} -> {frame_id}",
                      "depth": "too few depth points for the table plane"}.get(self._params.plane_source, "no table plane")
            self._panels_pub.publish(self._to_img_msg(
                self._pipeline.build_debug_panels(rgb, {"valid": False, "reason": reason}), stamp, frame_id))
            return
        try:
            result = self._pipeline.process(rgb, K, plane[0], plane[1])
        except Exception as exc:
            import traceback
            self.get_logger().error(f"Pipeline error: {exc}\n{traceback.format_exc()}")
            return

        self._stats["processed"] += 1
        self._panels_pub.publish(self._to_img_msg(self._pipeline.build_debug_panels(rgb, result),
                                                  stamp, frame_id))
        t = result["timings_ms"]
        if not result["valid"]:
            self.get_logger().warn(f"No die pose: {result['reason']} ({t['total']:.0f} ms)",
                                   throttle_duration_sec=1.0)
            return
        result["frame_id"] = frame_id
        self._last_result = result
        self._stats["poses"] += 1
        centroid, quat = result["centroid"], result["quaternion"]
        self._publish_pose(centroid, quat, stamp, frame_id)
        self._broadcast_tf("die_top_face", centroid, quat, stamp, frame_id)
        self._broadcast_tf("die_centroid", result["die_centroid_tf"], quat, stamp, frame_id)
        self._publish_plane_marker(result["table_surface_tf"], quat, stamp, frame_id)
        self.get_logger().info(
            f"Die: top {result['top_face_str']}, front {result['front_face_str']} | "
            f"IoU {result['iou']:.3f} ({result['source']}) | yaw {result['yaw_deg']:+.1f}° | "
            f"xyz=({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}) m in {frame_id} | "
            f"{t['total']:.0f} ms", throttle_duration_sec=1.0)

    # ──────────────────────────────────────────────────────────────────
    def _publish_pose(self, centroid, quat, stamp, frame_id) -> None:
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.pose.position.x = float(centroid[0])
        msg.pose.position.y = float(centroid[1])
        msg.pose.position.z = float(centroid[2])
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        self._pose_pub.publish(msg)

    def _publish_plane_marker(self, centroid, quat, stamp, frame_id) -> None:
        """Publish transparent fitted plane surface marker to RViz."""
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = frame_id
        marker.ns = "fitted_plane"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = float(centroid[0])
        marker.pose.position.y = float(centroid[1])
        marker.pose.position.z = float(centroid[2])
        marker.pose.orientation.x = float(quat[0])
        marker.pose.orientation.y = float(quat[1])
        marker.pose.orientation.z = float(quat[2])
        marker.pose.orientation.w = float(quat[3])
        marker.scale.x = 0.40  # 40 cm width
        marker.scale.y = 0.40  # 40 cm length
        marker.scale.z = 0.002 # 2 mm thin surface sheet
        marker.color.r = 0.0
        marker.color.g = 0.8
        marker.color.b = 1.0
        marker.color.a = 0.35  # Transparent
        self._plane_marker_pub.publish(marker)

    def _broadcast_tf(self, child_id, translation, quat, stamp, frame_id) -> None:
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = frame_id
        t.child_frame_id = child_id
        t.transform.translation.x = float(translation[0])
        t.transform.translation.y = float(translation[1])
        t.transform.translation.z = float(translation[2])
        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])
        self._tf_broadcaster.sendTransform(t)

    def _to_img_msg(self, bgr_img, stamp, frame_id) -> Image:
        msg = self._bridge.cv2_to_imgmsg(bgr_img, encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        return msg

    def _die_identification_cb(self, request, response):
        """ROS 2 service handler returning top face, front face, and die top TF."""
        if self._last_result is None:
            response.top_face = "None"
            response.front_face = "None"
            response.top_face_pips = 0
            response.front_face_pips = 0
            response.success = False
            return response

        res = self._last_result
        top_str = res.get("top_face_str", "None")
        front_str = res.get("front_face_str", "None")
        top_pips = res.get("top_face_pips", None)
        front_pips = res.get("front_face_pips", None)

        centroid = res["centroid"]
        quat = res["quaternion"]
        stamp = self.get_clock().now().to_msg()
        frame_id = res.get("frame_id", self._params.camera_frame_id)   # frame of the image it came from

        # Populate die_top_tf (TransformStamped)
        response.die_top_tf.header.stamp = stamp
        response.die_top_tf.header.frame_id = frame_id
        response.die_top_tf.child_frame_id = "die_top_face"
        response.die_top_tf.transform.translation.x = float(centroid[0])
        response.die_top_tf.transform.translation.y = float(centroid[1])
        response.die_top_tf.transform.translation.z = float(centroid[2])
        response.die_top_tf.transform.rotation.x = float(quat[0])
        response.die_top_tf.transform.rotation.y = float(quat[1])
        response.die_top_tf.transform.rotation.z = float(quat[2])
        response.die_top_tf.transform.rotation.w = float(quat[3])

        # Populate die_top_pose (PoseStamped)
        response.die_top_pose.header.stamp = stamp
        response.die_top_pose.header.frame_id = frame_id
        response.die_top_pose.pose.position.x = float(centroid[0])
        response.die_top_pose.pose.position.y = float(centroid[1])
        response.die_top_pose.pose.position.z = float(centroid[2])
        response.die_top_pose.pose.orientation.x = float(quat[0])
        response.die_top_pose.pose.orientation.y = float(quat[1])
        response.die_top_pose.pose.orientation.z = float(quat[2])
        response.die_top_pose.pose.orientation.w = float(quat[3])

        response.top_face = str(top_str)
        response.front_face = str(front_str)
        response.top_face_pips = int(top_pips) if top_pips is not None else 0
        response.front_face_pips = int(front_pips) if front_pips is not None else 0
        response.success = True
        return response

    def _on_param_change(self, params) -> SetParametersResult:
        """Dynamic ROS 2 parameter update callback."""
        for p in params:
            if hasattr(self._params, p.name):
                val = list(p.value) if hasattr(p.value, '__iter__') and not isinstance(p.value, (str, bytes)) else p.value
                setattr(self._params, p.name, val)
                self.get_logger().info(f"Dynamic param update: {p.name} = {val}")
        return SetParametersResult(successful=True)


# ──────────────────────────────────────────────────────────────────────────
def main(args=None) -> None:
    if not HAS_ROS2:
        print("[ERROR] rclpy / cv_bridge not available. "
              "Source ROS 2 Humble and rebuild the workspace.")
        return

    rclpy.init(args=args)
    node = DieDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():          # Ctrl-C under ros2 launch has already shut the context down
            rclpy.shutdown()


if __name__ == "__main__":
    main()
