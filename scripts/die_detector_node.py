#!/usr/bin/env python3
"""
ROS 2 Die Detector Node
=======================
Subscribes to synchronised RGB + aligned depth topics and a CameraInfo topic,
runs the DieDetectorPipeline, and publishes:

  * /dice/pose          — geometry_msgs/PoseStamped
  * /dice/debug_panels  — sensor_msgs/Image  (6-panel collage)
  * /dice/top_down      — sensor_msgs/Image  (pip-annotated top-down crop)
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
    from sensor_msgs.msg import Image, CameraInfo
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from visualization_msgs.msg import Marker
    from rcl_interfaces.msg import SetParametersResult
    from tf2_ros import TransformBroadcaster
    from cv_bridge import CvBridge
    import message_filters
    HAS_ROS2 = True
    try:
        from easy_motion_msgs.srv import DieIdentification3D
        HAS_EASY_MOTION_MSGS = True
    except ImportError:
        HAS_EASY_MOTION_MSGS = False
        DieIdentification3D = None
except ImportError:
    HAS_ROS2 = False
    HAS_EASY_MOTION_MSGS = False
    DieIdentification3D = None

# The die detection library is always importable regardless of ROS
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from drims_die_detection import DieDetectorParams, DieDetectorPipeline


def _declare_and_get(node: "Node", name: str, default):
    """Declare a ROS 2 parameter and return its value."""
    node.declare_parameter(name, default)
    return node.get_parameter(name).value


class DieDetectorNode(Node):
    """ROS 2 Humble node for real-time 3D die detection."""

    def __init__(self) -> None:
        super().__init__("die_detector_node")
        self.get_logger().info("Initialising DieDetectorNode…")

        # ── Read ROS 2 parameters ──────────────────────────────────────
        p = DieDetectorParams(
            debug=_declare_and_get(self, "debug", False),
            visualize=_declare_and_get(self, "visualize", False),
            save=_declare_and_get(self, "save", False),
            fx=_declare_and_get(self, "fx", 615.0),
            fy=_declare_and_get(self, "fy", 615.0),
            depth_model=_declare_and_get(self, "depth_model", "Intel/dpt-hybrid-midas"),
            depth_scale=_declare_and_get(self, "depth_scale", 0.001),
            use_monocular_fallback=_declare_and_get(self, "use_monocular_fallback", True),
            ransac_distance_threshold=_declare_and_get(self, "ransac_distance_threshold", 0.015),
            ransac_max_iterations=_declare_and_get(self, "ransac_max_iterations", 1000),
            die_color=_declare_and_get(self, "die_color", "white"),
            num_color_clusters=_declare_and_get(self, "num_color_clusters", 5),
            die_size_m=_declare_and_get(self, "die_size_m", 0.050),
            detection_mode=_declare_and_get(self, "detection_mode", "hsv"),
            hsv_min=_declare_and_get(self, "hsv_min", [0, 0, 150]),
            hsv_max=_declare_and_get(self, "hsv_max", [180, 80, 255]),
            glare_v_thresh=_declare_and_get(self, "glare_v_thresh", 245),
            clahe_clip_limit=_declare_and_get(self, "clahe_clip_limit", 3.0),
            pip_min_circularity=_declare_and_get(self, "pip_min_circularity", 0.45),
            canny_low_thresh=_declare_and_get(self, "canny_low_thresh", 40),
            canny_high_thresh=_declare_and_get(self, "canny_high_thresh", 130),
            min_pips=_declare_and_get(self, "min_pips", 1),
            max_pips=_declare_and_get(self, "max_pips", 6),
            min_die_height_m=_declare_and_get(self, "min_die_height_m", 0.003),
            max_die_height_m=_declare_and_get(self, "max_die_height_m", 0.065),
            output_dir=_declare_and_get(self, "output_dir", "output"),
            rgb_topic=_declare_and_get(self, "rgb_topic", "/camera/color/image_raw"),
            depth_topic=_declare_and_get(self, "depth_topic",
                                         "/camera/aligned_depth_to_color/image_raw"),
            camera_info_topic=_declare_and_get(self, "camera_info_topic",
                                               "/camera/color/camera_info"),
            pose_topic=_declare_and_get(self, "pose_topic", "/dice/pose"),
            debug_panels_topic=_declare_and_get(self, "debug_panels_topic", "/dice/debug_panels"),
            top_down_topic=_declare_and_get(self, "top_down_topic", "/dice/top_down"),
            service_name=_declare_and_get(self, "service_name", "die_identification"),
        )
        self._params = p
        p.log(self.get_logger())

        # Parameter change callback
        self.add_on_set_parameters_callback(self._on_param_change)

        # ── Pipeline ───────────────────────────────────────────────────
        self._pipeline = DieDetectorPipeline(p)
        self._last_result: dict | None = None

        # ── ROS infrastructure ─────────────────────────────────────────
        self._bridge = CvBridge()
        self._tf_broadcaster = TransformBroadcaster(self)

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

        # CameraInfo subscriber (latched once)
        self._info_sub = self.create_subscription(
            CameraInfo, p.camera_info_topic, self._camera_info_cb, 10
        )

        # Synchronised RGB + Depth
        self._rgb_sub = message_filters.Subscriber(self, Image, p.rgb_topic)
        self._depth_sub = message_filters.Subscriber(self, Image, p.depth_topic)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub], queue_size=10, slop=0.05
        )
        self._sync.registerCallback(self._rgbd_cb)

        self.get_logger().info(
            f"Listening — RGB: {p.rgb_topic} | Depth: {p.depth_topic} | "
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

        # Run pipeline
        try:
            result = self._pipeline.process_rgbd(rgb, depth_m)
            self._last_result = result
        except Exception as exc:
            self.get_logger().error(f"Pipeline error: {exc}")
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
        frame_id = self._params.camera_frame_id

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
        rclpy.shutdown()


if __name__ == "__main__":
    main()
