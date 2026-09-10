#!/usr/bin/env python3
"""
ROS 2 / RealSense Node for Angled Camera Die Detection & 6DOF Pose Estimation

Subscribes to RealSense aligned RGB and Depth topics, computes surface plane RANSAC,
segments the die, extracts an orthographic top-down face crop, counts pips, and broadcasts
a 6DOF TF transform (`die_frame`) relative to `camera_color_optical_frame`.

Required ROS 2 packages:
- rclpy
- sensor_msgs
- geometry_msgs
- tf2_ros
- cv_bridge
"""

import math
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R_sci

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo
    from geometry_msgs.msg import TransformStamped, PoseStamped
    from tf2_ros import TransformBroadcaster
    from cv_bridge import CvBridge
    import message_filters
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False


# Import pipeline classes from detect_die
from detect_die import (
    PointCloudProcessor,
    AlignmentAndProjection,
    PipDetector,
    PoseEstimator
)


class DiceDetectorROS2Node:
    """ROS 2 Node for real-time die detection and 6DOF pose publishing."""

    def __init__(self):
        if not HAS_ROS2:
            raise RuntimeError("ROS 2 libraries (rclpy, cv_bridge, tf2_ros) are not installed in current environment.")

        super().__init__('dice_detector_node')
        self.get_logger().info("Initializing Dice Detector ROS 2 Node...")

        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)

        # Camera Intrinsics (updated dynamically via CameraInfo)
        self.fx = 525.0
        self.fy = 525.0
        self.cx = 320.0
        self.cy = 240.0
        self.intrinsics_received = False

        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/dice/pose', 10)
        self.debug_img_pub = self.create_publisher(Image, '/dice/debug_image', 10)

        # Subscribers (Synchronized RGB + Aligned Depth)
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.05
        )
        self.ts.registerCallback(self.rgbd_callback)

        self.get_logger().info("Dice Detector ROS 2 Node initialized and waiting for RGB-D topics...")

    def camera_info_callback(self, msg: CameraInfo):
        """Extracts camera intrinsic matrix K [fx, 0, cx; 0, fy, cy; 0, 0, 1]."""
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.intrinsics_received = True

    def rgbd_callback(self, rgb_msg: Image, depth_msg: Image):
        """Callback processing synchronized RGB and Depth frames from RealSense."""
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            # Convert depth from millimeters to meters if uint16
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) / 1000.0
            else:
                depth = depth.astype(np.float32)

        except Exception as e:
            self.get_logger().error(f"Failed to convert ROS Image messages: {e}")
            return

        processor = PointCloudProcessor(fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy)

        # 1. Point Cloud Generation
        points, colors, pixel_coords, _ = processor.create_point_cloud(rgb, depth)
        if len(points) < 100:
            self.get_logger().warn("Point cloud has too few valid depth points.")
            return

        # 2. RANSAC Surface Plane Fitting
        plane_model, inliers, outliers, normal = processor.fit_plane_ransac(points)

        # 3. Die Cluster Segmentation
        outlier_points = points[outliers]
        die_points, die_indices = processor.cluster_die(outlier_points, plane_model)

        if die_points is None or len(die_points) < 15:
            self.get_logger().warn("No die cluster detected above surface plane.")
            return

        outlier_indices_arr = np.array(outliers)
        orig_die_indices = outlier_indices_arr[die_indices]

        # 4. 3D Rotation Alignment to Z-axis
        R_plane = AlignmentAndProjection.get_rotation_to_z(normal)

        # 5. Orthographic Top-Down Crop
        top_down_crop, raw_crop, (x1, y1, x2, y2), yaw_angle = AlignmentAndProjection.extract_top_down_rgb(
            rgb, die_points, pixel_coords, orig_die_indices, R_plane, crop_size=300
        )

        # 6. Pip Detection
        pip_count, annotated_top_down, _ = PipDetector.detect_pips(top_down_crop)

        # 7. Compute 6DOF Pose
        centroid, quat, _, _ = PoseEstimator.compute_pose(die_points, normal, R_plane)

        stamp = rgb_msg.header.stamp
        frame_id = rgb_msg.header.frame_id if rgb_msg.header.frame_id else "camera_color_optical_frame"

        # 8. Broadcast TF2 Transform
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = frame_id
        t.child_frame_id = 'die_frame'

        t.transform.translation.x = float(centroid[0])
        t.transform.translation.y = float(centroid[1])
        t.transform.translation.z = float(centroid[2])

        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])

        self.tf_broadcaster.sendTransform(t)

        # Publish PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = frame_id
        pose_msg.pose.position.x = float(centroid[0])
        pose_msg.pose.position.y = float(centroid[1])
        pose_msg.pose.position.z = float(centroid[2])
        pose_msg.pose.orientation.x = float(quat[0])
        pose_msg.pose.orientation.y = float(quat[1])
        pose_msg.pose.orientation.z = float(quat[2])
        pose_msg.pose.orientation.w = float(quat[3])
        self.pose_pub.publish(pose_msg)

        # Publish Debug Image
        annotated = rgb.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"Die Pips: {pip_count}", (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        debug_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        debug_msg.header.stamp = stamp
        debug_msg.header.frame_id = frame_id
        self.debug_img_pub.publish(debug_msg)

        self.get_logger().info(f"Die Detected! Pips: {pip_count} | Centroid: ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})m | TF published.")


def main(args=None):
    if not HAS_ROS2:
        print("ROS 2 is not available in current python environment.")
        return

    rclpy.init(args=args)
    node = DiceDetectorROS2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
