#!/usr/bin/env python3
"""
ROS 2 Humble launch file for drims_die_detection.

Loads all parameters from config/die_detector_params.yaml and starts the
die_detector_node.

Usage
-----
::

    ros2 launch drims_die_detection die_detector.launch.py
    ros2 launch drims_die_detection die_detector.launch.py debug:=true
    ros2 launch drims_die_detection die_detector.launch.py \\
        rgb_topic:=/my_camera/rgb/image_raw \\
        depth_topic:=/my_camera/depth/image_rect_raw \\
        camera_info_topic:=/my_camera/rgb/camera_info
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("drims_die_detection")
    default_config = os.path.join(pkg_share, "config", "die_detector_params.yaml")

    # ── Overrideable launch arguments ──────────────────────────────────
    args = [
        DeclareLaunchArgument("config",
                              default_value=default_config,
                              description="Path to YAML parameter file"),
        DeclareLaunchArgument("debug",
                              default_value="false",
                              description="Enable verbose debug prints"),
        DeclareLaunchArgument("visualize",
                              default_value="false",
                              description="Show cv2.imshow windows (requires display)"),
        DeclareLaunchArgument("save",
                              default_value="false",
                              description="Save output files to disk"),
        DeclareLaunchArgument("rviz",
                              default_value="false",
                              description="Launch RViz2 visualizer with die detection profile"),
        DeclareLaunchArgument("launch_mock_publisher",
                              default_value="false",
                              description="Launch RGB-D mock publisher for testing"),
        DeclareLaunchArgument("image_index",
                              default_value="0",
                              description="Select test image by 0-based index for mock publisher"),
        DeclareLaunchArgument("image_name",
                              default_value="",
                              description="Select test image by filename for mock publisher"),
        # Standard RealSense camera defaults (commented): /camera/color/image_raw, /camera/aligned_depth_to_color/image_raw, /camera/color/camera_info
        DeclareLaunchArgument("rgb_topic",
                              default_value="/camera/color/image_raw",
                              description="RGB image topic"),
        DeclareLaunchArgument("depth_topic",
                              default_value="/camera/aligned_depth_to_color/image_raw",
                              description="Aligned depth image topic"),
        DeclareLaunchArgument("camera_info_topic",
                              default_value="/camera/color/camera_info",
                              description="Camera intrinsics topic"),
    ]

    mock_node = Node(
        package="drims_die_detection",
        executable="rgbd_mock_publisher.py",
        name="rgbd_mock_publisher",
        output="screen",
        condition=IfCondition(LaunchConfiguration("launch_mock_publisher")),
        parameters=[
            LaunchConfiguration("config"),
            {
                "image_index": LaunchConfiguration("image_index"),
                "image_name": LaunchConfiguration("image_name"),
            },
        ],
    )

    rviz_config_path = os.path.join(pkg_share, "rviz", "die_detector.rviz")
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_path],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    node = Node(
        package="drims_die_detection",
        executable="die_detector_node.py",
        name="die_detector_node",
        output="screen",
        parameters=[
            LaunchConfiguration("config"),
        ],
    )

    return LaunchDescription(args + [mock_node, node, rviz_node])
