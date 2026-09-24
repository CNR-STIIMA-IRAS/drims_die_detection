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

    # YOLOE + CNN silhouette method (needs torch / ultralytics: activate the venv first)
    source src/drims_die_detection/.venv-docker/bin/activate
    ros2 launch drims_die_detection die_detector.launch.py pose_method:=silhouette plane_source:=tf

    # Static camera / bag replay without TF + CameraInfo: fixed plane and intrinsics
    ros2 launch drims_die_detection die_detector.launch.py plane_source:=fixed \\
        plane_normal_cam:=[-0.0349,-0.5982,-0.8006] plane_height_m:=0.62 fx:=1031 fy:=1031

Every node parameter in NODE_OVERRIDES can be passed as name:=value; arguments
left empty keep the value from the YAML config.

The node runs with the Python of the active virtualenv ($VIRTUAL_ENV), if any,
or with ``python_executable:=/path/to/python3``.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


def _floats(text):
    """'[a, b, c]' or 'a,b,c' -> [a, b, c]"""
    return [float(v) for v in text.strip().strip("[]").split(",") if v.strip()]


def _bool(text):
    return text.strip().lower() in ("1", "true", "yes", "on")


# (launch argument = node parameter, type, description)
NODE_OVERRIDES = [
    ("debug", _bool, "Verbose debug prints"),
    ("visualize", _bool, "Show cv2.imshow windows (requires display)"),
    ("save", _bool, "Save output files to disk"),
    ("rgb_topic", str, "RGB image topic"),
    ("depth_topic", str, "Aligned depth image topic"),
    ("camera_info_topic", str, "Camera intrinsics topic"),
    ("pose_method", str, "silhouette (YOLOE + cube fit + CNN) | yoloe_faces (YOLOE + face/pip polygons) | classical"),
    ("device", str, "YOLOE + CNN device: auto | cuda | cpu"),
    ("plane_source", str, "Table plane: tf | fixed | depth"),
    ("table_frame", str, "tf: frame whose z = table_height_m plane is the table"),
    ("table_height_m", float, "tf: table top height in table_frame [m]"),
    ("plane_normal_cam", _floats, "fixed: table normal (up) in the camera optical frame, e.g. [0.0,-0.6,-0.8]"),
    ("plane_height_m", float, "fixed: camera height above the table plane [m]"),
    ("fx", float, "Focal length [px], used only while no CameraInfo is received"),
    ("fy", float, "Focal length [px], used only while no CameraInfo is received"),
    ("cx", float, "Principal point [px] without CameraInfo (-1 = image centre)"),
    ("cy", float, "Principal point [px] without CameraInfo (-1 = image centre)"),
]


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("drims_die_detection")
    default_config = os.path.join(pkg_share, "config", "die_detector_params.yaml")

    # ── Overrideable launch arguments ──────────────────────────────────
    args = [
        DeclareLaunchArgument("config",
                              default_value=default_config,
                              description="Path to YAML parameter file"),
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
    ]
    # Node parameter overrides: empty (default) = keep the value from the YAML config
    args += [DeclareLaunchArgument(name, default_value="", description=desc)
             for name, _, desc in NODE_OVERRIDES]
    args += [
        DeclareLaunchArgument("python_executable", default_value="",
                              description="Python used to run the node (default: $VIRTUAL_ENV/bin/python3 if set)"),
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

    def make_node(context):
        overrides = {}
        for name, cast, _ in NODE_OVERRIDES:
            val = LaunchConfiguration(name).perform(context)
            if val != "":
                overrides[name] = cast(val)
        python = LaunchConfiguration("python_executable").perform(context)
        if not python and os.environ.get("VIRTUAL_ENV"):
            python = os.path.join(os.environ["VIRTUAL_ENV"], "bin", "python3")
        return [Node(
            package="drims_die_detection",
            executable="die_detector_node.py",
            name="die_detector_node",
            output="screen",
            prefix=python or None,
            parameters=[LaunchConfiguration("config"), overrides],
        )]

    return LaunchDescription(args + [mock_node, OpaqueFunction(function=make_node), rviz_node])
