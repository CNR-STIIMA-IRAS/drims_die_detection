"""
drims_die_detection
===================
ROS-free 3D die detection library for angled RGBD cameras.

Classes
-------
DieDetectorParams       — Central parameter container (load from YAML or dict)
MonocularDepthEstimator — HuggingFace DPT depth inference with heuristic fallback
PointCloudProcessor     — 3D point cloud generation and RANSAC plane fitting
RGBDieDetector          — 2D color-clustering + pip detection
AlignmentAndProjection  — Plane-normal-to-Z rotation and top-down crop
PipDetector             — Adaptive threshold / blob pip counting
PoseEstimator           — 6-DOF pose (centroid + quaternion)
PlotlyVisualizer        — Interactive 3D HTML plots
DieDetectorPipeline     — Full orchestration pipeline
"""

from .die_detector_params import DieDetectorParams
from .depth_estimator import MonocularDepthEstimator
from .point_cloud_processor import PointCloudProcessor
from .rgb_die_detector import RGBDieDetector
from .alignment_projection import AlignmentAndProjection
from .pip_detector import PipDetector
from .pose_estimator import PoseEstimator
from .pose_stabilizer import PoseStabilizer
from .plotly_visualizer import PlotlyVisualizer
from .die_detector_pipeline import DieDetectorPipeline
from .die_orientation import resolve_die_orientation

__all__ = [
    "DieDetectorParams",
    "MonocularDepthEstimator",
    "PointCloudProcessor",
    "RGBDieDetector",
    "AlignmentAndProjection",
    "PipDetector",
    "PoseEstimator",
    "PoseStabilizer",
    "PlotlyVisualizer",
    "DieDetectorPipeline",
    "resolve_die_orientation",
]
