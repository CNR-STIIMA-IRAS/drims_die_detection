"""
DieDetectorParams
=================
Central parameter container for the Die Detector 3D pipeline.

Can be instantiated directly, loaded from a YAML file, or built from a dict.
All pipeline classes accept a DieDetectorParams instance so that every tunable
knob lives in one place and can be versioned / launched via ROS 2 parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Absolute path to the package root directory (drims_die_detection)
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class DieDetectorParams:
    """All configurable parameters for the die detection pipeline.

    Flags
    -----
    debug : bool
        If True, verbose print statements are emitted throughout the pipeline.
    visualize : bool
        If True, intermediate processing images are shown via cv2.imshow().
    save : bool
        If True, diagnostic images and 3D HTML plots are written to *output_dir*.

    Camera intrinsics
    -----------------
    fx, fy : float
        Focal lengths in pixels (overridden by CameraInfo in ROS 2 mode).
    cx, cy : float | None
        Principal point; defaults to image centre when None.

    Depth
    -----
    depth_model : str
        HuggingFace model name for monocular depth estimation.
    depth_scale : float
        Multiplier applied to uint16 depth values to convert to metres (1/1000
        for RealSense millimetre encoding).
    use_monocular_fallback : bool
        If True and no depth stream is available (or depth is all-zero), the
        monocular estimator is invoked automatically.

    RANSAC
    ------
    ransac_distance_threshold : float
        Inlier distance threshold in metres for plane fitting.
    ransac_max_iterations : int
        Maximum RANSAC iterations.

    Detection
    ---------
    die_color : str
        Target die body colour ('white', 'black', 'red', …).
    num_color_clusters : int
        K for K-Means colour quantisation (Step 1 of RGBDieDetector).
    die_size_m : float
        Physical die side length in metres (used for pose ray backprojection).

    Clustering
    ----------
    min_die_height_m : float
        Minimum perpendicular height above the table plane to include a point
        in the die blob (metres).
    max_die_height_m : float
        Maximum perpendicular height above the table plane.

    Output
    ------
    output_dir : str
        Directory for saving diagnostic files.
    """

    # ── Flags ──────────────────────────────────────────────────────────────
    debug: bool = False
    visualize: bool = False
    save: bool = True

    # ── Camera intrinsics ──────────────────────────────────────────────────
    fx: float = 500.0
    fy: float = 500.0
    cx: Optional[float] = None
    cy: Optional[float] = None

    # ── Depth ──────────────────────────────────────────────────────────────
    depth_model: str = "Intel/dpt-hybrid-midas"
    depth_scale: float = 0.001          # mm → m for uint16 RealSense streams
    use_monocular_fallback: bool = True

    # ── RANSAC ─────────────────────────────────────────────────────────────
    ransac_distance_threshold: float = 0.015
    ransac_max_iterations: int = 1000

    # ── Detection ──────────────────────────────────────────────────────────
    die_color: str = "white"
    num_color_clusters: int = 5
    die_size_m: float = 0.050           # 5 cm standard die
    detection_mode: str = "hsv"         # "hsv", "kmeans"
    hsv_min: list[int] = field(default_factory=lambda: [0, 0, 150])
    hsv_max: list[int] = field(default_factory=lambda: [180, 80, 255])
    glare_v_thresh: int = 245
    clahe_clip_limit: float = 3.0
    pip_min_circularity: float = 0.45
    canny_low_thresh: int = 40
    canny_high_thresh: int = 130
    min_pips: int = 1
    max_pips: int = 6

    # ── Clustering / height filter ─────────────────────────────────────────
    min_die_height_m: float = 0.003     # 3 mm above table
    max_die_height_m: float = 0.065     # 6.5 cm above table

    # ── Output ─────────────────────────────────────────────────────────────
    output_dir: str = "output"

    @property
    def resolved_output_dir(self) -> str:
        """Return output_dir as an absolute path relative to the package root if relative."""
        if os.path.isabs(self.output_dir):
            return self.output_dir
        return os.path.abspath(os.path.join(PACKAGE_ROOT, self.output_dir))

    # ── ROS topics & services (used only by die_detector_node) ─────────────
    rgb_topic: str = "/camera/color/image_raw"
    depth_topic: str = "/camera/aligned_depth_to_color/image_raw"
    camera_info_topic: str = "/camera/color/camera_info"
    pose_topic: str = "/dice/pose"
    debug_panels_topic: str = "/dice/debug_panels"
    top_down_topic: str = "/dice/top_down"
    service_name: str = "die_identification_3d"

    # ── Internal (not in YAML) ─────────────────────────────────────────────
    camera_frame_id: str = "camera_color_optical_frame"

    # ──────────────────────────────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: str) -> "DieDetectorParams":
        """Load parameters from a YAML file.

        Only recognised field names are applied; unknown keys are silently
        ignored so that partial override files work correctly.
        """
        if not HAS_YAML:
            raise ImportError("PyYAML is required to load parameters from YAML. "
                              "Install it with: pip install pyyaml")
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}

        # Support both flat dict and ROS 2 parameter-server style
        # (e.g. {die_detector_node: {ros__parameters: {...}}})
        if "die_detector_node" in raw and "ros__parameters" in raw["die_detector_node"]:
            raw = raw["die_detector_node"]["ros__parameters"]
        elif "/**" in raw and "ros__parameters" in raw["/**"]:
            raw = raw["/**"]["ros__parameters"]

        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: dict) -> "DieDetectorParams":
        """Build a DieDetectorParams from a plain dict (ignores unknown keys)."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    def log(self, logger=None) -> None:
        """Print all parameters (to *logger* if provided, otherwise stdout)."""
        lines = ["DieDetectorParams:"]
        for k, v in self.__dict__.items():
            lines.append(f"  {k}: {v}")
        msg = "\n".join(lines)
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)
