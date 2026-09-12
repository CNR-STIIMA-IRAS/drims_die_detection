# drims_die_detection

> **3D die detection and 6-DOF pose estimation for angled RGBD cameras.**  
> Works standalone (ROS-free) and as a ROS 2 Humble package.

---

## Overview

This package processes RGB-D images of a scene containing a die resting on a flat
surface, captured by an angled (non-vertical) camera.

**Pipeline steps:**

| Step | Module | Description |
|------|--------|-------------|
| 1 | `depth_estimator` | Infer metric depth via HuggingFace DPT model (requires `torch` & `transformers`) |
| 2 | `point_cloud_processor` | Back-project RGB+Depth → 3D point cloud |
| 3 | `point_cloud_processor` | RANSAC plane fitting to identify the table surface |
| 4 | `rgb_die_detector` | 5-step 2D colour-cluster + polygon detection |
| 5 | `alignment_projection` | Rotate cloud so plane normal → Z-axis |
| 6 | `alignment_projection` | Extract ortho-rectified top-down crop |
| 7 | `pip_detector` | Adaptive threshold + blob analysis → pip count |
| 8 | `pose_estimator` | 6-DOF pose (centroid + quaternion) via ray backprojection |

---

## Repository Structure

```
drims_die_detection/
├── drims_die_detection/         ← Python package (importable without ROS)
│   ├── __init__.py
│   ├── die_detector_params.py   ← DieDetectorParams dataclass (YAML-loadable)
│   ├── depth_estimator.py
│   ├── point_cloud_processor.py
│   ├── rgb_die_detector.py
│   ├── alignment_projection.py
│   ├── pip_detector.py
│   ├── pose_estimator.py
│   ├── plotly_visualizer.py
│   └── die_detector_pipeline.py ← Full pipeline orchestrator
│
├── scripts/
│   ├── run_die_detector.py      ← ROS-free CLI
│   └── die_detector_node.py     ← ROS 2 Humble node
│
├── config/
│   └── die_detector_params.yaml ← All tunable parameters
│
├── launch/
│   └── die_detector.launch.py
│
├── rviz/
│   └── die_detector.rviz        ← RViz2 config with 3 image panels + TF axes
│
├── images/
│   ├── rgb/                     ← Test RGB images
│   └── depth/                   ← Depth colourmap PNGs (for visual reference)
│
├── output/                      ← Generated outputs (gitignored)
│
├── tests/
│   ├── test_params.py
│   ├── test_point_cloud.py
│   ├── test_rgb_detector.py
│   ├── test_alignment.py
│   ├── test_pip_detector.py
│   ├── test_pose_estimator.py
│   ├── test_pipeline_rgbd.py    ← Integration test with real images
│   └── test_ros2_node.py
│
├── package.xml
├── setup.py
├── setup.cfg
└── CMakeLists.txt
```

---

## Installation

### Standalone (ROS-free) — Automated Setup

You can set up the virtual environment and install all dependencies automatically with a single command:

```bash
# Automated setup (creates .venv and installs requirements.txt)
./setup_venv.sh

# Activate virtual environment
source .venv/bin/activate
```

Alternatively, install manually using `requirements.txt`:

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install all dependencies from requirements.txt
pip install -r requirements.txt
```

### ROS 2 Humble

```bash
# 1. Source ROS 2 Humble
source /opt/ros/humble/setup.bash

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Build the package
colcon build --packages-select drims_die_detection --symlink-install
source install/setup.bash
```

---

## Usage

### Standalone CLI

```bash
# Process all images in images/rgb/ with default parameters:
python scripts/run_die_detector.py

# Use the YAML config file:
python scripts/run_die_detector.py --config config/die_detector_params.yaml

# Process a single image with debug output:
python scripts/run_die_detector.py --image images/rgb/sample.jpg --debug

# Process a single image with a pre-saved depth map:
python scripts/run_die_detector.py \
    --image  images/rgb/sample.jpg \
    --depth  output/depth_sample.npy

# Show intermediate steps interactively:
python scripts/run_die_detector.py --visualize
```

### Interactive Parameter Tuning GUI

To visually tune detection parameters (HSV/LAB color bounds, specular glare threshold, CLAHE clip limit, and pip circularity) under real lighting conditions:

```bash
# Launch interactive trackbar GUI:
python scripts/tune_die_detector_gui.py --image test_images/rgb/5834717044920750589.jpg
```
- **Live Preview:** View real-time color segmentation and pip detection output.
- **Save Config:** Press `'s'` or `'S'` in the GUI window to automatically save the tuned parameters directly to `config/die_detector_params.yaml`.

Output files (written to `output/` by default):
- `detected_<image>.jpg` — 6-panel debug collage
- `depth_<stem>.npy` — raw metric depth map (float32)
- `depth_<stem>.png` — depth colourmap (Viridis PNG)
- `plotly_3d_<stem>.html` — interactive 3D point cloud

### Python API

```python
from drims_die_detection import DieDetectorParams, DieDetectorPipeline
import cv2, numpy as np

# Load parameters from YAML
params = DieDetectorParams.from_yaml("config/die_detector_params.yaml")
params.debug = True

pipeline = DieDetectorPipeline(params)

# With RGB only (monocular depth estimation):
rgb = cv2.imread("images/rgb/sample.jpg")
result = pipeline.process_rgb(rgb, image_name="sample.jpg")

# With RGB + metric depth (float32, metres):
depth_m = np.load("output/depth_sample.npy")
result = pipeline.process_rgbd(rgb, depth_m, image_name="sample.jpg")

print(f"Pip count : {result['pip_count']}")
print(f"Faces     : {result['num_visible_faces']}")
print(f"Centroid  : {result['centroid']}")
print(f"Quaternion: {result['quaternion']}")

# Build and display the 6-panel collage
collage = pipeline.build_debug_panels(rgb, result)
cv2.imshow("Die Detection", collage)
cv2.waitKey(0)
```

### ROS 2 Node

#### Launch

```bash
# Default topics (Intel RealSense layout):
ros2 launch drims_die_detection die_detector.launch.py

# Custom topics:
ros2 launch drims_die_detection die_detector.launch.py \
    rgb_topic:=/my_cam/rgb/image_raw \
    depth_topic:=/my_cam/depth/image_rect_raw \
    camera_info_topic:=/my_cam/rgb/camera_info

# Enable debug prints:
ros2 launch drims_die_detection die_detector.launch.py debug:=true
```

#### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/color/image_raw` | `sensor_msgs/Image` | RGB image |
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | Aligned depth (uint16 mm) |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | Camera intrinsics |

All topic names are configurable via YAML or launch arguments.

#### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/dice/pose` | `geometry_msgs/PoseStamped` | 6-DOF top-face pose |
| `/dice/debug_panels` | `sensor_msgs/Image` | 6-panel processing collage |
| `/dice/top_down` | `sensor_msgs/Image` | Pip-annotated top-down crop |

#### TF2 Transforms

| Child frame | Description |
|-------------|-------------|
| `die_top_face` | Pose of the top face (origin at face centre) |
| `die_centroid` | Pose at the geometric centroid of the die body |

#### RViz2

```bash
rviz2 -d $(ros2 pkg prefix drims_die_detection)/share/drims_die_detection/rviz/die_detector.rviz
```

The pre-configured layout shows:
- **Camera RGB** — live feed from `/camera/color/image_raw`
- **Debug Panels (6-panel)** — processing steps from `/dice/debug_panels`
- **Top-Down Crop** — pip view from `/dice/top_down`
- **TF axes** — `die_top_face` and `die_centroid` frames in 3D view

---

## Parameters

All parameters live in [`config/die_detector_params.yaml`](config/die_detector_params.yaml).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `debug` | bool | `false` | Verbose step-by-step prints |
| `visualize` | bool | `false` | cv2.imshow intermediate images |
| `save` | bool | `true` | Write output files |
| `fx`, `fy` | float | `615.0` | Camera focal lengths (px) |
| `depth_model` | str | `Intel/dpt-hybrid-midas` | HuggingFace DPT model |
| `depth_scale` | float | `0.001` | uint16 depth → metres |
| `use_monocular_fallback` | bool | `true` | Fall back to DPT if depth invalid |
| `ransac_distance_threshold` | float | `0.015` | Plane inlier distance (m) |
| `die_color` | str | `white` | Target die colour |
| `num_color_clusters` | int | `5` | K for K-Means colour clustering |
| `die_size_m` | float | `0.050` | Die physical size (m) |
| `min_die_height_m` | float | `0.003` | Min height above table (m) |
| `max_die_height_m` | float | `0.065` | Max height above table (m) |
| `rgb_topic` | str | `/camera/color/image_raw` | RGB input topic |
| `depth_topic` | str | `/camera/aligned_depth_to_color/image_raw` | Depth input topic |
| `camera_info_topic` | str | `/camera/color/camera_info` | CameraInfo topic |

---

## Testing

### Standalone (no ROS required)

```bash
python -m pytest tests/ -v
```

### With ROS 2 (colcon)

```bash
colcon test --packages-select drims_die_detection
colcon test-result --verbose
```

### Individual test modules

```bash
python -m pytest tests/test_params.py          -v   # DieDetectorParams
python -m pytest tests/test_point_cloud.py     -v   # PointCloudProcessor
python -m pytest tests/test_rgb_detector.py    -v   # RGBDieDetector
python -m pytest tests/test_alignment.py       -v   # AlignmentAndProjection
python -m pytest tests/test_pip_detector.py    -v   # PipDetector
python -m pytest tests/test_pose_estimator.py  -v   # PoseEstimator
python -m pytest tests/test_pipeline_rgbd.py   -v   # Full pipeline (real images)
python -m pytest tests/test_ros2_node.py       -v   # ROS2 node smoke tests
```

---

## Depth Data

The pipeline stores depth information in two formats:
- **`output/depth_<stem>.npy`** — raw `float32` depth map in metres (usable as pipeline input)
- **`output/depth_<stem>.png`** — Viridis colourmap PNG for visual inspection

The `images/depth/` directory contains pre-computed colourmap PNGs for visual reference only.
To use a previously saved depth map with the CLI:
```bash
python scripts/run_die_detector.py \
    --image images/rgb/5834717044920750589.jpg \
    --depth output/depth_5834717044920750589.npy
```

---

## License

MIT
