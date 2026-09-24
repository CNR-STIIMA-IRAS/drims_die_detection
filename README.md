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

## Recommended: YOLOE + CNN silhouette pipeline

`SilhouettePipeline` (`pose_method: silhouette`) replaces the colour/pip steps
above with learned detection and a model fit:

| Step | Module | Description |
|------|--------|-------------|
| 1 | `silhouette_pipeline` | YOLOE instance segmentation → up to 3 die candidates (HSV fallback if none fits) |
| 2 | `silhouette_pose.die_silhouette` | Convex die outline (fills pip notches and shaded faces the mask misses) |
| 3 | `silhouette_pose.SilhouettePoseFitter` | Fit a cube of known size on the table plane: (x, y, yaw) maximising outline IoU; best candidate wins, IoU < `min_fit_iou` is rejected |
| 4 | `cnn_die_classifier` | MobileNetV3 top / front face classification on the die crop |

No face or pip segmentation and no depth on the die are needed: only the table
plane (from TF + table height, a fixed calibration, or depth RANSAC) and the
camera intrinsics. Die frame convention: **z** = table normal (up), **x** =
outward normal of the *front face* (the side face turned most towards the
camera), **y** = z × x.

### The three pose methods (`pose_method`)

| `pose_method` | Detection | Top / front face | Pose | Table plane |
|---|---|---|---|---|
| `silhouette` (recommended) | YOLOE (+ HSV fallback) | CNN | cube fit to the die outline | `plane_source` |
| `yoloe_faces` | YOLOE (+ HSV fallback) | pip count per face polygon | top-face polygon edges (`PoseEstimator`) | `plane_source` |
| `classical` | HSV / k-means | pip count per face polygon | top-face polygon edges | RANSAC on aligned depth |

`silhouette` and `yoloe_faces` share the plane sources, the axis convention
and the quality check (silhouette IoU of the resulting cube against the die
outline; `min_fit_iou` / `min_face_pose_iou`), so they are interchangeable.

### Why the CNN pipeline — comparison on the real bag

60 frames of `test_bags/rosbag2_2026_09_22-15_05_03` (TIAGo head camera,
1280×720, die moved by hand into 13 resting positions); 44 frames labelled by
eye, 16 with a hand on the die. RTX 5060 laptop GPU, same table plane and
`config/die_detector_params.yaml` for every run. "Classical" here is the
classical detection + face/pip + pose steps on the same table plane (the
black table gives no depth for its RANSAC).

| | **silhouette (GPU)** | silhouette (CPU) | yoloe_faces (GPU) | classical (HSV) |
|---|---|---|---|---|
| Time per frame, median / p95 | **38.7 / 80.9 ms** | 170.4 / 230.5 ms | 35.3 / 42.0 ms | 18.5 / 23.5 ms |
| Frames per second | **25.8** | 5.9 | 28.3 | 54.1 |
| Top-face accuracy (44 frames) | **100%** | 100% | 90.9% (3 no answer, 1 wrong) | 81.8% (8 no answer, 0 wrong) |
| Clean frames rejected by the quality check | **0** | 0 | 1 (die cut off by the image border) | 7 (wrong object) |
| Occluded frames rejected | 5 of 16 | 5 of 16 | 7 of 16 | 14 of 16 |
| Yaw spread while the die is still, median / max | **0.7° / 3.0°** | 0.7° / 3.0° | 1.7° / 5.5° | 1.9° / 5.2° |
| Position spread while still, median | 0.3 mm | 0.3 mm | 0.2 mm | 0.3 mm |
| Silhouette IoU of the pose (clean frames, mean) | **0.940** | 0.940 | 0.857 | 0.790 |

- **GPU vs CPU** give identical results (same top face on every frame, yaw
  within 0.4°); only YOLOE is slower on the CPU (≈150 ms vs ≈20 ms).
- **YOLOE fixes detection**: without it, HSV picks other bright objects on 7
  of the 44 frames (caught by the quality check, which is why `classical`
  rejects them and answers fewer frames). With YOLOE the face-polygon pose agrees with the
  cube fit to ≈1 mm / ≈3° (median), max 12 mm / 9°.
- **Pip counting** works once each pip is counted once (a double count of
  top-face pips, fixed in `rgb_die_detector.py`, made it read 2×, e.g. 6 → 12
  → rejected). It still misses where a face is marked untrusted or the die is
  cut off; the CNN answers on every frame.
- **Caveats:** one bag, labels by eye, and the thresholds were tuned on this
  same bag — confirm on a second recording. Front-face accuracy is not
  measured (no front-face labels). The table plane was self-calibrated from
  the die silhouettes, so absolute positions are approximate; yaw and
  repeatability are not affected.

Reproduce (writes annotated images and `results.json` to `output/bag_pose_<mode>_<device>/`):

```bash
.venv-docker/bin/python scripts/extract_bag_frames.py --num_frames 60   # + output/bag_test_frames/labels.json
.venv-docker/bin/python scripts/run_bag_pose.py --device cuda               # silhouette
.venv-docker/bin/python scripts/run_bag_pose.py --device cpu
.venv-docker/bin/python scripts/run_bag_pose.py --mode yoloe_faces --device cuda
.venv-docker/bin/python scripts/run_bag_pose.py --mode classical
.venv-docker/bin/python scripts/compare_bag_runs.py
```

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

### Docker container — CNN training & YOLOE

Inside the ROS 2 Humble container, use the dedicated `.venv-docker` (Python 3.10, CUDA 12.8 PyTorch, ultralytics, wandb). It sees the system/ROS packages and lives on the mounted workspace, so it survives container restarts:

```bash
./setup_venv_docker.sh                    # one-time
./download_weights.sh                     # one-time: model weights -> weights/ (not in git)

# Either activate it...
source .venv-docker/bin/activate
python3 scripts/train_die_classifier.py --device cuda --wandb

# ...or use the wrapper, which needs no activation
scripts/train_die_classifier.sh --device cuda --wandb
```

Weights are not tracked by git: `./download_weights.sh` fetches the die CNN (GitHub release of this repo, or Google Drive via `DIE_CNN_URL=gdrive:<file-id>`) and the YOLOE / MobileCLIP checkpoints (Ultralytics releases) into `weights/`, verifying each by sha256. W&B runs are logged to `wandb/` (also gitignored).

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
# Launch GUI on offline saved images:
python scripts/tune_die_detector_gui.py

# Launch GUI on LIVE ROS 2 camera stream:
python scripts/tune_die_detector_gui.py --live

# Launch GUI on custom ROS 2 topic:
python scripts/tune_die_detector_gui.py --live --topic /head_front_camera/color/image_raw
```
- **Live Preview:** View real-time color segmentation and pip detection output (on static saved images or live ROS camera stream).
- **Save Config:** Press `'s'` or `'S'` (or click **Save to YAML**) to save parameters directly to `config/die_detector_params.yaml`.
- **Sync to ROS Node:** Click **Sync Live to ROS 2 Node** to instantly push slider values to a running `/die_detector_node` via `ros2 param set`.


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

#### 1. One-time setup

The YOLOE / CNN pipelines need torch + ultralytics from the container venv
(`./setup_venv_docker.sh`, see Installation). When a venv is active, the launch
file runs the node with that venv's Python automatically.

```bash
source /opt/ros/humble/setup.bash
cd /home/ws && colcon build --symlink-install --packages-select drims_die_detection
source install/setup.bash
source src/drims_die_detection/.venv-docker/bin/activate    # in every new terminal
```

#### 2. Choose a pipeline (`pose_method`)

| `pose_method` | What it does | Top / front face from | Needs | Bag results* |
|---|---|---|---|---|
| **`silhouette`** (default, recommended) | YOLOE segments the die → a 4.1 cm cube is fitted to the die outline on the table plane (position + yaw) → the CNN classifies the crop | CNN (MobileNetV3) | RGB, CameraInfo, a table plane (`plane_source`), GPU recommended | 100% top face, ≈26 fps (GPU) / 6 fps (CPU) |
| `yoloe_faces` | YOLOE segments the die → the original face / pip segmentation → pose from the top-face polygon (`PoseEstimator`) | pip count per face polygon | same as `silhouette` | 90.9% top face, ≈28 fps (GPU) |
| `classical` | the original pipeline: HSV/k-means detection → face / pip segmentation → RANSAC table plane on depth → top-face-polygon pose | pip count per face polygon | RGB + **aligned** depth that sees the table, CameraInfo; no GPU | 81.8% top face** |

\* Real bag, 44 labelled frames — see [the comparison](#why-the-cnn-pipeline--comparison-on-the-real-bag).
\** Measured on the same table plane as the others; in the node it needs a
table that returns depth (the black lab table does not).

`silhouette` and `yoloe_faces` publish the same outputs, use the same die frame
(**z** up, **x** out of the front face, **y** = z × x), and reject a frame when
the cube implied by the pose does not match the die outline (silhouette IoU
below `min_fit_iou` = 0.85 / `min_face_pose_iou` = 0.60): typically a hand on
the die, a die cut off by the image border, or a wrong object.

#### 3. Choose where the table plane comes from (`plane_source`, not for `classical`)

The die's distance is fixed by intersecting its image with the table plane, so
the plane must be right (1 cm of height error ≈ 1.3 cm of position error).

| `plane_source` | Needs | Use for |
|---|---|---|
| **`tf`** (default) | TF `table_frame` → camera frame, `table_height_m` = table top height in `table_frame` | the robot |
| `fixed` | `plane_normal_cam` (table normal, pointing up, in the camera optical frame) + `plane_height_m` (camera height above the table) | a static camera, a bag without TF |
| `depth` | an aligned depth topic that sees the table | tables that return depth |

With `tf` / `fixed` only the RGB image (+ CameraInfo) is subscribed. For the
TIAGo cell, `table_height_m = table_pos_z + leg_height + table_thickness` from
the `table_config_file` passed to `tiago_pro_start.launch.py`
(`tiago_utils_config.yaml`: 0 + 0.50 + 0.05 = **0.55 m**;
`tiago_utils_custom_table_config.yaml`: 0 + 0.71 + 0.02 = **0.73 m** above
`base_footprint`). Measure the real table once — the config describes the model.

#### 4. Launch

```bash
# Robot — recommended (everything below is also the YAML default):
ros2 launch drims_die_detection die_detector.launch.py \
    pose_method:=silhouette plane_source:=tf table_frame:=base_footprint table_height_m:=0.55

# Robot — YOLOE + original face / pip segmentation and top-face pose,
# cell started with table_config_file:=tiago_utils_custom_table_config (table top 0.73 m):
ros2 launch drims_die_detection die_detector.launch.py pose_method:=yoloe_faces \
    plane_source:=tf table_frame:=base_footprint table_height_m:=0.73 debug:=false

# Original classical pipeline (needs aligned depth of the table):
ros2 launch drims_die_detection die_detector.launch.py pose_method:=classical \
    depth_topic:=/head_front_camera/aligned_depth_to_color/image_raw

# Without a GPU:
ros2 launch drims_die_detection die_detector.launch.py device:=cpu

# Replay of test_bags/rosbag2_2026_09_22-15_05_03 (recorded without TF / CameraInfo):
# plane and focal length self-calibrated from the bag (output/bag_test_frames/camera_calib.json)
ros2 launch drims_die_detection die_detector.launch.py pose_method:=silhouette \
    plane_source:=fixed plane_normal_cam:=[-0.0349,-0.5982,-0.8006] plane_height_m:=0.62 \
    fx:=1031 fy:=1031 debug:=false
ros2 bag play src/drims_die_detection/test_bags/rosbag2_2026_09_22-15_05_03   # second terminal
```

Query the last detected die (used by the homework):

```bash
ros2 service call /die_identification_3d drims_homework_interfaces/srv/DieIdentification3D "{}"
```

`success: false` with empty fields means no valid pose has been produced yet —
check the node log (below).

#### 5. Launch arguments

Each argument overrides the YAML (`config/die_detector_params.yaml`) only when
given; everything else (thresholds, `cnn_model_path`, `die_size_m`, ...) is set
in the YAML.

| Argument | Meaning |
|---|---|
| `pose_method` | `silhouette` \| `yoloe_faces` \| `classical` |
| `device` | `auto` \| `cuda` \| `cpu` (YOLOE + CNN) |
| `plane_source` | `tf` \| `fixed` \| `depth` |
| `table_frame`, `table_height_m` | table plane for `tf` |
| `plane_normal_cam`, `plane_height_m` | table plane for `fixed`, e.g. `plane_normal_cam:=[0.0,-0.6,-0.8]` |
| `fx`, `fy`, `cx`, `cy` | intrinsics used only while no CameraInfo arrives (`cx`/`cy` −1 = image centre) |
| `rgb_topic`, `depth_topic`, `camera_info_topic` | input topics (YAML: `/head_front_camera/...`) |
| `debug`, `visualize`, `save` | verbose prints / OpenCV windows / files on disk |
| `config` | a different YAML file |
| `python_executable` | Python for the node (default: the active venv's) |
| `rviz` | also start RViz with `rviz/die_detector.rviz` |

#### 6. What the node does and logs

- Every processed frame publishes `/dice/debug_panels` (image + zoomed die with
  the outline, the fitted cube, the TF axes, IoU, faces, or the reason a frame
  was rejected).
- A valid frame publishes `/dice/pose`, the TF frames and the plane marker, and
  updates the service answer; it logs `Die: top …, front … | IoU … | yaw … | xyz …`.
- A rejected frame logs `No die pose: <reason>` (throttled).
- Every 10 s: `received / processed / published` counts — or, if nothing
  arrives, which input topics it is waiting for and how many publishers each has.
- Without CameraInfo it warns once and uses `fx`/`fy`/`cx`/`cy`.

To replay future bags exactly like the robot, record what the node uses there:

```bash
ros2 bag record /head_front_camera/color/image_raw /head_front_camera/color/camera_info /tf /tf_static
```

#### Topics, service and TF

| Direction | Name | Type | Notes |
|---|---|---|---|
| sub | `rgb_topic` (`/head_front_camera/color/image_raw`) | `sensor_msgs/Image` | best-effort, keeps the latest frame |
| sub | `camera_info_topic` (`/head_front_camera/color/camera_info`) | `sensor_msgs/CameraInfo` | intrinsics |
| sub | `depth_topic` | `sensor_msgs/Image` | only for `classical` / `plane_source: depth`; synchronised with RGB |
| pub | `/dice/pose` | `geometry_msgs/PoseStamped` | top-face centre, in the image's frame |
| pub | `/dice/debug_panels` | `sensor_msgs/Image` | every processed frame |
| pub | `/dice/fitted_plane_marker` | `visualization_msgs/Marker` | table plane under the die |
| srv | `/die_identification_3d` | `drims_homework_interfaces/DieIdentification3D` | last valid top / front face + pose |
| TF | `die_top_face`, `die_centroid` | | children of the camera optical frame |

#### RViz2

```bash
rviz2 -d $(ros2 pkg prefix drims_die_detection)/share/drims_die_detection/rviz/die_detector.rviz
```

Add an Image display on `/dice/debug_panels` and TF for `die_top_face` /
`die_centroid` (the TIAGo layout `moveit_die_detection.rviz` already has both).

---

## Parameters

All parameters live in [`config/die_detector_params.yaml`](config/die_detector_params.yaml).

Values below are the YAML's (the code defaults in `DieDetectorParams` can differ).

| Parameter | Type | YAML value | Description |
|-----------|------|---------|-------------|
| `pose_method` | str | `silhouette` | `silhouette` \| `yoloe_faces` \| `classical` (see ROS 2 Node) |
| `device` | str | `auto` | YOLOE + CNN device: `auto` \| `cuda` \| `cpu` |
| `die_size_m` | float | `0.041` | Die side length (m) — sets the fitted cube size |
| `cnn_model_path` | str | `weights/die_mobilenet_v3.pt` | CNN weights (relative to the package) |
| `cnn_conf_thresh` | float | `0.55` | Below this the CNN front face is reported as none |
| `min_fit_iou` | float | `0.85` | `silhouette`: reject fits below this silhouette IoU |
| `accept_fit_iou` | float | `0.90` | `silhouette`: stop trying YOLOE candidates once one fits this well |
| `min_face_pose_iou` | float | `0.60` | `yoloe_faces`: same check on the face-polygon pose |
| `max_yoloe_candidates` | int | `3` | YOLOE candidates tried per frame |
| `yoloe_candidate_conf` | float | `0.02` | YOLOE confidence for candidates (low on purpose; the fit check filters) |
| `color_fallback` | bool | `true` | HSV candidate when no YOLOE candidate fits |
| `plane_source` | str | `tf` | Table plane: `tf` \| `fixed` \| `depth` |
| `table_frame`, `table_height_m` | str, float | `base_footprint`, `0.55` | `tf`: table top at z = `table_height_m` in `table_frame` |
| `plane_normal_cam`, `plane_height_m` | list, float | `[0, -0.6, -0.8]`, `0.62` | `fixed`: table normal (up) in the camera optical frame, camera height above it |
| `fx`, `fy` (`cx`, `cy`) | float | `615.0` | Intrinsics used only until CameraInfo arrives |
| `detection_mode` | str | `hsv` | `classical` only: `hsv` \| `kmeans` |
| `hsv_min`, `hsv_max`, `glare_v_thresh` | | | HSV die segmentation (classical detection / colour fallback) |
| `clahe_clip_limit`, `pip_min_circularity`, `min_pips`, `max_pips` | | | Face / pip segmentation (`yoloe_faces`, `classical`) |
| `depth_scale`, `use_monocular_fallback`, `ransac_*` | | | Depth handling / RANSAC plane (`classical`, `plane_source: depth`) |
| `debug`, `visualize`, `save` | bool | `true` | Verbose prints / OpenCV windows / output files |
| `rgb_topic` | str | `/head_front_camera/color/image_raw` | RGB input topic |
| `depth_topic` | str | `/head_front_camera/aligned_depth_to_color/image_raw` | Aligned depth topic |
| `camera_info_topic` | str | `/head_front_camera/color/camera_info` | CameraInfo topic |

-----------|------|---------|-------------|
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
