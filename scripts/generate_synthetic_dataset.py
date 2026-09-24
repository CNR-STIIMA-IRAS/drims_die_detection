"""
Generate Synthetic Dataset for Die Orientation Classification
==============================================================
Uses PyRender PBR offscreen rendering + real table background compositing,
with the appearance matched to real TIAGo head-camera die crops:
- 24 canonical resting orientations + continuous yaw (0-360 deg) + slight tilt (+-3 deg)
- Camera elevation 40-66 deg, far (8-14 die widths) -> near-orthographic like the real camera
- 17-20% diameter pips, matte grey body, thick dark face edges and silhouette outline
- Mostly desaturated colour, occasional mild glare, subtle contact shadow
- Downscaled to the real native die size (80-130 px) before blur / noise / JPEG
- Cropped like inference: die bbox + 12 px detector margin, stretched to 224x224

Compare real vs synthetic crops before generating a full dataset:
    python3 scripts/generate_synthetic_dataset.py --preview 12
"""

import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time

import cv2
import numpy as np

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")  # headless (docker) rendering
import pyrender
import trimesh

# Canonical face normals of the CAD mesh
NORMALS = {
    1: np.array([0, 0, -1.0]),
    2: np.array([-1, 0, 0.0]),
    3: np.array([0, 1, 0.0]),
    4: np.array([0, -1, 0.0]),
    5: np.array([1, 0, 0.0]),
    6: np.array([0, 0, 1.0]),
}

# Joint 30-class label space (matches train_die_classifier.py's joint head):
# ids 0-23 = (top, front) pairs, ids 24-29 = top 1..6 seen as a single face
PAIRS = [(t, f) for t in range(1, 7) for f in range(1, 7) if f not in (t, 7 - t)]
CLASS_NAMES = [f"t{t}_f{f}" for t, f in PAIRS] + [f"t{t}_single" for t in range(1, 7)]


def joint_class_id(top_face, front_face):
    return 24 + top_face - 1 if front_face == 0 else PAIRS.index((top_face, front_face))


def build_pips_trimesh(pip_r=0.110, pip_h=0.002, D=0.2651, pip_surf=0.499):
    """Build flat 3D pip discs on all 6 faces of the canonical unit die."""
    pips_3d = []
    # Face 6 (+Z): 6 pips
    for x in [-D, D]:
        for y in [-D, 0.0, D]:
            pips_3d.append((np.array([x, y, pip_surf]), np.array([0, 0, 1.0])))
    # Face 1 (-Z): 1 pip
    pips_3d.append((np.array([0.0, 0.0, -pip_surf]), np.array([0, 0, -1.0])))
    # Face 5 (+X): 5 pips
    for y in [-D, D]:
        for z in [-D, D]:
            pips_3d.append((np.array([pip_surf, y, z]), np.array([1.0, 0, 0])))
    pips_3d.append((np.array([pip_surf, 0.0, 0.0]), np.array([1.0, 0, 0])))
    # Face 2 (-X): 2 pips
    pips_3d.append((np.array([-pip_surf, D, D]), np.array([-1.0, 0, 0])))
    pips_3d.append((np.array([-pip_surf, -D, -D]), np.array([-1.0, 0, 0])))
    # Face 3 (+Y): 3 pips
    pips_3d.append((np.array([-D, pip_surf, D]), np.array([0, 1.0, 0])))
    pips_3d.append((np.array([0.0, pip_surf, 0.0]), np.array([0, 1.0, 0])))
    pips_3d.append((np.array([D, pip_surf, -D]), np.array([0, 1.0, 0])))
    # Face 4 (-Y): 4 pips
    for x in [-D, D]:
        for z in [-D, D]:
            pips_3d.append((np.array([x, -pip_surf, z]), np.array([0, -1.0, 0])))

    pip_cylinders = []
    for pos, norm in pips_3d:
        cyl = trimesh.creation.cylinder(radius=pip_r, height=pip_h, sections=32)
        z_ax = np.array([0, 0, 1.0])
        v_cross = np.cross(z_ax, norm)
        c = np.dot(z_ax, norm)
        if np.linalg.norm(v_cross) < 1e-6:
            R = np.eye(3) if c > 0 else np.diag([1, -1, -1])
        else:
            v_skew = np.array([[0, -v_cross[2], v_cross[1]], [v_cross[2], 0, -v_cross[0]], [-v_cross[1], v_cross[0], 0]])
            R = np.eye(3) + v_skew + v_skew @ v_skew * ((1 - c) / (np.linalg.norm(v_cross) ** 2))
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos
        cyl.apply_transform(T)
        pip_cylinders.append(cyl)

    return trimesh.util.concatenate(pip_cylinders)


def make_rounded_rect_3d(E=0.415, rc=0.045, n_arc=8):
    """Generate 2D closed polygon of rounded rectangle."""
    corners = [
        (E - rc, E - rc, 0.0, np.pi/2),
        (-E + rc, E - rc, np.pi/2, np.pi),
        (-E + rc, -E + rc, np.pi, 3*np.pi/2),
        (E - rc, -E + rc, 3*np.pi/2, 2*np.pi),
    ]
    pts = []
    for cx_c, cy_c, a_start, a_end in corners:
        angles = np.linspace(a_start, a_end, n_arc)
        for a in angles:
            pts.append([cx_c + rc * np.cos(a), cy_c + rc * np.sin(a)])
    return np.array(pts, dtype=np.float32)


def get_rotation_to_top_face(target_face, yaw_deg=0.0, tilt_x_deg=0.0, tilt_y_deg=0.0):
    """Compute 4x4 matrix aligning target_face to +Z with yaw and tilt perturbation."""
    n = NORMALS[target_face]
    z = np.array([0, 0, 1.0])
    v = np.cross(n, z)
    c = np.dot(n, z)
    if np.linalg.norm(v) < 1e-6:
        R_align = np.eye(4) if c > 0 else trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    else:
        v_norm = v / np.linalg.norm(v)
        angle = np.arccos(np.clip(c, -1.0, 1.0))
        R_align = trimesh.transformations.rotation_matrix(angle, v_norm)

    R_yaw = trimesh.transformations.rotation_matrix(np.radians(yaw_deg), [0, 0, 1])
    R_tilt_x = trimesh.transformations.rotation_matrix(np.radians(tilt_x_deg), [1, 0, 0])
    R_tilt_y = trimesh.transformations.rotation_matrix(np.radians(tilt_y_deg), [0, 1, 0])

    return R_tilt_y @ R_tilt_x @ R_yaw @ R_align


class SyntheticGeneratorWorker:
    # Real-camera appearance model (tuned against real TIAGo head-camera crops,
    # see --preview). The CNN sees: detector bbox of a ~100 px die + 12 px
    # margin, stretched to 224x224 -- so we render large, then downscale to the
    # native die size, add sensor artefacts there, and crop the same way.
    INFER_MARGIN_PX = 12           # DieDetector bbox margin (rgb_die_detector Step 3)
    DIE_PX_RANGE = (80, 130)       # die bbox size in the real 1280x720 frames
    PIP_RADII = (0.085, 0.0925, 0.100)

    def __init__(self, cad_mesh_path, bg_image_paths, viewport_size=320, single_frac=0.15):
        self.single_frac = single_frac
        self.cad_mesh_path = cad_mesh_path
        self.bg_image_paths = bg_image_paths
        self.viewport_size = viewport_size
        self.raw_mesh = trimesh.load(cad_mesh_path, force='mesh')
        self.pips_meshes = [build_pips_trimesh(pip_r=r) for r in self.PIP_RADII]
        # Border loop sits on the cube edges (real dice have no inset frame line)
        self.rect_2d = make_rounded_rect_3d(E=0.49, rc=0.08)
        self.faces_borders = [
            (np.array([0, 0, 1.0]), [np.array([p[0], p[1], 0.50]) for p in self.rect_2d]),
            (np.array([0, 0, -1.0]), [np.array([p[0], p[1], -0.50]) for p in self.rect_2d]),
            (np.array([1.0, 0, 0]), [np.array([0.50, p[0], p[1]]) for p in self.rect_2d]),
            (np.array([-1.0, 0, 0]), [np.array([-0.50, p[0], p[1]]) for p in self.rect_2d]),
            (np.array([0, 1.0, 0]), [np.array([p[0], 0.50, p[1]]) for p in self.rect_2d]),
            (np.array([0, -1.0, 0]), [np.array([p[0], -0.50, p[1]]) for p in self.rect_2d]),
        ]

        # Load background images into memory
        self.bg_images = []
        for p in bg_image_paths:
            img = cv2.imread(p)
            if img is not None:
                self.bg_images.append(img)
        if not self.bg_images:
            # Fallback neutral tabletop
            self.bg_images.append(np.full((720, 1280, 3), 35, dtype=np.uint8))

    def sample_table_bg(self, size, out_size):
        """Crop a size x size table patch at native camera scale, resized to
        out_size. Rejects patches containing the real die or a hand (bright)."""
        for _ in range(30):
            src = random.choice(self.bg_images)
            H, W = src.shape[:2]
            y0 = random.randint(int(H * 0.15), max(int(H * 0.15) + 1, H - size - 1))
            x0 = random.randint(0, max(1, W - size - 1))
            patch = src[y0:y0 + size, x0:x0 + size]
            if patch.shape[:2] != (size, size):
                continue
            if np.mean(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) > 110) < 0.003:
                break
        return cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_CUBIC)

    def generate_single_sample(self, target_face, sample_id):
        # Domain randomized parameters
        yaw_deg = random.uniform(0.0, 360.0)
        tilt_x = random.uniform(-3.0, 3.0)
        tilt_y = random.uniform(-3.0, 3.0)
        # The real head camera always sees a side face (40-66 deg elevation);
        # a small share of near-overhead views gives the double-face / joint
        # heads genuine single-face examples.
        if random.random() < self.single_frac:
            pitch_deg = random.uniform(76.0, 88.0)
        else:
            pitch_deg = random.uniform(40.0, 66.0)
        # Real camera is ~10 die-widths away -> near-orthographic. (A close
        # camera exaggerates the top face and squashes the front face.)
        cam_dist = random.uniform(8.0, 14.0)
        roughness = random.uniform(0.35, 0.60)
        g = random.uniform(0.86, 0.94)
        tint = [g * random.uniform(0.98, 1.0), g, g * random.uniform(0.97, 1.0), 1.0]

        R_tot = get_rotation_to_top_face(target_face, yaw_deg, tilt_x, tilt_y)

        # Materials
        body_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=tint,
            metallicFactor=0.0,
            roughnessFactor=roughness
        )
        pip_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.02, 0.02, 0.02, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.9
        )

        m_rot = self.raw_mesh.copy()
        m_rot.apply_transform(R_tot)
        p_rot = random.choice(self.pips_meshes).copy()
        p_rot.apply_transform(R_tot)

        body_mesh = pyrender.Mesh.from_trimesh(m_rot, material=body_mat, smooth=True)
        pips_mesh = pyrender.Mesh.from_trimesh(p_rot, material=pip_mat, smooth=False)

        scene = pyrender.Scene(ambient_light=[random.uniform(0.45, 0.65)] * 3, bg_color=[0, 0, 0, 0])
        scene.add(body_mesh)
        scene.add(pips_mesh)

        # Soft overhead key light (lab ceiling lighting)
        light1 = pyrender.DirectionalLight(color=[1.0, 0.99, 0.97], intensity=random.uniform(2.0, 3.4))
        pose1 = np.eye(4)
        pose1[:3, :3] = trimesh.transformations.euler_matrix(
            -np.radians(random.uniform(55, 85)), np.radians(random.uniform(-30, 30)), 0)[:3, :3]
        scene.add(light1, pose=pose1)

        # Weak lateral fill light
        light2 = pyrender.DirectionalLight(color=[0.96, 0.97, 1.0], intensity=random.uniform(0.8, 1.8))
        pose2 = np.eye(4)
        pose2[:3, :3] = trimesh.transformations.euler_matrix(
            -np.radians(random.uniform(25, 45)), -np.radians(random.uniform(30, 75)), 0)[:3, :3]
        scene.add(light2, pose=pose2)

        # Camera: fov scaled with distance so the die fills ~65% of the viewport
        W = H = self.viewport_size
        yfov = 2.0 * np.arctan(random.uniform(1.10, 1.25) / cam_dist)
        cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
        pitch = np.radians(pitch_deg)
        cam_pos = np.array([0.0, -cam_dist * np.cos(pitch), cam_dist * np.sin(pitch)])
        target = np.array([0.0, 0.0, 0.0])
        forward = target - cam_pos
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        cam_up = np.cross(right, forward)

        R_cam = np.column_stack((right, cam_up, -forward))
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = R_cam
        cam_pose[:3, 3] = cam_pos
        scene.add(cam, pose=cam_pose)

        fy = (H / 2.0) / np.tan(yfov / 2.0)
        fx = fy
        cx = W / 2.0
        cy = H / 2.0

        r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        r.delete()

        die_rgb = cv2.cvtColor(color[:, :, :3], cv2.COLOR_RGB2BGR)
        die_alpha = (color[:, :, 3] > 0).astype(np.float32)
        ys, xs = np.nonzero(die_alpha)
        bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        die_render_px = max(bx1 - bx0, by1 - by0)

        # Native scale: real die bbox size in the camera image
        scale = random.uniform(*self.DIE_PX_RANGE) / die_render_px

        # Real table background, sampled at the same native scale
        table_bg = self.sample_table_bg(max(8, int(round(W * scale))), W)

        # Contact shadow (subtle)
        kernel_s = random.choice([11, 15, 19])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_s, kernel_s))
        shadow_mask = cv2.dilate(die_alpha, kernel)
        M = np.float32([[1, 0, 0], [0, 1, random.randint(4, 10)]])
        shadow_mask = cv2.warpAffine(shadow_mask, M, (W, H))
        shadow_outside = np.clip(shadow_mask - die_alpha, 0, 1)
        shadow_soft = cv2.GaussianBlur(shadow_outside, (23, 23), random.uniform(6.0, 9.0))
        table_bg = (table_bg.astype(np.float32) *
                    (1.0 - shadow_soft * random.uniform(0.25, 0.6))[:, :, None])

        # Die tone: real dice read as mid/light grey, not paper white
        die_rgb = die_rgb.astype(np.float32) * random.uniform(0.78, 0.95)

        die_alpha_smooth = cv2.GaussianBlur(die_alpha, (3, 3), 0.5)[:, :, np.newaxis]
        comp = table_bg * (1.0 - die_alpha_smooth) + die_rgb * die_alpha_smooth

        # Occasional mild specular glare (real dice are fairly matte)
        if random.random() < 0.25:
            glare = np.zeros((H, W), dtype=np.float32)
            cv2.circle(glare, (int(cx + random.uniform(-25, 25)), int(by0 + (by1 - by0) * random.uniform(0.2, 0.45))),
                       random.randint(18, 30), 1.0, -1)
            glare = cv2.GaussianBlur(glare, (41, 41), random.uniform(9.0, 14.0))
            comp += (glare * random.uniform(15.0, 40.0) * die_alpha)[:, :, None]

        # Dark face-border lines + silhouette outline. The real die shows thick
        # dark edges (~1.5-2% of die size) where the rounded bevels turn away.
        line_t = max(2, int(round(die_render_px * random.uniform(0.012, 0.022))))
        line_col = random.uniform(25.0, 55.0)

        def project(P_w):
            P_c = R_cam.T @ (P_w - cam_pos)
            if -P_c[2] <= 0.01:
                return None
            u = int(fx * P_c[0] / (-P_c[2]) + cx)
            v = int(fy * (-P_c[1]) / (-P_c[2]) + cy)
            return (u, v)

        to_cam_dir = -forward
        line_mask = np.zeros((H, W), dtype=np.uint8)
        for n_body, loop_3d in self.faces_borders:
            n_world = R_tot[:3, :3] @ n_body
            if np.dot(n_world, to_cam_dir) > 0.10:
                pts_2d = [pt for pt in (project(R_tot[:3, :3] @ p) for p in loop_3d) if pt is not None]
                if len(pts_2d) > 2:
                    cv2.polylines(line_mask, [np.array(pts_2d, dtype=np.int32)], isClosed=True,
                                  color=255, thickness=line_t, lineType=cv2.LINE_AA)
        erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * line_t + 1, 2 * line_t + 1))
        silhouette = die_alpha - cv2.erode(die_alpha, erode_k)
        edge_w = np.maximum((line_mask.astype(np.float32) / 255.0) * die_alpha, silhouette)
        edge_w = cv2.GaussianBlur(edge_w, (3, 3), 0.8) * random.uniform(0.75, 0.95)
        comp = comp * (1.0 - edge_w[:, :, None]) + line_col * edge_w[:, :, None]
        comp = np.clip(comp, 0, 255)

        # Colour: the bag frames are nearly grey with a faint cool cast
        gray = cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)[:, :, None]
        sat = random.uniform(0.0, 0.35)
        comp = gray * (1.0 - sat) + comp * sat
        comp *= np.array([random.uniform(1.0, 1.04), random.uniform(1.0, 1.03), random.uniform(0.95, 1.0)])
        comp = np.clip(comp, 0, 255).astype(np.uint8)

        # ── Sensor simulation at native resolution ──────────────────────────
        nW = max(16, int(round(W * scale)))
        small = cv2.resize(comp, (nW, nW), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), random.uniform(0.45, 1.0))
        noise = np.random.normal(0, random.uniform(2.0, 6.0), small.shape[:2]).astype(np.float32)[:, :, None]
        small = np.clip(small.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        ok, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, random.randint(55, 90)])
        small = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        # Inference-style crop: die bbox + detector margin (+ small bbox jitter),
        # stretched to 224x224 exactly like CNNDieClassifier.preprocess
        m = self.INFER_MARGIN_PX
        j = lambda: random.randint(-3, 3)
        sx0 = max(0, int(bx0 * scale) - m + j())
        sy0 = max(0, int(by0 * scale) - m + j())
        sx1 = min(nW, int(np.ceil(bx1 * scale)) + m + j())
        sy1 = min(nW, int(np.ceil(by1 * scale)) + m + j())
        final_crop = cv2.resize(small[sy0:sy1, sx0:sx1], (224, 224), interpolation=cv2.INTER_LINEAR)

        # Ground Truth Label Calculation
        # Top Face: face whose normal has highest alignment with +Z_world
        top_face = max(NORMALS.keys(), key=lambda k: np.dot(R_tot[:3, :3] @ NORMALS[k], [0, 0, 1.0]))

        # Front Face: prominent lateral face facing camera
        d_cam_dir = (cam_pos - target) / np.linalg.norm(cam_pos - target)
        candidate_faces = [k for k in NORMALS.keys() if k not in (top_face, 7 - top_face)]
        best_lat_face = max(candidate_faces, key=lambda k: np.dot(R_tot[:3, :3] @ NORMALS[k], d_cam_dir))
        best_lat_dot = float(np.dot(R_tot[:3, :3] @ NORMALS[best_lat_face], d_cam_dir))

        # Threshold for prominent lateral face
        if best_lat_dot >= 0.18:
            front_face = int(best_lat_face)
        else:
            front_face = 0  # None / Ambiguous / Top-down

        metadata = {
            "top_face": int(top_face),
            "front_face": front_face,
            "is_double_face": front_face != 0,
            "class_id": joint_class_id(int(top_face), front_face),
            "front_face_dot": round(best_lat_dot, 3),
            "yaw_deg": round(yaw_deg, 2),
            "pitch_deg": round(pitch_deg, 2),
        }

        return final_crop, metadata


# Global worker instance per process
_worker = None

def _init_worker(cad_path, bg_paths, viewport_size, single_frac):
    global _worker
    _worker = SyntheticGeneratorWorker(cad_path, bg_paths, viewport_size, single_frac)

def _generate_task(task_args):
    global _worker
    split, target_face, idx, out_img_dir = task_args
    img, meta = _worker.generate_single_sample(target_face, idx)
    fname = f"{split}_{target_face}_{idx:05d}.jpg"
    img_path = os.path.join(out_img_dir, fname)
    cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 94])
    meta["image_path"] = os.path.relpath(img_path, os.path.dirname(out_img_dir))
    return meta


def generate_dataset(
    output_dir="/home/ws/src/drims_die_detection/dataset",
    cad_mesh_path="/home/ws/src/drims_dice_simulator/urdf/Dice.obj",
    bg_dir="/home/ws/src/drims_die_detection/output/bag_eval_frames",
    num_train_per_face=600,
    num_val_per_face=100,
    num_workers=8,
    single_frac=0.15,
):
    print("=" * 70)
    print("Generating Synthetic Die Orientation Dataset (Two-Head MobileNetV3)")
    print(f"Output directory : {output_dir}")
    print(f"Train per face   : {num_train_per_face} (Total: {num_train_per_face * 6})")
    print(f"Val per face     : {num_val_per_face} (Total: {num_val_per_face * 6})")
    print(f"Workers          : {num_workers}")
    print("=" * 70)

    # Collect background images
    bg_paths = glob.glob(os.path.join(bg_dir, "*.jpg"))
    sample0 = "/home/ws/src/drims_die_detection/output/bag_samples/sample_rgb_0.jpg"
    if os.path.exists(sample0):
        bg_paths.append(sample0)
    print(f"Found {len(bg_paths)} background source images.")

    for split, count in [("train", num_train_per_face), ("val", num_val_per_face)]:
        split_dir = os.path.join(output_dir, split)
        img_dir = os.path.join(split_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        tasks = []
        for face in [1, 2, 3, 4, 5, 6]:
            for idx in range(count):
                tasks.append((split, face, idx, img_dir))

        print(f"\nRendering {split} split ({len(tasks)} images) with {num_workers} processes...")
        t0 = time.time()
        with mp.Pool(num_workers, initializer=_init_worker, initargs=(cad_mesh_path, bg_paths, 320, single_frac)) as pool:
            records = pool.map(_generate_task, tasks)
        elapsed = time.time() - t0
        fps = len(tasks) / elapsed
        print(f"Finished {split} in {elapsed:.1f}s ({fps:.1f} img/s)")

        labels_path = os.path.join(split_dir, "labels.json")
        with open(labels_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"Saved {len(records)} annotations to {labels_path}")

    with open(os.path.join(output_dir, "class_mapping.json"), "w") as f:
        json.dump({"classes": CLASS_NAMES,
                   "pairs": [list(p) for p in PAIRS] + [[t, 0] for t in range(1, 7)]}, f, indent=2)

    print("\nDataset generation complete!")


def preview(cad_mesh_path, bg_dir, n, out_path):
    """Side-by-side sheet: real detector crops (top rows) vs synthetic crops."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from drims_die_detection.die_detector_params import DieDetectorParams
    from drims_die_detection.rgb_die_detector import RGBDieDetector

    bg_paths = sorted(glob.glob(os.path.join(bg_dir, "*.jpg")))
    det = RGBDieDetector(DieDetectorParams(debug=False))
    real = []
    for f in bg_paths:
        img = cv2.imread(f)
        x, y, w, h = det.detect(img)["bbox"]
        if w > 10 and h > 10:
            real.append(cv2.resize(img[y:y + h, x:x + w], (224, 224)))
    worker = SyntheticGeneratorWorker(cad_mesh_path, bg_paths, 320)
    syn = []
    for i in range(n):
        crop, meta = worker.generate_single_sample(i % 6 + 1, i)
        crop = crop.copy()
        cv2.putText(crop, f"top{meta['top_face']} fr{meta['front_face']}", (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        syn.append(crop)

    def grid(tiles, cols=6):
        tiles = tiles + [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
        return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])

    sep = np.full((6, 224 * 6, 3), (0, 200, 255), dtype=np.uint8)
    cv2.imwrite(out_path, np.vstack([grid(real), sep, grid(syn)]))
    print(f"Preview ({len(real)} real / {len(syn)} synthetic) written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/home/ws/src/drims_die_detection/dataset")
    parser.add_argument("--cad_mesh", default="/home/ws/src/drims_dice_simulator/urdf/Dice.obj")
    parser.add_argument("--bg_dir", default="/home/ws/src/drims_die_detection/output/bag_eval_frames")
    parser.add_argument("--num_train", type=int, default=600)
    parser.add_argument("--num_val", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--single_frac", type=float, default=0.15,
                        help="share of near-overhead (single visible face) views")
    parser.add_argument("--preview", type=int, default=0, metavar="N",
                        help="only render N samples next to real crops (no dataset written)")
    parser.add_argument("--preview_out", default="/home/ws/src/drims_die_detection/output/synthetic_preview.jpg")
    args = parser.parse_args()

    if args.preview:
        preview(args.cad_mesh, args.bg_dir, args.preview, args.preview_out)
        sys.exit(0)

    generate_dataset(
        output_dir=args.output_dir,
        cad_mesh_path=args.cad_mesh,
        bg_dir=args.bg_dir,
        num_train_per_face=args.num_train,
        num_val_per_face=args.num_val,
        num_workers=args.num_workers,
        single_frac=args.single_frac,
    )

