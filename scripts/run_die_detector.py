#!/usr/bin/env python3
"""
run_die_detector.py
===================
ROS-free CLI entry point for the DieDetectorPipeline.

Usage
-----
::

    # Process all images in images/rgb/ using default params:
    python scripts/run_die_detector.py

    # Use a custom YAML config:
    python scripts/run_die_detector.py --config config/die_detector_params.yaml

    # Process a specific RGB file (monocular depth):
    python scripts/run_die_detector.py --image images/rgb/sample.jpg

    # Process RGB + a pre-saved depth .npy file:
    python scripts/run_die_detector.py \\
        --image  images/rgb/sample.jpg \\
        --depth  output/depth_sample.npy

    # Enable debug prints and visualise results:
    python scripts/run_die_detector.py --debug --visualize
"""

from __future__ import annotations

import os
import sys

# ── Venv auto-relaunch guard ──────────────────────────────────────────────────
# If the project .venv exists but its site-packages are not on sys.path
# (i.e. the venv is not active), transparently re-exec with the venv's python3
# so that torch, transformers, and plotly are always available.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_VENV_PYTHON = os.path.join(_REPO_ROOT, ".venv", "bin", "python3")
_VENV_LIB = os.path.join(_REPO_ROOT, ".venv", "lib")

def _venv_active() -> bool:
    """Return True if the venv's site-packages are already on sys.path."""
    if not os.path.isdir(_VENV_LIB):
        return True   # No venv → nothing to activate
    for p in sys.path:
        if p.startswith(_VENV_LIB):
            return True
    return False

if (
    not _venv_active()
    and os.path.isfile(_VENV_PYTHON)
    and os.environ.get("_DIE_DETECTOR_VENV_ACTIVE") != "1"
):
    import subprocess
    os.environ["_DIE_DETECTOR_VENV_ACTIVE"] = "1"
    result = subprocess.run([_VENV_PYTHON] + sys.argv)
    sys.exit(result.returncode)
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import glob

import cv2
import numpy as np

# Allow running directly from repo root without installation
sys.path.insert(0, _REPO_ROOT)

from drims_die_detection import DieDetectorParams, DieDetectorPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS-free Die Detector 3D CLI")
    parser.add_argument("--config", default="config/die_detector_params.yaml",
                        help="Path to YAML parameter file")
    parser.add_argument("--image", default=None,
                        help="Path to a single RGB image file")
    parser.add_argument("--depth", default=None,
                        help="Path to a matching .npy depth map (metres). "
                             "If omitted, monocular estimation is used.")
    parser.add_argument("--rgb-dir", default="images/rgb",
                        help="Directory of RGB images (used when --image is not set)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (overrides config value)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose debug prints")
    parser.add_argument("--visualize", action="store_true",
                        help="Show intermediate images with cv2.imshow")
    parser.add_argument("--no-save", action="store_true",
                        help="Disable saving output files")
    return parser.parse_args()

def _print_dependency_status() -> None:
    """Print a one-time summary of which optional packages are available."""
    lines = ["[run_die_detector] Optional dependency status:"]
    for pkg, label in [
        ("torch",        "torch          (monocular depth estimation)"),
        ("transformers", "transformers   (HuggingFace DPT model)"),
        ("plotly",       "plotly         (interactive 3D HTML plots)"),
        ("open3d",       "open3d         (faster RANSAC plane fitting)"),
    ]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            lines.append(f"  ✅ {label}  [{ver}]")
        except ImportError:
            lines.append(f"  ❌ {label}  [NOT INSTALLED — run: pip install {pkg}]")
    print("\n".join(lines))

def main() -> None:
    _print_dependency_status()
    args = parse_args()

    # ── Load params ───────────────────────────────────────────────────
    if os.path.isfile(args.config):
        params = DieDetectorParams.from_yaml(args.config)
        print(f"[run_die_detector] Loaded params from {args.config}")
    else:
        params = DieDetectorParams()
        print(f"[run_die_detector] Config not found at '{args.config}' — using defaults.")

    # CLI overrides
    if args.debug:
        params.debug = True
    if args.visualize:
        params.visualize = True
    if args.no_save:
        params.save = False
    if args.output_dir:
        params.output_dir = args.output_dir

    pipeline = DieDetectorPipeline(params)

    # ── Collect image paths ───────────────────────────────────────────
    if args.image:
        image_paths = [args.image]
        depth_paths = [args.depth] if args.depth else [None]
    else:
        rgb_dir = args.rgb_dir
        image_paths = sorted(
            glob.glob(os.path.join(rgb_dir, "*.jpg")) +
            glob.glob(os.path.join(rgb_dir, "*.png"))
        )
        depth_paths = [None] * len(image_paths)
        if not image_paths:
            print(f"[run_die_detector] No images found in '{rgb_dir}'. "
                  "Use --image or --rgb-dir to specify input.")
            return

    print(f"[run_die_detector] Processing {len(image_paths)} image(s)…")

    results = []
    for img_path, dep_path in zip(image_paths, depth_paths):
        rgb = cv2.imread(img_path)
        if rgb is None:
            print(f"  [WARN] Cannot read {img_path} — skipping.")
            continue
        img_name = os.path.basename(img_path)

        # Load or infer depth
        if dep_path and os.path.isfile(dep_path):
            depth_m = np.load(dep_path).astype(np.float32)
            print(f"  [{img_name}] Using depth map from {dep_path}")
            result = pipeline.process_rgbd(rgb, depth_m, image_name=img_name)
        else:
            print(f"  [{img_name}] Using monocular depth estimation.")
            result = pipeline.process_rgb(rgb, image_name=img_name)
            depth_m = result.get("depth_m", None)

        if params.save:
            collage = pipeline.build_debug_panels(rgb, result)
            pipeline.save_outputs(result, collage, depth_m)

        if params.visualize:
            collage = pipeline.build_debug_panels(rgb, result)
            cv2.imshow(f"Die Detector — {img_name}", collage)
            cv2.waitKey(0)

        results.append(result)

    if params.visualize:
        cv2.destroyAllWindows()

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DETECTION SUMMARY")
    print("=" * 70)
    for r in results:
        pos = r["centroid"]
        q = r["quaternion"]
        orient = r.get("die_orientation", {})
        fm = orient.get("face_map", {})
        det_status = "FULLY DETERMINED" if orient.get("is_fully_determined") else "PARTIAL (Top Face Only)"
        faces_str = f"Orientation [{det_status}]: +Z(top)={fm.get('+Z')}, -Z(bot)={fm.get('-Z')}, +X={fm.get('+X')}, -X={fm.get('-X')}, +Y={fm.get('+Y')}, -Y={fm.get('-Y')}"
        print(f"{r['image_name']:<30} | {faces_str} | "
              f"pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) m | "
              f"q=[{q[0]:.2f},{q[1]:.2f},{q[2]:.2f},{q[3]:.2f}]")


if __name__ == "__main__":
    main()
