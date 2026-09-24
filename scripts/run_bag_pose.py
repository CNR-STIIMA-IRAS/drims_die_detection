"""
Die pose (TF) + top face on real bag frames — YOLOE/CNN vs classical OpenCV
===========================================================================
--mode yoloe (default)
    YOLOE die segmentation -> convex die outline -> cube-silhouette fit of
    (x, y, yaw) on the table plane (drims_die_detection.silhouette_pose);
    top / front face from the CNN.
--mode yoloe_faces
    YOLOE die detection + segmentation, then the original face / pip
    segmentation and top-face-polygon pose (FacePolygonPipeline).
--mode classical
    The standard-CV path: HSV die detection, face / pip segmentation, top face
    = uppermost face polygon (pip count = class), pose from the top-face
    polygon edges (FacePolygonPipeline with colour detection).

Both modes use the same table plane / intrinsics and the same YAML tunables,
so only the method differs. Draws the die cube and its TF axes (x red,
y green, z blue) on every image, times every stage and scores the top face
against <frames_dir>/labels.json.

The test bag has no camera_info and the black table returns no depth, so the
table plane (and fx) are calibrated once from YOLOE die silhouettes (static
camera) and cached in <frames_dir>/camera_calib.json. On the robot, use
camera_info + TF (head camera -> base_footprint) and the table height.

    python3 scripts/run_bag_pose.py                          # yoloe, cuda
    python3 scripts/run_bag_pose.py --device cpu
    python3 scripts/run_bag_pose.py --mode yoloe_faces
    python3 scripts/run_bag_pose.py --mode classical
    python3 scripts/compare_bag_runs.py                      # compare runs
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PACKAGE_ROOT)

import torch  # noqa: E402

from drims_die_detection.die_detector_params import DieDetectorParams  # noqa: E402
from drims_die_detection.face_polygon_pipeline import FacePolygonPipeline  # noqa: E402
from drims_die_detection.rgb_die_detector import RGBDieDetector  # noqa: E402
from drims_die_detection.silhouette_pipeline import SilhouettePipeline  # noqa: E402
from drims_die_detection.silhouette_pose import (AXIS_COLORS, SilhouettePoseFitter,  # noqa: E402
                                                 calibrate_plane_from_masks, die_silhouette,
                                                 draw_die_pose)

ZOOM_PX = 400


def outline(img, det):
    """Convex die outline from the detector's segmentation mask (None if no die)."""
    x, y, w, h = det["bbox"]
    if w <= 10 or h <= 10:
        return None
    return die_silhouette(img, det["debug_steps"]["step2_whitest_mask"][:, :, 0], det["bbox"])


# ── Drawing ───────────────────────────────────────────────────────────────
def _put(img, text, org, color=(255, 255, 255), scale=0.6, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def build_visualization(img, fname, fitter, pose, contour, lines, times, legend):
    full = img.copy()
    zoom = np.full((ZOOM_PX, ZOOM_PX, 3), 30, np.uint8)
    if pose is not None:
        draw_die_pose(full, fitter, pose, contour)
        ref = contour if contour is not None else pose["corners_2d"].astype(np.float32)
        x, y, w, h = cv2.boundingRect(ref.reshape(-1, 2).astype(np.int32))
        side = int(max(w, h) * 2.0)
        x0 = int(np.clip(x + w / 2 - side / 2, 0, img.shape[1] - side))
        y0 = int(np.clip(y + h / 2 - side / 2, 0, img.shape[0] - side))
        zoom = cv2.resize(img[y0:y0 + side, x0:x0 + side], (ZOOM_PX, ZOOM_PX), interpolation=cv2.INTER_CUBIC)
        draw_die_pose(zoom, fitter, pose, contour, scale=ZOOM_PX / side, off=(x0, y0))
        p = pose["centroid"] * 1000
        lines = [f"xyz cam [mm]: {p[0]:.0f} {p[1]:.0f} {p[2]:.0f}",
                 f"yaw: {pose['theta_deg']:+.1f} deg" + (f"   IoU: {pose['iou']:.3f}" if pose["iou"] else "")] + lines
    else:
        lines = ["NO POSE"] + lines
    for i, t in enumerate(lines):
        _put(zoom, t, (8, 22 + 22 * i), scale=0.55)
    for i, (txt, col) in enumerate([("x", AXIS_COLORS[0]), ("y", AXIS_COLORS[1]), ("z", AXIS_COLORS[2])]):
        _put(zoom, txt, (8 + 20 * i, ZOOM_PX - 12), col, 0.6, 2)
    _put(zoom, legend, (75, ZOOM_PX - 12), (0, 255, 255), 0.5)

    _put(full, fname, (15, 30), scale=0.7, thick=2)
    _put(full, " | ".join(f"{k} {v:.1f}" for k, v in times.items()) + " ms", (15, 62), scale=0.55)
    side = np.full((img.shape[0], ZOOM_PX, 3), 30, np.uint8)
    side[:ZOOM_PX] = zoom
    return np.hstack([full, side])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["yoloe", "yoloe_faces", "classical"], default="yoloe")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="device for YOLOE and the CNN (classical mode is CPU-only)")
    parser.add_argument("--frames_dir", default=os.path.join(PACKAGE_ROOT, "output", "bag_test_frames"))
    parser.add_argument("--params", default=os.path.join(PACKAGE_ROOT, "config", "die_detector_params.yaml"))
    parser.add_argument("--weights", default=os.path.join(PACKAGE_ROOT, "weights", "die_mobilenet_v3.pt"))
    parser.add_argument("--out_dir", default=None, help="default: output/bag_pose_<mode>_<device>")
    parser.add_argument("--recalibrate", action="store_true", help="re-estimate plane / fx from the die masks")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available; use --device cpu")
    device = "cpu" if args.mode == "classical" else args.device
    run_name = f"{args.mode}_{device}"
    out_dir = args.out_dir or os.path.join(PACKAGE_ROOT, "output", f"bag_pose_{run_name}")
    os.makedirs(out_dir, exist_ok=True)

    params = DieDetectorParams.from_yaml(args.params)
    params.debug = False
    params.detection_mode = "yoloe" if args.mode == "yoloe" else "hsv"
    die_size = params.die_size_m
    with open(os.path.join(args.frames_dir, "labels.json")) as f:
        labels = json.load(f)
    frames = sorted(labels)
    first = cv2.imread(os.path.join(args.frames_dir, frames[0]))

    # ── Table plane / intrinsics (always from YOLOE silhouettes) ───────────
    calib_path = os.path.join(args.frames_dir, "camera_calib.json")
    if args.recalibrate or not os.path.exists(calib_path):
        print("Calibrating table plane from die silhouettes (one frame per resting position)...")
        cal_params = DieDetectorParams.from_yaml(args.params)
        cal_params.debug, cal_params.detection_mode = False, "yoloe"
        cal_det = RGBDieDetector(cal_params)
        contours, prev = [], None
        for fname in frames:
            if "top_face" not in labels[fname]:
                continue
            img = cv2.imread(os.path.join(args.frames_dir, fname))
            d = cal_det.detect(img)
            c = outline(img, d)
            x, y = d["bbox"][:2]
            if c is not None and (prev is None or abs(x - prev[0]) + abs(y - prev[1]) > 15):
                contours.append(c)
            prev = (x, y)
        H, W = first.shape[:2]
        calib = calibrate_plane_from_masks(contours, first.shape, 910.0, W / 2, H / 2, die_size)
        calib["n_calibration_frames"] = len(contours)
        with open(calib_path, "w") as f:
            json.dump(calib, f, indent=2)
    else:
        with open(calib_path) as f:
            calib = json.load(f)
    K = np.array([[calib["fx"], 0, calib["cx"]], [0, calib["fy"], calib["cy"]], [0, 0, 1.0]])
    fitter = SilhouettePoseFitter(K, np.array(calib["plane_n"]), calib["plane_h"], die_size)
    print(f"[{run_name}] plane: tilt={calib['tilt_deg']:.1f}°, height={calib['plane_h'] * 1000:.0f} mm, "
          f"fx={calib['fx']:.0f}px")

    # ── Models (load + first call reported separately) ─────────────────────
    t0 = time.perf_counter()
    if args.mode == "yoloe":
        params.device = device
        params.cnn_model_path = args.weights
        pipeline = SilhouettePipeline(params)
        pipeline.process(first, K, fitter.n, fitter.h)       # warm-up on the requested device
    else:
        params.device = device
        pipeline = FacePolygonPipeline(params, "yoloe" if args.mode == "yoloe_faces" else "hsv")
        pipeline.process(first, K, fitter.n, fitter.h)
    t_load = (time.perf_counter() - t0) * 1e3

    # ── Per-frame run ──────────────────────────────────────────────────────
    records, vis_list = [], []
    for fname in frames:
        img = cv2.imread(os.path.join(args.frames_dir, fname))
        entry = labels[fname]
        t_start = time.perf_counter()
        pose = contour = None
        top_pred = front_pred = None
        rejected = None
        lines = []
        if args.mode == "yoloe":
            res = pipeline.process(img, K, fitter.n, fitter.h)
            if device == "cuda":
                torch.cuda.synchronize()
            tm = res["timings_ms"]
            times = {"YOLOE": tm["yoloe"], "colour fallback": tm.get("color_fallback", 0.0),
                     "outline+fit": tm["outline+fit"], "CNN": tm.get("cnn", 0.0)}
            if res.get("source") == "color":
                lines.append("detected by colour fallback (YOLOE found nothing)")
            if "fit" in res:
                pose, contour = res["fit"], res["contour"]
            if res["valid"]:
                cnn = res["cnn_classification"]
                top_pred, front_pred = cnn["top_face"], cnn["front_face"]
                lines.append(f"CNN: top {top_pred} ({cnn['top_conf']:.2f}), front {front_pred or '-'}")
            else:
                rejected = res["reason"]
                lines.append("REJECTED: " + rejected[:40])
            if len(res["candidates"]) > 1:
                lines.append("candidates IoU: " + ", ".join(f"{c['iou']:.2f}" for c in res["candidates"]))
            legend = "mask  fitted cube"
        else:
            res = pipeline.process(img, K, fitter.n, fitter.h)
            if device == "cuda":
                torch.cuda.synchronize()
            tm = res["timings_ms"]
            times = ({"YOLOE": tm["yoloe"], "detector rest": tm["detector"] - tm["yoloe"]}
                     if "yoloe" in tm else {"detector (HSV+faces/pips)": tm["detector"]})
            times["pose"] = tm.get("pose", 0.0)
            if "fit" in res:
                pose, contour = res["fit"], res["contour"]
            if res["valid"]:
                top_pred, front_pred = res["top_face_pips"], res["front_face_pips"]
                lines.append(f"pips: top {top_pred or '-'}, front {front_pred or '-'}  "
                             f"({res['num_visible_faces']} faces)")
            else:
                rejected = res["reason"]
                lines.append("REJECTED: " + rejected[:40])
            legend = "top face  cube from pose"
        times["total"] = (time.perf_counter() - t_start) * 1e3

        gt = entry.get("top_face")
        if gt is not None:
            lines.append(f"GT top {gt} -> {'OK' if top_pred == gt else 'MISS'}")
        records.append({
            "frame": fname, "skip": "skip" in entry, "gt_top": gt, "rejected": rejected,
            "source": res.get("source"),
            "top_pred": top_pred, "front_pred": front_pred,
            "yaw_deg": None if pose is None else round(float(pose["theta_deg"]), 3),
            "position_m": None if pose is None else [round(float(v), 5) for v in pose["centroid"]],
            "iou": None if pose is None or pose.get("iou") is None else round(float(pose["iou"]), 4),
            "times_ms": {k: round(v, 3) for k, v in times.items()},
        })
        vis = build_visualization(img, fname, fitter, pose, contour, lines, times, legend)
        cv2.imwrite(os.path.join(out_dir, fname), vis)
        vis_list.append(vis)

    tw = 640
    tiles = [cv2.resize(v, (tw, int(v.shape[0] * tw / v.shape[1]))) for v in vis_list]
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 4)
    cv2.imwrite(os.path.join(out_dir, "_summary.jpg"),
                np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))

    scored = [r for r in records if r["gt_top"] is not None]
    acc = np.mean([r["top_pred"] == r["gt_top"] for r in scored]) if scored else float("nan")
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"run": run_name, "load_ms": t_load, "camera_calib": calib, "frames": records}, f, indent=2)

    print(f"\n[{run_name}] {len(frames)} frames, model load + first call {t_load / 1e3:.1f} s")
    print(f"{'stage':<28}{'median ms':>10}{'p95 ms':>10}")
    for k in records[0]["times_ms"]:
        v = np.array([r["times_ms"][k] for r in records])
        print(f"{k:<28}{np.median(v):>10.1f}{np.percentile(v, 95):>10.1f}")
    print(f"Top-face accuracy: {acc * 100:.1f}% on {len(scored)} labelled frames")
    print(f"Visualizations + results.json written to {out_dir}")


if __name__ == "__main__":
    main()
