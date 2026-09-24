"""
Evaluate CNN Die Orientation Classifier on Real Bag Frames
==========================================================
Runs the 2D die detector + CNN orientation classifier (the same crop path as
DieDetectorPipeline, without depth / point cloud) on held-out real frames and
scores them against per-frame ground truth.

Frames come from scripts/extract_bag_frames.py. Ground truth lives in
<frames_dir>/labels.json, one entry per frame file:

    "frame_0110_t006.70s.jpg": {"top_face": 3, "front_face": null},
    "frame_0184_t011.24s.jpg": {"skip": "hand occluding die"}

front_face is optional (null = not labelled, not scored). The die is moved by
hand during the bag, so labels are per frame, not a single constant.

    python3 scripts/evaluate_bag_orientation.py
    python3 scripts/evaluate_bag_orientation.py --weights weights/other.pt

For every frame (skipped ones included) a visualization is written to
output/cnn_bag_eval/: the full image with the detected bbox and segmented die,
the CNN input crop, the segmented die, and the predicted class probabilities
vs. ground truth. A contact sheet of all frames is saved as _summary.jpg.
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

from drims_die_detection.die_detector_params import DieDetectorParams
from drims_die_detection.cnn_die_classifier import CNNDieClassifier
from drims_die_detection.rgb_die_detector import RGBDieDetector


def print_confusion(name, gt, pred, classes):
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for g, p in zip(gt, pred):
        cm[idx[g], idx[p]] += 1
    print(f"\n{name} confusion (rows = ground truth, cols = predicted):")
    print("      " + "".join(f"{str(c):>5}" for c in classes))
    for c, row in zip(classes, cm):
        print(f"{str(c):>5} " + "".join(f"{v:>5}" for v in row))


GREEN, RED, GREY, YELLOW = (60, 200, 60), (50, 50, 230), (170, 170, 170), (0, 220, 255)
PANEL_W = 300


def _put(img, text, org, color=(255, 255, 255), scale=0.55, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _fit(img, size):
    """Letterbox *img* into a size x size black square."""
    out = np.zeros((size, size, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    s = size / max(h, w)
    r = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_CUBIC)
    y0, x0 = (size - r.shape[0]) // 2, (size - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


def _prob_bars(probs, labels, highlight, gt, title):
    """Horizontal bar chart of class probabilities; predicted bar yellow, GT marked."""
    row_h = 20
    img = np.full((28 + row_h * len(probs), PANEL_W, 3), 30, dtype=np.uint8)
    _put(img, title, (6, 18), scale=0.5)
    for i, (p, lab) in enumerate(zip(probs, labels)):
        y = 28 + i * row_h
        color = YELLOW if lab == highlight else GREY
        cv2.rectangle(img, (40, y + 3), (40 + int(p * (PANEL_W - 100)), y + row_h - 3), color, -1)
        _put(img, str(lab), (8, y + 15), GREEN if lab == gt else (255, 255, 255), scale=0.45)
        _put(img, f"{p:.2f}", (PANEL_W - 52, y + 15), scale=0.45)
    return img


def build_visualization(img, det, res, entry, fname):
    """Full image with bbox + segmentation overlay, plus a side panel with the
    CNN crop, the segmented die, and the predicted class probabilities."""
    x, y, w, h = det["bbox"]
    gt_top = entry.get("top_face")
    gt_front = entry.get("front_face")
    ok = None if gt_top is None or res is None else res["top_face"] == gt_top
    status_color = GREY if ok is None else (GREEN if ok else RED)

    # Segmentation mask: the detector's die contour (YOLOE mask or classical blob)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cnt = det.get("contour")
    if cnt is not None and len(cnt) >= 3:
        cv2.drawContours(mask, [cnt], -1, 255, -1)

    full = img.copy()
    overlay = full.copy()
    overlay[mask > 0] = (0.45 * overlay[mask > 0] + 0.55 * np.array(GREEN)).astype(np.uint8)
    full = overlay
    if cnt is not None and len(cnt) >= 3:
        cv2.drawContours(full, [cnt], -1, GREEN, 2)
    if w > 10 and h > 10:
        cv2.rectangle(full, (x, y), (x + w, y + h), YELLOW, 2)
    if res is not None:
        front = f", front {res['front_face']}" if res["front_face"] is not None else ""
        _put(full, f"pred: top {res['top_face']} ({res['top_conf']:.2f}){front}",
             (x, max(22, y - 10)), status_color, scale=0.7, thick=2)
    else:
        _put(full, "NO DETECTION", (20, 40), RED, scale=0.9, thick=2)

    gt_str = "GT: skipped (" + entry["skip"] + ")" if "skip" in entry else \
        f"GT: top {gt_top}" + (f", front {gt_front}" if gt_front is not None else "")
    verdict = "" if ok is None else ("  -> OK" if ok else "  -> MISS")
    _put(full, fname, (15, 30), scale=0.7, thick=2)
    _put(full, gt_str + verdict, (15, 62), status_color, scale=0.7, thick=2)

    # Side panel
    tile = PANEL_W // 2
    if res is not None:
        crop = img[y:y + h, x:x + w]
        seg = cv2.bitwise_and(crop, crop, mask=mask[y:y + h, x:x + w])
        crops = np.hstack([_fit(crop, tile), _fit(seg, tile)])
        _put(crops, "CNN input", (5, 16), scale=0.45)
        _put(crops, "segmented", (tile + 5, 16), scale=0.45)
        bars_top = _prob_bars(res["top_probs"], [1, 2, 3, 4, 5, 6], res["top_face"], gt_top, "Top face")
        panel = [crops, bars_top]
        if res["front_probs"]:
            panel.append(_prob_bars(res["front_probs"], ["-", 1, 2, 3, 4, 5, 6],
                                    res["front_face"] if res["front_face"] is not None else "-",
                                    gt_front, "Front face (- = single face)"))
        panel = np.vstack(panel)
    else:
        panel = np.full((tile, PANEL_W, 3), 30, dtype=np.uint8)

    H = full.shape[0]
    side = np.full((H, PANEL_W, 3), 30, dtype=np.uint8)
    ph = min(H, panel.shape[0])
    side[:ph] = panel[:ph]
    return np.hstack([full, side])


def save_summary(vis_list, path, cols=6, tile_w=480):
    tiles = [cv2.resize(v, (tile_w, int(v.shape[0] * tile_w / v.shape[1]))) for v in vis_list]
    th = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.zeros((th - t.shape[0], tile_w, 3), np.uint8)]) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(path, np.vstack(rows))


def evaluate(frames_dir, weights_path, conf_thresh, visualize):
    labels_path = os.path.join(frames_dir, "labels.json")
    with open(labels_path) as f:
        labels = json.load(f)

    classifier = CNNDieClassifier(model_path=weights_path, conf_thresh=conf_thresh)
    if not classifier.is_loaded:
        raise SystemExit(f"Could not load weights '{weights_path}'")
    detector = RGBDieDetector(DieDetectorParams(debug=False))
    print(f"Weights : {weights_path}")
    print(f"Frames  : {frames_dir}")

    # Latency benchmark (100 forward passes)
    dummy_crop = np.zeros((150, 150, 3), dtype=np.uint8)
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        classifier.predict(dummy_crop)
        times.append((time.perf_counter() - t0) * 1000.0)
    print(f"Latency (100 runs): median = {np.median(times):.2f} ms, p95 = {np.percentile(times, 95):.2f} ms")

    vis_dir = os.path.join(PACKAGE_ROOT, "output", "cnn_bag_eval")
    if visualize:
        os.makedirs(vis_dir, exist_ok=True)

    gt_top, pred_top, gt_front, pred_front = [], [], [], []
    vis_list = []
    n_skipped = n_nodet = 0
    print()
    for fname in sorted(labels):
        entry = labels[fname]
        scored = "skip" not in entry and entry.get("top_face") is not None
        if not scored:
            n_skipped += 1
            if not visualize:
                continue
        img = cv2.imread(os.path.join(frames_dir, fname))
        if img is None:
            print(f"[MISSING] {fname}")
            continue

        det = detector.detect(img)
        x, y, w, h = det["bbox"]
        res = classifier.predict(img[y:y + h, x:x + w]) if w > 10 and h > 10 else None

        if visualize:
            vis = build_visualization(img, det, res, entry, fname)
            cv2.imwrite(os.path.join(vis_dir, fname), vis)
            vis_list.append(vis)

        if not scored:
            continue
        if res is None:
            n_nodet += 1
            print(f"[NO DET ] {fname}")
            continue

        gt_top.append(entry["top_face"])
        pred_top.append(res["top_face"])
        top_ok = res["top_face"] == entry["top_face"]
        front_str = ""
        if entry.get("front_face") is not None:
            gt_front.append(entry["front_face"])
            pred_front.append(res["front_face"] or 0)
            front_str = f" | front gt={entry['front_face']} pred={res['front_face']} ({res['front_conf']:.2f})"

        print(f"[{'OK' if top_ok else 'MISS':<7}] {fname} | top gt={entry['top_face']} "
              f"pred={res['top_face']} ({res['top_conf']:.2f}){front_str}")

    n = len(gt_top)
    print("\n" + "=" * 70)
    print(f"Scored {n} frames ({n_skipped} skipped by label, {n_nodet} with no detection)")
    if n:
        acc = np.mean(np.array(gt_top) == np.array(pred_top))
        print(f"Top-face accuracy   : {int(acc * n)}/{n} ({acc * 100:.1f}%)  [chance ≈ 16.7%]")
        print_confusion("Top face", gt_top, pred_top, [1, 2, 3, 4, 5, 6])
    if gt_front:
        acc_f = np.mean(np.array(gt_front) == np.array(pred_front))
        print(f"\nFront-face accuracy : {acc_f * 100:.1f}% over {len(gt_front)} labelled frames")
    print("=" * 70)
    if visualize and vis_list:
        save_summary(vis_list, os.path.join(vis_dir, "_summary.jpg"))
        print(f"Visualizations written to {vis_dir} (overview: _summary.jpg)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames_dir", default=os.path.join(PACKAGE_ROOT, "output", "bag_test_frames"))
    parser.add_argument("--weights", default=os.path.join(PACKAGE_ROOT, "weights", "die_mobilenet_v3.pt"))
    parser.add_argument("--conf_thresh", type=float, default=0.55)
    parser.add_argument("--no_vis", action="store_true", help="skip writing visualizations to output/cnn_bag_eval/")
    args = parser.parse_args()
    evaluate(args.frames_dir, args.weights, args.conf_thresh, not args.no_vis)
