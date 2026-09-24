"""
Compare run_bag_pose.py runs (results.json) side by side
========================================================
    python3 scripts/compare_bag_runs.py                       # default three runs
    python3 scripts/compare_bag_runs.py output/bag_pose_a output/bag_pose_b

Reports, per run: timing, top-face accuracy, pose repeatability within each
resting position (the die does not move between hand touches, so yaw and
position should be constant there); and, per pair of runs, how often they
agree frame by frame.
"""

import json
import os
import sys

import numpy as np

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_RUNS = ["bag_pose_yoloe_cuda", "bag_pose_yoloe_cpu", "bag_pose_yoloe_faces_cuda", "bag_pose_classical_cpu"]


def load(run_dir):
    with open(os.path.join(run_dir, "results.json")) as f:
        return json.load(f)


def yaw_diff(a, b):
    """Smallest difference between two yaws modulo the cube's 90° symmetry."""
    d = (a - b + 45.0) % 90.0 - 45.0
    return abs(d)


def resting_runs(frames):
    """Consecutive labelled frames with the same GT top face (a skipped /
    occluded frame means the die was touched, so it breaks the run)."""
    runs, cur, prev_gt = [], [], None
    for fr in frames:
        if fr["skip"] or fr["gt_top"] is None:
            if cur:
                runs.append(cur)
            cur, prev_gt = [], None
            continue
        if cur and fr["gt_top"] != prev_gt:
            runs.append(cur)
            cur = []
        cur.append(fr["frame"])
        prev_gt = fr["gt_top"]
    if cur:
        runs.append(cur)
    return [r for r in runs if len(r) >= 2]


def main():
    dirs = sys.argv[1:] or [os.path.join(PACKAGE_ROOT, "output", d) for d in DEFAULT_RUNS]
    runs = [load(d) for d in dirs]
    names = [r["run"] for r in runs]
    by_frame = [{f["frame"]: f for f in r["frames"]} for r in runs]
    segs = resting_runs(runs[0]["frames"])

    col = max(len(n) for n in names) + 2
    print("=" * (34 + col * len(names)))
    print(f"{'':<34}" + "".join(f"{n:>{col}}" for n in names))
    print("-" * (34 + col * len(names)))

    # Timing
    stages = []
    for r in runs:
        for k in r["frames"][0]["times_ms"]:
            if k not in stages:
                stages.append(k)
    for k in stages:
        row = []
        for r in runs:
            v = [f["times_ms"][k] for f in r["frames"] if k in f["times_ms"]]
            row.append(f"{np.median(v):.1f}" if v else "-")
        print(f"{'time ' + k + ' [ms, median]':<34}" + "".join(f"{x:>{col}}" for x in row))
    row = []
    for r in runs:
        v = [f["times_ms"]["total"] for f in r["frames"]]
        row.append(f"{np.percentile(v, 95):.1f}")
    print(f"{'time total [ms, p95]':<34}" + "".join(f"{x:>{col}}" for x in row))
    print(f"{'model load + 1st call [s]':<34}" + "".join(f"{r['load_ms'] / 1e3:>{col}.1f}" for r in runs))
    print(f"{'-> frames per second':<34}" + "".join(
        f"{1000.0 / np.median([f['times_ms']['total'] for f in r['frames']]):>{col}.1f}" for r in runs))

    # Top face
    print("-" * (34 + col * len(names)))
    row_acc, row_none = [], []
    for r in runs:
        s = [f for f in r["frames"] if f["gt_top"] is not None]
        row_acc.append(f"{np.mean([f['top_pred'] == f['gt_top'] for f in s]) * 100:.1f}%")
        row_none.append(str(sum(f["top_pred"] is None for f in s)))
    n_lab = sum(f["gt_top"] is not None for f in runs[0]["frames"])
    print(f"{f'top-face accuracy ({n_lab} frames)':<34}" + "".join(f"{x:>{col}}" for x in row_acc))
    print(f"{'  of which: no answer':<34}" + "".join(f"{x:>{col}}" for x in row_none))
    row_rej_l, row_rej_s = [], []
    for r in runs:
        row_rej_l.append(str(sum(bool(f.get("rejected")) for f in r["frames"] if not f["skip"])))
        row_rej_s.append(str(sum(bool(f.get("rejected")) for f in r["frames"] if f["skip"])))
    print(f"{'rejected by fit check: clean':<34}" + "".join(f"{x:>{col}}" for x in row_rej_l))
    print(f"{'rejected by fit check: occluded':<34}" + "".join(f"{x:>{col}}" for x in row_rej_s))

    # Pose repeatability within resting positions
    print("-" * (34 + col * len(names)))
    row_y, row_ymax, row_p, row_iou = [], [], [], []
    for bf, r in zip(by_frame, runs):
        ys, ps = [], []
        for seg in segs:
            fr = [bf[k] for k in seg if bf[k]["yaw_deg"] is not None and not bf[k].get("rejected")]
            if len(fr) < 2:
                continue
            y = np.array([f["yaw_deg"] for f in fr])
            ref = y[0]
            spread = max(yaw_diff(a, b) for a in y for b in y)
            ys.append(spread)
            P = np.array([f["position_m"] for f in fr])
            ps.append(np.max(np.linalg.norm(P - P.mean(0), axis=1)) * 1000)
        row_y.append(f"{np.median(ys):.1f}")
        row_ymax.append(f"{np.max(ys):.1f}")
        row_p.append(f"{np.median(ps):.1f}")
        ious = [f["iou"] for f in r["frames"] if f["iou"] is not None and not f["skip"]]
        row_iou.append(f"{np.mean(ious):.3f}" if ious else "-")
    print(f"{f'yaw spread in rest [deg, median]':<34}" + "".join(f"{x:>{col}}" for x in row_y))
    print(f"{'yaw spread in rest [deg, max]':<34}" + "".join(f"{x:>{col}}" for x in row_ymax))
    print(f"{'position spread in rest [mm, med]':<34}" + "".join(f"{x:>{col}}" for x in row_p))
    print(f"{'silhouette IoU (unoccluded, mean)':<34}" + "".join(f"{x:>{col}}" for x in row_iou))
    print(f"({len(segs)} resting positions with >= 2 labelled frames)")

    # Pairwise agreement
    print("=" * (34 + col * len(names)))
    print("Pairwise agreement on labelled frames (first vs others):")
    base = by_frame[0]
    for name, bf in zip(names[1:], by_frame[1:]):
        common = [k for k in base if not base[k]["skip"] and base[k]["gt_top"] is not None
                  and base[k]["yaw_deg"] is not None and bf[k]["yaw_deg"] is not None
                  and not base[k].get("rejected") and not bf[k].get("rejected")]
        same_top = np.mean([base[k]["top_pred"] == bf[k]["top_pred"] for k in common]) * 100
        dy = np.array([yaw_diff(base[k]["yaw_deg"], bf[k]["yaw_deg"]) for k in common])
        dp = np.array([np.linalg.norm(np.subtract(base[k]["position_m"], bf[k]["position_m"])) * 1000
                       for k in common])
        diff_top = [k for k in common if base[k]["top_pred"] != bf[k]["top_pred"]]
        print(f"  {names[0]} vs {name}: same top face {same_top:.1f}% | yaw diff median {np.median(dy):.2f}°, "
              f"max {dy.max():.2f}° | position diff median {np.median(dp):.2f} mm, max {dp.max():.2f} mm")
        if diff_top and len(diff_top) <= 5:
            for k in diff_top:
                print(f"      top differs on {k}: {base[k]['top_pred']} vs {bf[k]['top_pred']} (GT {base[k]['gt_top']})")


if __name__ == "__main__":
    main()
