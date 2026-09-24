"""
Extract evenly-spaced RGB frames from a rosbag2 recording
=========================================================
Writes N colour frames, sampled uniformly over the bag's duration, as JPEGs
(named by their bag timestamp) for offline evaluation of the die detector /
CNN orientation classifier:

    python3 scripts/extract_bag_frames.py --num_frames 60
"""

import argparse
import glob
import os

import cv2
import numpy as np
import rosbag2_py
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def extract_frames(bag_dir, topic, num_frames, out_dir):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_dir, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr",
                                    output_serialization_format="cdr"),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))

    # First pass: timestamps only, so we can pick evenly-spaced indices
    stamps = []
    while reader.has_next():
        _, _, t = reader.read_next()
        stamps.append(t)
    if not stamps:
        raise RuntimeError(f"No messages on '{topic}' in {bag_dir}")

    n = min(num_frames, len(stamps))
    wanted = set(np.linspace(0, len(stamps) - 1, n).round().astype(int).tolist())
    print(f"{len(stamps)} messages on {topic}; extracting {len(wanted)} frames -> {out_dir}")

    os.makedirs(out_dir, exist_ok=True)
    reader.seek(stamps[0])
    bridge = CvBridge()
    t0 = stamps[0]
    idx = 0
    while reader.has_next():
        _, data, t = reader.read_next()
        if idx in wanted:
            msg = deserialize_message(data, Image)
            bgr = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            fname = f"frame_{idx:04d}_t{(t - t0) / 1e9:06.2f}s.jpg"
            cv2.imwrite(os.path.join(out_dir, fname), bgr)
        idx += 1


if __name__ == "__main__":
    bags = sorted(glob.glob(os.path.join(PACKAGE_ROOT, "test_bags", "rosbag2_*")))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bag", default=bags[-1] if bags else None, help="rosbag2 directory")
    parser.add_argument("--topic", default="/head_front_camera/color/image_raw")
    parser.add_argument("--num_frames", type=int, default=60)
    parser.add_argument("--out_dir", default=os.path.join(PACKAGE_ROOT, "output", "bag_test_frames"))
    args = parser.parse_args()
    if args.bag is None:
        parser.error("no bag found under test_bags/; pass --bag")
    extract_frames(args.bag, args.topic, args.num_frames, args.out_dir)
