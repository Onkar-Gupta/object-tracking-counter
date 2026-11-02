"""Measure camera motion in each MOT17 training sequence, from the images only.

For sampled frames, match ORB features against frame 1 and fit a similarity transform with
RANSAC. Walking people are outliers, so the fit follows the background. We report the largest
background shift as a percentage of image width. A sequence passes if that is below --max-shift-pct.

    python scripts/check_static_camera.py --dir D:/Datasets/MOT17 --out reports/static_camera.json
"""
import argparse
import json
import pathlib

import cv2
import numpy as np


def background_shift_pct(frames):
    """Largest background shift versus the first frame, in percent of image width."""
    orb = cv2.ORB_create(2000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    kp0, des0 = orb.detectAndCompute(frames[0], None)
    worst = 0.0
    for img in frames[1:]:
        kp, des = orb.detectAndCompute(img, None)
        matches = matcher.match(des0, des)
        if len(matches) < 20:
            return None  # too few features to judge
        src = np.float32([kp0[m.queryIdx].pt for m in matches])
        dst = np.float32([kp[m.trainIdx].pt for m in matches])
        m, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if m is None:
            return None
        h, w = img.shape
        corners = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        moved = corners @ m[:, :2].T + m[:, 2]
        worst = max(worst, float(np.abs(moved - corners).max()) / w * 100)
    return worst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--out", default="reports/static_camera.json")
    p.add_argument("--step", type=int, default=25, help="sample every Nth frame")
    p.add_argument("--max-shift-pct", type=float, default=0.5)
    args = p.parse_args()

    result = {}
    for seq in sorted(pathlib.Path(args.dir).glob("MOT17/train/MOT17-*-FRCNN")):
        files = sorted((seq / "img1").glob("*.jpg"))[:: args.step]
        frames = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in files]
        shift = background_shift_pct(frames)
        result[seq.name] = {
            "frames_sampled": len(frames),
            "max_background_shift_pct": None if shift is None else round(shift, 3),
            "static": shift is not None and shift < args.max_shift_pct,
        }
        print(seq.name, result[seq.name])
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"max_shift_pct_threshold": args.max_shift_pct, "sequences": result},
                              indent=2))


if __name__ == "__main__":
    main()
