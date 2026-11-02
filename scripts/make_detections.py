"""Run an ONNX detector over MOT17 frames and write MOT-format detection files.

  python scripts/make_detections.py --mot-dir D:/Datasets/MOT17/MOT17/train --model models/yolox_s.onnx \
      --out-dir D:/Datasets/detections/yolox_s

Writes <out-dir>/<sequence>/det/det.txt and timing.json. Score floor is low on purpose: choose the
operating threshold afterwards, from the scoring report. Timing here is indicative only; the CPU
benchmark is a separate step.
"""
import argparse
import configparser
import json
import time
from pathlib import Path

import cv2
import numpy as np

from track_counter.detectors import OnnxYoloxDetector

SEQUENCES = ("MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-09-FRCNN")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mot-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--score-floor", type=float, default=0.05)
    ap.add_argument("--sequences", nargs="*", default=SEQUENCES)
    args = ap.parse_args()
    det = OnnxYoloxDetector(args.model, score_floor=args.score_floor)
    timing = {}
    for seq in args.sequences:
        seq_dir = Path(args.mot_dir) / seq
        ini = configparser.ConfigParser()
        ini.read(seq_dir / "seqinfo.ini")
        n = int(ini["Sequence"]["seqLength"])
        lines, ms = [], []
        for f in range(1, n + 1):
            img = cv2.imread(str(seq_dir / "img1" / f"{f:06d}.jpg"))
            t = time.perf_counter()
            d = det(f, img)
            ms.append((time.perf_counter() - t) * 1000)
            lines += [f"{f},-1,{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{s:.4f},-1,-1,-1"
                      for x1, y1, x2, y2, s in d]
        out = Path(args.out_dir) / seq / "det"
        out.mkdir(parents=True, exist_ok=True)
        (out / "det.txt").write_text("\n".join(lines))
        timing[seq] = {"frames": n, "median_ms": float(np.median(ms)), "p90_ms": float(np.percentile(ms, 90))}
        print(seq, timing[seq], flush=True)
    (Path(args.out_dir) / "timing.json").write_text(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
