"""CPU speed of the full chain (decode, detect, track, count) on a real video, one model at a time.

  python scripts/benchmark_cpu.py --video clip.mp4 --model models/yolox_s.onnx --model models/yolox_m.onnx \\
      --line 0,650,1920,650 --out reports/cpu_benchmark.json

Method: CPU provider only, single pass at 640 x 640 (the frame is shrunk to fit). The first --warmup
frames are run but not timed. Each timed frame records decode, detect, track and count time with
time.perf_counter. fps is 1000 / median(detect + track + count); decoding is reported on its own
because a camera or a hardware decoder would replace it. Plug the machine in and close other
programs first. Numbers are for this machine only.
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from track_counter.cli import open_video, video_frames
from track_counter.counter import LineCounter
from track_counter.detectors import OnnxYoloxDetector
from track_counter.tracking import Tracker


def ms_stats(values):
    return {"median_ms": float(np.median(values)), "p90_ms": float(np.percentile(values, 90))}


def bench(model, video, line, frames, warmup, score_floor):
    cap, fps, _ = open_video(video)
    detect = OnnxYoloxDetector(model, score_floor=score_floor, device="cpu")
    tracker, counter = Tracker(fps), LineCounter(line)
    rows = {"decode": [], "detect": [], "track": [], "count": []}
    t = time.perf_counter()
    for i, image in tqdm(video_frames(cap, frames + warmup), total=frames + warmup,
                         desc=Path(model).stem, unit="frame"):
        t1 = time.perf_counter()
        dets = detect(i, image)
        t2 = time.perf_counter()
        tracks = tracker.update(dets)
        t3 = time.perf_counter()
        counter.update(i, tracks)
        t4 = time.perf_counter()
        if i > warmup:
            for k, v in zip(rows, (t1 - t, t2 - t1, t3 - t2, t4 - t3)):
                rows[k].append(v * 1000)
        t = time.perf_counter()
    total = np.array(rows["detect"]) + rows["track"] + rows["count"]
    return {"model": Path(model).name, "timed_frames": len(total), "frame_size": list(image.shape[1::-1]),
            "fps": float(1000 / np.median(total)), "chain": ms_stats(total),
            **{k: ms_stats(v) for k, v in rows.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--line", required=True, help="x1,y1,x2,y2")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--score-floor", type=float, default=0.4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    x1, y1, x2, y2 = (float(v) for v in args.line.split(","))
    import onnxruntime as ort
    results = [bench(m, args.video, [[x1, y1], [x2, y2]], args.frames, args.warmup, args.score_floor)
               for m in args.model]
    report = {"video": Path(args.video).name, "provider": "CPUExecutionProvider",
              "onnxruntime": ort.__version__, "machine": platform.processor(),
              "logical_cpus": os.cpu_count(), "os": platform.platform(), "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    for r in results:
        print(f"{r['model']:14s} {r['fps']:5.1f} fps  detect {r['detect']['median_ms']:6.1f} ms  "
              f"track {r['track']['median_ms']:4.1f} ms  count {r['count']['median_ms']:4.2f} ms  "
              f"decode {r['decode']['median_ms']:4.1f} ms")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
