import argparse
import json
from pathlib import Path

import yaml

from track_counter import __version__
from track_counter.detectors import MotFileDetector, OnnxYoloxDetector
from track_counter.evaluate import evaluate_sequences
from track_counter.pipeline import run_pipeline, write_outputs

DEFAULT_CONFIG = Path(__file__).parent / "default_config.yaml"


def _line(text):
    x1, y1, x2, y2 = (float(v) for v in text.split(","))
    return [[x1, y1], [x2, y2]]


def open_video(path, frame=0):
    """Open a video file (or webcam index). Returns (capture, fps, total frames or 0)."""
    import cv2

    cap = cv2.VideoCapture(int(path) if str(path).isdigit() else str(path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    if frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    return cap, cap.get(cv2.CAP_PROP_FPS) or 0.0, max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0)


def video_frames(cap, limit=0):
    """Yield (frame_idx, BGR image), 1-based like the MOT files. Stops at the end or at limit."""
    i = 0
    while not limit or i < limit:
        ok, image = cap.read()
        if not ok:
            return
        i += 1
        yield i, image


def run(args):
    cfg = yaml.safe_load(Path(args.config or DEFAULT_CONFIG).read_text())
    line = args.line or cfg["counter"]["line"]
    min_conf = args.min_conf if args.min_conf is not None else cfg["detector"]["confidence"]
    if bool(args.video) == bool(args.detections):
        raise SystemExit("Give exactly one of --video or --detections")
    if args.video:
        model = args.model or cfg["detector"]["model"]
        if model == "TBC":
            raise SystemExit("Give --model <yolox .onnx file> or set detector.model in the config")
        cap, video_fps, total = open_video(args.video)
        fps = args.fps or video_fps
        if not fps:
            raise SystemExit("The video has no frame rate; pass --fps")
        detect = OnnxYoloxDetector(model, score_floor=min_conf, device=args.device)
        limit = args.frames or 0
        from tqdm import tqdm
        frames = tqdm(video_frames(cap, limit), total=limit or total or None, unit="frame")
    else:
        if not args.fps:
            raise SystemExit("--fps is required with --detections")
        fps = args.fps
        detect = MotFileDetector(args.detections, min_conf)
        n = args.frames or detect.last_frame()
        frames = ((i, None) for i in range(1, n + 1))  # MOT frames are 1-based
    counter_params = {k: v for k, v in cfg["counter"].items() if k != "line"}
    tracker_params = cfg["tracker"].get("params") or {}
    events, n_frames = run_pipeline(frames, detect, {"line": line}, fps,
                                    tracker_params, counter_params)
    out_dir = args.out or cfg["output"]["dir"]
    summary = write_outputs(out_dir, events, n_frames, fps, {"line": line},
                            {"min_conf": min_conf, "counter": counter_params,
                             "tracker": tracker_params})
    c = summary["lines"]["line"]
    print(f"{n_frames} frames: in {c['in']}, out {c['out']}. Wrote {out_dir}/events.csv and summary.json")


def draw_line(args):
    """Click two points on a video frame; print the --line value. Enter keeps it, r resets, Esc quits."""
    import cv2
    import numpy as np

    cap, _, _ = open_video(args.video, args.frame)
    ok, image = cap.read()
    if not ok:
        raise SystemExit("Cannot read that frame")
    scale = min(1.0, args.width / image.shape[1])  # shrink to fit the screen, map clicks back
    shown = cv2.resize(image, None, fx=scale, fy=scale)
    pts = []
    name = "draw-line: click A then B, Enter to keep, r to reset, Esc to quit"

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))

    cv2.namedWindow(name)
    cv2.setMouseCallback(name, on_click)
    while True:
        canvas = shown.copy()
        for q, label in zip(pts, "AB"):
            cv2.circle(canvas, q, 5, (0, 255, 255), -1)
            cv2.putText(canvas, label, (q[0] + 8, q[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if len(pts) == 2:
            a, b = np.array(pts, float)
            cv2.line(canvas, pts[0], pts[1], (0, 255, 255), 2)
            ab = b - a
            if np.hypot(*ab):  # arrow from the midpoint toward the "in" side (positive to negative)
                n_in = np.array([ab[1], -ab[0]]) / np.hypot(*ab)
                mid = (a + b) / 2
                cv2.arrowedLine(canvas, tuple(int(v) for v in mid), tuple(int(v) for v in mid + 50 * n_in),
                                (0, 200, 0), 3, tipLength=0.3)
                cv2.putText(canvas, "in", tuple(int(v) for v in mid + 60 * n_in), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 200, 0), 2)
        cv2.imshow(name, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("Cancelled")
        if key == ord("r"):
            pts.clear()
        if key in (13, 10) and len(pts) == 2:
            break
    cv2.destroyAllWindows()
    (x1, y1), (x2, y2) = [(x / scale, y / scale) for x, y in pts]
    print(f"--line {x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}")
    print('"in" is a move toward the green arrow; the other direction is "out". Swap A and B to flip them.')


def evaluate(args):
    cfg = yaml.safe_load(Path(args.config or DEFAULT_CONFIG).read_text())
    ev = cfg["evaluation"]
    mot_dir = args.dir or ev["mot17_dir"]
    if mot_dir == "TBC":
        raise SystemExit("Set evaluation.mot17_dir in the config or pass --dir")
    lines_cfg = yaml.safe_load(Path(args.lines or ev["lines_file"]).read_text())
    report = evaluate_sequences(
        mot_dir, lines_cfg, Path(args.work_dir or cfg["output"]["dir"]) / "mot",
        ev["min_conf"], {k: v for k, v in cfg["counter"].items() if k != "line"},
        cfg["tracker"].get("params") or {}, ev["tolerance_s"], ev["position_tolerance"], ev["gt_margin_px"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    p = report["pooled_over_all_lines"]
    print(f"{p['gt_events']} ground-truth events, {p['pred_events']} predicted. "
          f"Precision {p['precision']:.3f}, recall {p['recall']:.3f}, F1 {p['f1']:.3f}. "
          f"Wrote {args.out}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="track-counter", description=__doc__)
    parser.add_argument("--version", action="version", version=f"track-counter {__version__}")
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("run", help="track and count from a video, or from a MOT-format detections file")
    p.add_argument("--video", help="video file or webcam index")
    p.add_argument("--model", help="YOLOX .onnx file for --video (default: detector.model in the config)")
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="gpu needs onnxruntime-directml")
    p.add_argument("--detections", help="MOT det.txt (instead of --video)")
    p.add_argument("--fps", type=float, help="required with --detections; default: the video's own rate")
    p.add_argument("--line", type=_line, help="x1,y1,x2,y2 (default: counter.line in the config)")
    p.add_argument("--config", help="default: the packaged default_config.yaml")
    p.add_argument("--min-conf", type=float, help="score floor (default: detector.confidence)")
    p.add_argument("--frames", type=int, help="stop after this many frames (default: all)")
    p.add_argument("--out", help="default: output.dir in the config")
    d = sub.add_parser("draw-line", help="click two points on a video frame to get the --line value")
    d.add_argument("--video", required=True)
    d.add_argument("--frame", type=int, default=0, help="which frame to show (default: the first)")
    d.add_argument("--width", type=int, default=1280, help="largest window width in pixels")
    e = sub.add_parser("evaluate", help="score counting on MOT17 sequences against ground truth")
    e.add_argument("--dir", help="MOT17 train folder (default: evaluation.mot17_dir)")
    e.add_argument("--lines", help="default: evaluation.lines_file in the config")
    e.add_argument("--config", help="default: the packaged default_config.yaml")
    e.add_argument("--out", default="reports/eval.json")
    e.add_argument("--work-dir", help="where MOT tracker files go (default: output.dir)")
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args)
    elif args.command == "draw-line":
        draw_line(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        parser.print_help()
