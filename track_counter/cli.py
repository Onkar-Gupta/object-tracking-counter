import argparse
import json
from pathlib import Path

import yaml

from track_counter import __version__
from track_counter.detectors import MotFileDetector
from track_counter.evaluate import evaluate_sequences
from track_counter.pipeline import run_pipeline, write_outputs

DEFAULT_CONFIG = Path(__file__).parent / "default_config.yaml"


def _line(text):
    x1, y1, x2, y2 = (float(v) for v in text.split(","))
    return [[x1, y1], [x2, y2]]


def run(args):
    cfg = yaml.safe_load(Path(args.config or DEFAULT_CONFIG).read_text())
    line = args.line or cfg["counter"]["line"]
    min_conf = args.min_conf if args.min_conf is not None else cfg["detector"]["confidence"]
    detect = MotFileDetector(args.detections, min_conf)
    n = args.frames or detect.last_frame()
    frames = ((i, None) for i in range(1, n + 1))  # MOT frames are 1-based
    counter_params = {k: v for k, v in cfg["counter"].items() if k != "line"}
    tracker_params = cfg["tracker"].get("params") or {}
    events, n_frames = run_pipeline(frames, detect, {"line": line}, args.fps,
                                    tracker_params, counter_params)
    out_dir = args.out or cfg["output"]["dir"]
    summary = write_outputs(out_dir, events, n_frames, args.fps, {"line": line},
                            {"min_conf": min_conf, "counter": counter_params,
                             "tracker": tracker_params})
    c = summary["lines"]["line"]
    print(f"{n_frames} frames: in {c['in']}, out {c['out']}. Wrote {out_dir}/events.csv and summary.json")


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
    p = sub.add_parser("run", help="track and count from a MOT-format detections file")
    p.add_argument("--detections", required=True, help="MOT det.txt")
    p.add_argument("--fps", type=float, required=True)
    p.add_argument("--line", type=_line, help="x1,y1,x2,y2 (default: counter.line in the config)")
    p.add_argument("--config", help="default: the packaged default_config.yaml")
    p.add_argument("--min-conf", type=float, help="score floor (default: detector.confidence)")
    p.add_argument("--frames", type=int, help="default: last frame in the detections file")
    p.add_argument("--out", help="default: output.dir in the config")
    e = sub.add_parser("evaluate", help="score counting on MOT17 sequences against ground truth")
    e.add_argument("--dir", help="MOT17 train folder (default: evaluation.mot17_dir)")
    e.add_argument("--lines", help="default: evaluation.lines_file in the config")
    e.add_argument("--config", help="default: the packaged default_config.yaml")
    e.add_argument("--out", default="reports/eval.json")
    e.add_argument("--work-dir", help="where MOT tracker files go (default: output.dir)")
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        parser.print_help()
