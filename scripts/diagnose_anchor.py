"""Step A diagnostic (eval/step_a_protocol.md): foot-point error against ground-truth visibility,
and bottom-centre versus box-centre anchor. Uses the settings chosen in reports/ablation.json.

  python scripts/diagnose_anchor.py --mot-dir D:/Datasets/MOT17/MOT17/train \\
      --detector "public FRCNN=D:/Datasets/MOT17/MOT17/train" \\
      --detector "YOLOX-s tiled=D:/Datasets/detections/yolox_s_tiled" --out reports/anchor_diagnostic.json
"""
import argparse
import configparser
import json
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from track_counter.detectors import MotFileDetector
from track_counter.evaluate import (
    evaluate_sequences,
    event_visibility,
    gt_events,
    load_mot_gt,
    match_events,
)
from track_counter.pipeline import run_pipeline

BINS = (("under 0.5", 0.0, 0.5), ("0.5 to 0.7", 0.5, 0.7), ("0.7 or more", 0.7, 1.01))


def quartiles(values):
    return None if not len(values) else [float(x) for x in np.percentile(values, [25, 50, 75])]


def fmt(q):
    return "-" if q is None else f"{q[1]:+.3f} [{q[0]:+.3f},{q[2]:+.3f}]"


def matched_rows(args, ev, lines_cfg, det_dir, counter, tracker):
    """One dict per correctly matched crossing: visibility, time, position and foot errors."""
    rows = []
    for seq, line_list in tqdm(lines_cfg["sequences"].items(), desc="matched crossings"):
        d = Path(args.mot_dir) / seq
        ini = configparser.ConfigParser()
        ini.read(d / "seqinfo.ini")
        fps, n = float(ini["Sequence"]["frameRate"]), int(ini["Sequence"]["seqLength"])
        lines = {e["name"]: e["line"] for e in line_list}
        sink = {}
        det = MotFileDetector(Path(det_dir) / seq / "det" / "det.txt", ev["min_conf"])
        events, _ = run_pipeline(((i, None) for i in range(1, n + 1)), det, lines, fps, tracker,
                                 counter, track_sink=sink)
        gt = load_mot_gt(d / "gt" / "gt.txt")
        for lname, line in lines.items():
            g = gt_events(gt, line, counter["anchor"], ev["gt_margin_px"])
            pairs, _, _ = match_events(events[lname], g, fps, ev["tolerance_s"],
                                       ev["position_tolerance"])
            for pi, gi in pairs:
                p, ge = events[lname][pi], g[gi]
                trk = next((t for t in sink.get(p.frame, []) if int(t[4]) == p.track_id), None)
                gtr = next((r for r in gt.get(p.frame, []) if int(r[4]) == ge.track_id), None)
                row = {"vis": event_visibility(gt, ge), "dt_s": (p.frame - ge.frame) / fps,
                       "dpos": abs(p.position - ge.position)}
                if trk is not None and gtr is not None:
                    gh = gtr[3] - gtr[1]
                    row["foot_offset"] = (trk[3] - gtr[3]) / gh
                    row["height_ratio"] = (trk[3] - trk[1]) / gh
                rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mot-dir", required=True)
    ap.add_argument("--detector", action="append", required=True)
    ap.add_argument("--ablation", default="reports/ablation.json")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--lines", default="eval/lines.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    ev, lines_cfg = cfg["evaluation"], yaml.safe_load(Path(args.lines).read_text())
    base = {k: v for k, v in cfg["counter"].items() if k != "line"}
    abl = json.loads(Path(args.ablation).read_text())["detectors"]
    out = {}
    for spec in args.detector:
        name, det_dir = spec.split("=", 1)
        c, t = abl[name]["chosen"], abl[name]["t"]
        counter = {**base, "min_frames_on_far_side": c["n"], "margin_px": c["margin_px"]}
        tracker = {"high_conf_det_threshold": t, "track_activation_threshold": t,
                   "minimum_consecutive_frames": c["min_consecutive"]}
        rows = matched_rows(args, ev, lines_cfg, det_dir, counter, tracker)
        table = {}
        for label, lo, hi in BINS:
            sel = [r for r in rows if lo <= r["vis"] < hi]
            table[label] = {
                "matched": len(sel),
                "dt_s_quartiles": quartiles([r["dt_s"] for r in sel]),
                "abs_dt_s_quartiles": quartiles([abs(r["dt_s"]) for r in sel]),
                "dpos_quartiles": quartiles([r["dpos"] for r in sel]),
                "foot_offset_quartiles": quartiles([r["foot_offset"] for r in sel if "foot_offset" in r]),
                "height_ratio_quartiles": quartiles([r["height_ratio"] for r in sel if "height_ratio" in r]),
            }
        anchors = {}
        for anchor in ("bottom_center", "center"):
            r = evaluate_sequences(args.mot_dir, lines_cfg, "runs/latest/mot", ev["min_conf"],
                                   {**counter, "anchor": anchor}, tracker, ev["tolerance_s"],
                                   ev["position_tolerance"], ev["gt_margin_px"], det_dir, False)
            p = r["pooled_over_all_lines"]
            anchors[anchor] = {
                "gt_events": p["gt_events"], "pred_events": p["pred_events"],
                "precision": p["precision"], "recall": p["recall"], "f1": p["f1"],
                "recall_visibility_lt_0.5":
                    r["confidence_intervals_95"]["recall_visibility_lt_0.5"]["estimate"]}
        out[name] = {"settings": {"t": t, **c}, "by_visibility": table, "anchor_comparison": anchors}
    Path(args.out).write_text(json.dumps(out, indent=2))
    for name, d in out.items():
        print(f"\n{name}  settings {d['settings']}")
        print(f"  {'visibility':12s} {'matched':>7s} | time error s: median [Q1,Q3]    | "
              f"foot offset (x GT height)     | height ratio")
        for label, v in d["by_visibility"].items():
            print(f"  {label:12s} {v['matched']:7d} | {fmt(v['dt_s_quartiles']):30s} | "
                  f"{fmt(v['foot_offset_quartiles']):28s} | {fmt(v['height_ratio_quartiles'])}")
        for a, v in d["anchor_comparison"].items():
            print(f"  anchor {a:13s}: GT {v['gt_events']:3d} pred {v['pred_events']:3d}  "
                  f"P {v['precision']:.3f} R {v['recall']:.3f} F1 {v['f1']:.3f}  "
                  f"recall(vis<0.5) {v['recall_visibility_lt_0.5']:.3f}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
