"""Counting evaluation. Pure Python and NumPy: no OpenCV, no dataset needed for the tests.

Events are CrossingEvent objects. Matching, metrics and definitions follow P2_Build_Plan.md.
"""
import configparser
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from track_counter.counter import LineCounter


def load_mot_gt(path):
    """MOT17 gt.txt -> {frame: [[x1, y1, x2, y2, id, score, class_id], ...]}.

    Columns are frame, id, x, y, w, h, consider flag, class, visibility. Keeps pedestrians (class 1)
    whose consider flag is 1. Low-visibility rows stay in: a person behind an occluder still crosses.
    """
    g = np.loadtxt(path, delimiter=",", ndmin=2)
    g = g[(g[:, 6] == 1) & (g[:, 7] == 1)]
    by_frame = defaultdict(list)
    for f, tid, x, y, w, h, *_ in g:
        by_frame[int(f)].append([x, y, x + w, y + h, tid, 1.0, 0])
    return dict(by_frame)


def gt_events(by_frame, line, anchor="bottom_center", margin_px=0.5):
    """Ground-truth crossings: the same LineCounter with the N-frame rule off.

    MOT boxes sit on whole pixels, so a bottom edge often lands exactly on a line at an integer y.
    With no dead band that tie falls on an arbitrary side and the count moves by several events.
    A half-pixel dead band makes the count independent of how the line is nudged.
    """
    lc = LineCounter(line, anchor, min_frames_on_far_side=1, margin_px=margin_px,
                     max_age_frames=10**9)
    events = []
    for f in sorted(by_frame):
        events += lc.update(f, np.asarray(by_frame[f], dtype=float))
    return events


def match_events(pred, gt, fps, tolerance_s=1.0, position_tolerance=0.1):
    """One-to-one greedy match by smallest time gap, ties by ground-truth order.

    Returns (pairs, unmatched_pred_idx, unmatched_gt_idx). A pair needs the same direction,
    |dt| <= tolerance_s and, when both events have a position, |dpos| <= position_tolerance.
    Greedy, not optimal: the two can differ only when crossings are dense.
    """
    cands = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            if not _compatible(p, g, fps, tolerance_s, position_tolerance):
                continue
            cands.append((abs(p.frame - g.frame), gi, pi))
    cands.sort()
    used_p, used_g, pairs = set(), set(), []
    for _, gi, pi in cands:
        if pi not in used_p and gi not in used_g:
            used_p.add(pi), used_g.add(gi), pairs.append((pi, gi))
    return (pairs,
            [i for i in range(len(pred)) if i not in used_p],
            [i for i in range(len(gt)) if i not in used_g])


def _compatible(p, g, fps, tolerance_s, position_tolerance):
    if p.direction != g.direction or abs(p.frame - g.frame) / fps > tolerance_s:
        return False
    if p.position is None or g.position is None:
        return True
    return abs(p.position - g.position) <= position_tolerance


def evaluate_events(pred, gt, fps, tolerance_s=1.0, position_tolerance=0.1):
    """Event precision, recall, F1, signed count error and double-count rate.

    Count error can be 0 while every event is wrong (false positives and misses cancel), so both
    are reported. When there are no events on either side the scores are 1.0; otherwise a zero
    denominator gives 0.0. count_error_pct is None when the ground-truth count is 0.
    """
    pairs, fp_idx, fn_idx = match_events(pred, gt, fps, tolerance_s, position_tolerance)
    matched_gt = [gt[gi] for _, gi in pairs]
    double = sum(any(_compatible(pred[pi], g, fps, tolerance_s, position_tolerance)
                     for g in matched_gt) for pi in fp_idx)
    tp, fp, fn = len(pairs), len(fp_idx), len(fn_idx)
    if not pred and not gt:
        precision = recall = f1 = 1.0
    else:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out = {"gt_events": len(gt), "pred_events": len(pred), "tp": tp, "fp": fp, "fn": fn,
           "precision": precision, "recall": recall, "f1": f1,
           "double_count": double, "double_count_rate": double / len(pred) if pred else 0.0}
    for name, sel in (("in", "in"), ("out", "out"), ("total", None)):
        n_pred = sum(1 for e in pred if sel is None or e.direction == sel)
        n_gt = sum(1 for e in gt if sel is None or e.direction == sel)
        out[f"count_error_{name}"] = n_pred - n_gt
        out[f"count_error_pct_{name}"] = abs(n_pred - n_gt) / n_gt * 100 if n_gt else None
    return out


def write_mot(path, tracks_by_frame):
    """Write tracker output in MOT format: frame,id,x,y,w,h,conf,-1,-1,-1 (frames are 1-based)."""
    with open(path, "w") as f:
        for frame in sorted(tracks_by_frame):
            f.writelines(f"{frame},{int(tid)},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},"
                        f"{score:.4f},-1,-1,-1\n" for x1, y1, x2, y2, tid, score, *_ in tracks_by_frame[frame])


def events_to_dicts(events):
    """For JSON reports: one dict per event."""
    return [asdict(e) for e in events]


def pool(results):
    """Sum per-line results into one micro-averaged result (rates recomputed from the sums)."""
    keys = ("gt_events", "pred_events", "tp", "fp", "fn", "double_count")
    t = {k: sum(r[k] for r in results) for k in keys}
    p = t["tp"] / (t["tp"] + t["fp"]) if t["tp"] + t["fp"] else 0.0
    r_ = t["tp"] / (t["tp"] + t["fn"]) if t["tp"] + t["fn"] else 0.0
    t.update(precision=p, recall=r_, f1=2 * p * r_ / (p + r_) if p + r_ else 0.0,
             double_count_rate=t["double_count"] / t["pred_events"] if t["pred_events"] else 0.0,
             count_error_total=t["pred_events"] - t["gt_events"])
    t["count_error_pct_total"] = (abs(t["count_error_total"]) / t["gt_events"] * 100
                                  if t["gt_events"] else None)
    return t


def _event_row(e):
    return {"frame": e.frame, "track_id": e.track_id, "direction": e.direction,
            "position": round(e.position, 4)}


def evaluate_sequences(mot_dir, lines_cfg, work_dir, min_conf, counter_params, tracker_params,
                       tolerance_s, position_tolerance, gt_margin_px=0.5):
    """Run detect, track, count on each sequence in lines_cfg and score it against ground truth.

    Counting metrics are the headline. HOTA, IDF1 and MOTA come from trackers.eval (it follows
    TrackEval's matching) and are context only. Returns a JSON-ready dict.
    """
    from trackers.eval import evaluate_mot_sequence

    from track_counter.detectors import MotFileDetector
    from track_counter.pipeline import run_pipeline
    from track_counter.tracking import resolved_params

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    anchor = counter_params.get("anchor", "bottom_center")
    report, all_lines = {}, []
    for seq, line_list in lines_cfg["sequences"].items():
        d = Path(mot_dir) / seq
        ini = configparser.ConfigParser()
        ini.read(d / "seqinfo.ini")
        fps, n = float(ini["Sequence"]["frameRate"]), int(ini["Sequence"]["seqLength"])
        lines = {entry["name"]: entry["line"] for entry in line_list}
        sink = {}
        events, _ = run_pipeline(((i, None) for i in range(1, n + 1)),
                                 MotFileDetector(d / "det" / "det.txt", min_conf), lines, fps,
                                 tracker_params, counter_params, track_sink=sink)
        tracker_file = work / f"{seq}.txt"
        write_mot(tracker_file, sink)
        gt = load_mot_gt(d / "gt" / "gt.txt")
        per_line = {}
        for name, line in lines.items():
            g = gt_events(gt, line, anchor, gt_margin_px)
            res = evaluate_events(events[name], g, fps, tolerance_s, position_tolerance)
            per_line[name] = {"line": line, **res,
                              "gt_event_list": [_event_row(e) for e in g],
                              "pred_event_list": [_event_row(e) for e in events[name]]}
            all_lines.append(res)
        ctx = evaluate_mot_sequence(d / "gt" / "gt.txt", tracker_file,
                                    metrics=["CLEAR", "HOTA", "Identity"])
        report[seq] = {"fps": fps, "frames": n, "lines": per_line,
                       "context_metrics": {"HOTA": ctx.HOTA.HOTA, "IDF1": ctx.Identity.IDF1,
                                           "MOTA": ctx.CLEAR.MOTA, "IDSW": ctx.CLEAR.IDSW}}
    return {
        "settings": {"min_conf": min_conf, "counter": counter_params,
                     "tracker": resolved_params(**tracker_params),
                     "tolerance_s": tolerance_s, "position_tolerance": position_tolerance,
                     "gt_margin_px": gt_margin_px,
                     "context_metric_tool": "trackers.eval (TrackEval-compatible), context only",
                     "detections": "MOT17 public FRCNN detections"},
        "pooled_over_all_lines": pool(all_lines),
        "sequences": report,
    }
