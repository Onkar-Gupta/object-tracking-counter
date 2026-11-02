"""Counting evaluation. Pure Python and NumPy: no OpenCV, no dataset needed for the tests.

Events are CrossingEvent objects. Matching, metrics and definitions follow P2_Build_Plan.md.
"""
from collections import defaultdict
from dataclasses import asdict

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


def gt_events(by_frame, line, anchor="bottom_center"):
    """Ground-truth crossings: the same LineCounter, with the N-frame rule and dead band off."""
    lc = LineCounter(line, anchor, min_frames_on_far_side=1, margin_px=0.0, max_age_frames=10**9)
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
           "double_count_rate": double / len(pred) if pred else 0.0}
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
