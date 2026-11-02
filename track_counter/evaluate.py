"""Counting evaluation. Pure Python and NumPy: no OpenCV, no dataset needed for the tests.

Events are CrossingEvent objects. Matching, metrics and definitions follow P2_Build_Plan.md.
"""
import configparser
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from track_counter.counter import LineCounter


def load_mot_gt(path):
    """MOT17 gt.txt -> {frame: [[x1, y1, x2, y2, id, score, class_id], ...]}.

    Columns are frame, id, x, y, w, h, consider flag, class, visibility. The visibility (0 to 1)
    is returned in the score slot. Keeps pedestrians (class 1)
    whose consider flag is 1. Low-visibility rows stay in: a person behind an occluder still crosses.
    """
    g = np.loadtxt(path, delimiter=",", ndmin=2)
    g = g[(g[:, 6] == 1) & (g[:, 7] == 1)]
    by_frame = defaultdict(list)
    for f, tid, x, y, w, h, _, _, vis in g:  # visibility rides in the score slot
        by_frame[int(f)].append([x, y, x + w, y + h, tid, vis, 0])
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


VIS_BINS = ((0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01))


def event_visibility(by_frame, event, half_window=5):
    """Mean GT visibility of the crossing person over +/- half_window frames of the crossing."""
    vals = [row[5] for f in range(event.frame - half_window, event.frame + half_window + 1)
            for row in by_frame.get(f, []) if int(row[4]) == event.track_id]
    return float(np.mean(vals)) if vals else float("nan")


def recall_by_visibility(rows):
    """rows: dicts with 'visibility' and 'matched'. Recall per visibility bin of the crossing person."""
    out = {}
    for lo, hi in VIS_BINS:
        sel = [r for r in rows if lo <= r["visibility"] < hi]
        found = sum(r["matched"] for r in sel)
        out[f"{lo:.1f}-{min(hi, 1.0):.1f}"] = {"events": len(sel), "found": found,
                                              "recall": found / len(sel) if sel else None}
    return out


def cluster_bootstrap_ci(clusters, n_boot=2000, seed=0):
    """95% percentile interval for sum(hits) / sum(total), resampling whole clusters.

    clusters: {key: (hits, total)}. A cluster is one person, so events by the same person
    stay together. Fixed seed, so the report is reproducible. None when there is no data.
    """
    arr = np.array(list(clusters.values()), dtype=float).reshape(-1, 2)
    if arr[:, 1].sum() == 0:
        return None
    idx = np.random.default_rng(seed).integers(0, len(arr), size=(n_boot, len(arr)))
    sums = arr[idx].sum(axis=1)
    ok = sums[:, 1] > 0
    r = sums[ok, 0] / sums[ok, 1]
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def _ci_entry(items, key_of, hit_of):
    """items -> {"estimate", "ci", "n"} with clusters from key_of and hits from hit_of."""
    clusters = defaultdict(lambda: [0, 0])
    for it in items:
        c = clusters[key_of(it)]
        c[0] += hit_of(it)
        c[1] += 1
    hits = sum(c[0] for c in clusters.values())
    total = sum(c[1] for c in clusters.values())
    return {"estimate": hits / total if total else None,
            "ci": cluster_bootstrap_ci(clusters), "events": total, "clusters": len(clusters)}


def false_positive_cause(p, gt, matched_gt, fps, tolerance_s, position_tolerance,
                         gt_person_under_track):
    """Why an unmatched predicted event is a false positive. See the categories below."""
    for gi in matched_gt:
        if _compatible(p, gt[gi], fps, tolerance_s, position_tolerance):
            return "double count"
    near = [g for g in gt if abs(p.frame - g.frame) / fps <= tolerance_s]
    if any(g.direction == p.direction for g in near):
        return "position off"
    if near:
        return "wrong direction"
    if any(g.direction == p.direction and abs(p.frame - g.frame) / fps <= 3 * tolerance_s
           for g in gt):
        return "timing off"
    if not gt_person_under_track:
        return "no ground-truth person under the track"
    return "person without a ground-truth crossing here"


def _person_under_track(by_frame, tracks_by_frame, event, min_iou=0.5):
    """True if a ground-truth pedestrian overlaps the event's track box at the event frame."""
    box = next((t[:4] for t in tracks_by_frame.get(event.frame, []) if int(t[4]) == event.track_id),
               None)
    if box is None:
        return False
    for r in by_frame.get(event.frame, []):
        iw = min(box[2], r[2]) - max(box[0], r[0])
        ih = min(box[3], r[3]) - max(box[1], r[1])
        inter = max(iw, 0) * max(ih, 0)
        union = ((box[2] - box[0]) * (box[3] - box[1]) + (r[2] - r[0]) * (r[3] - r[1]) - inter)
        if union > 0 and inter / union >= min_iou:
            return True
    return False


def paired_recall_difference(report_new, report_base, n_boot=2000, seed=0):
    """Event recall of new minus base on the same ground truth, with a person-level paired
    bootstrap interval. Both reports must come from the same ground-truth events."""
    clusters = defaultdict(lambda: [0, 0])
    for seq, v in report_new["sequences"].items():
        for name, line in v["lines"].items():
            base_line = report_base["sequences"][seq]["lines"][name]
            new_rows, base_rows = line["gt_event_list"], base_line["gt_event_list"]
            if [(r["frame"], r["track_id"]) for r in new_rows] != [
                    (r["frame"], r["track_id"]) for r in base_rows]:
                raise ValueError("reports use different ground-truth events")
            for a, b in zip(new_rows, base_rows, strict=True):
                c = clusters[f"{seq}:{a['track_id']}"]
                c[0] += int(a["matched"]) - int(b["matched"])
                c[1] += 1
    hits = sum(c[0] for c in clusters.values())
    total = sum(c[1] for c in clusters.values())
    return {"difference": hits / total if total else None,
            "ci": cluster_bootstrap_ci(clusters, n_boot, seed), "events": total}


def _margin_row(seq_data, margin, anchor, tolerance_s, position_tolerance):
    """Pooled counting metrics when ground truth is rebuilt with dead band `margin`."""
    results, vis = [], []
    for gt, lines, events, fps in seq_data:
        for name, line in lines.items():
            g = gt_events(gt, line, anchor, margin)
            results.append(evaluate_events(events[name], g, fps, tolerance_s, position_tolerance))
            pairs, _, _ = match_events(events[name], g, fps, tolerance_s, position_tolerance)
            found = {gi for _, gi in pairs}
            vis += [(event_visibility(gt, e), i in found) for i, e in enumerate(g)]
    p = pool(results)
    seen = [m for v, m in vis if v >= 0.7]
    return {"gt_margin_px": margin, "gt_events": p["gt_events"], "precision": p["precision"],
            "recall": p["recall"], "f1": p["f1"],
            "recall_visibility_ge_0.7": sum(seen) / len(seen) if seen else None}


def _event_row(e):
    return {"frame": e.frame, "track_id": e.track_id, "direction": e.direction,
            "position": round(e.position, 4)}


def evaluate_sequences(mot_dir, lines_cfg, work_dir, min_conf, counter_params, tracker_params,
                       tolerance_s, position_tolerance, gt_margin_px=0.5, dets_dir=None,
                       context_metrics=True):
    """Run detect, track, count on each sequence in lines_cfg and score it against ground truth.

    Counting metrics are the headline. HOTA, IDF1 and MOTA come from trackers.eval (it follows
    TrackEval's matching) and are context only. Returns a JSON-ready dict.

    dets_dir: where <sequence>/det/det.txt lives (default: mot_dir, the public detections).
    context_metrics=False skips HOTA, IDF1 and MOTA, which is faster for threshold searches.
    """
    from trackers.eval import evaluate_mot_sequence

    from track_counter.detectors import MotFileDetector
    from track_counter.pipeline import run_pipeline
    from track_counter.tracking import resolved_params

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    anchor = counter_params.get("anchor", "bottom_center")
    report, all_lines, vis_rows = {}, [], []
    recall_items, precision_items, line_data, fp_causes = [], [], [], Counter()
    seq_data = []
    for seq, line_list in lines_cfg["sequences"].items():
        d = Path(mot_dir) / seq
        ini = configparser.ConfigParser()
        ini.read(d / "seqinfo.ini")
        fps, n = float(ini["Sequence"]["frameRate"]), int(ini["Sequence"]["seqLength"])
        lines = {entry["name"]: entry["line"] for entry in line_list}
        sink = {}
        events, _ = run_pipeline(((i, None) for i in range(1, n + 1)),
                                 MotFileDetector(Path(dets_dir or mot_dir) / seq / "det" / "det.txt", min_conf), lines, fps,
                                 tracker_params, counter_params, track_sink=sink)
        tracker_file = work / f"{seq}.txt"
        if context_metrics:
            write_mot(tracker_file, sink)
        gt = load_mot_gt(d / "gt" / "gt.txt")
        seq_data.append((gt, lines, events, fps))
        per_line = {}
        for name, line in lines.items():
            g = gt_events(gt, line, anchor, gt_margin_px)
            res = evaluate_events(events[name], g, fps, tolerance_s, position_tolerance)
            pairs, _, _ = match_events(events[name], g, fps, tolerance_s, position_tolerance)
            found = {gi for _, gi in pairs}
            gt_rows = [{**_event_row(e), "visibility": round(event_visibility(gt, e), 3),
                        "matched": i in found} for i, e in enumerate(g)]
            vis_rows += gt_rows
            recall_items += [(f"{seq}:{e.track_id}", r["matched"], r["visibility"])
                             for e, r in zip(g, gt_rows, strict=True)]
            matched_pred = {pi for pi, _ in pairs}
            pred_rows = []
            for i, e in enumerate(events[name]):
                row = {**_event_row(e), "matched": i in matched_pred}
                if i not in matched_pred:
                    row["fp_cause"] = false_positive_cause(
                        e, g, found, fps, tolerance_s, position_tolerance,
                        _person_under_track(gt, sink, e))
                    fp_causes[row["fp_cause"]] += 1
                pred_rows.append(row)
                precision_items.append((f"{seq}:{e.track_id}", row["matched"]))
            line_data.append((events[name], g, fps))
            per_line[name] = {"line": line, **res, "gt_event_list": gt_rows,
                              "pred_event_list": pred_rows}
            all_lines.append(res)
        ctx_out = None
        if context_metrics:
            ctx = evaluate_mot_sequence(d / "gt" / "gt.txt", tracker_file,
                                        metrics=["CLEAR", "HOTA", "Identity"])
            ctx_out = {"HOTA": ctx.HOTA.HOTA, "IDF1": ctx.Identity.IDF1,
                       "MOTA": ctx.CLEAR.MOTA, "IDSW": ctx.CLEAR.IDSW}
        report[seq] = {"fps": fps, "frames": n, "lines": per_line, "context_metrics": ctx_out}
    return {
        "settings": {"min_conf": min_conf, "counter": counter_params,
                     "tracker": resolved_params(**tracker_params),
                     "tolerance_s": tolerance_s, "position_tolerance": position_tolerance,
                     "gt_margin_px": gt_margin_px,
                     "context_metric_tool": "trackers.eval (TrackEval-compatible), context only",
                     "detections": "MOT17 public FRCNN detections",
                     "note": "Visibility is the MOT17 annotation of the crossing person. "
                             "Matching uses all ground-truth events, so precision is unaffected."},
        "pooled_over_all_lines": pool(all_lines),
        "recall_by_gt_visibility": recall_by_visibility(vis_rows),
        "confidence_intervals_95": {
            "method": "bootstrap, 2000 resamples, seed 0, resampling people (recall: ground-truth "
                      "person per sequence; precision: tracker ID per sequence)",
            "recall": _ci_entry(recall_items, lambda i: i[0], lambda i: int(i[1])),
            "recall_visibility_ge_0.7": _ci_entry(
                [i for i in recall_items if i[2] >= 0.7], lambda i: i[0], lambda i: int(i[1])),
            "recall_visibility_lt_0.5": _ci_entry(
                [i for i in recall_items if i[2] < 0.5], lambda i: i[0], lambda i: int(i[1])),
            "precision": _ci_entry(precision_items, lambda i: i[0], lambda i: int(i[1])),
        },
        "false_positive_causes": dict(fp_causes),
        "gt_margin_sensitivity": {
            "note": "Same tracker output, ground truth rebuilt with a wider dead band. The primary "
                    "result uses the smallest margin (strictest). A person swaying on the line "
                    "gives many ground-truth crossings at small margins.",
            "rows": [_margin_row(seq_data, m, anchor, tolerance_s, position_tolerance)
                     for m in (gt_margin_px, 2.0, 4.0, 8.0)]},
        "tolerance_sensitivity": [
            {"tolerance_s": ts, "position_tolerance": pt,
             **{k: pool([evaluate_events(p_, g_, f_, ts, pt) for p_, g_, f_ in line_data])[k]
                for k in ("precision", "recall", "f1")}}
            for ts in (0.5, 1.0, 2.0) for pt in (0.05, 0.1, 0.2)],
        "sequences": report,
    }
