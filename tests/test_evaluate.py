"""Hand-made event lists for the evaluation metrics. No video, no dataset."""
import numpy as np

from track_counter.counter import CrossingEvent
from track_counter.evaluate import (
    evaluate_events,
    evaluate_sequences,
    gt_events,
    load_mot_gt,
    match_events,
    pool,
    write_mot,
)

FPS = 10.0


def ev(frame, direction="in", pos=0.5, tid=1):
    return CrossingEvent(frame, frame, tid, 0, direction, pos)


def test_perfect_match():
    gt = [ev(10), ev(50, "out")]
    r = evaluate_events([ev(12), ev(49, "out")], gt, FPS)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 0, 0)
    assert r["precision"] == r["recall"] == r["f1"] == 1.0
    assert r["count_error_total"] == 0 and r["double_count_rate"] == 0.0


def test_all_missed():
    r = evaluate_events([], [ev(10), ev(50)], FPS)
    assert (r["tp"], r["fp"], r["fn"]) == (0, 0, 2)
    assert r["recall"] == 0.0 and r["f1"] == 0.0
    assert r["count_error_total"] == -2 and r["count_error_pct_total"] == 100.0


def test_false_positives_and_misses_cancel():
    # Two real events missed, two spurious events elsewhere: count error 0, F1 0.
    gt = [ev(10), ev(30)]
    pred = [ev(100), ev(200)]
    r = evaluate_events(pred, gt, FPS)
    assert r["count_error_total"] == 0 and r["count_error_pct_total"] == 0.0
    assert (r["tp"], r["fp"], r["fn"]) == (0, 2, 2)
    assert r["f1"] == 0.0


def test_double_count():
    # One real event, counted twice: the second is a false positive near a matched event.
    r = evaluate_events([ev(10), ev(13, tid=2)], [ev(11)], FPS)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 0)
    assert r["double_count_rate"] == 0.5
    assert r["count_error_total"] == 1


def test_far_false_positive_is_not_a_double_count():
    r = evaluate_events([ev(10), ev(500)], [ev(11)], FPS)
    assert r["fp"] == 1 and r["double_count_rate"] == 0.0


def test_time_tolerance_edges():
    gt = [ev(100)]
    assert len(match_events([ev(110)], gt, FPS, tolerance_s=1.0)[0]) == 1   # exactly 1.0 s
    assert len(match_events([ev(111)], gt, FPS, tolerance_s=1.0)[0]) == 0   # 1.1 s


def test_position_tolerance_edges():
    gt = [ev(100, pos=0.5)]
    assert len(match_events([ev(100, pos=0.55)], gt, FPS, position_tolerance=0.1)[0]) == 1
    assert len(match_events([ev(100, pos=0.7)], gt, FPS, position_tolerance=0.1)[0]) == 0


def test_missing_position_skips_position_check():
    gt = [ev(100, pos=None)]
    assert len(match_events([ev(100, pos=0.9)], gt, FPS)[0]) == 1


def test_direction_mismatch():
    r = evaluate_events([ev(10, "out")], [ev(10, "in")], FPS)
    assert (r["tp"], r["fp"], r["fn"]) == (0, 1, 1)
    assert r["count_error_in"] == -1 and r["count_error_out"] == 1 and r["count_error_total"] == 0


def test_one_to_one_closest_wins():
    # Two predictions near one ground-truth event: the closer one is matched.
    pairs, fp_idx, _ = match_events([ev(14), ev(11)], [ev(10)], FPS)
    assert pairs == [(1, 0)] and fp_idx == [0]


def test_no_events_at_all_scores_one():
    r = evaluate_events([], [], FPS)
    assert r["f1"] == 1.0 and r["count_error_pct_total"] is None


def test_count_error_pct_is_none_without_ground_truth():
    r = evaluate_events([ev(10)], [], FPS)
    assert r["count_error_pct_total"] is None and r["precision"] == 0.0


def _gt_file(tmp_path, rows):
    p = tmp_path / "gt.txt"
    p.write_text("\n".join(",".join(str(v) for v in r) for r in rows))
    return p


def test_gt_events_from_mot_file(tmp_path):
    # Pedestrian walks up from y=150 to y=50 (bottom edge). Line at y=100: one "in" crossing.
    # Rows: frame,id,x,y,w,h,consider,class,visibility. Bottom edge = y + h.
    rows = [(f, 1, 100, y - 20, 10, 20, 1, 1, 1.0) for f, y in [(1, 150), (2, 120), (3, 80), (4, 50)]]
    rows.append((2, 2, 300, 100, 10, 20, 0, 1, 1.0))   # consider=0: ignored
    rows.append((2, 3, 300, 100, 10, 20, 1, 2, 1.0))   # class 2 (not a pedestrian): ignored
    by_frame = load_mot_gt(_gt_file(tmp_path, rows))
    events = gt_events(by_frame, [(0, 100), (400, 100)])
    assert [(e.frame, e.direction) for e in events] == [(3, "in")]


def test_write_mot_round_trip(tmp_path):
    p = tmp_path / "pred.txt"
    write_mot(p, {2: [[10, 20, 30, 60, 7, 0.9, 0]], 1: [[0, 0, 5, 5, 3, 0.5, 0]]})
    lines = p.read_text().splitlines()
    assert lines[0].startswith("1,3,") and lines[1] == "2,7,10.00,20.00,20.00,40.00,0.9000,-1,-1,-1"
    arr = np.loadtxt(p, delimiter=",")
    assert arr.shape == (2, 10)


def test_gt_events_do_not_depend_on_ties_with_the_line():
    # Bottom edge sits exactly on y=100 for two frames. Nudging the line by 1e-9 must not change events.
    path = [(1, 150), (2, 100), (3, 100), (4, 50)]
    by_frame = {f: [[100, y - 20, 110, y, 1, 1.0, 0]] for f, y in path}
    a = gt_events(by_frame, [(0, 100), (400, 100)])
    b = gt_events(by_frame, [(0, 100 + 1e-9), (400, 100 + 1e-9)])
    c = gt_events(by_frame, [(0, 100 - 1e-9), (400, 100 - 1e-9)])
    assert [(e.frame, e.direction) for e in a] == [(e.frame, e.direction) for e in b] \
        == [(e.frame, e.direction) for e in c] == [(4, "in")]


def test_pool_sums_counts_and_recomputes_rates():
    a = evaluate_events([ev(10)], [ev(10)], FPS)                  # tp 1
    b = evaluate_events([], [ev(10), ev(30)], FPS)                # fn 2
    p = pool([a, b])
    assert (p["tp"], p["fp"], p["fn"], p["gt_events"], p["pred_events"]) == (1, 0, 2, 3, 1)
    assert p["precision"] == 1.0 and abs(p["recall"] - 1 / 3) < 1e-9
    assert p["count_error_total"] == -2


def test_evaluate_sequences_end_to_end(tmp_path):
    seq = tmp_path / "SEQ"
    n = 40
    (seq / "det").mkdir(parents=True)
    (seq / "gt").mkdir()
    (seq / "seqinfo.ini").write_text(f"[Sequence]\nframeRate=10\nseqLength={n}\n")
    det, gt = [], []
    # Three 20x40 rectangles: one moving up across y=120, one moving down, one still.
    boxes = (lambda i: (40, 160 - 4 * i), lambda i: (140, 4 + 4 * i), lambda i: (240, 100))
    for i in range(n):
        for k, kind in enumerate(boxes):
            x, y = kind(i)
            det.append(f"{i + 1},-1,{x},{y},20,40,0.95,-1,-1,-1")
            gt.append(f"{i + 1},{k + 1},{x},{y},20,40,1,1,1.0")
    (seq / "det" / "det.txt").write_text("\n".join(det))
    (seq / "gt" / "gt.txt").write_text("\n".join(gt))
    cfg = {"sequences": {"SEQ": [{"name": "mid", "line": [[0, 120], [320, 120]]}]}}
    rep = evaluate_sequences(tmp_path, cfg, tmp_path / "work", 0.1,
                             {"min_frames_on_far_side": 3, "margin_px": 4.0}, {}, 1.0, 0.1)
    line = rep["sequences"]["SEQ"]["lines"]["mid"]
    assert (line["gt_events"], line["pred_events"], line["tp"]) == (2, 2, 2)
    assert rep["pooled_over_all_lines"]["f1"] == 1.0
    assert set(rep["sequences"]["SEQ"]["context_metrics"]) == {"HOTA", "IDF1", "MOTA", "IDSW"}
