"""Detector scoring on hand-made boxes. No dataset, no model."""
import numpy as np

from track_counter.detection_eval import (
    iou_matrix,
    load_gt_for_detection,
    pool_rows,
    recall_at_precision,
    score_detections,
    summarise,
)


def gt_file(tmp_path, rows):
    p = tmp_path / "gt.txt"
    p.write_text("\n".join(",".join(str(v) for v in r) for r in rows))
    return load_gt_for_detection(p)


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]])
    assert iou_matrix(a, np.array([[0, 0, 10, 10]]))[0, 0] == 1.0
    assert iou_matrix(a, np.array([[20, 20, 30, 30]]))[0, 0] == 0.0
    assert iou_matrix(np.zeros((0, 4)), a).shape == (0, 1)


def test_recall_by_visibility_and_ignored_regions(tmp_path):
    # frame 1: a visible person, a hidden person, and a distractor (class 8). Rows: frame,id,x,y,w,h,flag,class,vis
    gt = gt_file(tmp_path, [(1, 1, 0, 0, 10, 20, 1, 1, 0.9), (1, 2, 100, 0, 10, 20, 1, 1, 0.1),
                            (1, 3, 200, 0, 10, 20, 1, 8, 1.0)])
    dets = {1: np.array([[0, 0, 10, 20, 0.9],        # finds the visible person
                         [200, 0, 210, 20, 0.8],     # sits on the distractor: dropped, not a false positive
                         [300, 0, 310, 20, 0.7]])}   # nothing there: false positive
    rows = score_detections(dets, gt, 1)
    s = summarise([r for r in rows if r["score_threshold"] == 0.5])[0]
    assert s["precision"] == 0.5                      # 1 TP, 1 FP
    assert s["recall_vis_0.7-1.0"] == 1.0 and s["recall_vis_0.0-0.5"] == 0.0
    assert s["recall"] == 0.5


def test_score_threshold_changes_precision(tmp_path):
    gt = gt_file(tmp_path, [(1, 1, 0, 0, 10, 20, 1, 1, 1.0)])
    dets = {1: np.array([[0, 0, 10, 20, 0.9], [300, 0, 310, 20, 0.3]])}
    rows = summarise(score_detections(dets, gt, 1))
    by_t = {r["score_threshold"]: r for r in rows}
    assert by_t[0.1]["precision"] == 0.5 and by_t[0.5]["precision"] == 1.0
    assert recall_at_precision(rows, "recall", 0.9) == 1.0


def test_pooling_sums_counts_not_ratios(tmp_path):
    gt_a = gt_file(tmp_path, [(1, 1, 0, 0, 10, 20, 1, 1, 1.0)])
    dets_hit = {1: np.array([[0, 0, 10, 20, 0.9]])}
    big = gt_file(tmp_path, [(1, i, i * 20, 0, 10, 20, 1, 1, 1.0) for i in range(9)])
    pooled = summarise(pool_rows([score_detections(dets_hit, gt_a, 1), score_detections({}, big, 1)]))
    assert abs(pooled[4]["recall"] - 1 / 10) < 1e-9   # 1 found of 10, not the mean of 1.0 and 0.0
