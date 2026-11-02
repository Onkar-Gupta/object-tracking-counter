"""Score a detector on its own, before any tracking or counting. Pure NumPy.

Matches detections to MOT ground truth per frame at IoU >= 0.5, highest score first, and reports
recall split by how visible the person is. Detections that land on ignored ground truth (distractor
classes, persons marked not-to-consider) are dropped rather than counted as false positives,
which follows TrackEval's MOT17 handling.
"""
from collections import defaultdict

import numpy as np

IGNORE_CLASSES = (2, 7, 8, 12)  # person on vehicle, static person, distractor, reflection
VIS_BINS = ((0.0, 0.5), (0.5, 0.7), (0.7, 1.01))
THRESHOLDS = tuple(round(t, 2) for t in np.arange(0.1, 0.95, 0.1))


def load_gt_for_detection(path):
    """-> {frame: (considered boxes Nx4, their visibility N, ignored boxes Mx4)} (xyxy)."""
    g = np.loadtxt(path, delimiter=",", ndmin=2)
    out = defaultdict(lambda: [[], [], []])
    for f, _, x, y, w, h, flag, cls, vis in g:
        box = [x, y, x + w, y + h]
        slot = out[int(f)]
        if flag == 1 and cls == 1:
            slot[0].append(box), slot[1].append(vis)
        elif cls in IGNORE_CLASSES or (flag == 0 and cls == 1):
            slot[2].append(box)
    return {f: (np.array(a).reshape(-1, 4), np.array(b), np.array(c).reshape(-1, 4))
            for f, (a, b, c) in out.items()}


def iou_matrix(a, b):
    """IoU between every box in a (Nx4) and b (Mx4), xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    iw = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    ih = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def _match_frame(dets, gt_boxes, ignore_boxes, threshold, min_iou=0.5):
    """-> (matched gt flags, number of false positives) for detections scoring >= threshold."""
    keep = dets[dets[:, 4] >= threshold]
    keep = keep[np.argsort(-keep[:, 4], kind="stable")]
    matched = np.zeros(len(gt_boxes), bool)
    fp = 0
    ious = iou_matrix(keep[:, :4], gt_boxes)
    ign = iou_matrix(keep[:, :4], ignore_boxes)
    for i in range(len(keep)):
        cand = np.where(~matched & (ious[i] >= min_iou))[0] if ious.shape[1] else []
        if len(cand):
            matched[cand[np.argmax(ious[i, cand])]] = True
        elif ign.shape[1] and ign[i].max() >= min_iou:
            continue
        else:
            fp += 1
    return matched, fp


def score_detections(dets_by_frame, gt, n_frames):
    """dets_by_frame: {frame: Nx5 [x1, y1, x2, y2, score]}; gt from load_gt_for_detection.

    Returns one dict of raw counts per score threshold. Use `summarise` to turn counts into
    precision and recall, and to pool several sequences by summing counts.
    """
    rows = []
    for t in THRESHOLDS:
        row = {"score_threshold": t, "tp": 0, "fp": 0}
        for lo, hi in VIS_BINS:
            row[f"found_{_bin(lo, hi)}"] = row[f"n_{_bin(lo, hi)}"] = 0
        for f in range(1, n_frames + 1):
            boxes, vis, ign = gt.get(f, (np.zeros((0, 4)), np.zeros(0), np.zeros((0, 4))))
            matched, nfp = _match_frame(dets_by_frame.get(f, np.zeros((0, 5))), boxes, ign, t)
            row["tp"] += int(matched.sum())
            row["fp"] += nfp
            for lo, hi in VIS_BINS:
                sel = (vis >= lo) & (vis < hi)
                row[f"found_{_bin(lo, hi)}"] += int(matched[sel].sum())
                row[f"n_{_bin(lo, hi)}"] += int(sel.sum())
        rows.append(row)
    return rows


def _bin(lo, hi):
    return f"vis_{lo:.1f}-{min(hi, 1.0):.1f}"


def pool_rows(list_of_rows):
    """Sum raw counts over sequences, threshold by threshold."""
    return [{k: (rows[0]["score_threshold"] if k == "score_threshold" else sum(r[k] for r in rows))
             for k in rows[0]} for rows in zip(*list_of_rows, strict=True)]


def summarise(rows):
    """Raw counts -> precision, overall recall and recall per visibility bin."""
    out = []
    for r in rows:
        n_all = sum(r[f"n_{_bin(lo, hi)}"] for lo, hi in VIS_BINS)
        s = {"score_threshold": r["score_threshold"],
             "precision": r["tp"] / (r["tp"] + r["fp"]) if r["tp"] + r["fp"] else None,
             "recall": r["tp"] / n_all if n_all else None}
        for lo, hi in VIS_BINS:
            b = _bin(lo, hi)
            s[f"recall_{b}"] = r[f"found_{b}"] / r[f"n_{b}"] if r[f"n_{b}"] else None
        out.append(s)
    return out


def recall_at_precision(summary, key, min_precision=0.9):
    """Highest recall (column `key`) over thresholds whose precision is at least min_precision."""
    ok = [r[key] for r in summary if r["precision"] is not None and r["precision"] >= min_precision
          and r[key] is not None]
    return max(ok) if ok else None
