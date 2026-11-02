"""Cut MOT17 training sequences into native-resolution tiles with COCO-format person boxes, for
fine-tuning YOLOX (eval/finetune_protocol.md). Needs only numpy, OpenCV and tqdm.

Sequences 02, 04 and 09 are the evaluation scenes and are refused here.

  python training/prepare_mot17_tiles.py --mot-dir D:/Datasets/MOT17/MOT17/train --out-dir D:/Datasets/mot17_tiles

MOT17 is CC BY-NC-SA 3.0. The tiles and anything trained on them stay on your machine.
"""
import argparse
import configparser
import json
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

TRAIN_SEQS = ("MOT17-05-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN")
EVAL_SEQS = ("MOT17-02", "MOT17-04", "MOT17-09")
TILE, OVERLAP = 640, 128


def tile_starts(length, tile=TILE, overlap=OVERLAP):
    """Same layout as the inference tiles in track_counter.detectors.tile_starts."""
    if length <= tile:
        return [0]
    n = int(np.ceil((length - tile) / (tile - overlap))) + 1
    return [round(v) for v in np.linspace(0, length - tile, n)]


def load_boxes(gt_path, min_vis):
    """{frame: array of full-body pedestrian boxes x1, y1, x2, y2}: considered pedestrians only."""
    g = np.loadtxt(gt_path, delimiter=",", ndmin=2)
    g = g[(g[:, 6] == 1) & (g[:, 7] == 1) & (g[:, 8] >= min_vis)]
    by_frame = {}
    for f, _, x, y, w, h, *_ in g:
        by_frame.setdefault(int(f), []).append([x, y, x + w, y + h])
    return {f: np.array(b) for f, b in by_frame.items()}


def clip_to_tile(boxes, x0, y0, w, h, min_inside):
    """Clip boxes to the tile and keep those with at least min_inside of their area inside."""
    if len(boxes) == 0:
        return np.zeros((0, 4))
    shifted = boxes - np.array([x0, y0, x0, y0])
    clipped = np.column_stack([np.clip(shifted[:, 0], 0, w), np.clip(shifted[:, 1], 0, h),
                               np.clip(shifted[:, 2], 0, w), np.clip(shifted[:, 3], 0, h)])
    area = (shifted[:, 2] - shifted[:, 0]) * (shifted[:, 3] - shifted[:, 1])
    inside = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])
    keep = (inside >= min_inside * area) & (clipped[:, 2] - clipped[:, 0] >= 4) \
        & (clipped[:, 3] - clipped[:, 1] >= 8)
    return clipped[keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mot-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stride", type=int, default=2, help="keep every Nth frame")
    ap.add_argument("--min-vis", type=float, default=0.1)
    ap.add_argument("--min-inside", type=float, default=0.6)
    ap.add_argument("--empty-fraction", type=float, default=0.1)
    ap.add_argument("--val-tiles", type=int, default=60, help="size of the placeholder validation list")
    ap.add_argument("--max-frames", type=int, default=0, help="per sequence, for a quick test (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert not any(e in s for s in TRAIN_SEQS for e in EVAL_SEQS), "evaluation scenes must not be used"
    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    (out / "train2017").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for seq in TRAIN_SEQS:
        d = Path(args.mot_dir) / seq
        ini = configparser.ConfigParser()
        ini.read(d / "seqinfo.ini")
        W, H, n = (int(ini["Sequence"][k]) for k in ("imWidth", "imHeight", "seqLength"))
        boxes = load_boxes(d / "gt" / "gt.txt", args.min_vis)
        frames = range(1, n + 1, args.stride)
        if args.max_frames:
            frames = list(frames)[:args.max_frames]
        for f in tqdm(frames, desc=seq, unit="frame"):
            img = cv2.imread(str(d / "img1" / f"{f:06d}.jpg"))
            for y0 in tile_starts(H):
                for x0 in tile_starts(W):
                    tile = img[y0:y0 + TILE, x0:x0 + TILE]
                    th, tw = tile.shape[:2]
                    kept = clip_to_tile(boxes.get(f, np.zeros((0, 4))), x0, y0, tw, th, args.min_inside)
                    if len(kept) == 0 and rng.random() > args.empty_fraction:
                        continue
                    name = f"{seq[6:8]}_{f:06d}_{x0}_{y0}.jpg"
                    cv2.imwrite(str(out / "train2017" / name), tile, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    image_id = len(images) + 1
                    images.append({"id": image_id, "file_name": name, "width": tw, "height": th})
                    for x1, y1, x2, y2 in kept:
                        annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": 1,
                                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                            "area": float((x2 - x1) * (y2 - y1)), "iscrowd": 0})
    cats = [{"id": 1, "name": "person"}]
    (out / "annotations" / "instances_train2017.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": cats}))
    # Placeholder validation list: a few training tiles, only because the trainer asks for one.
    # Nothing is selected with it (eval/finetune_protocol.md).
    (out / "val2017").mkdir(exist_ok=True)
    val_ids = set(rng.sample(range(1, len(images) + 1), min(args.val_tiles, len(images))))
    val_images = [im for im in images if im["id"] in val_ids]
    for im in val_images:
        (out / "val2017" / im["file_name"]).write_bytes((out / "train2017" / im["file_name"]).read_bytes())
    (out / "annotations" / "instances_val2017.json").write_text(json.dumps({
        "images": val_images, "annotations": [a for a in annotations if a["image_id"] in val_ids],
        "categories": cats}))
    print(f"{len(images)} tiles, {len(annotations)} boxes -> {out}")


if __name__ == "__main__":
    main()
