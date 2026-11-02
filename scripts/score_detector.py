"""Score a detector on MOT17 sequences before any tracking. Any detector works if it can write
MOT-format detection files (frame,-1,x,y,w,h,score,-1,-1,-1), one det.txt per sequence.

  python scripts/score_detector.py --mot-dir D:/Datasets/MOT17/MOT17/train \\
      --dets-dir <folder with MOT17-02-FRCNN/det/det.txt style layout> --out reports/detector_public_frcnn.json

The public MOT17 detections already have that layout, so --dets-dir can be the MOT17 folder itself.
"""
import argparse
import configparser
import json
from pathlib import Path

from track_counter.detection_eval import (
    load_gt_for_detection,
    pool_rows,
    recall_at_precision,
    score_detections,
    summarise,
)
from track_counter.detectors import load_mot_detections

SEQUENCES = ("MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-09-FRCNN")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mot-dir", required=True, help="MOT17 train folder (ground truth)")
    ap.add_argument("--dets-dir", required=True, help="folder holding <sequence>/det/det.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="detector")
    args = ap.parse_args()
    per_seq = {}
    for seq in SEQUENCES:
        ini = configparser.ConfigParser()
        ini.read(Path(args.mot_dir) / seq / "seqinfo.ini")
        gt = load_gt_for_detection(Path(args.mot_dir) / seq / "gt" / "gt.txt")
        dets = load_mot_detections(Path(args.dets_dir) / seq / "det" / "det.txt", 0.0)
        per_seq[seq] = score_detections(dets, gt, int(ini["Sequence"]["seqLength"]))
    pooled = summarise(pool_rows(list(per_seq.values())))
    key = "recall_vis_0.5-0.7"
    report = {"detector": args.name, "iou_threshold": 0.5,
              "pooled_over_sequences": pooled,
              "recall_at_precision_0.9": {
                  k: recall_at_precision(pooled, k)
                  for k in ("recall", "recall_vis_0.0-0.5", key, "recall_vis_0.7-1.0")},
              "per_sequence": {s: summarise(r) for s, r in per_seq.items()}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
