# Step D: how well does the shipped default count? (pre-registered)

Written on 3 Nov 2026, before the run. Not changed after results.

## Question

Every counting result so far used tiled detection (overlapping 640 px tiles plus a whole-frame pass,
about 9 model runs per frame). The default that `track-counter run` ships is the single pass: the
whole frame shrunk to 640 x 640, one model run, 19 fps on CPU (`reports/cpu_benchmark.json`). How
does that default count on MOT17, next to the tuned public FRCNN detections?

## Data and detector

MOT17 02, 04 and 09 (the same static scenes and the same lines as every other step,
`eval/lines.yaml`). Detections: official COCO YOLOX-s ONNX, single pass at 640, score floor 0.05,
made on 3 Nov 2026 (`D:/Datasets/detections/yolox_s`). Nothing is re-detected for this step.

## Procedure

The same chain as step C2, each stage tuned on 02 and 04 only, with 09 held out:
`compare_detectors.py`, `ablate_counter.py`, `ablate_cleanup.py`, `ablate_trackers.py`, run with two
detectors: the public FRCNN detections (baseline, first) and the YOLOX-s single pass. Outputs are
`reports/sp_*.json`.

## What is reported

Event precision, recall and F1 with person-level bootstrap intervals, recall by visibility, and the
paired recall difference against FRCNN, at the feet anchor (headline) and the box-centre anchor.
The step C rule is applied as before (a counting gain over the tuned FRCNN needs a recall interval
that excludes zero AND precision at least 0.80), and the result is reported whichever way it falls.

## Known limits

One detector, one run. The public FRCNN detections may have been trained on these scenes
(UNVERIFIED), which would favour them. Tiled detection is not re-run, so tiled and single-pass
numbers come from different earlier runs of the same chain.
