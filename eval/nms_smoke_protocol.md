# Smoke test: does a higher NMS threshold recover hidden people? (pre-registered)

Written on 3 Nov 2026, before the run. Not changed after results. This is a smoke test, not a claim.

## Idea

Non-maximum suppression (NMS) deletes a box that overlaps a better-scoring box by more than a
threshold (0.45 so far). In a crowd, a hidden person's box overlaps the person in front, so NMS may
delete a real person. A higher threshold keeps more of them, at the cost of more duplicate boxes.

## Test

Detector: official COCO YOLOX-s ONNX, single pass at 640 (the shipped default), score floor 0.05,
on MOT17 02, 04 and 09, run on the GPU. Only the NMS IoU threshold changes: 0.45 (current), 0.60, 0.70.
Each is scored with `scripts/score_detector.py` (detector only, no tracking, IoU 0.5 matching).

## Measure

Recall at precision 0.9 (`recall_at_precision_0.9`, which already charges duplicates against the
setting), overall and for people under 50% visible.

## Decision rule (fixed now)

Worth a full test (counting chain, tracker, held-out 09) only if, against 0.45, a higher threshold gives
(a) under-50%-visible recall at precision 0.9 higher by at least 0.03 absolute, AND (b) overall recall at
precision 0.9 not lower. Otherwise the idea is dropped and the three numbers are reported as they are.

## Limits

Single pass only; tiled detection is not tested. One detector, one run. Detector-level only: counting
is not measured here.
