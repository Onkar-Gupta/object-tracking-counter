# Step C2: fine-tune YOLOX on other MOT17 scenes (pre-registered)

Written on 3 Nov 2026, before any training code was written or run. Not changed after results.

## Question

A detector trained on everyday photos (COCO) boxes only what it can see, while MOT17's ground truth
boxes the whole body even when mostly hidden. Does fine-tuning on other MOT17 scenes teach the
detector that labelling style, and does that improve counting on scenes it has never seen?

## Data and the train/test wall

- **Training:** MOT17 sequences 05, 10, 11 and 13 (moving cameras), every second frame. Boxes are
  the ground-truth pedestrian boxes with the consider flag set and visibility at least 0.1.
- **Never used for training, for choosing a checkpoint, or for tuning the detector:** sequences
  02, 04 and 09, the three static scenes used for every result so far. They are different scenes
  and cameras from the training ones.
- Training images are native-resolution 640 x 640 tiles cut with the same layout as inference
  (overlap 128), plus a tenth as many empty tiles. A box is kept if at least 60% of it lies inside
  the tile and is then clipped to the tile.
- No checkpoint is selected by validation. The recipe is fixed in advance: 20 epochs, the final
  checkpoint. A small validation list exists only because the trainer asks for one; it is cut from
  training tiles and never used to choose anything.

## Model and recipe

YOLOX-s from the official COCO weights (Apache-2.0), one class (person), 640 input, mixed
precision, batch 16, the YOLOX default schedule (warm-up, cosine decay, mosaic off for the last 3
epochs), seed 0. The detector head for the new class is trained from scratch; everything else
starts from COCO.

## How it is evaluated

Export to ONNX and run it exactly like the pretrained YOLOX-s: tiled (overlap 128) plus a
whole-frame pass, on 02, 04 and 09. Then the same chain of pre-registered steps used for the other
detectors: the exploratory score setting, the counter ablation, the clean-up ablation and the
tracker and smoothing ablation (`eval/*_protocol.md`), each tuned on 02 and 04 only, with 09 held
out. Stage 1 detector scores (`scripts/score_detector.py`) are reported with no pass mark.

## Verdict rules

1. **Against the tuned public FRCNN detections, same anchor:** a counting gain is claimed only if the
   paired bootstrap interval for event recall excludes zero AND event precision is at least 0.80.
2. **Against the pretrained YOLOX-s tiled:** fine-tuning is said to help if event F1 is higher both
   on all three sequences and on the held-out 09. Intervals are reported.

Otherwise the result is reported as no gain, with the numbers.

## Licence note

MOT17 is CC BY-NC-SA 3.0. Weights trained on it inherit that restriction, so they are never
committed or published. Only the training scripts, this protocol and the measured results are
shared. Nothing here is a route to a commercial product.

## Known limits

The fine-tuned detector learns MOT17's labelling style, so a gain shows that the style gap was a
large part of the problem, not that it generalises to other datasets. Only one training run and
one seed. Sequence 05 has a lower resolution than the others. A real test of generalisation would
need the held-out clip.
