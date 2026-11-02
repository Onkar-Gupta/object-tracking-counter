# Counter and tracker ablation protocol (pre-registered)

Written on 3 Nov 2026, before the ablation was run. Not changed after results.

## Why

The exploratory comparison (`reports/exploratory_counting.json`) showed that YOLOX detections
raise recall on mostly hidden people but give about four times too many crossing events, mostly
near real crossings (wrong direction, position or timing, and double counts). The counter and
tracker settings were made for steadier detections. This ablation asks whether tougher settings
recover precision without losing the recall gain.

## What varies (full grid, every setting is run and reported)

| Setting | Values | Where it acts |
|---|---|---|
| `min_frames_on_far_side` (N) | 3, 5, 8, 12 | counter: frames a person must stay past the line |
| `margin_px` | 4, 8, 12 | counter: dead band either side of the line |
| `minimum_consecutive_frames` | 2, 4 | tracker: frames before a track gets an ID |

24 combinations. The score setting `t` (tracker high split and activation threshold) is fixed at
the value each detector chose in the exploratory run: FRCNN 0.3, YOLOX-s 0.6, YOLOX-m 0.7.

## How a setting is chosen

On MOT17-02 and MOT17-04 only, the setting with the highest pooled event F1. Ties go to the
larger N, then the larger margin. Every detector, including the public FRCNN detections, is tuned
by the same rule on the same grid, so the comparison is between tuned versions.

## What is reported

The whole grid for every detector (not just the winner). The chosen setting run on all three
sequences, with MOT17-09 shown on its own because it was never used for choosing. Person-level
bootstrap intervals. The paired recall difference against the tuned public FRCNN detections.

## Verdict rule (same as stage 2 in `success_criteria.md`)

A counting gain over the tuned public detections is claimed only if the paired bootstrap interval
for the recall difference excludes zero AND event precision is at least 0.80. Otherwise it is
reported as no gain, with the numbers.

## Known limits

Only 106 ground-truth events sit in the tuning sequences. The grid is small and `t` is not
re-tuned jointly. A sustained gain would still need the held-out clip.
