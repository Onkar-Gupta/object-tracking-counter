# Step B: counter clean-up ablation (pre-registered)

Written on 3 Nov 2026, before the clean-up features were built or run. Not changed after results.
It follows the foot-point diagnostic (`eval/step_a_protocol.md`, `reports/anchor_diagnostic.json`),
which found jitter, not a fixed foot-point shift, and a steadier box-centre anchor that scored better.

## What is added to the counter (all off by default, so earlier results are unchanged)

| Setting | Meaning |
|---|---|
| `cooldown_frames` | the same track cannot emit another event for this many frames after one |
| `min_track_frames` | a track must have been followed this many frames before it may emit an event |
| `merge_frames`, `merge_px` | drop an event if another event in the same direction was emitted on the same line within `merge_frames` frames and within `merge_px` pixels along the line (merges duplicates from different IDs) |

## Grid (every setting is run and reported)

For each anchor (`bottom_center`, `center`) and each detector (tuned public FRCNN, YOLOX-s tiled,
YOLOX-m tiled), 32 combinations:

| Setting | Values |
|---|---|
| `min_frames_on_far_side` (N) | 5, 12, 20, 30 |
| `cooldown_frames` | 0, 30 |
| merge | off (0 frames), on (15 frames, 60 px) |
| `min_track_frames` | 0, 10 |

Fixed: margin 4 px, the score setting `t` from the exploratory run (FRCNN 0.3, YOLOX-s 0.6,
YOLOX-m 0.7) and the tracker's `minimum_consecutive_frames` from the first ablation (FRCNN 2,
YOLOX-s 2, YOLOX-m 4).

## The two anchors are two tasks

Ground truth is rebuilt with the same anchor, so bottom-centre (feet) and centre (body) crossings
are different questions with different numbers of true events. Each anchor is tuned and judged on
its own, and a detector is only compared with the public detections under the same anchor. The
headline definition stays bottom-centre unless the project owner decides otherwise after reading
both.

## How a setting is chosen

On MOT17-02 and MOT17-04 only, the highest pooled event F1. Ties go to the larger N, then the
longer cooldown, then merging on, then the larger minimum track age. All detectors are tuned the
same way, including the public FRCNN detections. MOT17-09 is never used for choosing.

## What is reported

Every grid for every detector and anchor. The chosen setting on all three sequences and on 09
alone. Person-level bootstrap intervals. The paired recall difference against the tuned public
detections under the same anchor.

## Verdict rule (unchanged from stage 2 in `success_criteria.md`)

A counting gain over the tuned public detections is claimed only if the paired bootstrap interval
for recall excludes zero AND event precision is at least 0.80. Otherwise: no gain, with numbers.

## Known limits

Only 106 ground-truth events sit in the tuning sequences (fewer under the centre anchor). Each
detector's score setting is not re-tuned jointly. The merge distance and cooldown have two values
each, so this is a coarse search. A real gain would still need the held-out clip.
