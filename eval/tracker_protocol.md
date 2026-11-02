# Step C1: tracker choice and anchor smoothing (pre-registered)

Written on 3 Nov 2026, before the code for this step was written or run. Not changed after results.
It follows the counter clean-up ablation (`eval/cleanup_protocol.md`, `reports/cleanup_ablation.json`),
which cut false crossings a little but left YOLOX precision below 0.80.

## Question

Do trackers built for occlusion (OC-SORT, BoT-SORT) and a smoothed foot point reduce the jitter that
produces false crossings, without losing the recall gain on hidden people?

## What varies (every setting is run and reported)

For each anchor (`bottom_center`, `center`) and each detector (tuned public FRCNN, YOLOX-s tiled,
YOLOX-m tiled), 36 combinations:

| Setting | Values |
|---|---|
| tracker | ByteTrack (current), OC-SORT, BoT-SORT with camera-motion compensation off (static cameras) |
| `smooth_alpha` | 1.0 (off), 0.5, 0.3. Per-track exponential moving average of the anchor point: smoothed = alpha x new + (1 - alpha) x previous |
| `min_frames_on_far_side` (N) | 5, 12, 20, 30 |

All trackers get the detector's score setting `t` as their high-confidence threshold (and as the
activation threshold where the tracker has one; OC-SORT has none), and the same
`minimum_consecutive_frames` chosen in the first ablation. Other tracker settings stay at library
defaults. Counter settings not listed (margin 4 px, cooldown, minimum track age, merge) are fixed at
the values chosen for that detector and anchor in the clean-up ablation.

## How a setting is chosen

On MOT17-02 and MOT17-04 only, the highest pooled event F1. Ties go to the larger N, then the
smaller smooth_alpha, then the tracker in the order ByteTrack, OC-SORT, BoT-SORT. All detectors are
tuned the same way, including the public FRCNN detections. MOT17-09 is never used for choosing.

## What is reported

Every grid for every detector and anchor. The chosen setting on all three sequences and on 09
alone. Person-level bootstrap intervals. The paired recall difference against the tuned public
detections under the same anchor.

## Verdict rule (unchanged from stage 2 in `success_criteria.md`)

A counting gain over the tuned public detections is claimed only if the paired bootstrap interval
for recall excludes zero AND event precision is at least 0.80. Otherwise: no gain, with numbers.
This step may also show that a tracker or smoothing setting improves F1 for every detector; that is
reported as an improvement to the counter, separate from the detector-versus-detector verdict.

## Known limits

Only 106 ground-truth events sit in the tuning sequences. Trackers keep their library defaults
apart from the score thresholds, so a tracker could do better with its own tuning. The smoothing
adds lag, which can shift crossing times. A real gain would still need the held-out clip.
