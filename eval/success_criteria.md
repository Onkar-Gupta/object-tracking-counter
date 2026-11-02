# Detector success criteria (pre-registered)

Written on 3 Nov 2026 and committed before any detector other than the public MOT17 FRCNN
detections has produced output on these sequences. They are not changed after results.

## What is being tested

Whether a modern, permissively licensed person detector finds more of the people the public
FRCNN detections miss, without losing the people they already find. The tracker, the counter,
the six counting lines and the matching settings stay as they are.

## Baseline (public FRCNN detections, `reports/detector_public_frcnn.json`)

Pooled over MOT17-02, 04 and 09, IoU 0.5, the highest recall at precision of at least 0.9:

| Ground-truth people by visibility | Recall |
|---|---|
| all | 0.502 |
| under 0.5 | 0.022 |
| 0.5 to 0.7 | 0.589 |
| 0.7 or more | 0.987 |

## Stage 1: the detector alone (`scripts/score_detector.py`)

Measured at the best score threshold that keeps precision at or above 0.9, on the same three
sequences. A detector must meet all of:

1. **No loss on clear people:** recall at visibility 0.7 or more is at least 0.95.
2. **A real gain somewhere:** at least one of
   - recall at visibility 0.5 to 0.7 is at least 0.70 (baseline 0.589), or
   - recall at visibility under 0.5 is at least 0.10 (baseline 0.022).
3. **Overall recall** at precision of at least 0.9 is at least 0.55 (baseline 0.502).

A detector that fails stage 1 is not run through the counting pipeline.

## Stage 2: the full counting pipeline (`track-counter evaluate`)

Same tracker defaults, same lines, same ground truth (0.5 px dead band). A gain in counting is
claimed only if both hold:

1. The person-level paired bootstrap interval (2000 resamples, seed 0) for the difference in
   overall event recall (new minus public FRCNN) excludes zero.
2. Event precision stays at or above 0.80.

If the interval includes zero, the result is reported as a tie, not a gain.

## Speed

A detector can become the default only if its median CPU time per frame, measured in step 9 at
the input size used here, gives at least 5 frames per second on the development laptop. A slower
detector that passes stages 1 and 2 may be offered as an optional "accurate" mode.

## Reporting rule

Every detector and input size that is tried is reported, including the ones that fail. The
score threshold for each detector is picked on MOT17-02 and MOT17-04 and then reported on
MOT17-09 as well as pooled, so a threshold is never tuned on the sequence it is judged on alone.
