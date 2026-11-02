# Step A: do partly hidden people shift the foot point? (pre-registered)

Written on 3 Nov 2026, before the diagnostic was run. A diagnostic only: it gives numbers, not a
pass or fail. Step B (counter clean-up) gets its own protocol once this is read.

## Hypothesis

For partly hidden people a detector's box covers only the visible part, so the bottom-centre foot
point sits higher than the ground-truth foot point. The crossing is then registered late or in the
wrong place, which would explain false events near real crossings.

## What is measured

Detectors and settings: the ones chosen in the ablation (`reports/ablation.json`): tuned public
FRCNN and YOLOX-s tiled. Only correctly matched crossings are used (a predicted event matched to a
ground-truth event). Split by the ground-truth person's visibility at the crossing: under 0.5,
0.5 to 0.7, 0.7 or more.

For each matched pair, at the predicted crossing frame:
- time error: predicted minus ground-truth crossing frame, in seconds (signed and absolute);
- position error along the line, as a fraction of the line length;
- foot offset: (tracker box bottom minus ground-truth box bottom) divided by ground-truth box
  height. Negative means the tracker's foot point is higher than the true one;
- height ratio: tracker box height divided by ground-truth box height.

Reported: count, median and quartiles per visibility bin.

## Anchor comparison

The same two detectors at the same settings with the anchor changed from bottom-centre to box
centre (ground truth rebuilt with the same anchor). Reported: pooled precision, recall, F1 and
recall on crossings by people under 0.5 visibility. This is one comparison, not a search.

## How the hypothesis is read

Supported if, as visibility falls, the foot offset becomes clearly negative and the time and
position errors grow. Not supported if they stay about the same across bins.
