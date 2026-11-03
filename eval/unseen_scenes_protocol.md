# Unseen scenes: a small hand-labelled check of the shipped default (pre-registered)

Written on 3 Nov 2026, before the run. Not changed after results. This is a sanity check, not a benchmark.

## Why

Every number in the README comes from MOT17, where the settings were tuned on sequences 02 and 04. A real
held-out test needs a new scene with labels made by someone else. No such clip exists yet (nobody has
filmed and labelled one), so this is the most honest smaller version: scenes the tool was never tuned on,
labelled by hand from the video.

## Scenes and lines (fixed now, before any result)

| Scene | Source | Line (pixels) | Note |
|---|---|---|---|
| Busy crosswalk | Mixkit 4000, 1920x1080, 13 s | x = 1100, y 470 to 740 | Chosen before any tracker output. "in" = left to right. |
| Campus steps | Pexels 7969488, 1920x1080, 10 s | y = 930, full width | **Chosen after looking at where the two walkers' feet travel, so it is not a held-out choice.** |
| Corridor | Pexels 11903990, 1920x1080, 20 s | y = 700, full width | Chosen from a still frame only. |
| Quiet Paris street | Mixkit 4348, 1280x720, 22 s | y = 360, full width | Chosen from a still frame only. |

## Method

Shipped defaults, no tuning: `config.yaml`, YOLOX-s single pass, ByteTrack, CPU. Every predicted crossing
is reviewed on a contact sheet (the tracked box and its feet point, a few frames before and after the
event, with the line drawn). A predicted crossing is **correct** if the highlighted box is one real
pedestrian whose feet point is on one side of the line before the event and on the other side after it, in
the direction the tool reported. Otherwise it is **wrong**.

For the three small scenes (campus, corridor, Paris) every true crossing is also counted by watching the
clip, so recall is reported there. For the busy crosswalk only the predicted crossings are reviewed
(precision); recall is **not** measured because people overlap too much to label every crossing reliably.

## Reported

Per scene: predicted crossings, correct, wrong, true crossings (small scenes only), missed. No pass mark.

## Limits

Labels are made by the author of the code (reviewed by eye from the video), not by an independent person.
Tiny samples. The campus line is not a clean held-out choice. Footage is stock video, not surveillance.
