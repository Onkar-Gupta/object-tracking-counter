# Demo footage credit

`demo.gif` is made from "Busy street in the city" (clip 4000), a stock video on Mixkit:
https://mixkit.co/free-stock-video/busy-street-in-the-city-4000/

Licence: Mixkit Stock Video Free License, https://mixkit.co/license/#videoFree
(commercial and non-commercial use, copying, modifying and distribution allowed, attribution not required).

The GIF is a modified version: the video was processed by this project (detection, tracking, line
counting), annotated with boxes, track IDs, the counting line and a running count, resized and
shortened to 8 frames per second. The original clip is not in this repository.

The counts in the GIF are not validated against ground truth. The GIF shows what the tool does, not how
accurate it is. Accuracy is measured on MOT17 (see the README).

Reproduce it (download the clip from the Mixkit page first):

    track-counter run --video 4000-1080.mp4 --model models/yolox_s.onnx --line 1100,470,1100,740 --save-video demo.mp4
    python scripts/make_gif.py --video demo.mp4 --out docs/demo.gif --width 560 --fps 8 --colours 48
