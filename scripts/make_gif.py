"""Turn the annotated video from `track-counter run --save-video` into a small GIF.

  python scripts/make_gif.py --video runs/demo.mp4 --out docs/demo.gif --width 720 --fps 10

Keeps every (source fps / --fps)th frame, shrinks to --width, and uses an adaptive palette.
"""
import argparse
from pathlib import Path

import cv2
from PIL import Image

from track_counter.cli import open_video, video_frames


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--colours", type=int, default=128)
    args = ap.parse_args()
    cap, src_fps, _ = open_video(args.video)
    step = max(1, round(src_fps / args.fps))
    frames = []
    for i, image in video_frames(cap):
        if (i - 1) % step == 0:
            h, w = image.shape[:2]
            small = cv2.resize(image, (args.width, round(h * args.width / w)), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                          .quantize(args.colours, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=round(1000 * step / src_fps),
                   loop=0, optimize=True)
    print(f"{len(frames)} frames, {Path(args.out).stat().st_size / 1e6:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
