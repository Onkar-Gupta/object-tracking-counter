"""Detection sources. Only `mot_file` for now: reads MOT-format detections (public MOT17 det.txt)."""
from collections import defaultdict

import numpy as np

EMPTY = np.zeros((0, 5))


def load_mot_detections(path, min_conf=0.0):
    """MOT det.txt (frame,id,x,y,w,h,score,...) -> {frame: array of [x1, y1, x2, y2, score]}.

    min_conf is a floor on the score. ByteTrack's second stage needs low scores, so keep it low.
    """
    d = np.loadtxt(path, delimiter=",", ndmin=2)
    by_frame = defaultdict(list)
    for f, _, x, y, w, h, s, *_ in d:
        if s >= min_conf:
            by_frame[int(f)].append([x, y, x + w, y + h, s])
    return {f: np.asarray(v) for f, v in by_frame.items()}


class MotFileDetector:
    """Callable detector: (frame_idx, image) -> Nx5 array. The image is ignored."""

    def __init__(self, path, min_conf=0.0):
        self.by_frame = load_mot_detections(path, min_conf)

    def __call__(self, frame_idx, image=None):
        return self.by_frame.get(frame_idx, EMPTY)

    def last_frame(self):
        return max(self.by_frame, default=0)
