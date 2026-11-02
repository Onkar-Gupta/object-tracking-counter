"""Tracker adapter. ByteTrack-style tracker from the `trackers` package (Apache-2.0, pinned).

Not paper-exact: it uses one IoU threshold for both stages, no score fusion and no minimum
score. See the decision notes. Detections below the tracker's confirmation count come back
with id -1 and are dropped here.
"""
import inspect

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker


class Tracker:
    def __init__(self, fps, name="bytetrack", **params):
        if name != "bytetrack":
            raise ValueError(f"unknown tracker: {name}")
        self._tracker = ByteTrackTracker(frame_rate=fps, **params)

    def update(self, dets):
        """dets: Nx5 [x1, y1, x2, y2, score]. Returns Mx7 [x1, y1, x2, y2, id, score, class_id]."""
        d = np.asarray(dets, dtype=float).reshape(-1, 5)
        out = self._tracker.update(sv.Detections(
            xyxy=d[:, :4], confidence=d[:, 4], class_id=np.zeros(len(d), dtype=int)))
        if len(out) == 0:
            return np.zeros((0, 7))
        keep = out.tracker_id >= 0
        return np.column_stack([out.xyxy[keep], out.tracker_id[keep], out.confidence[keep],
                                np.zeros(keep.sum())])


def resolved_params(**params):
    """The tracker's real parameters: library defaults overridden by params. For reports."""
    sig = inspect.signature(ByteTrackTracker.__init__).parameters
    base = {k: v.default for k, v in sig.items()
            if isinstance(v.default, (int, float)) and k != "frame_rate"}
    return {**base, **params}
