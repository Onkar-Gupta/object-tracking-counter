"""Tracker adapter. Trackers from the `trackers` package (Apache-2.0, pinned): ByteTrack, OC-SORT
and BoT-SORT.

These are re-implementations, not the original code. ByteTrack here is ByteTrack-style, not
paper-exact: it uses one IoU threshold for both stages, no score fusion and no minimum score.
Detections below the tracker's confirmation count come back with id -1 and are dropped here.
"""
import inspect

import numpy as np
import supervision as sv
from trackers import BoTSORTTracker, ByteTrackTracker, OCSORTTracker

TRACKERS = {"bytetrack": ByteTrackTracker, "ocsort": OCSORTTracker, "botsort": BoTSORTTracker}


class Tracker:
    def __init__(self, fps, name="bytetrack", **params):
        if name not in TRACKERS:
            raise ValueError(f"unknown tracker: {name}")
        if name == "botsort":
            params.setdefault("enable_cmc", False)  # static cameras, and no frames are passed
        self._tracker = TRACKERS[name](frame_rate=fps, **params)

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
    cls = TRACKERS[params.get("name", "bytetrack")]
    sig = inspect.signature(cls.__init__).parameters
    base = {k: v.default for k, v in sig.items()
            if isinstance(v.default, (int, float)) and k != "frame_rate"}
    return {**base, **params}
