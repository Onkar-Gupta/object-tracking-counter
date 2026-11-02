"""Line-crossing counter. Pure Python and NumPy: no OpenCV, no model.

Side of the line is the sign of the signed distance d = cross(B - A, P - A) / |B - A|.
A move from the positive side to the negative side is "in", the reverse is "out".
"""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CrossingEvent:
    frame: int            # frame where the object flipped sides (not when it was confirmed)
    confirmed_frame: int  # frame where the N-frame rule was satisfied
    track_id: int
    class_id: int
    direction: str        # "in" or "out"
    position: float       # where along A->B the path met the line, 0 to 1


class _State:
    __slots__ = ("first_frame", "flip_frame", "flip_pos", "last_event", "last_pt", "last_seen",
                 "pending", "side")

    def __init__(self, side, pt, frame):
        self.side, self.last_pt, self.pending = side, pt, 0
        self.flip_frame, self.flip_pos, self.last_seen = frame, 0.0, frame
        self.first_frame, self.last_event = frame, None


class LineCounter:
    """Counts crossings of one line.

    Optional clean-up rules, all off by default:
      cooldown_frames: the same track cannot emit another event for this many frames.
      min_track_frames: a track must be followed this long before it may emit an event.
      merge_frames, merge_px: drop an event if another event in the same direction was emitted
        within merge_frames frames and merge_px pixels along the line (duplicates from other IDs).
    A dropped event still updates the track's side, so it does not flap.
    """

    def __init__(self, line, anchor="bottom_center", min_frames_on_far_side=3,
                 margin_px=4.0, max_age_frames=60, cooldown_frames=0, min_track_frames=0,
                 merge_frames=0, merge_px=0.0, smooth_alpha=1.0):
        if anchor not in ("bottom_center", "center"):
            raise ValueError(f"unknown anchor: {anchor}")
        self.a = np.asarray(line[0], dtype=float)
        self.b = np.asarray(line[1], dtype=float)
        self.ab = self.b - self.a
        self.length = float(np.hypot(*self.ab))
        if self.length == 0:
            raise ValueError("line has zero length")
        self.anchor, self.n, self.margin, self.max_age = (
            anchor, min_frames_on_far_side, margin_px, max_age_frames)
        self.cooldown, self.min_track = cooldown_frames, min_track_frames
        self.merge_frames, self.merge_px = merge_frames, merge_px
        if not 0 < smooth_alpha <= 1:
            raise ValueError("smooth_alpha must be in (0, 1]")
        self.alpha = smooth_alpha
        self._smooth = {}  # track id -> (smoothed anchor, last frame seen)
        self._state = {}
        self._recent = []  # (direction, confirmed_frame, position) of emitted events
        self.counts = defaultdict(int)  # (class_id, direction) -> count

    def _point(self, x1, y1, x2, y2):
        y = y2 if self.anchor == "bottom_center" else (y1 + y2) / 2
        return np.array([(x1 + x2) / 2, y])

    def _dist(self, p):
        v = p - self.a
        return float(self.ab[0] * v[1] - self.ab[1] * v[0]) / self.length

    def _dropped(self, st, frame_idx, direction):
        if self.cooldown and st.last_event is not None and frame_idx - st.last_event < self.cooldown:
            return True
        if frame_idx - st.first_frame < self.min_track:
            return True
        if self.merge_frames:
            self._recent = [r for r in self._recent if frame_idx - r[1] <= self.merge_frames]
            return any(r[0] == direction and abs(r[2] - st.flip_pos) * self.length <= self.merge_px
                       for r in self._recent)
        return False

    def update(self, frame_idx, tracks):
        """tracks: rows of [x1, y1, x2, y2, track_id, score, class_id]. Returns new events."""
        for tid in [t for t, s in self._state.items() if frame_idx - s.last_seen > self.max_age]:
            del self._state[tid]
        for tid in [t for t, (_, seen) in self._smooth.items() if frame_idx - seen > self.max_age]:
            del self._smooth[tid]
        events = []
        for row in tracks:
            x1, y1, x2, y2, tid, _, cls = row[:7]
            tid, cls = int(tid), int(cls)
            p = self._point(x1, y1, x2, y2)
            if self.alpha < 1.0:  # exponential moving average of the anchor, per track
                prev = self._smooth.get(tid)
                if prev is not None:
                    p = self.alpha * p + (1 - self.alpha) * prev[0]
                self._smooth[tid] = (p, frame_idx)
            d = self._dist(p)
            st = self._state.get(tid)
            if st:
                st.last_seen = frame_idx
            if abs(d) < self.margin:  # dead band: ignored, resets nothing
                continue
            side = 1 if d > 0 else -1
            if st is None:
                self._state[tid] = _State(side, p, frame_idx)
            elif side == st.side:
                st.pending, st.last_pt = 0, p
            else:
                if st.pending == 0:
                    d0 = self._dist(st.last_pt)
                    x = st.last_pt + d0 / (d0 - d) * (p - st.last_pt)
                    t = float(np.dot(x - self.a, self.ab)) / self.length**2
                    if not 0 <= t <= 1:  # went round the end of the segment
                        st.side, st.last_pt = side, p
                        continue
                    st.flip_frame, st.flip_pos = frame_idx, t
                st.pending += 1
                if st.pending >= self.n:
                    direction = "in" if st.side > 0 else "out"
                    if not self._dropped(st, frame_idx, direction):
                        events.append(CrossingEvent(st.flip_frame, frame_idx, tid, cls,
                                                    direction, st.flip_pos))
                        self.counts[(cls, direction)] += 1
                        st.last_event = frame_idx
                        self._recent.append((direction, frame_idx, st.flip_pos))
                    st.side, st.last_pt, st.pending = side, p, 0
        return events
