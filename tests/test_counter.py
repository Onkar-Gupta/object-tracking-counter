"""Synthetic-track tests for LineCounter. No video, no dataset.

Line runs left to right at y=100, so d = y - 100: y > 100 is the positive side,
and moving from y > 100 to y < 100 is "in".
"""
import pathlib

from track_counter import counter
from track_counter.counter import LineCounter

LINE = [(0, 100), (400, 100)]


def row(tid, x, y, cls=0):
    """A track row whose bottom-centre anchor is (x, y)."""
    return [x - 5, y - 20, x + 5, y, tid, 0.9, cls]


def run(lc, paths):
    """paths: {track_id: [(frame, x, y), ...]} or {track_id: (cls, [...])}. Returns all events."""
    by_frame = {}
    for tid, path in paths.items():
        cls, path = path if isinstance(path, tuple) else (0, path)
        for f, x, y in path:
            by_frame.setdefault(f, []).append(row(tid, x, y, cls))
    events = []
    for f in range(max(by_frame) + 1):
        events += lc.update(f, by_frame.get(f, []))
    return events


def ys(start, ys_):
    return [(start + i, 200, y) for i, y in enumerate(ys_)]


def test_01_straight_crossing_in():
    ev = run(LineCounter(LINE, min_frames_on_far_side=1), {1: ys(0, [130, 110, 90, 70])})
    assert [(e.direction, e.frame, e.track_id) for e in ev] == [("in", 2, 1)]
    assert abs(ev[0].position - 0.5) < 1e-9


def test_02_reverse_path_is_out():
    ev = run(LineCounter(LINE, min_frames_on_far_side=1), {1: ys(0, [70, 90, 110, 130])})
    assert [e.direction for e in ev] == ["out"]


def test_03_swapping_a_and_b_flips_direction():
    ev = run(LineCounter(LINE[::-1], min_frames_on_far_side=1), {1: ys(0, [130, 110, 90, 70])})
    assert [e.direction for e in ev] == ["out"]


def test_04_jitter_needs_the_rules():
    wobble = ys(0, [102, 98] * 15)
    assert run(LineCounter(LINE, min_frames_on_far_side=1, margin_px=4), {1: wobble}) == []
    assert len(run(LineCounter(LINE, min_frames_on_far_side=1, margin_px=0), {1: wobble})) > 5
    assert run(LineCounter(LINE, min_frames_on_far_side=3, margin_px=0), {1: wobble}) == []


def test_05_stay_on_far_side_rule():
    path = ys(0, [120, 120, 80, 80, 120, 120])  # 2 frames on the far side, then back
    assert run(LineCounter(LINE, min_frames_on_far_side=3), {1: path}) == []
    ev = run(LineCounter(LINE, min_frames_on_far_side=2), {1: path})
    assert [e.direction for e in ev] == ["in", "out"]


def test_06_event_frame_is_the_flip_frame():
    ev = run(LineCounter(LINE, min_frames_on_far_side=5), {1: ys(0, [120, 120] + [80] * 6)})
    assert (ev[0].frame, ev[0].confirmed_frame) == (2, 6)


def test_07_crossing_beyond_segment_end_is_ignored():
    path = [(0, 450, 120), (1, 450, 80), (2, 450, 80),   # outside x range: silent flip
            (3, 200, 80), (4, 200, 120), (5, 200, 120)]  # real crossing back through
    ev = run(LineCounter(LINE, min_frames_on_far_side=2), {1: path})
    assert [e.direction for e in ev] == ["out"]


def test_08_first_seen_on_far_side_then_leaves():
    assert run(LineCounter(LINE, min_frames_on_far_side=1), {1: ys(0, [80, 60, 40])}) == []


def test_09_gap_in_detections():
    path = [(10, 200, 130), (15, 200, 70), (16, 200, 70), (17, 200, 70)]
    ev = run(LineCounter(LINE, min_frames_on_far_side=3), {1: path})
    assert [(e.direction, e.frame) for e in ev] == [("in", 15)]


def test_10_reborn_track_double_counts():
    paths = {1: ys(0, [130, 110, 90, 70]), 2: ys(10, [130, 110, 90, 70])}
    assert len(run(LineCounter(LINE, min_frames_on_far_side=1), paths)) == 2


def test_11_id_switch_on_the_line_gives_a_miss():
    paths = {1: ys(0, [130, 110, 100]), 2: ys(3, [80, 70, 60])}
    assert run(LineCounter(LINE, min_frames_on_far_side=1, margin_px=4), paths) == []


def test_12_independent_state_and_per_class_counts():
    lc = LineCounter(LINE, min_frames_on_far_side=1)
    paths = {1: (0, ys(0, [130, 110, 90, 70])), 2: (2, ys(0, [70, 90, 110, 130])),
             3: (0, ys(0, [130, 110, 90, 70]))}
    ev = run(lc, paths)
    assert len(ev) == 3
    assert dict(lc.counts) == {(0, "in"): 2, (2, "out"): 1}


def test_13_stale_track_is_forgotten():
    lc = LineCounter(LINE, min_frames_on_far_side=1, max_age_frames=5)
    path = [(0, 200, 130), (20, 200, 70), (21, 200, 70)]  # gone 20 frames, back on the far side
    assert run(lc, {1: path}) == []  # fresh state on return, so no crossing was seen
    assert 1 in lc._state and lc._state[1].side == -1


def test_14_dead_band_does_not_reset_pending():
    path = ys(0, [120, 80, 99, 80])  # flip, dead band, still far side
    ev = run(LineCounter(LINE, min_frames_on_far_side=2, margin_px=4), {1: path})
    assert [(e.direction, e.frame) for e in ev] == [("in", 1)]


def test_15_counter_has_no_opencv():
    assert "cv2" not in pathlib.Path(counter.__file__).read_text()


# --- optional clean-up rules (all off by default) ---------------------------------------------

def _cross(tid, start, up=True, y_from=150, y_to=50, x=100):
    """A track that walks from y_from to y_to in steps of 20 starting at frame `start`."""
    ys = range(y_from, y_to - 1, -20) if up else range(y_from, y_to + 1, 20)
    return {tid: [(start + i, x, y) for i, y in enumerate(ys)]}


def test_defaults_leave_events_unchanged():
    paths = _cross(1, 0)
    assert len(run(LineCounter(LINE, min_frames_on_far_side=1, margin_px=0), paths)) == 1


def test_cooldown_suppresses_an_immediate_second_event_from_the_same_track():
    # in, then straight back out within 5 frames
    path = [(0, 100, 150), (1, 100, 130), (2, 100, 70), (3, 100, 50), (4, 100, 70), (5, 100, 130), (6, 100, 150)]
    free = run(LineCounter(LINE, min_frames_on_far_side=1, margin_px=0), {1: path})
    held = run(LineCounter(LINE, min_frames_on_far_side=1, margin_px=0, cooldown_frames=30), {1: path})
    assert [e.direction for e in free] == ["in", "out"]
    assert [e.direction for e in held] == ["in"]


def test_cooldown_still_updates_the_side_so_it_does_not_flap():
    path = [(0, 100, 150), (1, 100, 70), (2, 100, 130), (3, 100, 70)]
    lc = LineCounter(LINE, min_frames_on_far_side=1, margin_px=0, cooldown_frames=30)
    ev = run(lc, {1: path})
    assert [e.direction for e in ev] == ["in"]      # the later flips are dropped, not queued


def test_min_track_frames_blocks_young_tracks():
    young = _cross(1, 0, y_from=130, y_to=70)          # crosses within 3 frames of birth
    assert len(run(LineCounter(LINE, min_frames_on_far_side=1, margin_px=0, min_track_frames=10), young)) == 0
    old = {1: [(i, 100, 150) for i in range(12)] + [(12 + i, 100, y) for i, y in enumerate((130, 70))]}
    assert len(run(LineCounter(LINE, min_frames_on_far_side=1, margin_px=0, min_track_frames=10), old)) == 1


def test_merge_drops_a_nearby_duplicate_from_another_id():
    paths = {**_cross(1, 0, x=100), **_cross(2, 1, x=120)}          # two IDs, same person, 20 px apart
    merged = run(LineCounter(LINE, min_frames_on_far_side=1, margin_px=0, merge_frames=15, merge_px=60), paths)
    assert len(merged) == 1
    far = {**_cross(1, 0, x=50), **_cross(2, 1, x=350)}              # 300 px apart: two people
    assert len(run(LineCounter(LINE, min_frames_on_far_side=1, margin_px=0, merge_frames=15, merge_px=60), far)) == 2


def test_merge_ignores_the_opposite_direction():
    paths = {**_cross(1, 0, up=True, x=100), **_cross(2, 0, up=False, y_from=50, y_to=150, x=120)}
    ev = run(LineCounter(LINE, min_frames_on_far_side=1, margin_px=0, merge_frames=15, merge_px=60), paths)
    assert sorted(e.direction for e in ev) == ["in", "out"]


def test_smoothing_calms_a_wobbling_track_and_is_off_by_default():
    # A person hovering at the line whose foot point jitters across it every frame.
    wobble = [(i, 100, 100 + (12 if i % 2 else -12)) for i in range(20)]
    raw = run(LineCounter(LINE, min_frames_on_far_side=1, margin_px=4), {1: wobble})
    calm = run(LineCounter(LINE, min_frames_on_far_side=1, margin_px=4, smooth_alpha=0.3), {1: wobble})
    assert len(raw) >= 10 and len(calm) == 0   # smoothed wobble stays inside the 4 px dead band


def test_smoothing_still_counts_a_real_crossing():
    ev = run(LineCounter(LINE, min_frames_on_far_side=1, margin_px=0, smooth_alpha=0.5), _cross(1, 0, y_from=200, y_to=0))
    assert [e.direction for e in ev] == ["in"]


def test_smooth_alpha_is_validated():
    import pytest
    for bad in (0, 1.5, -1):
        with pytest.raises(ValueError):
            LineCounter(LINE, smooth_alpha=bad)
