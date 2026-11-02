"""Detect, track, count. One tracker run feeds one counter per named line."""
import csv
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from track_counter.counter import LineCounter
from track_counter.tracking import Tracker


def run_pipeline(frames, detect, lines, fps, tracker_params=None, counter_params=None):
    """frames: iterable of (frame_idx, image). detect(frame_idx, image) -> Nx5 detections.

    lines: {name: [[x1, y1], [x2, y2]]}. Returns ({name: [CrossingEvent]}, n_frames).
    """
    tracker = Tracker(fps, **(tracker_params or {}))
    counters = {n: LineCounter(line, **(counter_params or {})) for n, line in lines.items()}
    events = {n: [] for n in lines}
    n_frames = 0
    for idx, image in frames:
        tracks = tracker.update(detect(idx, image))
        for n, counter in counters.items():
            events[n] += counter.update(idx, tracks)
        n_frames += 1
    return events, n_frames


def write_outputs(out_dir, events, n_frames, fps, lines, settings=None):
    """Write events.csv and summary.json into out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields = ["line", "frame", "confirmed_frame", "time_s", "track_id", "class_id",
              "direction", "position"]
    with open(out / "events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, evs in events.items():
            for e in evs:
                w.writerow({"line": name, **asdict(e), "time_s": round(e.frame / fps, 3)})
    summary = {
        "frames": n_frames, "fps": fps, "trackers_version": version("trackers"),
        "settings": settings or {},
        "lines": {n: {"line": lines[n], "in": sum(e.direction == "in" for e in evs),
                      "out": sum(e.direction == "out" for e in evs), "events": len(evs)}
                  for n, evs in events.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
