"""Pipeline tests on a synthetic video of moving rectangles. No dataset, no model."""
import csv
import json

import cv2
import numpy as np

from track_counter.cli import main
from track_counter.detectors import MotFileDetector
from track_counter.pipeline import run_pipeline, write_outputs

W, H, LINE_Y, N = 320, 240, 120, 40
LINES = {"mid": [[0, LINE_Y], [W, LINE_Y]]}
COUNTER = {"min_frames_on_far_side": 3, "margin_px": 4.0}


def box(i, kind):
    """Rectangle top-left (x, y), size 20x40, for frame i. All rectangles stay apart."""
    if kind == "up":      # bottom edge moves 200 -> 44: crosses y=120 upwards ("in")
        return 40, 160 - 4 * i
    if kind == "down":    # bottom edge moves 44 -> 200: crosses downwards ("out")
        return 140, 4 + 4 * i
    return 240, 100       # stays put, crosses nothing


def image(i):
    img = np.zeros((H, W), np.uint8)
    for kind in ("up", "down", "still"):
        x, y = box(i, kind)
        img[y:y + 40, x:x + 20] = 255
    return img


def detect(i, img):
    """Fake detector: one box per white blob."""
    _, _, stats, _ = cv2.connectedComponentsWithStats((img > 128).astype(np.uint8))
    return np.array([[x, y, x + w, y + h, 0.95] for x, y, w, h, _ in stats[1:]]).reshape(-1, 5)


def test_synthetic_video_gives_exact_events(tmp_path):
    frames = ((i, image(i)) for i in range(N))
    events, n = run_pipeline(frames, detect, LINES, fps=10.0, counter_params=COUNTER)
    assert n == N
    assert sorted(e.direction for e in events["mid"]) == ["in", "out"]
    summary = write_outputs(tmp_path, events, n, 10.0, LINES)
    assert summary["lines"]["mid"] == {"line": LINES["mid"], "in": 1, "out": 1, "events": 2}
    rows = list(csv.DictReader((tmp_path / "events.csv").read_text().splitlines()))
    assert sorted(r["direction"] for r in rows) == ["in", "out"] and {r["line"] for r in rows} == {"mid"}
    assert json.loads((tmp_path / "summary.json").read_text())["trackers_version"]


def test_no_detections_gives_no_events():
    events, n = run_pipeline(((i, None) for i in range(5)), lambda i, im: np.zeros((0, 5)),
                             LINES, fps=10.0, counter_params=COUNTER)
    assert events == {"mid": []} and n == 5


def _write_det(path):
    rows = []
    for i in range(N):
        for kind in ("up", "down", "still"):
            x, y = box(i, kind)
            rows.append(f"{i + 1},-1,{x},{y},20,40,0.95,-1,-1,-1")
    path.write_text("\n".join(rows))


def test_mot_file_detector_applies_score_floor(tmp_path):
    p = tmp_path / "det.txt"
    p.write_text("1,-1,10,20,30,40,0.9,-1,-1,-1\n1,-1,50,60,10,10,0.2,-1,-1,-1\n")
    assert MotFileDetector(p, 0.0)(1).shape == (2, 5)
    kept = MotFileDetector(p, 0.5)(1)
    assert kept.tolist() == [[10, 20, 40, 60, 0.9]]
    assert MotFileDetector(p)(99).shape == (0, 5)


def test_cli_run_writes_outputs(tmp_path, capsys):
    det = tmp_path / "det.txt"
    _write_det(det)
    out = tmp_path / "out"
    main(["run", "--detections", str(det), "--fps", "10", "--line", f"0,{LINE_Y},{W},{LINE_Y}",
          "--out", str(out)])
    assert "in 1, out 1" in capsys.readouterr().out
    assert (out / "events.csv").exists() and (out / "summary.json").exists()
