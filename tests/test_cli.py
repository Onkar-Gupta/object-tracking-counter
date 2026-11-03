import pathlib

import yaml

import track_counter
from track_counter.cli import main

ROOT = pathlib.Path(track_counter.__file__).parent.parent


def test_default_config_matches_config():
    shipped = (ROOT / "track_counter" / "default_config.yaml").read_text()
    assert shipped == (ROOT / "config.yaml").read_text()
    assert yaml.safe_load(shipped)["counter"]["min_frames_on_far_side"] == 3


def test_version(capsys):
    try:
        main(["--version"])
    except SystemExit:
        pass
    assert track_counter.__version__ in capsys.readouterr().out


def _make_video(path, n=40, w=320, h=240, fps=20):
    import cv2
    import numpy as np

    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for _ in range(n):
        out.write(np.zeros((h, w, 3), np.uint8))
    out.release()


def test_video_frames_are_one_based_and_limited(tmp_path):
    from track_counter.cli import open_video, video_frames

    _make_video(tmp_path / "v.avi", n=12)
    cap, fps, total = open_video(tmp_path / "v.avi")
    assert (fps, total) == (20.0, 12)
    assert [i for i, _ in video_frames(cap)] == list(range(1, 13))
    cap, _, _ = open_video(tmp_path / "v.avi")
    assert len(list(video_frames(cap, limit=5))) == 5


def test_run_on_video_counts_a_person_crossing(tmp_path, monkeypatch, capsys):
    import numpy as np

    from track_counter import cli

    _make_video(tmp_path / "v.avi", n=40)

    class Walker:  # one person walking up the frame, feet from y=220 to y=20
        def __init__(self, *a, **k):
            pass

        def __call__(self, idx, image):
            y = 220 - 5 * idx
            return np.array([[150.0, y - 60, 170.0, y, 0.9]])

    monkeypatch.setattr(cli, "OnnxYoloxDetector", Walker)
    cli.main(["run", "--video", str(tmp_path / "v.avi"), "--model", "x.onnx",
              "--line", "0,120,320,120", "--out", str(tmp_path / "out")])
    assert "in 1, out 0" in capsys.readouterr().out


def test_run_needs_exactly_one_source():
    import pytest

    with pytest.raises(SystemExit):
        main(["run", "--fps", "30"])


def test_save_video_writes_every_frame(tmp_path, monkeypatch):
    import numpy as np

    from track_counter import cli

    _make_video(tmp_path / "v.avi", n=30)

    class Walker:
        def __init__(self, *a, **k):
            pass

        def __call__(self, idx, image):
            y = 220 - 5 * idx
            return np.array([[150.0, y - 60, 170.0, y, 0.9]])

    monkeypatch.setattr(cli, "OnnxYoloxDetector", Walker)
    cli.main(["run", "--video", str(tmp_path / "v.avi"), "--model", "x.onnx", "--line", "0,120,320,120",
              "--start", "5", "--save-video", str(tmp_path / "o.mp4"), "--out", str(tmp_path / "out")])
    _, _, total = cli.open_video(tmp_path / "o.mp4")
    assert total == 25  # 30 frames minus the 5 skipped


def test_tiled_flag_picks_the_tiled_detector(tmp_path, monkeypatch):
    from track_counter import cli

    _make_video(tmp_path / "v.avi", n=3)
    used = []

    def fake(name):
        class D:
            def __init__(self, *a, **k):
                used.append(name)

            def __call__(self, idx, image):
                return cli.np_empty

        return D

    monkeypatch.setattr(cli, "np_empty", __import__("numpy").zeros((0, 5)), raising=False)
    monkeypatch.setattr(cli, "OnnxYoloxDetector", fake("single"))
    monkeypatch.setattr(cli, "TiledOnnxYoloxDetector", fake("tiled"))
    for extra, expected in (([], "single"), (["--tiled"], "tiled")):
        used.clear()
        cli.main(["run", "--video", str(tmp_path / "v.avi"), "--model", "x.onnx", "--out", str(tmp_path / "o")] + extra)
        assert used == [expected]
