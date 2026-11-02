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
