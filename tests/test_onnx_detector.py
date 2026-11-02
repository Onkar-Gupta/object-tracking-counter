"""Pre- and post-processing for the ONNX YOLOX adapter. No model file needed."""
from itertools import pairwise

import numpy as np

from track_counter.detectors import (
    OnnxYoloxDetector,
    TiledOnnxYoloxDetector,
    decode_yolox,
    drop_border_boxes,
    letterbox,
    nms,
    tile_starts,
)


def test_letterbox_keeps_aspect_and_pads_grey():
    img = np.zeros((1080, 1920, 3), np.uint8)
    blob, r = letterbox(img, 640)
    assert blob.shape == (3, 640, 640) and abs(r - 640 / 1920) < 1e-9
    assert blob[0, 639, 0] == 114                       # padded bottom rows are grey
    assert blob[0, 0, 0] == 0                           # image area is untouched


def test_decode_places_a_box_on_its_grid_cell():
    raw = np.zeros((8400, 85), np.float32)
    raw[0, :4] = [0.5, 0.5, np.log(2), np.log(4)]        # first cell of the stride-8 grid
    box = decode_yolox(raw)[0]
    assert np.allclose(box, [(0 + 0.5) * 8, (0 + 0.5) * 8, 16, 32])
    last = 6400 + 1600 + 399                             # last cell of the stride-32 grid (19, 19)
    raw[last, :4] = [0, 0, 0, 0]
    assert np.allclose(decode_yolox(raw)[last][:2], [19 * 32, 19 * 32])


def test_nms_suppresses_overlaps_keeps_best():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    assert nms(boxes, np.array([0.5, 0.9, 0.7])) == [1, 2]


def test_postprocess_filters_by_score_and_rescales():
    det = OnnxYoloxDetector.__new__(OnnxYoloxDetector)   # skip loading a model
    det.score_floor, det.nms_iou, det.size = 0.3, 0.45, 640
    raw = np.zeros((8400, 85), np.float32)
    raw[0, :4] = [0.5, 0.5, np.log(2), np.log(4)]        # box 16 x 32 centred at (4, 4)
    raw[0, 4], raw[0, 5] = 0.9, 0.9                       # score 0.81
    raw[1, 4], raw[1, 5] = 0.9, 0.2                       # score 0.18: below the floor
    out = det.postprocess(raw, ratio=0.5)
    assert out.shape == (1, 5) and abs(out[0, 4] - 0.81) < 1e-6
    assert np.allclose(out[0, :4], [(4 - 8) / 0.5, (4 - 16) / 0.5, (4 + 8) / 0.5, (4 + 16) / 0.5])


def test_tile_starts_cover_the_frame_with_overlap():
    assert tile_starts(1920, 640, 128) == [0, 427, 853, 1280]
    assert tile_starts(1080, 640, 128) == [0, 440]
    assert tile_starts(500, 640, 128) == [0]
    for length in (1920, 1080):
        starts = tile_starts(length, 640, 128)
        assert starts[-1] + 640 == length and all(b - a <= 640 - 128 for a, b in pairwise(starts))


def test_drop_border_boxes_keeps_image_edges_only():
    dets = np.array([[0, 100, 50, 200, 0.9],      # touches the tile's left edge
                     [300, 100, 350, 200, 0.9]])  # inside
    # tile in the middle of the image: the left-edge box is cut by the tile, so it is dropped
    assert len(drop_border_boxes(dets, 500, 0, 640, 640, 1920, 1080)) == 1
    # tile at the image's left border: the same box is genuinely at the image edge, so it stays
    assert len(drop_border_boxes(dets, 0, 0, 640, 640, 1920, 1080)) == 2


def test_tiled_detector_merges_duplicates_and_shifts_tiles(monkeypatch):
    det = TiledOnnxYoloxDetector.__new__(TiledOnnxYoloxDetector)
    det.tile, det.overlap, det.nms_iou = 640, 128, 0.45
    calls = []

    def fake_single(self, idx, image):                       # stands in for one model run
        calls.append(image.shape[:2])
        if image.shape[:2] == (1080, 1920):
            return np.array([[10.0, 10, 60, 120, 0.4]])     # whole-frame: a low-score small person
        return np.array([[10.0, 10, 60, 120, 0.8]]) if len(calls) == 2 else np.zeros((0, 5))
    monkeypatch.setattr(OnnxYoloxDetector, "__call__", fake_single)
    out = det(1, np.zeros((1080, 1920, 3), np.uint8))
    assert len(calls) == 1 + 4 * 2                            # whole frame + 4 x 2 tiles
    assert out.shape == (1, 5) and out[0, 4] == 0.8           # duplicate merged, higher score kept
