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


# --- ONNX YOLOX adapter ---------------------------------------------------------------------
# YOLOX (Megvii, Apache-2.0). The released ONNX files output raw predictions that need decoding.

def letterbox(image, size=640):
    """Resize keeping aspect ratio, pad with grey (114) to size x size. Returns (CHW float32, ratio)."""
    import cv2

    h, w = image.shape[:2]
    r = min(size / h, size / w)
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[:int(h * r), :int(w * r)] = cv2.resize(image, (int(w * r), int(h * r)),
                                                   interpolation=cv2.INTER_LINEAR)
    return padded.transpose(2, 0, 1).astype(np.float32), r  # BGR, 0-255, no normalisation


def decode_yolox(raw, size=640, strides=(8, 16, 32)):
    """Raw output (N, 85) -> boxes (N, 4) as centre x, centre y, width, height in letterbox pixels."""
    grids, expanded = [], []
    for s in strides:
        n = size // s
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        expanded.append(np.full((n * n, 1), s))
    grid, stride = np.concatenate(grids), np.concatenate(expanded)
    out = raw[:, :4].copy()
    out[:, :2] = (out[:, :2] + grid) * stride
    out[:, 2:4] = np.exp(out[:, 2:4]) * stride
    return out


def nms(boxes, scores, iou_threshold=0.45):
    """Greedy non-maximum suppression. boxes Nx4 xyxy. Returns kept indices, best score first."""
    from track_counter.detection_eval import iou_matrix

    order = np.argsort(-scores, kind="stable")
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        rest = order[1:]
        order = rest[iou_matrix(boxes[i:i + 1], boxes[rest])[0] <= iou_threshold]
    return keep


class OnnxYoloxDetector:
    """Callable detector: (frame_idx, BGR image) -> Nx5 [x1, y1, x2, y2, score], person class only."""

    def __init__(self, model_path, score_floor=0.1, nms_iou=0.45, size=640):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.score_floor, self.nms_iou, self.size = score_floor, nms_iou, size

    def __call__(self, frame_idx, image):
        blob, ratio = letterbox(image, self.size)
        raw = self.session.run(None, {self.input_name: blob[None]})[0][0]
        return self.postprocess(raw, ratio)

    def postprocess(self, raw, ratio):
        scores = raw[:, 4] * raw[:, 5]  # objectness x person-class probability (COCO class 0)
        keep = scores >= self.score_floor
        if not keep.any():
            return EMPTY
        cxcywh = decode_yolox(raw, self.size)[keep]
        xyxy = np.column_stack([cxcywh[:, 0] - cxcywh[:, 2] / 2, cxcywh[:, 1] - cxcywh[:, 3] / 2,
                                cxcywh[:, 0] + cxcywh[:, 2] / 2, cxcywh[:, 1] + cxcywh[:, 3] / 2]) / ratio
        s = scores[keep]
        idx = nms(xyxy, s, self.nms_iou)
        return np.column_stack([xyxy[idx], s[idx]])


def tile_starts(length, tile=640, overlap=128):
    """Start offsets of tiles of size `tile` covering `length`, spread evenly, last one flush."""
    if length <= tile:
        return [0]
    n = int(np.ceil((length - tile) / (tile - overlap))) + 1
    return [round(v) for v in np.linspace(0, length - tile, n)]


def drop_border_boxes(dets, x0, y0, tile_w, tile_h, image_w, image_h, margin=3):
    """Drop boxes touching an inner tile edge (a neighbouring tile sees that person whole).

    Edges on the image border are kept. dets are in tile coordinates.
    """
    if len(dets) == 0:
        return dets
    cut = np.zeros(len(dets), bool)
    if x0 > 0:
        cut |= dets[:, 0] < margin
    if x0 + tile_w < image_w:
        cut |= dets[:, 2] > tile_w - margin
    if y0 > 0:
        cut |= dets[:, 1] < margin
    if y0 + tile_h < image_h:
        cut |= dets[:, 3] > tile_h - margin
    return dets[~cut]


class TiledOnnxYoloxDetector(OnnxYoloxDetector):
    """YOLOX on overlapping native-resolution tiles plus one whole-frame pass, merged with NMS.

    The whole-frame pass finds large people, the tiles find small ones that a 640 px resize loses.
    About 9 model runs per 1080p frame, so much slower than the single pass.
    """

    def __init__(self, model_path, tile=640, overlap=128, **kwargs):
        super().__init__(model_path, size=tile, **kwargs)
        self.tile, self.overlap = tile, overlap

    def __call__(self, frame_idx, image):
        h, w = image.shape[:2]
        parts = [super().__call__(frame_idx, image)]            # whole frame, shrunk to the tile size
        for y0 in tile_starts(h, self.tile, self.overlap):
            for x0 in tile_starts(w, self.tile, self.overlap):
                crop = image[y0:y0 + self.tile, x0:x0 + self.tile]
                d = drop_border_boxes(super().__call__(frame_idx, crop), x0, y0,
                                      crop.shape[1], crop.shape[0], w, h)
                parts.append(d + np.array([x0, y0, x0, y0, 0]))
        merged = np.vstack(parts)
        return merged[nms(merged[:, :4], merged[:, 4], self.nms_iou)] if len(merged) else EMPTY
