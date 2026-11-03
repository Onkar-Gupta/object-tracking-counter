"""Gradio showcase for track-counter: upload a short video, click a counting line, get the count.

  pip install -e ".[onnx]" gradio     (from the repo root)
  python demo/app.py

CPU only. Only the first MAX_SECONDS of a video are used. The counts are a demonstration of the tool,
not a measure of its accuracy: that is measured on MOT17 and reported in the README.
"""
import csv
import tempfile
import time
from pathlib import Path

import av
import cv2
import gradio as gr
import numpy as np
import yaml

from track_counter.cli import open_video, render_video, video_frames
from track_counter.detectors import OnnxYoloxDetector
from track_counter.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
MODEL = ROOT / CFG["detector"]["model"]
MAX_SECONDS, MAX_MB, SHOW_WIDTH = 10, 50, 960
_detector = None


def detector():
    global _detector
    if _detector is None:
        _detector = OnnxYoloxDetector(MODEL, score_floor=CFG["detector"]["confidence"])
    return _detector


def draw_setup(frame, pts):
    """Frame with the clicked points, the line and a green arrow toward the "in" side."""
    img = frame.copy()
    for p, label in zip(pts, "AB"):
        cv2.circle(img, tuple(p), 6, (255, 212, 0), -1)
        cv2.putText(img, label, (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 212, 0), 2)
    if len(pts) == 2:
        a, b = np.array(pts, float)
        cv2.line(img, tuple(pts[0]), tuple(pts[1]), (255, 212, 0), 3)
        ab = b - a
        n = np.hypot(*ab)
        if n:  # "in" is a move from the positive to the negative side of A to B
            nin, mid = np.array([ab[1], -ab[0]]) / n, (a + b) / 2
            tip = (mid + 50 * nin).astype(int)
            cv2.arrowedLine(img, tuple(mid.astype(int)), tuple(tip), (25, 179, 90), 4, tipLength=0.3)
            cv2.putText(img, "in", tuple((mid + 62 * nin).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (25, 179, 90), 2)
    return img


def line_text(pts, scale):
    if len(pts) < 2:
        return "Click point A, then point B."
    (x1, y1), (x2, y2) = (np.array(p) * scale for p in pts)
    return f"--line {x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}"


def load_frame(path):
    """First frame of the upload, shrunk to the display width. Returns image and state."""
    if not path:
        return None, {"pts": [], "scale": 1.0, "frame": None}, "Upload a video first."
    if Path(path).stat().st_size > MAX_MB * 1e6:
        raise gr.Error(f"The video is over {MAX_MB} MB. Please use a shorter clip.")
    cap, _, _ = open_video(path)
    ok, bgr = cap.read()
    if not ok:
        raise gr.Error("Could not read that video.")
    scale = max(bgr.shape[1] / SHOW_WIDTH, 1.0)
    small = cv2.resize(bgr, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    return rgb, {"pts": [], "scale": scale, "frame": rgb}, "Click point A, then point B."


def click(state, evt: gr.SelectData):
    if state["frame"] is None:
        return None, state, "Upload a video first."
    if len(state["pts"]) >= 2:
        state["pts"] = []  # a third click starts a new line
    state["pts"].append([int(evt.index[0]), int(evt.index[1])])
    return draw_setup(state["frame"], state["pts"]), state, line_text(state["pts"], state["scale"])


def reset(state):
    state["pts"] = []
    if state["frame"] is None:
        return None, state, "Upload a video first."
    return state["frame"], state, "Click point A, then point B."


def to_h264(src, dst, width=SHOW_WIDTH):
    """Re-encode the OpenCV output so every browser can play it (smaller and H.264)."""
    cap = cv2.VideoCapture(str(src))
    fps = round(cap.get(cv2.CAP_PROP_FPS)) or 25
    out, stream = av.open(str(dst), "w"), None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        s = min(1.0, width / img.shape[1])
        img = cv2.resize(img, (int(img.shape[1] * s) // 2 * 2, int(img.shape[0] * s) // 2 * 2))
        if stream is None:
            stream = out.add_stream("libx264", rate=fps)
            stream.width, stream.height, stream.pix_fmt = img.shape[1], img.shape[0], "yuv420p"
            stream.options = {"crf": "26", "preset": "veryfast"}
        for pkt in stream.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
            out.mux(pkt)
    for pkt in stream.encode():
        out.mux(pkt)
    out.close()


def count(path, state, progress=gr.Progress()):  # noqa: B008 (Gradio injects the progress bar here)
    if not path:
        raise gr.Error("Upload a video first.")
    if len(state["pts"]) != 2:
        raise gr.Error("Click two points on the frame to draw the counting line.")
    (x1, y1), (x2, y2) = (np.array(p) * state["scale"] for p in state["pts"])
    line = [[float(x1), float(y1)], [float(x2), float(y2)]]
    cap, fps, total = open_video(path)
    limit = int(MAX_SECONDS * (fps or 25))
    counter_params = {k: v for k, v in CFG["counter"].items() if k != "line"}
    tracker_params = CFG["tracker"].get("params") or {}
    sink, t0 = {}, time.perf_counter()
    frames = progress.tqdm(video_frames(cap, limit), total=min(total or limit, limit), unit="frame")
    events, n = run_pipeline(frames, detector(), {"line": line}, fps, tracker_params, counter_params,
                             track_sink=sink)
    work = Path(tempfile.mkdtemp())
    render_video(path, work / "raw.mp4", 0, line, sink, events["line"], fps, limit)
    to_h264(work / "raw.mp4", work / "counted.mp4")
    with open(work / "events.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "track_id", "direction"])
        w.writerows([e.confirmed_frame, round(e.confirmed_frame / fps, 2), e.track_id, e.direction]
                    for e in events["line"])
    n_in = sum(e.direction == "in" for e in events["line"])
    n_out = len(events["line"]) - n_in
    secs = time.perf_counter() - t0
    summary = (f"### in **{n_in}** &nbsp;&nbsp; out **{n_out}**\n"
               f"{n} frames in {secs:.0f} s ({n / secs:.1f} frames per second on this machine's CPU).")
    return str(work / "counted.mp4"), summary, str(work / "events.csv")


EXAMPLES = sorted(str(p) for p in (Path(__file__).parent / "examples").glob("*.mp4"))

with gr.Blocks(title="track-counter demo") as demo:
    gr.Markdown(
        "# track-counter\n"
        "Count people crossing a line in a short video. YOLOX-s detects, ByteTrack follows, "
        "and a line-crossing rule counts. Runs on CPU.\n\n"
        f"Only the first {MAX_SECONDS} seconds are used (up to {MAX_MB} MB). "
        "The counts here show what the tool does, **not how accurate it is**: accuracy is measured on "
        "MOT17 and reported in the [README](https://github.com/Onkar-Gupta/object-tracking-counter). "
        "The fast single-pass detector misses small and mostly hidden people."
    )
    state = gr.State({"pts": [], "scale": 1.0, "frame": None})
    with gr.Row():
        with gr.Column():
            video = gr.Video(label="1. Upload a short video", sources=["upload"])
            frame = gr.Image(label="2. Click point A, then point B (a third click starts again)",
                             interactive=False)
            line_box = gr.Textbox(label="Counting line", value="Upload a video first.", interactive=False)
            with gr.Row():
                go = gr.Button("3. Count", variant="primary")
                clear = gr.Button("Reset line")
        with gr.Column():
            out_video = gr.Video(label="Result", interactive=False)
            summary = gr.Markdown()
            events_file = gr.File(label="events.csv")
    if EXAMPLES:
        gr.Examples(EXAMPLES, inputs=video, label="Example clips")
    video.change(load_frame, video, [frame, state, line_box])
    frame.select(click, state, [frame, state, line_box])
    clear.click(reset, state, [frame, state, line_box])
    go.click(count, [video, state], [out_video, summary, events_file])

if __name__ == "__main__":
    demo.queue(max_size=4, default_concurrency_limit=1).launch(inbrowser=False)
