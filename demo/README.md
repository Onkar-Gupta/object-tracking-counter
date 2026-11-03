# Browser demo (Gradio)

Upload a short video, click two points to draw the counting line, press Count. You get the annotated
video, the in and out counts and `events.csv`. CPU only, first 10 seconds of the clip, up to 50 MB.

From the repo root:

    pip install -e ".[onnx]" gradio==6.31.0
    curl -L --create-dirs -o models/yolox_s.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx
    python demo/app.py

Then open http://127.0.0.1:7860. Put your own `.mp4` files in `demo/examples/` to get clickable
examples. That folder is gitignored on purpose: never commit footage you do not have the right to share.

The counts show what the tool does, not how accurate it is. Accuracy is measured on MOT17 and reported
in the main README. The fast single-pass detector misses small and mostly hidden people, and people who
overlap are harder to detect and track, as in a crowded scene.
