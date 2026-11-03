FROM python:3.11-slim

# CPU image. The model and the videos are mounted at run time (see the README), not baked in:
# one image serves any YOLOX ONNX file, and no model or footage is stored in the image.
# OpenCV needs these two system libraries on the slim base.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home app

WORKDIR /build
COPY pyproject.toml README.md ./
COPY track_counter ./track_counter
RUN pip install --no-cache-dir ".[onnx]"

USER app
WORKDIR /data
ENTRYPOINT ["track-counter"]
