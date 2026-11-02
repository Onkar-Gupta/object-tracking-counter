# Fine-tuning YOLOX on MOT17 person tiles (research only)

Recipe for the step C2 experiment (`eval/finetune_protocol.md`). Weights trained on MOT17 inherit
its CC BY-NC-SA 3.0 licence, so they are never committed or published. Only these scripts and the
measured results are shared.

## One-off setup (separate environment, outside the project)

A second Python environment holds PyTorch and the official YOLOX training code (Apache-2.0):

    python -m venv D:\ml_env\yolox_train
    D:\ml_env\yolox_train\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    git clone https://github.com/Megvii-BaseDetection/YOLOX D:\ml_env\YOLOX
    D:\ml_env\yolox_train\Scripts\python.exe -m pip install loguru opencv-python thop tabulate psutil tensorboard pycocotools onnx tqdm
    D:\ml_env\yolox_train\Scripts\python.exe -m pip install -e D:\ml_env\YOLOX --no-deps

Pretrained COCO weights (official release, Apache-2.0):

    curl.exe -L -o D:\ml_env\yolox_s.pth https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth

## 1. Cut the training sequences into tiles

Uses sequences 05, 10, 11 and 13 only (the script refuses 02, 04 and 09):

    python training\prepare_mot17_tiles.py --mot-dir D:\Datasets\MOT17\MOT17\train --out-dir D:\Datasets\mot17_tiles

## 2. Train (20 epochs, final checkpoint, no checkpoint selection)

    cd D:\ml_env\YOLOX
    $env:PYTHONPATH = "D:\ml_env\YOLOX"
    D:\ml_env\yolox_train\Scripts\python.exe tools\train.py -f <path to>\training\exp_mot17_person.py -d 1 -b 16 --fp16 -c D:\ml_env\yolox_s.pth

PyTorch 2.6 and later load checkpoints in a restricted "weights only" mode. Try without any override
first. Only if the official `yolox_s.pth` fails to load for that reason, set
`$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = "1"` for that one run, which switches the safety check off.

## 3. Export to ONNX (same input size as the other models)

    D:\ml_env\yolox_train\Scripts\python.exe tools\export_onnx.py --output-name D:\ml_env\yolox_mot17_s.onnx -f <path to>\training\exp_mot17_person.py -c D:\ml_env\YOLOX_outputs\mot17_person_s\last_epoch_ckpt.pth --no-onnxsim

The exported model has one class, so its output is 6 columns (box, objectness, person). The
adapter in `track_counter/detectors.py` already reads that layout.
