"""YOLOX experiment file: fine-tune YOLOX-s on MOT17 person tiles (eval/finetune_protocol.md).

Run with the YOLOX trainer from a separate environment that has PyTorch (see training/README.md).
Recipe fixed in advance: 20 epochs, final checkpoint, no checkpoint selection, seed 0.
"""
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.depth, self.width = 0.33, 0.50                     # YOLOX-s
        self.exp_name = "mot17_person_s"
        self.num_classes = 1                                      # person
        self.data_dir = os.environ.get("MOT_TILES_DIR", "D:/Datasets/mot17_tiles")
        self.train_ann, self.val_ann = "instances_train2017.json", "instances_val2017.json"
        self.input_size = (640, 640)
        self.multiscale_range = 3
        self.max_epoch = 20
        self.warmup_epochs = 1
        self.no_aug_epochs = 3
        self.eval_interval = 1000       # no scheduled evaluation: no checkpoint is chosen by validation
        self.data_num_workers = 4
        self.print_interval = 20
        self.seed = 0
        self.output_dir = os.environ.get("YOLOX_OUTPUT_DIR", "D:/ml_env/YOLOX_outputs")
        self.test_conf = 0.01
        self.nmsthre = 0.65
