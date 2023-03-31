import torch, detectron2
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json

def get_class_names(txt_path):
    with open(txt_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

class_names = get_class_names("./robot/classes.txt")
coins_metadata = MetadataCatalog.get("robot").set(thing_classes=class_names)  # Set metadata here
dataset_dicts = load_coco_json("./robot/robot.json", "./robot", "robot")
DatasetCatalog.register("robot", lambda: dataset_dicts)  # Register dataset

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("robot",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0004
cfg.SOLVER.MAX_ITER = (
    1500
)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    128
)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)

ckpt_folder = os.path.join(os.getcwd(), "ckpt/robot")
os.makedirs(ckpt_folder, exist_ok=True)
cfg.OUTPUT_DIR = ckpt_folder

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
