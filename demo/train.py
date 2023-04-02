import os
import json
import cv2
import random
import torch
import detectron2
import numpy as np
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

CLASSES_TXT_PATH = "./img_train/classes.txt"
IMG_TRAIN_JSON_PATH = "./img_train/img_train.json"
IMG_TRAIN_PATH = "./img_train"
TRAIN_DATASET_NAME = "img_train"
MODEL_CONFIG_PATH = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
CKPT_FOLDER_PATH = os.path.join(os.getcwd(), "ckpt/img_train")

def get_class_names(txt_path):
    with open(txt_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

class_names = get_class_names(CLASSES_TXT_PATH)
coins_metadata = MetadataCatalog.get(TRAIN_DATASET_NAME).set(thing_classes=class_names)  # Set metadata here
dataset_dicts = load_coco_json(IMG_TRAIN_JSON_PATH, IMG_TRAIN_PATH, TRAIN_DATASET_NAME)
DatasetCatalog.register(TRAIN_DATASET_NAME, lambda: dataset_dicts)  # Register dataset

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG_PATH))
cfg.DATASETS.TRAIN = (TRAIN_DATASET_NAME,)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_CONFIG_PATH)
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0004
cfg.SOLVER.MAX_ITER = 1500
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)
os.makedirs(CKPT_FOLDER_PATH, exist_ok=True)
cfg.OUTPUT_DIR = CKPT_FOLDER_PATH

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

setup_logger()