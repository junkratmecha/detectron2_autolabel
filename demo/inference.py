import os
import glob
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import random
import json
from pathlib import Path

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def get_class_names(txt_path):
    with open(txt_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def get_image_files(folder_path):
    extensions = ['jpg', 'jpeg', 'png']
    return [f for ext in extensions for f in glob.glob(os.path.join(folder_path, f'*.{ext}'))]

def convert_to_yolo_label(box, width, height):
    x_center = (box[0] + box[2]) / 2 / width
    y_center = (box[1] + box[3]) / 2 / height
    w = (box[2] - box[0]) / width
    h = (box[3] - box[1]) / height
    return x_center, y_center, w, h

def detect_and_visualize(image_path, predictor, cfg, class_names):
    im = cv2.imread(image_path)
    if im is None:
        print(f"画像ファイルの読み込みに失敗しました: {image_path}")
        return

    # coco dataset
    # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)

    # custom dataset
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get("robot"), scale=1.2)

    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    outputs = predictor(im)

    # 少量で画像を確認する
    cv2.imshow("Object Detection", v.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    instances = outputs['instances']
    height, width = im.shape[:2]
    annotations = []
    for idx, bbox in enumerate(instances.pred_boxes):
        class_id = instances.pred_classes[idx].item()
        x1, y1, x2, y2 = bbox.tolist()
        width, height = x2 - x1, y2 - y1
        coco_bbox = [x1, y1, width, height]
        annotations.append({"class_id": class_id, "bbox": coco_bbox})
        # YOLOアノテーションの保存
        x_center, y_center, w, h = convert_to_yolo_label(bbox, width, height)
        save_yolo_label(image_path, class_id, x_center, y_center, w, h)

    return {"image_path": image_path, "annotations": annotations}


def save_coco_annotation(image_folder, all_annotations, output_folder, class_names):
    coco_annotations = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": idx, "name": name} for idx, name in enumerate(class_names)
        ],
    }

    annotation_id = 0
    for item in all_annotations:
        image_path = item["image_path"]
        image_id = Path(image_path).stem
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        coco_annotations["images"].append({
            "id": image_id,
            "dataset_id": 1,  # or other dataset id you want to use
            "category_ids": [annotation["class_id"] for annotation in item["annotations"]],
            "path": image_path,
            "width": width,
            "height": height,
            "file_name": os.path.basename(image_path),
            "annotated": True,
            "annotating": [],
            "num_annotations": len(item["annotations"]),
            "metadata": {},
            "milliseconds": 0,
            "events": [],
            "regenerate_thumbnail": False,
            "is_modified": False
        })

        for annotation in item["annotations"]:
            coco_annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": annotation["class_id"],
                "bbox": annotation["bbox"],
                "segmentation": [],
                "iscrowd": 0,
            }
            coco_annotations["annotations"].append(coco_annotation)
            annotation_id += 1

    output_path = os.path.join(output_folder, "annotations_coco.json")
    with open(output_path, "w") as f:
        json.dump(coco_annotations, f, indent=2)


def save_yolo_label(image_path, class_id, x_center, y_center, w, h):
    label_path = os.path.join(os.path.dirname(os.path.abspath(image_path)), "..", "labels")
    os.makedirs(label_path, exist_ok=True)
    label_file = os.path.join(label_path, os.path.splitext(os.path.basename(image_path))[0] + ".txt")

    with open(label_file, "a") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

def main():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    # pre-trained model
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    # custom dataset
    cfg.MODEL.WEIGHTS = os.path.join("ckpt/robot", "model_final.pth")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    class_names = get_class_names("./img_train/classes.txt")
    MetadataCatalog.get("robot").set(thing_classes=class_names)
    image_folder = "./img_test"

    predictor = DefaultPredictor(cfg)
    all_annotations = []
    image_files = get_image_files(image_folder)

    for image_file in image_files:
        annotations = detect_and_visualize(image_file, predictor, cfg, class_names)
        if annotations is not None:
            all_annotations.append(annotations)

    #coco-annotatorの形式で保存
    save_coco_annotation(image_folder, all_annotations, "labels", class_names)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()