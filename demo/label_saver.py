import os
import glob
import json
import cv2
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from detectron2.utils.visualizer import Visualizer

from detectron2.data import DatasetCatalog, MetadataCatalog

YOLO_LABELS_OUTPUT_FOLDER = "labels/yolo"
PASCAL_VOC_ANNOTATIONS_OUTPUT_FOLDER = "labels/pascal_voc"

def get_class_names(txt_path):
    with open(txt_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def get_image_files(folder_path):
    extensions = ['jpg', 'jpeg', 'png']
    return [f for ext in extensions for f in glob.glob(os.path.join(folder_path, f'*.{ext}'))]

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

    output_path = os.path.join(output_folder, "coco.json")
    with open(output_path, "w") as f:
        json.dump(coco_annotations, f, indent=2)

def save_yolo_label(image_path, class_id, x_center, y_center, w, h):
    label_path = os.path.join(os.path.dirname(os.path.abspath(image_path)), "..", YOLO_LABELS_OUTPUT_FOLDER)
    os.makedirs(label_path, exist_ok=True)
    label_file = os.path.join(label_path, os.path.splitext(os.path.basename(image_path))[0] + ".txt")

    with open(label_file, "a") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

def convert_to_yolo_label(box, width, height):
    x_center = (box[0] + box[2]) / 2 / width
    y_center = (box[1] + box[3]) / 2 / height
    w = (box[2] - box[0]) / width
    h = (box[3] - box[1]) / height
    return x_center, y_center, w, h

def save_pascal_voc_annotation(image_path, class_id, x1, y1, x2, y2, class_names):
    image_folder_path = os.path.dirname(os.path.abspath(image_path))
    annotation_folder = os.path.join(image_folder_path, "..", PASCAL_VOC_ANNOTATIONS_OUTPUT_FOLDER)
    os.makedirs(annotation_folder, exist_ok=True)
    annotation_file = os.path.join(annotation_folder, os.path.splitext(os.path.basename(image_path))[0] + ".xml")

    if not os.path.exists(annotation_file):
        create_pascal_voc_xml(image_path, annotation_file)
    tree = ET.parse(annotation_file)

    root = tree.getroot()
    obj = ET.SubElement(root, "object")
    ET.SubElement(obj, "name").text = class_names[class_id]
    ET.SubElement(obj, "pose").text = "Unspecified"
    ET.SubElement(obj, "truncated").text = "0"
    ET.SubElement(obj, "difficult").text = "0"
    bndbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(x1)
    ET.SubElement(bndbox, "ymin").text = str(y1)
    ET.SubElement(bndbox, "xmax").text = str(x2)
    ET.SubElement(bndbox, "ymax").text = str(y2)

    tree.write(annotation_file)

def create_pascal_voc_xml(image_path, annotation_file):
    img = cv2.imread(image_path)
    height, width, depth = img.shape

    root = ET.Element("annotation")
    folder = ET.SubElement(root, "folder")
    folder.text = "images"
    filename = ET.SubElement(root, "filename")
    filename.text = os.path.basename(image_path)
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"
    tree = ET.ElementTree(root)
    tree.write(annotation_file)