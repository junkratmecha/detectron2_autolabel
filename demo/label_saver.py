import os
import cv2
import xml.etree.ElementTree as ET

def save_yolo_label(image_path, class_id, x_center, y_center, w, h):
    label_path = os.path.join(os.path.dirname(os.path.abspath(image_path)), "..", "labels")
    os.makedirs(label_path, exist_ok=True)
    label_file = os.path.join(label_path, os.path.splitext(os.path.basename(image_path))[0] + ".txt")

    with open(label_file, "a") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

def save_pascal_voc_annotation(image_path, class_id, x1, y1, x2, y2, class_names):
    image_folder_path = os.path.dirname(os.path.abspath(image_path))
    annotation_folder = os.path.join(image_folder_path, "..", "annotations_voc")
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
