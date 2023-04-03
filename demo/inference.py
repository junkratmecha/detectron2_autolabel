import os
import glob
import cv2

from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from label_saver import get_class_names, get_image_files, save_coco_annotation, save_yolo_label, save_pascal_voc_annotation, convert_to_yolo_label

setup_logger()
MODEL_WEIGHTS_PATH = os.path.join("ckpt/img_train", "model_final.pth")
CLASS_NAMES_PATH = "./img_train/classes.txt"
IMAGE_FOLDER = "./img_test"
COCO_ANNOTATIONS_OUTPUT_FOLDER = "labels"
CONFIDENCE_THRESHOLD = 0.7

def detect_and_visualize(image_path, predictor, cfg, class_names, save_yolo_label, save_pascal_voc_annotation):
    im = cv2.imread(image_path)
    if im is None:
        print(f"画像ファイルの読み込みに失敗しました: {image_path}")
        return
    # coco dataset
    # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # custom dataset
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get("img_train"), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    outputs = predictor(im)

    # 画像を確認する
    # cv2.imshow("Object Detection", v.get_image()[:, :, ::-1])
    # cv2.waitKey(0)

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
        # Save Pascal VOC annotations
        save_pascal_voc_annotation(image_path, class_id, x1, y1, x2, y2, class_names)

    return {"image_path": image_path, "annotations": annotations}


def main():
    cfg = get_cfg()
    # pre-trained model
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")]
    # custom dataset
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.WEIGHTS = MODEL_WEIGHTS_PATH
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    class_names = get_class_names(CLASS_NAMES_PATH)
    MetadataCatalog.get("img_train").set(thing_classes=class_names)
    predictor = DefaultPredictor(cfg)
    all_annotations = []
    image_files = get_image_files(IMAGE_FOLDER)

    for image_file in image_files:
        annotations = detect_and_visualize(image_file, predictor, cfg, class_names, save_yolo_label, save_pascal_voc_annotation)
        if annotations is not None:
            all_annotations.append(annotations)

    save_coco_annotation(IMAGE_FOLDER, all_annotations, COCO_ANNOTATIONS_OUTPUT_FOLDER, class_names)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()