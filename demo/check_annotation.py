import os
import json
import cv2
import glob

def get_image_files(folder_path):
    extensions = ['jpg', 'jpeg', 'png']
    return [f for ext in extensions for f in glob.glob(os.path.join(folder_path, f'*.{ext}'))]

def load_annotations(annotation_file):
    with open(annotation_file, "r") as f:
        annotations = json.load(f)
    return annotations

def load_class_names(class_file):
    with open(class_file, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def display_annotations(image_folder, annotations, class_names):
    for image_data in annotations["images"]:
        image_path = os.path.join(image_folder, image_data["file_name"])
        img = cv2.imread(image_path)
        if img is None:
            print(f"画像ファイルの読み込みに失敗しました: {image_path}")
            continue

        for annotation in annotations["annotations"]:
            if annotation["image_id"] == image_data["id"]:
                category_id = annotation["category_id"]
                class_name = class_names[category_id]
                bbox = annotation["bbox"]
                x1, y1, width, height = [int(coord) for coord in bbox]
                x2, y2 = x1 + width, y1 + height
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Annotated Image", img)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break

    cv2.destroyAllWindows()

def main():
    image_folder = "./img_test"  # 画像フォルダを指定
    annotation_file = "./labels/annotations_coco.json"  # アノテーションファイルを指定
    class_file = "./labels/classes.txt"  # クラス名ファイルを指定

    annotations = load_annotations(annotation_file)
    class_names = load_class_names(class_file)
    display_annotations(image_folder, annotations, class_names)

if __name__ == "__main__":
    main()
