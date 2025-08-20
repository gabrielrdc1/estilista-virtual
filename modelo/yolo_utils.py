import os
import cv2
from ultralytics import YOLO


def segmentar_pecas(img_path, output_folder='static/temp_crops', yolo_model_path='yolov8n.pt'):
    """
    Detecta e recorta peças de roupa usando YOLOv8, salvando cortes em disco e retornando caminhos.
    """
    os.makedirs(output_folder, exist_ok=True)
    model = YOLO(yolo_model_path)
    results = model(img_path)
    img = cv2.imread(img_path)
    crop_paths = []
    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]
        crop_filename = f"crop_{i}_{os.path.basename(img_path)}"
        crop_path = os.path.join(output_folder, crop_filename)
        cv2.imwrite(crop_path, crop)
        crop_paths.append(crop_path)
    return crop_paths
