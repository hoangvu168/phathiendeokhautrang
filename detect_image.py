import cv2
import os
from ultralytics import YOLO

# Load model YOLO
model = YOLO("best.pt")

def detect_mask(image_path):
    img = cv2.imread(image_path)
    results = model(image_path)

    total_mask, total_no_mask = 0, 0

    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            label = "Mask" if cls == 0 else "No Mask"

            if cls == 0:
                total_mask += 1
                color = (0, 255, 0)
            else:
                total_no_mask += 1
                color = (0, 0, 255)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    output_path = os.path.join("static/results", os.path.basename(image_path))
    cv2.imwrite(output_path, img)

    return output_path, total_mask, total_no_mask
