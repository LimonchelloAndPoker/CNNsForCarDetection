import os
import cv2
import torch
from pathlib import Path

# Load the YOLOv5 model via PyTorch Hub (this avoids serialization errors)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# Target classes
TARGET_CLASSES = ['person', 'car']

def find_person_and_cars_in_folder(folder_path):
    # Create output directories
    base_output_dir = Path('Additional_results')
    full_image_dir = base_output_dir / 'full_images'
    cropped_dir = base_output_dir / 'cropped_images'

    full_image_dir.mkdir(parents=True, exist_ok=True)
    cropped_dir.mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            # Inference
            results = model(image)
            detections = results.xyxy[0]  # (x1, y1, x2, y2, conf, cls)

            detection_found = False
            for idx, (*box, conf, cls) in enumerate(detections):
                label = model.names[int(cls)]
                if label in TARGET_CLASSES:
                    detection_found = True
                    x1, y1, x2, y2 = map(int, box)

                    # Draw box and label
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Save cropped detection
                    crop = image[y1:y2, x1:x2]
                    class_dir = cropped_dir / label
                    class_dir.mkdir(parents=True, exist_ok=True)
                    crop_filename = class_dir / f"{Path(filename).stem}_{idx}.jpg"
                    cv2.imwrite(str(crop_filename), crop)

            if detection_found:
                full_image_output_path = full_image_dir / filename
                cv2.imwrite(str(full_image_output_path), image)

    print("Processing complete. Results saved in 'Additional_results/'.")


find_person_and_cars_in_folder("../images")
