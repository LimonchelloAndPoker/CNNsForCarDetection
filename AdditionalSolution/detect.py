import cv2
import torch
import numpy as np
from pathlib import Path

def detect_and_crop_yolov5(model_path, image_path, output_dir='output', conf_thresh=0.5):
    # YOLOv5 spezifisches Loading
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    
    # Ordner erstellen
    Path(output_dir).mkdir(exist_ok=True)
    full_output = Path(output_dir) / 'full'
    crops_output = Path(output_dir) / 'crops'
    full_output.mkdir(exist_ok=True)
    crops_output.mkdir(exist_ok=True)

    # Inference mit YOLOv5
    results = model(image_path)
    
    # Originalbild laden
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")

    # Ergebnisparsing für YOLOv5
    pandas_results = results.pandas().xyxy[0]
    
    # Boxen zeichnen und Crops speichern
    for i, row in pandas_results.iterrows():
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        
        # Zeichne Box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, f"{row['name']} {row['confidence']:.2f}", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        
        # Speichere Crop
        crop = img[y1:y2, x1:x2]
        cv2.imwrite(str(crops_output / f"{Path(image_path).stem}_crop_{i}.jpg"), crop)

    # Speichere volles Bild
    full_path = full_output / f"{Path(image_path).stem}_detected.jpg"
    cv2.imwrite(str(full_path), img)
    
    return str(full_path), [str(p) for p in crops_output.glob('*')]



# Erste Ausführung: YOLOv5 Dependencies laden
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True  # Fix für Colab




MODEL_PFAD = "yolov5/runs/train/exp/weights/best.pt"
BILD_PFAD = "../images/bild6.jpg"
AUSGABE_ORDNER = "ergebnisse"
KONFIDENZ = 0.4

# Führe Detection aus
vollbild, crops = detect_and_crop_yolov5(
    model_path=MODEL_PFAD,
    image_path=BILD_PFAD,
    output_dir=AUSGABE_ORDNER,
    conf_thresh=KONFIDENZ
)

# Ausgabe der Ergebnisse
print(f"Ergebnisbild: {vollbild}")
print(f"Gefundene Objekte: {len(crops)}")
for crop in crops:
    print(f" - {crop}")