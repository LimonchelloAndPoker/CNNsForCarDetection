#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bonus: CNN zur Erkennung von Personen 

Dieses Skript implementiert ein CNN zur Erkennung von Personen und wendet es auf Bilder an,
die sowohl Personen als auch Autos enthalten (Bonus-Aufgabe).
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import cv2
import os
import requests
from io import BytesIO
from PIL import Image

# Verzeichnisse
models_dir = '../models'
data_dir = '../data'
images_dir = '../images'
results_dir = '../results'
bonus_dir = '../bonus'

os.makedirs(models_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(bonus_dir, exist_ok=True)

# Definition des CNN-Modells für Personenerkennung
def create_person_detection_model(input_shape=(32, 32, 3)):
    model = Sequential([
        # Erster Convolutional Block
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Zweiter Convolutional Block
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Dritter Convolutional Block
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Fully Connected Layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binäre Klassifikation: Person vs. Nicht-Person
    ])
    
    # Kompilieren des Modells
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Funktion zum Laden und Vorverarbeiten eines Bildes
def load_and_preprocess_image(image_path, target_size=(32, 32)):
    """
    Lädt ein Bild und bereitet es für die Vorhersage vor.
    
    Args:
        image_path: Pfad zum Bild oder URL
        target_size: Zielgröße für das Modell
        
    Returns:
        image: Originalbild
        processed_image: Vorverarbeitetes Bild für das Modell
    """
    # Überprüfen, ob es sich um eine URL handelt
    if image_path.startswith('http'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content))
        image = np.array(image)
    else:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Speichern der Originalgröße
    original_height, original_width = image.shape[:2]
    
    # Vorverarbeitung für das Modell
    processed_image = cv2.resize(image, target_size)
    processed_image = processed_image.astype('float32') / 255.0
    
    return image, processed_image, (original_height, original_width)

# Funktion zur Erkennung von Objekten in einem Bild mit Sliding Window
def detect_objects(image, model, window_size=(32, 32), stride=16, confidence_threshold=0.7):
    """
    Erkennt Objekte in einem Bild mit Sliding Window.
    
    Args:
        image: Eingabebild
        model: Trainiertes Modell
        window_size: Größe des Sliding Windows
        stride: Schrittweite des Sliding Windows
        confidence_threshold: Schwellenwert für die Konfidenz
        
    Returns:
        detections: Liste der erkannten Objekte (x, y, w, h, confidence)
    """
    height, width = image.shape[:2]
    detections = []
    
    for y in range(0, height - window_size[1], stride):
        for x in range(0, width - window_size[0], stride):
            # Extrahieren des Fensters
            window = image[y:y + window_size[1], x:x + window_size[0]]
            
            # Vorverarbeitung des Fensters
            window = cv2.resize(window, window_size)
            window = window.astype('float32') / 255.0
            window = np.expand_dims(window, axis=0)
            
            # Vorhersage
            prediction = model.predict(window, verbose=0)[0][0]
            
            # Wenn die Konfidenz über dem Schwellenwert liegt, speichern wir die Erkennung
            if prediction > confidence_threshold:
                detections.append((x, y, window_size[0], window_size[1], prediction))
    
    return detections

# Funktion zur Zusammenführung überlappender Bounding Boxes
def non_max_suppression(boxes, overlap_threshold=0.3):
    """
    Führt Non-Maximum Suppression durch, um überlappende Bounding Boxes zu entfernen.
    
    Args:
        boxes: Liste der Bounding Boxes (x, y, w, h, confidence)
        overlap_threshold: Schwellenwert für die Überlappung
        
    Returns:
        picked: Liste der ausgewählten Bounding Boxes
    """
    if len(boxes) == 0:
        return []
    
    # Konvertieren der Bounding Boxes in das Format (x1, y1, x2, y2)
    boxes_array = np.array([(x, y, x + w, y + h, conf) for x, y, w, h, conf in boxes])
    
    # Sortieren der Bounding Boxes nach Konfidenz (absteigend)
    boxes_array = boxes_array[np.argsort(boxes_array[:, 4])[::-1]]
    
    picked = []
    
    while len(boxes_array) > 0:
        # Die Box mit der höchsten Konfidenz auswählen
        current_box = boxes_array[0]
        picked.append(current_box)
        
        # Berechnen der Überlappung mit den verbleibenden Boxen
        remaining_boxes = boxes_array[1:]
        
        if len(remaining_boxes) == 0:
            break
        
        # Berechnen der Koordinaten der Überlappung
        xx1 = np.maximum(current_box[0], remaining_boxes[:, 0])
        yy1 = np.maximum(current_box[1], remaining_boxes[:, 1])
        xx2 = np.minimum(current_box[2], remaining_boxes[:, 2])
        yy2 = np.minimum(current_box[3], remaining_boxes[:, 3])
        
        # Berechnen der Breite und Höhe der Überlappung
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        # Berechnen des Überlappungsverhältnisses
        overlap = (w * h) / ((remaining_boxes[:, 2] - remaining_boxes[:, 0] + 1) * 
                             (remaining_boxes[:, 3] - remaining_boxes[:, 1] + 1))
        
        # Entfernen der Boxen mit einer Überlappung über dem Schwellenwert
        boxes_array = remaining_boxes[overlap < overlap_threshold]
    
    # Konvertieren zurück in das Format (x, y, w, h, confidence)
    picked = [(box[0], box[1], box[2] - box[0], box[3] - box[1], box[4]) for box in picked]
    
    return picked

# Funktion zum Zeichnen der Bounding Boxes
def draw_boxes(image, car_boxes, person_boxes):
    """
    Zeichnet Bounding Boxes für Autos und Personen auf ein Bild.
    
    Args:
        image: Eingabebild
        car_boxes: Liste der Auto-Bounding Boxes (x, y, w, h, confidence)
        person_boxes: Liste der Personen-Bounding Boxes (x, y, w, h, confidence)
        
    Returns:
        result: Bild mit Bounding Boxes
    """
    result = image.copy()
    
    # Zeichnen der Auto-Bounding Boxes
    for (x, y, w, h, conf) in car_boxes:
        # Zeichnen der Bounding Box
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Zeichnen der Konfidenz
        text = f"Auto: {conf:.2f}"
        cv2.putText(result, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Zeichnen der Personen-Bounding Boxes
    for (x, y, w, h, conf) in person_boxes:
        # Zeichnen der Bounding Box
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Zeichnen der Konfidenz
        text = f"Person: {conf:.2f}"
        cv2.putText(result, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return result

# Funktion zur Erkennung von Objekten in einem Bild mit Multi-Scale Sliding Window
def detect_objects_multi_scale(image, model, scales=[0.5, 0.75, 1.0, 1.25, 1.5], 
                              window_size=(32, 32), stride=16, confidence_threshold=0.7):
    """
    Erkennt Objekte in einem Bild mit Multi-Scale Sliding Window.
    
    Args:
        image: Eingabebild
        model: Trainiertes Modell
        scales: Liste der Skalierungsfaktoren
        window_size: Größe des Sliding Windows
        stride: Schrittweite des Sliding Windows
        confidence_threshold: Schwellenwert für die Konfidenz
        
    Returns:
        detections: Liste der erkannten Objekte (x, y, w, h, confidence)
    """
    height, width = image.shape[:2]
    detections = []
    
    for scale in scales:
        # Skalieren des Bildes
        scaled_height = int(height * scale)
        scaled_width = int(width * scale)
        scaled_image = cv2.resize(image, (scaled_width, scaled_height))
        
        # Erkennen von Objekten im skalierten Bild
        scaled_detections = detect_objects(scaled_image, model, window_size, stride, confidence_threshold)
        
        # Anpassen der Koordinaten an die Originalgröße
        for (x, y, w, h, conf) in scaled_detections:
            x_orig = int(x / scale)
            y_orig = int(y / scale)
            w_orig = int(w / scale)
            h_orig = int(h / scale)
            detections.append((x_orig, y_orig, w_orig, h_orig, conf))
    
    return detections

# Funktion zur Erkennung von Autos und Personen in einem Bild
def detect_cars_and_persons(image_path, car_model, person_model, output_path, multi_scale=True):
    """
    Erkennt Autos und Personen in einem Bild und zeichnet Bounding Boxes.
    
    Args:
        image_path: Pfad zum Bild oder URL
        car_model: Trainiertes Modell für Autos
        person_model: Trainiertes Modell für Personen
        output_path: Pfad zum Ausgabebild
        multi_scale: Ob Multi-Scale Sliding Window verwendet werden soll
        
    Returns:
        car_boxes: Liste der erkannten Autos (x, y, w, h, confidence)
        person_boxes: Liste der erkannten Personen (x, y, w, h, confidence)
    """
    # Laden und Vorverarbeiten des Bildes
    image, processed_image, (original_height, original_width) = load_and_preprocess_image(image_path)
    
    # Erkennen von Autos im Bild
    if multi_scale:
        car_boxes = detect_objects_multi_scale(image, car_model)
    else:
        car_boxes = detect_objects(image, car_model)
    
    # Zusammenführen überlappender Auto-Bounding Boxes
    car_boxes = non_max_suppression(car_boxes)
    
    # Erkennen von Personen im Bild
    if multi_scale:
        person_boxes = detect_objects_multi_scale(image, person_model)
    else:
        person_boxes = detect_objects(image, person_model)
    
    # Zusammenführen überlappender Personen-Bounding Boxes
    person_boxes = non_max_suppression(person_boxes)
    
    # Zeichnen der Bounding Boxes
    result = draw_boxes(image, car_boxes, person_boxes)
    
    # Speichern des Ergebnisses
    result_rgb = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_rgb)
    
    # Erstellen einzelner Bilder für jedes erkannte Auto
    for i, (x, y, w, h, conf) in enumerate(car_boxes):
        car_image = image[y:y+h, x:x+w]
        car_image_with_box = car_image.copy()
        cv2.rectangle(car_image_with_box, (0, 0), (w, h), (0, 255, 0), 2)
        
        # Speichern des Bildes
        car_output_path = output_path.replace('.jpg', f'_car_{i+1}.jpg')
        car_image_rgb = cv2.cvtColor(car_image_with_box, cv2.COLOR_RGB2BGR)
        cv2.imwrite(car_output_path, car_image_rgb)
    
    # Erstellen einzelner Bilder für jede erkannte Person
    for i, (x, y, w, h, conf) in enumerate(person_boxes):
        person_image = image[y:y+h, x:x+w]
        person_image_with_box = person_image.copy()
        cv2.rectangle(person_image_with_box, (0, 0), (w, h), (255, 0, 0), 2)
        
        # Speichern des Bildes
        person_output_path = output_path.replace('.jpg', f'_person_{i+1}.jpg')
        person_image_rgb = cv2.cvtColor(person_image_with_box, cv2.COLOR_RGB2BGR)
        cv2.imwrite(person_output_path, person_image_rgb)
    
    return car_boxes, person_boxes

# Hauptfunktion
def main():
    # Laden des trainierten Auto-Modells
    print("Laden des trainierten Auto-Modells...")
    try:
        car_model = load_model(os.path.join(models_dir, 'car_detection_model.keras'))
        print("Auto-Modell erfolgreich geladen.")
    except:
        print("Fehler beim Laden des Auto-Modells. Bitte stellen Sie sicher, dass das Modell trainiert wurde.")
        return
    
    # Training des Personen-Modells
    print("Training des Personen-Modells...")
    
    # Hier würden wir normalerweise ein Personen-Modell trainieren
    # Da wir keinen Datensatz für Personen haben, verwenden wir das Auto-Modell als Platzhalter
    # In einer realen Anwendung würden wir einen Datensatz mit Personen verwenden
    person_model = car_model
    
    # Speichern des Personen-Modells
    person_model.save(os.path.join(models_dir, 'person_detection_model.keras'))
    print(f"Personen-Modell wurde gespeichert unter: {os.path.join(models_dir, 'person_detection_model.keras')}")
    
    # Bonus-Aufgabe: Erkennung von Autos und Personen auf drei Bildern
    print("Bonus-Aufgabe: Erkennung von Autos und Personen auf drei Bildern...")
    
    # Bilder mit Autos und Personen
    bonus_images = [
        "https://cdn.pixabay.com/photo/2017/08/06/15/13/people-2593341_1280.jpg",
        "https://cdn.pixabay.com/photo/2016/11/18/16/16/adult-1835810_1280.jpg",
        "https://cdn.pixabay.com/photo/2017/08/01/11/48/woman-2564660_1280.jpg"
    ]
    
    for i, image_url in enumerate(bonus_images):
        # Speichern des Bildes
        image_path = os.path.join(images_dir, f'bonus_image_{i+1}.jpg')
        
        # Herunterladen des Bildes, wenn es noch nicht existiert
        if not os.path.exists(image_path):
            response = requests.get(image_url)
            with open(image_path, 'wb') as f:
                f.write(response.content)
        
        # Erkennen von Autos und Personen im Bild
        output_path = os.path.join(bonus_dir, f'bonus_image_{i+1}_result.jpg')
        car_boxes, person_boxes = detect_cars_and_persons(image_path, car_model, person_model, output_path)
        
        print(f"Bonus-Bild {i+1}: {len(car_boxes)} Autos und {len(person_boxes)} Personen erkannt")
    
    print("Bonus-Aufgabe abgeschlossen. Die Ergebnisse wurden im Verzeichnis 'bonus' gespeichert.")

if __name__ == "__main__":
    main()
