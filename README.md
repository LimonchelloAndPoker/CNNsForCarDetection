# Anwendung von Convolutional Neural Networks (CNN) zur Bilderkennung von Autos

Dieses Repository enthält die Implementierung verschiedener CNN-Modelle zur Bilderkennung von Autos im Rahmen eines Praxisprojekts.

## Projektstruktur

- `notebooks_jupyter/`: Jupyter Notebooks für die Implementierung (primärer Teil des Projekts)
  - `01_cifar10_dataset_preparation.ipynb`: Vorbereitung des CIFAR-10 Datensatzes
  - `02_cnn_keras_tensorflow.ipynb`: Implementierung eines CNN mit Keras/TensorFlow
  - `03_custom_cnn_implementation.ipynb`: Implementierung eines CNN ohne keras.models oder keras.layers
  - `04_pretrained_cnn.ipynb`: Laden und Anpassen eines vortrainierten CNN
  - `05_car_detection_on_images.ipynb`: Automerkennung auf Bildern
  - `05_car_detection_on_images_improved.ipynb`: Verbesserte Automerkennung mit Selective Search
  - `06_bonus_person_detection.ipynb`: Bonus-Aufgabe zur Personenerkennung
  - `07_documentation.ipynb`: Dokumentation des Projekts
- `notebooks/`: Python-Skript-Versionen der Jupyter Notebooks (für Batch-Verarbeitung)
- `data/`: Datensätze und vorverarbeitete Daten
- `models/`: Trainierte Modelle und Visualisierungen
  - `keras_cnn/`: Modelle mit Keras/TensorFlow
  - `custom_cnn/`: Benutzerdefinierte CNN-Implementierungen
  - `pretrained_cnn/`: Vortrainierte CNN-Modelle
- `images/`: Testbilder für die Automerkennung
- `results/`: Ergebnisse der Automerkennung
- `docs/`: Dokumentation und Präsentation

## Aufgaben

### Aufgabe 1: Entwicklung von CNN-Modellen

- a) Entwicklung eines CNN mit Keras/TensorFlow zur Erkennung von Autos im CIFAR-10 Datensatz
- b) Implementierung eines CNN ohne Verwendung von keras.models oder keras.layers
- c) Laden und Anpassen eines vortrainierten CNN

### Aufgabe 2: Automerkennung auf Bildern

- a) Erkennung von Autos auf drei gegebenen Bildern
- b) Erkennung von Autos auf drei weiteren Bildern aus dem Internet

### Bonus: Personenerkennung

- Entwicklung eines zweiten CNN zur Erkennung von Personen
- Erkennung von Personen und Autos auf drei weiteren Bildern

## Verwendung der Jupyter Notebooks

Die Jupyter Notebooks sind der primäre Teil dieses Projekts und bieten eine interaktive Umgebung zur Ausführung und Visualisierung der Ergebnisse.

1. Vorbereitung des CIFAR-10 Datensatzes:
   ```
   jupyter notebook notebooks_jupyter/01_cifar10_dataset_preparation.ipynb
   ```

2. Training des CNN mit Keras/TensorFlow:
   ```
   jupyter notebook notebooks_jupyter/02_cnn_keras_tensorflow.ipynb
   ```

3. Implementierung eines benutzerdefinierten CNN:
   ```
   jupyter notebook notebooks_jupyter/03_custom_cnn_implementation.ipynb
   ```

4. Laden und Anpassen eines vortrainierten CNN:
   ```
   jupyter notebook notebooks_jupyter/04_pretrained_cnn.ipynb
   ```

5. Automerkennung auf Bildern:
   ```
   jupyter notebook notebooks_jupyter/05_car_detection_on_images.ipynb
   ```

6. Verbesserte Automerkennung mit Selective Search:
   ```
   jupyter notebook notebooks_jupyter/05_car_detection_on_images_improved.ipynb
   ```

7. Bonus-Aufgabe zur Personenerkennung:
   ```
   jupyter notebook notebooks_jupyter/06_bonus_person_detection.ipynb
   ```

8. Dokumentation des Projekts:
   ```
   jupyter notebook notebooks_jupyter/07_documentation.ipynb
   ```

## Verbesserungen in der Automerkennung

Die verbesserte Version der Automerkennung (`05_car_detection_on_images_improved.ipynb`) verwendet folgende Techniken:

1. **Selective Search für Region Proposals** - Anstelle des Sliding-Window-Ansatzes wird Selective Search verwendet, um potenzielle Regionen vorzuschlagen, in denen sich Autos befinden könnten.
2. **Verbesserte Multi-Scale-Erkennung** - Erweiterte Skalierungsfaktoren für eine bessere Erkennung von Autos unterschiedlicher Größen.
3. **HOG-Feature-Extraktion** - Histogram of Oriented Gradients zur Verbesserung der Erkennungsgenauigkeit.
4. **Optimierte Non-Maximum Suppression** - Verbesserte Algorithmen zur Entfernung überlappender Bounding Boxes.

Diese Verbesserungen ermöglichen eine zuverlässigere Erkennung von Autos in Bildern, insbesondere in komplexen Szenen mit mehreren Autos.

## Ergebnisse

Die Ergebnisse der Automerkennung sind im Verzeichnis `results/` gespeichert. Die Ergebnisse der Bonus-Aufgabe sind im Verzeichnis `bonus/` gespeichert.

Eine ausführliche Dokumentation der Ergebnisse und Lösungsschritte ist im Jupyter Notebook `07_documentation.ipynb` zu finden.
