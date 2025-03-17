# Anwendung von Convolutional Neural Networks (CNN) zur Bilderkennung von Autos

Dieses Repository enthält die Implementierung verschiedener CNN-Modelle zur Bilderkennung von Autos im Rahmen eines Praxisprojekts.

## Projektstruktur

- `notebooks/`: Jupyter Notebooks und Python-Skripte für die Implementierung
  - `01_cifar10_dataset_preparation.py`: Vorbereitung des CIFAR-10 Datensatzes
  - `02_cnn_keras_tensorflow.py`: Implementierung eines CNN mit Keras/TensorFlow
  - `03_custom_cnn_implementation.py`: Implementierung eines CNN ohne keras.models oder keras.layers
  - `04_pretrained_cnn.py`: Laden und Anpassen eines vortrainierten CNN
  - `05_car_detection_on_images.py`: Automerkennung auf Bildern
  - `06_bonus_person_detection.py`: Bonus-Aufgabe zur Personenerkennung
- `data/`: Datensätze und vorverarbeitete Daten
- `models/`: Trainierte Modelle und Visualisierungen
- `images/`: Testbilder für die Automerkennung
- `results/`: Ergebnisse der Automerkennung
- `bonus/`: Ergebnisse der Bonus-Aufgabe
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

## Verwendung

1. Vorbereitung des CIFAR-10 Datensatzes:
   ```
   python notebooks/01_cifar10_dataset_preparation.py
   ```

2. Training des CNN mit Keras/TensorFlow:
   ```
   python notebooks/02_cnn_keras_tensorflow.py
   ```

3. Implementierung eines benutzerdefinierten CNN:
   ```
   python notebooks/03_custom_cnn_implementation.py
   ```

4. Laden und Anpassen eines vortrainierten CNN:
   ```
   python notebooks/04_pretrained_cnn.py
   ```

5. Automerkennung auf Bildern:
   ```
   python notebooks/05_car_detection_on_images.py
   ```

6. Bonus-Aufgabe zur Personenerkennung:
   ```
   python notebooks/06_bonus_person_detection.py
   ```

## Ergebnisse

Die Ergebnisse der Automerkennung sind im Verzeichnis `results/` gespeichert. Die Ergebnisse der Bonus-Aufgabe sind im Verzeichnis `bonus/` gespeichert.

Eine ausführliche Dokumentation der Ergebnisse und Lösungsschritte ist im Verzeichnis `docs/` zu finden.
