#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dokumentation der Ergebnisse und Lösungsschritte

Dieses Skript dokumentiert die Ergebnisse und Lösungsschritte des Projekts
zur Anwendung von Convolutional Neural Networks (CNN) zur Bilderkennung von Autos.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

# Verzeichnisse
models_dir = '../models'
data_dir = '../data'
images_dir = '../images'
results_dir = '../results'
docs_dir = '../docs'

os.makedirs(docs_dir, exist_ok=True)

# Dokumentation erstellen
documentation = """# Anwendung von Convolutional Neural Networks (CNN) zur Bilderkennung von Autos

## Einleitung

Dieses Projekt befasst sich mit der Anwendung von Convolutional Neural Networks (CNN) zur Bilderkennung von Autos. Es wurden verschiedene CNN-Modelle implementiert und trainiert, um Autos in Bildern zu erkennen und zu lokalisieren.

## Aufgabe 1: Entwicklung von CNN-Modellen

### a) CNN mit Keras/TensorFlow

Für die erste Aufgabe wurde ein CNN mit Keras/TensorFlow entwickelt, um Autos im CIFAR-10 Datensatz zu erkennen. Der CIFAR-10 Datensatz enthält 60.000 Farbbilder in 10 Klassen, wobei eine der Klassen 'automobile' (Auto) ist.

#### Architektur des Modells

Das CNN-Modell besteht aus mehreren Convolutional Blocks, gefolgt von Fully Connected Layers:

```python
model = Sequential([
    # Erster Convolutional Block
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
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
    Dense(1, activation='sigmoid')  # Binäre Klassifikation: Auto vs. Nicht-Auto
])
```

#### Training und Evaluierung

Das Modell wurde mit dem CIFAR-10 Datensatz trainiert, wobei die Klasse 'automobile' als positive Klasse und alle anderen Klassen als negative Klasse verwendet wurden. Die Bilder wurden normalisiert und in Trainings- und Validierungsdaten aufgeteilt.

Das Training wurde mit dem Adam-Optimizer und der Binary Crossentropy Loss-Funktion durchgeführt. Early Stopping und Model Checkpointing wurden verwendet, um das beste Modell zu speichern.

Die Evaluierung auf den Testdaten zeigte eine hohe Genauigkeit bei der Erkennung von Autos.

### b) CNN ohne Verwendung von keras.models oder keras.layers

Für die zweite Aufgabe wurde ein CNN ohne Verwendung von keras.models oder keras.layers implementiert. Stattdessen wurden die grundlegenden Operationen eines CNN (Faltung, Pooling, Aktivierungsfunktionen, etc.) manuell implementiert.

#### Implementierung der CNN-Funktionen

Es wurden Funktionen für die folgenden Operationen implementiert:
- Initialisierung der Parameter
- Faltung (Convolution)
- ReLU-Aktivierungsfunktion
- Pooling
- Flatten
- Fully Connected Layer
- Sigmoid-Aktivierungsfunktion
- Kostenfunktion (Binary Crossentropy)
- Backpropagation
- Parameteraktualisierung

#### Training und Evaluierung

Das Modell wurde mit einem reduzierten CIFAR-10 Datensatz trainiert, um die Rechenzeit zu reduzieren. Die Implementierung von Backpropagation ermöglichte das Training des Modells von Grund auf.

Die Evaluierung zeigte, dass das selbst implementierte CNN in der Lage ist, Autos zu erkennen, wenn auch mit geringerer Genauigkeit als das Keras/TensorFlow-Modell.

### c) Vortrainiertes CNN

Für die dritte Aufgabe wurde ein vortrainiertes CNN-Modell (MobileNetV2) geladen und für die Autoerkennung angepasst. MobileNetV2 ist ein effizientes CNN, das auf dem ImageNet-Datensatz vortrainiert wurde.

#### Transfer Learning

Die vortrainierten Schichten von MobileNetV2 wurden eingefroren, und es wurden benutzerdefinierte Schichten hinzugefügt, um das Modell für die Autoerkennung anzupassen:

```python
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Einfrieren der vortrainierten Schichten
for layer in base_model.layers:
    layer.trainable = False

# Hinzufügen von benutzerdefinierten Schichten für die Autoerkennung
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Erstellen des Modells
model = Model(inputs=base_model.input, outputs=predictions)
```

#### Training und Evaluierung

Das angepasste Modell wurde mit dem CIFAR-10 Datensatz trainiert, wobei die Bilder auf die Eingabegröße von MobileNetV2 (224x224) skaliert wurden.

Die Evaluierung zeigte, dass das vortrainierte Modell sehr gut in der Lage ist, Autos zu erkennen, auch wenn es auf einem anderen Datensatz vortrainiert wurde.

## Aufgabe 2: Automerkennung auf Bildern

### a) Erkennung von Autos auf drei gegebenen Bildern

Für die erste Aufgabe wurden drei Bilder mit Autos verwendet, um die Automerkennung zu testen. Die Bilder wurden aus dem Internet heruntergeladen und enthielten jeweils mindestens ein Auto.

#### Sliding Window-Ansatz

Zur Erkennung der Autos wurde ein Sliding Window-Ansatz verwendet. Dabei wird ein Fenster über das Bild geschoben und für jede Position eine Vorhersage mit dem trainierten CNN-Modell getroffen. Wenn die Konfidenz über einem Schwellenwert liegt, wird ein Auto erkannt.

Um Autos in verschiedenen Größen zu erkennen, wurde ein Multi-Scale Sliding Window-Ansatz verwendet, bei dem das Bild in verschiedenen Skalierungen analysiert wird.

#### Non-Maximum Suppression

Um überlappende Bounding Boxes zu entfernen, wurde Non-Maximum Suppression angewendet. Dabei werden Bounding Boxes mit hoher Überlappung zusammengeführt, wobei die Box mit der höchsten Konfidenz beibehalten wird.

#### Ergebnisse

Für jedes Bild wurden die erkannten Autos mit Bounding Boxes markiert. Zusätzlich wurden einzelne Bilder für jedes erkannte Auto erstellt.

### b) Erkennung von Autos auf drei weiteren Bildern

Für die zweite Aufgabe wurden drei weitere Bilder aus dem Internet verwendet, die jeweils etwa 5-10 Autos enthielten. Die Bilder wurden mit dem gleichen Ansatz wie in Aufgabe 2a analysiert.

#### Ergebnisse

Die Ergebnisse zeigten, dass das Modell in der Lage ist, Autos in verschiedenen Szenarien zu erkennen, auch wenn die Bilder komplexer sind und mehrere Autos enthalten.

## Bonus: Personenerkennung

Für die Bonus-Aufgabe wurde ein zweites CNN zur Erkennung von Personen entwickelt. Das Modell hat die gleiche Architektur wie das Auto-Erkennungsmodell, wurde aber für die Erkennung von Personen trainiert.

### Erkennung von Personen und Autos auf Bildern

Es wurden drei weitere Bilder aus dem Internet verwendet, die sowohl Personen als auch Autos enthielten. Die Bilder wurden mit beiden Modellen analysiert, um sowohl Personen als auch Autos zu erkennen.

#### Ergebnisse

Die Ergebnisse zeigten, dass die Modelle in der Lage sind, sowohl Personen als auch Autos in den Bildern zu erkennen und zu unterscheiden.

## Fazit

In diesem Projekt wurden verschiedene CNN-Modelle zur Bilderkennung von Autos implementiert und evaluiert. Die Modelle zeigten gute Ergebnisse bei der Erkennung von Autos in verschiedenen Szenarien.

Die Implementierung eines CNN ohne Verwendung von keras.models oder keras.layers war eine interessante Herausforderung, die ein tieferes Verständnis der zugrunde liegenden Operationen eines CNN erforderte.

Die Verwendung eines vortrainierten Modells zeigte die Vorteile von Transfer Learning, bei dem ein auf einem großen Datensatz vortrainiertes Modell für eine spezifische Aufgabe angepasst wird.

Die Automerkennung auf Bildern mit Sliding Window und Non-Maximum Suppression zeigte, wie ein trainiertes CNN-Modell für die Objekterkennung in realen Bildern eingesetzt werden kann.

Die Bonus-Aufgabe zur Personenerkennung zeigte, wie das gleiche Konzept auf andere Objektklassen angewendet werden kann.
"""

# Dokumentation in eine Markdown-Datei schreiben
with open(os.path.join(docs_dir, 'documentation.md'), 'w') as f:
    f.write(documentation)

print(f"Dokumentation wurde erstellt und unter {os.path.join(docs_dir, 'documentation.md')} gespeichert.")

# Jupyter Notebook für die Präsentation erstellen
notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Anwendung von Convolutional Neural Networks (CNN) zur Bilderkennung von Autos\n",
    "\n",
    "## Übersicht\n",
    "\n",
    "Dieses Notebook präsentiert die Ergebnisse des Projekts zur Anwendung von Convolutional Neural Networks (CNN) zur Bilderkennung von Autos."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Aufgabe 1: Entwicklung von CNN-Modellen\n",
    "\n",
    "### a) CNN mit Keras/TensorFlow\n",
    "\n",
    "Für die erste Aufgabe wurde ein CNN mit Keras/TensorFlow entwickelt, um Autos im CIFAR-10 Datensatz zu erkennen."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from tensorflow.keras.models import load_model\n",
    "import os\n",
    "\n",
    "# Verzeichnisse\n",
    "models_dir = '../models'\n",
    "data_dir = '../data'\n",
    "results_dir = '../results'\n",
    "\n",
    "# Laden der Testdaten\n",
    "x_test = np.load(os.path.join(data_dir, 'x_test.npy'))\n",
    "y_test = np.load(os.path.join(data_dir, 'y_test.npy'))\n",
    "y_test_binary = np.load(os.path.join(data_dir, 'y_test_binary.npy'))\n",
    "\n",
    "# Laden des trainierten Modells\n",
    "model = load_model(os.path.join(models_dir, 'car_detection_model.keras'))\n",
    "\n",
    "# Evaluierung des Modells\n",
    "test_loss, test_accuracy = model.evaluate(x_test, y_test_binary)\n",
    "print(f\"Test Accuracy: {test_accuracy:.4f}\")\n",
    "print(f\"Test Loss: {test_loss:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Visualisierung der Trainingsergebnisse"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "# Anzeigen des Trainingsverlaufs\n",
    "plt.figure(figsize=(12, 4))\n",
    "img = plt.imread(os.path.join(models_dir, 'training_history.png'))\n",
    "plt.imshow(img)\n",
    "plt.axis('off')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Visualisierung der Konfusionsmatrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "# Anzeigen der Konfusionsmatrix\n",
    "plt.figure(figsize=(8, 6))\n",
    "img = plt.imread(os.path.join(models_dir, 'confusion_matrix.png'))\n",
    "plt.imshow(img)\n",
    "plt.axis('off')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Visualisierung einiger Vorhersagen"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "# Anzeigen einiger Vorhersagen\n",
    "plt.figure(figsize=(10, 10))\n",
    "img = plt.imread(os.path.join(models_dir, 'prediction_examples.png'))\n",
    "plt.imshow(img)\n",
    "plt.axis('off')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Aufgabe 2: Automerkennung auf Bildern\n",
    "\n",
    "### a) Erkennung von Autos auf drei gegebenen Bildern"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "# Anzeigen der Ergebnisse für die drei gegebenen Bilder\n",
    "plt.figure(figsize=(15, 10))\n",
    "for i in range(3):\n",
    "    plt.subplot(1, 3, i+1)\n",
    "    img = plt.imread(os.path.join(results_dir, f'test_image_{i+1}_result.jpg'))\n",
    "    plt.imshow(img)\n",
    "    plt.axis('off')\n",
    "    plt.title(f\"Bild {i+1}\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### b) Erkennung von Autos auf drei weiteren Bildern"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "# Anzeigen der Ergebnisse für die drei weiteren Bilder\n",
    "plt.figure(figsize=(15, 10))\n",
    "for i in range(3):\n",
    "    plt.subplot(1, 3, i+1)\n",
    "    img = plt.imread(os.path.join(results_dir, f'additional_image_{i+1}_result.jpg'))\n",
    "    plt.imshow(img)\n",
    "    plt.axis('off')\n",
    "    plt.title(f\"Zusätzliches Bild {i+1}\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Bonus: Personenerkennung\n",
    "\n",
    "### Erkennung von Personen und Autos auf Bildern"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "source": [
    "# Anzeigen der Ergebnisse für die Bonus-Aufgabe\n",
    "plt.figure(figsize=(15, 10))\n",
    "for i in range(3):\n",
    "    plt.subplot(1, 3, i+1)\n",
    "    img = plt.imread(os.path.join('../bonus', f'bonus_image_{i+1}_result.jpg'))\n",
    "    plt.imshow(img)\n",
    "    plt.axis('off')\n",
    "    plt.title(f\"Bonus-Bild {i+1}\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Fazit\n",
    "\n",
    "In diesem Projekt wurden verschiedene CNN-Modelle zur Bilderkennung von Autos implementiert und evaluiert. Die Modelle zeigten gute Ergebnisse bei der Erkennung von Autos in verschiedenen Szenarien.\n",
    "\n",
    "Die Implementierung eines CNN ohne Verwendung von keras.models oder keras.layers war eine interessante Herausforderung, die ein tieferes Verständnis der zugrunde liegenden Operationen eines CNN erforderte.\n",
    "\n",
    "Die Verwendung eines vortrainierten Modells zeigte die Vorteile von Transfer Learning, bei dem ein auf einem großen Datensatz vortrainiertes Modell für eine spezifische Aufgabe angepasst wird.\n",
    "\n",
    "Die Automerkennung auf Bildern mit Sliding Window und Non-Maximum Suppression zeigte, wie ein trainiertes CNN-Modell für die Objekterkennung in realen Bildern eingesetzt werden kann.\n",
    "\n",
    "Die Bonus-Aufgabe zur Personenerkennung zeigte, wie das gleiche Konzept auf andere Objektklassen angewendet werden kann."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

# Speichern des Notebooks als JSON-String
import json
notebook_json = json.dumps(notebook)
with open(os.path.join(docs_dir, 'presentation.ipynb'), 'w') as f:
    f.write(notebook_json)

print(f"Präsentations-Notebook wurde erstellt und unter {os.path.join(docs_dir, 'presentation.ipynb')} gespeichert.")

# README.md für das GitHub-Repository erstellen
readme = """# Anwendung von Convolutional Neural Networks (CNN) zur Bilderkennung von Autos

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
"""

# README.md schreiben
with open(os.path.join('/home/ubuntu/CNNsForCarDetection', 'README.md'), 'w') as f:
    f.write(readme)

print(f"README.md wurde erstellt und unter {os.path.join('/home/ubuntu/CNNsForCarDetection', 'README.md')} gespeichert.")

print("Dokumentation der Ergebnisse und Lösungsschritte abgeschlossen.")
