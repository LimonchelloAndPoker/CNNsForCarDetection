# Anwendung von Convolutional Neural Networks (CNN) zur Bilderkennung von Autos

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
