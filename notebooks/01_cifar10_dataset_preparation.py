#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CIFAR-10 Dataset für CNN-Autoerkennung

Dieses Skript lädt den CIFAR-10 Datensatz und bereitet ihn für das Training 
unseres CNN zur Autoerkennung vor.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import os

# Erstellen des Datenverzeichnisses
data_dir = '../data'
os.makedirs(data_dir, exist_ok=True)

print("Laden des CIFAR-10 Datensatzes...")
# Laden des CIFAR-10 Datensatzes
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Klassen im CIFAR-10 Datensatz
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Ausgabe der Datensatzgröße
print(f"Trainingsdaten: {x_train.shape[0]} Bilder")
print(f"Testdaten: {x_test.shape[0]} Bilder")
print(f"Bildgröße: {x_train.shape[1]}x{x_train.shape[2]} Pixel mit {x_train.shape[3]} Farbkanälen")

# Anzeigen einiger Beispielbilder
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    # Die Labels sind in einem 2D-Array, daher benötigen wir den Index [0]
    plt.xlabel(class_names[y_train[i][0]])
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'cifar10_examples.png'))
print("Beispielbilder wurden gespeichert.")

# Index der Auto-Klasse im CIFAR-10 Datensatz
automobile_class_index = class_names.index('automobile')
print(f"Index der Auto-Klasse: {automobile_class_index}")

# Erstellen von binären Labels (Auto = 1, Nicht-Auto = 0)
y_train_binary = (y_train == automobile_class_index).astype(int)
y_test_binary = (y_test == automobile_class_index).astype(int)

# Anzahl der Auto-Bilder im Trainings- und Testdatensatz
train_car_count = np.sum(y_train_binary)
test_car_count = np.sum(y_test_binary)

print(f"Anzahl der Auto-Bilder im Trainingsdatensatz: {train_car_count}")
print(f"Anzahl der Auto-Bilder im Testdatensatz: {test_car_count}")

# Normalisierung der Pixelwerte auf den Bereich [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print("Daten wurden normalisiert.")

# Speichern der Daten
np.save(os.path.join(data_dir, 'x_train.npy'), x_train)
np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
np.save(os.path.join(data_dir, 'x_test.npy'), x_test)
np.save(os.path.join(data_dir, 'y_test.npy'), y_test)

# Speichern der binären Labels
np.save(os.path.join(data_dir, 'y_train_binary.npy'), y_train_binary)
np.save(os.path.join(data_dir, 'y_test_binary.npy'), y_test_binary)

print("Daten wurden erfolgreich gespeichert.")

# Indizes der Auto-Bilder im Trainingsdatensatz
car_indices = np.where(y_train == automobile_class_index)[0]

# Anzeigen einiger Auto-Beispiele
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[car_indices[i]])
    plt.xlabel('automobile')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'cifar10_car_examples.png'))
print("Auto-Beispielbilder wurden gespeichert.")

print("Datensatzvorbereitung abgeschlossen.")
