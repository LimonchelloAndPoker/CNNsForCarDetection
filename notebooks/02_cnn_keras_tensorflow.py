#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
CNN mit Keras/TensorFlow zur Erkennung von Autos

Dieses Skript implementiert ein CNN mit Keras/TensorFlow zur Erkennung von Autos
im CIFAR-10 Datensatz (Aufgabe 1a).
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import os

# Verzeichnisse
data_dir = '../data'
models_dir = '../models'
os.makedirs(models_dir, exist_ok=True)

# Laden der vorbereiteten Daten
print("Laden der vorbereiteten Daten...")
x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
x_test = np.load(os.path.join(data_dir, 'x_test.npy'))
y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
y_train_binary = np.load(os.path.join(data_dir, 'y_train_binary.npy'))
y_test_binary = np.load(os.path.join(data_dir, 'y_test_binary.npy'))

# Klassen im CIFAR-10 Datensatz
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
automobile_class_index = class_names.index('automobile')

print(f"Trainingsdaten: {x_train.shape[0]} Bilder")
print(f"Testdaten: {x_test.shape[0]} Bilder")
print(f"Anzahl der Auto-Bilder im Trainingsdatensatz: {np.sum(y_train_binary)}")
print(f"Anzahl der Auto-Bilder im Testdatensatz: {np.sum(y_test_binary)}")

# Definition des CNN-Modells
def create_car_detection_model():
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
    
    # Kompilieren des Modells
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Modell erstellen
print("Erstellen des CNN-Modells...")
model = create_car_detection_model()
model.summary()

# Callbacks für das Training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(
        filepath=os.path.join(models_dir, 'car_detection_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Training des Modells
print("Training des CNN-Modells...")
history = model.fit(
    x_train, y_train_binary,
    batch_size=64,
    epochs=50,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Speichern des Modells
model.save(os.path.join(models_dir, 'car_detection_model.keras'))
print(f"Modell wurde gespeichert unter: {os.path.join(models_dir, 'car_detection_model.keras')}")

# Evaluierung des Modells auf den Testdaten
print("Evaluierung des Modells auf den Testdaten...")
test_loss, test_accuracy = model.evaluate(x_test, y_test_binary)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Visualisierung des Trainingsverlaufs
plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(models_dir, 'training_history.png'))
print(f"Trainingsverlauf wurde gespeichert unter: {os.path.join(models_dir, 'training_history.png')}")

# Vorhersagen auf den Testdaten
y_pred = model.predict(x_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# Berechnung von Precision, Recall und F1-Score
from sklearn.metrics import classification_report, confusion_matrix

print("Klassifikationsbericht:")
print(classification_report(y_test_binary, y_pred_binary))

# Konfusionsmatrix
cm = confusion_matrix(y_test_binary, y_pred_binary)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Konfusionsmatrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Nicht-Auto', 'Auto'])
plt.yticks(tick_marks, ['Nicht-Auto', 'Auto'])

# Beschriftung der Zellen mit den Werten
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('Tatsächliche Klasse')
plt.xlabel('Vorhergesagte Klasse')
plt.savefig(os.path.join(models_dir, 'confusion_matrix.png'))
print(f"Konfusionsmatrix wurde gespeichert unter: {os.path.join(models_dir, 'confusion_matrix.png')}")

# Visualisierung einiger Vorhersagen
def plot_predictions(x, y_true, y_pred, class_names, num_images=25):
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[i])
        
        predicted = "Auto" if y_pred[i] > 0.5 else "Nicht-Auto"
        actual = "Auto" if y_true[i] == 1 else "Nicht-Auto"
        
        color = 'green' if predicted == actual else 'red'
        plt.xlabel(f"P: {predicted}, A: {actual}", color=color)
    
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'prediction_examples.png'))
    print(f"Vorhersagebeispiele wurden gespeichert unter: {os.path.join(models_dir, 'prediction_examples.png')}")

# Zufällige Auswahl von Testbildern
np.random.seed(42)
random_indices = np.random.choice(len(x_test), 25, replace=False)
plot_predictions(
    x_test[random_indices],
    y_test_binary[random_indices],
    y_pred[random_indices],
    class_names
)

print("CNN-Modell mit Keras/TensorFlow wurde erfolgreich trainiert und evaluiert.")
