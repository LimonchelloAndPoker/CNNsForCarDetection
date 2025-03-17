#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vortrainiertes CNN für Autoerkennung

Dieses Skript lädt ein vortrainiertes CNN-Modell und trainiert es für die Autoerkennung
im CIFAR-10 Datensatz (Aufgabe 1c).
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
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

print(f"Trainingsdaten: {x_train.shape[0]} Bilder")
print(f"Testdaten: {x_test.shape[0]} Bilder")
print(f"Anzahl der Auto-Bilder im Trainingsdatensatz: {np.sum(y_train_binary)}")
print(f"Anzahl der Auto-Bilder im Testdatensatz: {np.sum(y_test_binary)}")

# Die Bilder müssen auf die Eingabegröße des vortrainierten Modells skaliert werden
# MobileNetV2 erwartet Bilder der Größe 224x224
def preprocess_images(images, target_size=(224, 224)):
    processed_images = []
    for img in images:
        # Skalieren des Bildes auf die Zielgröße
        resized_img = tf.image.resize(img, target_size)
        processed_images.append(resized_img)
    return np.array(processed_images)

# Vorverarbeitung der Bilder
print("Vorverarbeitung der Bilder...")
x_train_processed = preprocess_images(x_train)
x_test_processed = preprocess_images(x_test)

print(f"Vorverarbeitete Trainingsdaten: {x_train_processed.shape}")
print(f"Vorverarbeitete Testdaten: {x_test_processed.shape}")

# Laden des vortrainierten MobileNetV2-Modells
print("Laden des vortrainierten MobileNetV2-Modells...")
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

# Kompilieren des Modells
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Zusammenfassung des Modells
model.summary()

# Callbacks für das Training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(
        filepath=os.path.join(models_dir, 'pretrained_car_detection_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Training des Modells
print("Training des vortrainierten Modells...")
history = model.fit(
    x_train_processed, y_train_binary,
    batch_size=32,
    epochs=20,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Speichern des Modells
model.save(os.path.join(models_dir, 'pretrained_car_detection_model.keras'))
print(f"Modell wurde gespeichert unter: {os.path.join(models_dir, 'pretrained_car_detection_model.keras')}")

# Evaluierung des Modells auf den Testdaten
print("Evaluierung des Modells auf den Testdaten...")
test_loss, test_accuracy = model.evaluate(x_test_processed, y_test_binary)
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
plt.savefig(os.path.join(models_dir, 'pretrained_training_history.png'))
print(f"Trainingsverlauf wurde gespeichert unter: {os.path.join(models_dir, 'pretrained_training_history.png')}")

# Vorhersagen auf den Testdaten
y_pred = model.predict(x_test_processed)
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
plt.savefig(os.path.join(models_dir, 'pretrained_confusion_matrix.png'))
print(f"Konfusionsmatrix wurde gespeichert unter: {os.path.join(models_dir, 'pretrained_confusion_matrix.png')}")

# Visualisierung einiger Vorhersagen
def plot_predictions(x, y_true, y_pred, num_images=25):
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
    plt.savefig(os.path.join(models_dir, 'pretrained_prediction_examples.png'))
    print(f"Vorhersagebeispiele wurden gespeichert unter: {os.path.join(models_dir, 'pretrained_prediction_examples.png')}")

# Zufällige Auswahl von Testbildern
np.random.seed(42)
random_indices = np.random.choice(len(x_test), 25, replace=False)
plot_predictions(
    x_test[random_indices],
    y_test_binary[random_indices],
    y_pred[random_indices],
)

print("Vortrainiertes CNN-Modell wurde erfolgreich trainiert und evaluiert.")
