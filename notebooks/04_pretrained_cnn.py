import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# Verzeichnisse erstellen
models_dir = os.path.join('..', 'models')
os.makedirs(models_dir, exist_ok=True)

# Laden der vorbereiteten Daten
print("Laden der vorbereiteten Daten...")
x_train = np.load(os.path.join('..', 'data', 'x_train.npy'))
y_train = np.load(os.path.join('..', 'data', 'y_train.npy'))
x_test = np.load(os.path.join('..', 'data', 'x_test.npy'))
y_test = np.load(os.path.join('..', 'data', 'y_test.npy'))

print(f"Trainingsdaten: {len(x_train)} Bilder")
print(f"Testdaten: {len(x_test)} Bilder")

# Bilder auf die Größe anpassen, die das vortrainierte Modell erwartet
input_shape = (224, 224, 3)
x_train_resized = tf.image.resize(x_train, (input_shape[0], input_shape[1]))
x_test_resized = tf.image.resize(x_test, (input_shape[0], input_shape[1]))

# Vortrainiertes Modell laden
print("Laden des vortrainierten MobileNetV2-Modells...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

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
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Zusammenfassung des Modells anzeigen
model.summary()

# Callbacks für das Training
checkpoint = ModelCheckpoint(
    os.path.join(models_dir, 'pretrained_car_detection_model.keras'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    mode='max',
    verbose=1
)

# Training des Modells
print("Training des vortrainierten Modells...")
history = model.fit(
    x_train_resized, y_train,
    validation_data=(x_test_resized, y_test),
    epochs=20,
    batch_size=32,
    callbacks=[checkpoint, early_stopping]
)

# Evaluierung des Modells
print("Evaluierung des Modells auf den Testdaten...")
test_loss, test_accuracy = model.evaluate(x_test_resized, y_test)
print(f"Testgenauigkeit: {test_accuracy:.4f}")

# Visualisierung der Trainingsergebnisse
plt.figure(figsize=(12, 4))

# Genauigkeit
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Modellgenauigkeit')
plt.ylabel('Genauigkeit')
plt.xlabel('Epoche')
plt.legend(['Training', 'Validierung'], loc='lower right')

# Verlust
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Modellverlust')
plt.ylabel('Verlust')
plt.xlabel('Epoche')
plt.legend(['Training', 'Validierung'], loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(models_dir, 'pretrained_model_training.png'))
print(f"Trainingsverlauf wurde gespeichert unter: {os.path.join(models_dir, 'pretrained_model_training.png')}")

# Speichern des Modells
model.save(os.path.join(models_dir, 'pretrained_car_detection_model.keras'))
print(f"Modell wurde gespeichert unter: {os.path.join(models_dir, 'pretrained_car_detection_model.keras')}")

# Einige Vorhersagen visualisieren
def plot_predictions(X, y, predictions, num_images=25):
    plt.figure(figsize=(10, 10))
    for i in range(min(num_images, len(X))):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[i])
        
        color = 'green' if predictions[i] == y[i] else 'red'
        label = "Auto" if predictions[i] == 1 else "Kein Auto"
        plt.xlabel(label, color=color)
    
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'pretrained_model_predictions.png'))
    print(f"Vorhersagen wurden gespeichert unter: {os.path.join(models_dir, 'pretrained_model_predictions.png')}")

# Vorhersagen für Testdaten
print("Generieren von Vorhersagen für Testdaten...")
predictions = (model.predict(x_test_resized) > 0.5).astype(int).flatten()
plot_predictions(x_test, y_test, predictions)

print("Training und Evaluierung des vortrainierten Modells abgeschlossen.")
