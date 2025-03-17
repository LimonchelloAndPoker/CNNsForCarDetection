#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benutzerdefiniertes CNN ohne keras.models oder keras.layers

Dieses Skript implementiert ein CNN ohne Verwendung von keras.models oder keras.layers
zur Erkennung von Autos im CIFAR-10 Datensatz (Aufgabe 1b).
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

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
y_train_binary = np.load(os.path.join(data_dir, 'y_train_binary.npy')).reshape(-1, 1)
y_test_binary = np.load(os.path.join(data_dir, 'y_test_binary.npy')).reshape(-1, 1)

print(f"Trainingsdaten: {x_train.shape[0]} Bilder")
print(f"Testdaten: {x_test.shape[0]} Bilder")
print(f"Anzahl der Auto-Bilder im Trainingsdatensatz: {np.sum(y_train_binary)}")
print(f"Anzahl der Auto-Bilder im Testdatensatz: {np.sum(y_test_binary)}")

# Für das Training verwenden wir einen kleineren Datensatz, um die Rechenzeit zu reduzieren
# Wir verwenden 10% der Trainingsdaten und 10% der Testdaten
np.random.seed(42)
train_indices = np.random.choice(len(x_train), size=int(len(x_train) * 0.1), replace=False)
test_indices = np.random.choice(len(x_test), size=int(len(x_test) * 0.1), replace=False)

x_train_small = x_train[train_indices]
y_train_small = y_train_binary[train_indices]
x_test_small = x_test[test_indices]
y_test_small = y_test_binary[test_indices]

print(f"Reduzierte Trainingsdaten: {x_train_small.shape[0]} Bilder")
print(f"Reduzierte Testdaten: {x_test_small.shape[0]} Bilder")

# Implementierung der CNN-Funktionen

def initialize_parameters(filter_sizes, num_filters):
    """
    Initialisiert die Parameter für ein CNN.
    
    Args:
        filter_sizes: Liste der Filter-Größen für jede Schicht
        num_filters: Liste der Anzahl der Filter für jede Schicht
        
    Returns:
        parameters: Dictionary mit den initialisierten Parametern
    """
    np.random.seed(1)
    parameters = {}
    L = len(num_filters)
    
    for l in range(1, L + 1):
        parameters[f'W{l}'] = np.random.randn(filter_sizes[l-1], filter_sizes[l-1], 3 if l == 1 else num_filters[l-2], num_filters[l-1]) * 0.01
        parameters[f'b{l}'] = np.zeros((1, 1, 1, num_filters[l-1]))
    
    # Fully connected layer
    parameters['W_fc'] = np.random.randn(4 * 4 * num_filters[-1], 1) * 0.01
    parameters['b_fc'] = np.zeros((1, 1))
        
    return parameters

def zero_pad(X, pad):
    """
    Fügt Nullen um die Bilder herum hinzu.
    
    Args:
        X: Eingabedaten der Form (m, h, w, c)
        pad: Anzahl der Nullen, die hinzugefügt werden sollen
        
    Returns:
        X_pad: Gepolsterte Eingabedaten
    """
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)
    return X_pad

def conv_single_step(a_slice_prev, W, b):
    """
    Führt einen einzelnen Faltungsschritt durch.
    
    Args:
        a_slice_prev: Ausschnitt der Eingabedaten
        W: Gewichte
        b: Bias
        
    Returns:
        Z: Ergebnis der Faltung
    """
    Z = np.sum(a_slice_prev * W) + float(b)
    return Z

def conv_forward(A_prev, W, b, hparameters):
    """
    Führt einen Vorwärtsdurchlauf für eine Faltungsschicht durch.
    
    Args:
        A_prev: Ausgabe der vorherigen Schicht (m, h_prev, w_prev, c_prev)
        W: Gewichte (f, f, c_prev, c)
        b: Bias (1, 1, 1, c)
        hparameters: Dictionary mit Hyperparametern
        
    Returns:
        Z: Ausgabe der Faltungsschicht
        cache: Cache für die Rückwärtspropagierung
    """
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (f, f, c_prev, c) = W.shape
    
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    n_H = int((h_prev - f + 2 * pad) / stride) + 1
    n_W = int((w_prev - f + 2 * pad) / stride) + 1
    
    Z = np.zeros((m, n_H, n_W, c))
    A_prev_pad = zero_pad(A_prev, pad)
    
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c_out in range(c):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    Z[i, h, w, c_out] = conv_single_step(a_slice_prev, W[:, :, :, c_out], b[:, :, :, c_out])
    
    cache = (A_prev, W, b, hparameters)
    return Z, cache

def relu(Z):
    """
    Wendet die ReLU-Aktivierungsfunktion an.
    
    Args:
        Z: Eingabedaten
        
    Returns:
        A: Ausgabe nach Anwendung von ReLU
        cache: Cache für die Rückwärtspropagierung
    """
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def pool_forward(A_prev, hparameters, mode="max"):
    """
    Führt einen Vorwärtsdurchlauf für eine Pooling-Schicht durch.
    
    Args:
        A_prev: Ausgabe der vorherigen Schicht (m, h_prev, w_prev, c_prev)
        hparameters: Dictionary mit Hyperparametern
        mode: Pooling-Modus ("max" oder "average")
        
    Returns:
        A: Ausgabe der Pooling-Schicht
        cache: Cache für die Rückwärtspropagierung
    """
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    n_H = int(1 + (h_prev - f) / stride)
    n_W = int(1 + (w_prev - f) / stride)
    n_C = c_prev
    
    A = np.zeros((m, n_H, n_W, n_C))
    
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    
    cache = (A_prev, hparameters)
    return A, cache

def flatten(A):
    """
    Flacht die Ausgabe der letzten Pooling-Schicht ab.
    
    Args:
        A: Ausgabe der letzten Pooling-Schicht (m, h, w, c)
        
    Returns:
        A_flat: Abgeflachte Ausgabe (m, h*w*c)
    """
    return A.reshape(A.shape[0], -1)

def fc_forward(A_prev, W, b):
    """
    Führt einen Vorwärtsdurchlauf für eine Fully-Connected-Schicht durch.
    
    Args:
        A_prev: Ausgabe der vorherigen Schicht (m, n_prev)
        W: Gewichte (n_prev, n)
        b: Bias (1, n)
        
    Returns:
        Z: Ausgabe der Fully-Connected-Schicht
        cache: Cache für die Rückwärtspropagierung
    """
    Z = np.dot(A_prev, W) + b
    cache = (A_prev, W, b)
    return Z, cache

def sigmoid(Z):
    """
    Wendet die Sigmoid-Aktivierungsfunktion an.
    
    Args:
        Z: Eingabedaten
        
    Returns:
        A: Ausgabe nach Anwendung von Sigmoid
        cache: Cache für die Rückwärtspropagierung
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def compute_cost(AL, Y):
    """
    Berechnet die Kostenfunktion (binäre Kreuzentropie).
    
    Args:
        AL: Ausgabe des Modells (m, 1)
        Y: Tatsächliche Labels (m, 1)
        
    Returns:
        cost: Wert der Kostenfunktion
    """
    m = Y.shape[0]
    cost = -1/m * np.sum(Y * np.log(AL + 1e-8) + (1 - Y) * np.log(1 - AL + 1e-8))
    cost = np.squeeze(cost)
    return cost

def conv_backward(dZ, cache):
    """
    Führt einen Rückwärtsdurchlauf für eine Faltungsschicht durch.
    
    Args:
        dZ: Gradient der Kostenfunktion bezüglich der Ausgabe der Faltungsschicht
        cache: Cache aus dem Vorwärtsdurchlauf
        
    Returns:
        dA_prev: Gradient der Kostenfunktion bezüglich der Eingabe der Faltungsschicht
        dW: Gradient der Kostenfunktion bezüglich der Gewichte
        db: Gradient der Kostenfunktion bezüglich des Bias
    """
    (A_prev, W, b, hparameters) = cache
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (f, f, c_prev, c) = W.shape
    
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    (m, n_H, n_W, n_C) = dZ.shape
    
    dA_prev = np.zeros((m, h_prev, w_prev, c_prev))
    dW = np.zeros((f, f, c_prev, c))
    db = np.zeros((1, 1, 1, c))
    
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        
        for h in range(n_H):
            for w in range(n_W):
                for c_out in range(c):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c_out] * dZ[i, h, w, c_out]
                    dW[:, :, :, c_out] += a_slice * dZ[i, h, w, c_out]
                    db[:, :, :, c_out] += dZ[i, h, w, c_out]
        
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
    
    return dA_prev, dW, db

def relu_backward(dA, cache):
    """
    Führt einen Rückwärtsdurchlauf für die ReLU-Aktivierungsfunktion durch.
    
    Args:
        dA: Gradient der Kostenfunktion bezüglich der Ausgabe der ReLU-Funktion
        cache: Cache aus dem Vorwärtsdurchlauf
        
    Returns:
        dZ: Gradient der Kostenfunktion bezüglich der Eingabe der ReLU-Funktion
    """
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def pool_backward(dA, cache, mode="max"):
    """
    Führt einen Rückwärtsdurchlauf für eine Pooling-Schicht durch.
    
    Args:
        dA: Gradient der Kostenfunktion bezüglich der Ausgabe der Pooling-Schicht
        cache: Cache aus dem Vorwärtsdurchlauf
        mode: Pooling-Modus ("max" oder "average")
        
    Returns:
        dA_prev: Gradient der Kostenfunktion bezüglich der Eingabe der Pooling-Schicht
    """
    (A_prev, hparameters) = cache
    
    stride = hparameters["stride"]
    f = hparameters["f"]
    
    m, h_prev, w_prev, c_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    
    dA_prev = np.zeros(A_prev.shape)
    
    for i in range(m):
        a_prev = A_prev[i]
        
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        if a_prev_slice.size > 0:  # Überprüfen, ob das Array nicht leer ist
                            mask = (a_prev_slice == np.max(a_prev_slice))
                            dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += mask * dA[i, h, w, c]
                    elif mode == "average":
                        da = dA[i, h, w, c]
                        size = (vert_end - vert_start) * (horiz_end - horiz_start)
                        if size > 0:  # Überprüfen, ob die Größe nicht null ist
                            dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += da / size
    
    return dA_prev

def fc_backward(dZ, cache):
    """
    Führt einen Rückwärtsdurchlauf für eine Fully-Connected-Schicht durch.
    
    Args:
        dZ: Gradient der Kostenfunktion bezüglich der Ausgabe der Fully-Connected-Schicht
        cache: Cache aus dem Vorwärtsdurchlauf
        
    Returns:
        dA_prev: Gradient der Kostenfunktion bezüglich der Eingabe der Fully-Connected-Schicht
        dW: Gradient der Kostenfunktion bezüglich der Gewichte
        db: Gradient der Kostenfunktion bezüglich des Bias
    """
    A_prev, W, b = cache
    m = A_prev.shape[0]
    
    dW = 1/m * np.dot(A_prev.T, dZ)
    db = 1/m * np.sum(dZ, axis=0, keepdims=True)
    dA_prev = np.dot(dZ, W.T)
    
    return dA_prev, dW, db

def sigmoid_backward(dA, cache):
    """
    Führt einen Rückwärtsdurchlauf für die Sigmoid-Aktivierungsfunktion durch.
    
    Args:
        dA: Gradient der Kostenfunktion bezüglich der Ausgabe der Sigmoid-Funktion
        cache: Cache aus dem Vorwärtsdurchlauf
        
    Returns:
        dZ: Gradient der Kostenfunktion bezüglich der Eingabe der Sigmoid-Funktion
    """
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

def model_forward(X, parameters):
    """
    Führt einen Vorwärtsdurchlauf für das gesamte Modell durch.
    
    Args:
        X: Eingabedaten (m, h, w, c)
        parameters: Dictionary mit den Parametern des Modells
        
    Returns:
        AL: Ausgabe des Modells
        caches: Liste der Caches für die Rückwärtspropagierung
    """
    caches = []
    A = X
    L = len(parameters) // 2 - 1  # Anzahl der Faltungsschichten
    
    # Faltungsschichten
    for l in range(1, L + 1):
        A_prev = A
        
        # Faltung
        Z, conv_cache = conv_forward(A_prev, parameters[f'W{l}'], parameters[f'b{l}'], 
                                    {'stride': 1, 'pad': 1})
        
        # ReLU
        A, relu_cache = relu(Z)
        
        # Pooling
        A, pool_cache = pool_forward(A, {'stride': 2, 'f': 2}, mode="max")
        
        caches.append((conv_cache, relu_cache, pool_cache))
    
    # Flatten
    A_flat = flatten(A)
    
    # Fully connected layer
    Z_fc, fc_cache = fc_forward(A_flat, parameters['W_fc'], parameters['b_fc'])
    
    # Sigmoid
    AL, sigmoid_cache = sigmoid(Z_fc)
    
    caches.append((fc_cache, sigmoid_cache))
    
    return AL, caches

def model_backward(AL, Y, caches):
    """
    Führt einen Rückwärtsdurchlauf für das gesamte Modell durch.
    
    Args:
        AL: Ausgabe des Modells
        Y: Tatsächliche Labels
        caches: Liste der Caches aus dem Vorwärtsdurchlauf
        
    Returns:
        gradients: Dictionary mit den Gradienten
    """
    gradients = {}
    L = len(caches)
    m = AL.shape[0]
    Y = Y.reshape(AL.shape)
    
    # Initialisierung des Gradienten der Ausgabeschicht
    dAL = - (np.divide(Y, AL + 1e-8) - np.divide(1 - Y, 1 - AL + 1e-8))
    
    # Rückwärtsdurchlauf für die Fully-Connected-Schicht
    fc_cache, sigmoid_cache = caches[L-1]
    dZ_fc = sigmoid_backward(dAL, sigmoid_cache)
    dA_flat, dW_fc, db_fc = fc_backward(dZ_fc, fc_cache)
    
    gradients['dW_fc'] = dW_fc
    gradients['db_fc'] = db_fc
    
    # Reshape dA_flat zurück in die Form der letzten Pooling-Schicht
    # Berechne die korrekte Form basierend auf dem letzten Pool-Cache
    last_pool_shape = caches[L-2][2][0].shape
    # Prüfe, ob die Dimensionen kompatibel sind
    if dA_flat.size != np.prod(last_pool_shape):
        # Wenn nicht kompatibel, verwende eine sichere Reshape-Operation
        # Berechne die neue Form basierend auf der Größe von dA_flat
        n_samples = last_pool_shape[0]
        height = last_pool_shape[1]
        width = last_pool_shape[2]
        n_channels = dA_flat.size // (n_samples * height * width)
        dA = dA_flat.reshape(n_samples, height, width, n_channels)
    else:
        # Wenn kompatibel, verwende die ursprüngliche Form
        dA = dA_flat.reshape(last_pool_shape)
    
    # Rückwärtsdurchlauf für die Faltungsschichten
    for l in reversed(range(L-1)):
        conv_cache, relu_cache, pool_cache = caches[l]
        
        # Pooling
        dA = pool_backward(dA, pool_cache, mode="max")
        
        # ReLU
        dZ = relu_backward(dA, relu_cache)
        
        # Faltung
        dA, dW, db = conv_backward(dZ, conv_cache)
        
        gradients[f'dW{l+1}'] = dW
        gradients[f'db{l+1}'] = db
    
    return gradients

def update_parameters(parameters, gradients, learning_rate):
    """
    Aktualisiert die Parameter des Modells.
    
    Args:
        parameters: Dictionary mit den Parametern des Modells
        gradients: Dictionary mit den Gradienten
        learning_rate: Lernrate
        
    Returns:
        parameters: Aktualisierte Parameter
    """
    L = len(parameters) // 2 - 1  # Anzahl der Faltungsschichten
    
    # Aktualisierung der Parameter der Faltungsschichten
    for l in range(1, L + 1):
        parameters[f'W{l}'] -= learning_rate * gradients[f'dW{l}']
        parameters[f'b{l}'] -= learning_rate * gradients[f'db{l}']
    
    # Aktualisierung der Parameter der Fully-Connected-Schicht
    parameters['W_fc'] -= learning_rate * gradients['dW_fc']
    parameters['b_fc'] -= learning_rate * gradients['db_fc']
    
    return parameters

def predict(X, parameters):
    """
    Führt Vorhersagen mit dem trainierten Modell durch.
    
    Args:
        X: Eingabedaten
        parameters: Trainierte Parameter des Modells
        
    Returns:
        predictions: Vorhersagen (0 oder 1)
    """
    AL, _ = model_forward(X, parameters)
    predictions = (AL > 0.5).astype(int)
    return predictions

def compute_accuracy(predictions, Y):
    """
    Berechnet die Genauigkeit der Vorhersagen.
    
    Args:
        predictions: Vorhersagen des Modells
        Y: Tatsächliche Labels
        
    Returns:
        accuracy: Genauigkeit
    """
    return np.mean(predictions == Y)

def mini_batches(X, Y, batch_size):
    """
    Erstellt Mini-Batches für das Training.
    
    Args:
        X: Eingabedaten
        Y: Labels
        batch_size: Größe der Mini-Batches
        
    Returns:
        mini_batches: Liste der Mini-Batches
    """
    m = X.shape[0]
    mini_batches = []
    
    # Shuffle
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]
    
    # Erstellen der Mini-Batches
    num_complete_minibatches = m // batch_size
    
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * batch_size:(k + 1) * batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * batch_size:(k + 1) * batch_size, :]
        mini_batches.append((mini_batch_X, mini_batch_Y))
    
    # Letzter Mini-Batch
    if m % batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * batch_size:, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * batch_size:, :]
        mini_batches.append((mini_batch_X, mini_batch_Y))
    
    return mini_batches

def model(X_train, Y_train, X_test, Y_test, learning_rate=0.01, num_epochs=10, batch_size=32, print_cost=True):
    """
    Trainiert ein CNN-Modell.
    
    Args:
        X_train: Trainingsdaten
        Y_train: Trainingslabels
        X_test: Testdaten
        Y_test: Testlabels
        learning_rate: Lernrate
        num_epochs: Anzahl der Epochen
        batch_size: Größe der Mini-Batches
        print_cost: Ob die Kosten ausgegeben werden sollen
        
    Returns:
        parameters: Trainierte Parameter des Modells
        costs: Liste der Kosten während des Trainings
    """
    np.random.seed(1)
    costs = []
    
    # Initialisierung der Parameter
    parameters = initialize_parameters([3, 3, 3], [8, 16, 32])
    
    for epoch in range(num_epochs):
        epoch_cost = 0
        num_batches = 0
        
        # Mini-Batches erstellen
        minibatches = mini_batches(X_train, Y_train, batch_size)
        
        start_time = time.time()
        
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            
            # Vorwärtsdurchlauf
            AL, caches = model_forward(minibatch_X, parameters)
            
            # Kosten berechnen
            cost = compute_cost(AL, minibatch_Y)
            epoch_cost += cost
            num_batches += 1
            
            # Rückwärtsdurchlauf
            gradients = model_backward(AL, minibatch_Y, caches)
            
            # Parameter aktualisieren
            parameters = update_parameters(parameters, gradients, learning_rate)
        
        epoch_cost /= num_batches
        costs.append(epoch_cost)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        
        if print_cost and epoch % 1 == 0:
            print(f"Kosten nach Epoche {epoch}: {epoch_cost:.4f} (Zeit: {epoch_time:.2f}s)")
            
            # Genauigkeit auf dem Trainingsdatensatz
            train_predictions = predict(X_train, parameters)
            train_accuracy = compute_accuracy(train_predictions, Y_train)
            print(f"Trainingsgenauigkeit: {train_accuracy:.4f}")
            
            # Genauigkeit auf dem Testdatensatz
            test_predictions = predict(X_test, parameters)
            test_accuracy = compute_accuracy(test_predictions, Y_test)
            print(f"Testgenauigkeit: {test_accuracy:.4f}")
    
    # Speichern der Parameter
    np.save(os.path.join(models_dir, 'custom_cnn_parameters.npy'), parameters)
    print(f"Parameter wurden gespeichert unter: {os.path.join(models_dir, 'custom_cnn_parameters.npy')}")
    
    # Visualisierung der Kosten
    plt.figure(figsize=(10, 5))
    plt.plot(costs)
    plt.title('Kosten während des Trainings')
    plt.xlabel('Epochen')
    plt.ylabel('Kosten')
    plt.savefig(os.path.join(models_dir, 'custom_cnn_costs.png'))
    print(f"Kostenverlauf wurde gespeichert unter: {os.path.join(models_dir, 'custom_cnn_costs.png')}")
    
    return parameters, costs

# Training des Modells
print("Training des benutzerdefinierten CNN-Modells...")
parameters, costs = model(x_train_small, y_train_small, x_test_small, y_test_small, 
                         learning_rate=0.01, num_epochs=5, batch_size=32, print_cost=True)

# Evaluierung des Modells auf den Testdaten
print("Evaluierung des Modells auf den Testdaten...")
test_predictions = predict(x_test_small, parameters)
test_accuracy = compute_accuracy(test_predictions, y_test_small)
print(f"Testgenauigkeit: {test_accuracy:.4f}")

# Visualisierung einiger Vorhersagen
def plot_predictions(X, Y, predictions, num_images=25):
    plt.figure(figsize=(10, 10))
    for i in range(min(num_images, len(X))):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[i])
        
        predicted = "Auto" if predictions[i] == 1 else "Nicht-Auto"
        actual = "Auto" if Y[i] == 1 else "Nicht-Auto"
        
        color = 'green' if predicted == actual else 'red'
        plt.xlabel(f"P: {predicted}, A: {actual}", color=color)
    
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'custom_cnn_predictions.png'))
    print(f"Vorhersagebeispiele wurden gespeichert unter: {os.path.join(models_dir, 'custom_cnn_predictions.png')}")

# Zufällige Auswahl von Testbildern
np.random.seed(42)
random_indices = np.random.choice(len(x_test_small), 25, replace=False)
plot_predictions(
    x_test_small[random_indices],
    y_test_small[random_indices],
    test_predictions[random_indices]
)

print("Benutzerdefiniertes CNN-Modell ohne keras.models oder keras.layers wurde erfolgreich trainiert und evaluiert.")
