import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import time
import glob
import random
from sklearn.model_selection import train_test_split

# Directory setup
models_dir = '../models'
data_dir = '../data'
images_dir = '../images'
results_dir = '../results'
bonus_dir = '../bonus'
human_dataset_dir = '../HumanBinaryClassification'

os.makedirs(models_dir, exist_ok=True)
os.makedirs(os.path.join(models_dir, 'human_detection'), exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(bonus_dir, exist_ok=True)

def load_and_prepare_dataset(dataset_dir, target_size=(32, 32), batch_size=32, validation_split=0.2):
    """
    Loads the dataset and prepares it for training.
    
    Args:
        dataset_dir: Directory with subfolders 'human' and 'nonhuman'
        target_size: Target size for images (e.g., (64, 64) for persons)
        batch_size: Batch size for training
        validation_split: Fraction of data for validation
        
    Returns:
        train_generator: Generator for training data
        validation_generator: Generator for validation data
        class_weights: Class weights for unbalanced data
    """
    print(f"Loading dataset from {dataset_dir}...")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )
    
    human_count = len(os.listdir(os.path.join(dataset_dir, 'human')))
    nonhuman_count = len(os.listdir(os.path.join(dataset_dir, 'nonhuman')))
    total_count = human_count + nonhuman_count
    
    print(f"Number of images: Human = {human_count}, Nonhuman = {nonhuman_count}, Total = {total_count}")
    
    weight_for_0 = (1 / nonhuman_count) * (total_count / 2.0)  # nonhuman
    weight_for_1 = (1 / human_count) * (total_count / 2.0)     # human
    class_weights = {0: weight_for_0, 1: weight_for_1}
    print(f"Class weights: {class_weights}")
    
    return train_generator, validation_generator, class_weights

# Load dataset for person detection with 64x64 images
train_generator, validation_generator, class_weights = load_and_prepare_dataset(
    human_dataset_dir,
    target_size=(64, 64),  # Changed to 64x64 for more detail
    batch_size=64,
    validation_split=0.2
)

def create_person_detection_model(input_shape=(64, 64, 3)):  # Changed to 64x64
    """
    Creates a CNN model for person detection.
    
    Args:
        input_shape: Shape of input images (e.g., (64, 64, 3))
    
    Returns:
        model: Compiled Keras model
    """
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification: Person vs. Non-Person
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create and train the person detection model
person_model = create_person_detection_model(input_shape=(64, 64, 3))
person_model.summary()

checkpoint_path = os.path.join(models_dir, 'human_detection', 'human_detection_model.keras')
checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

print("Training the person detection model...")
history = person_model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[checkpoint, early_stopping],
    class_weight=class_weights,
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show(block=False)

# Load models
print("Loading the trained person detection model...")
try:
    person_model = load_model(os.path.join(models_dir, 'human_detection', 'human_detection_model.keras'))
    print("Person detection model loaded successfully.")
except:
    print("Error loading person model. Using the last trained model.")

print("Loading the trained car detection model...")
try:
    car_model = load_model(os.path.join(models_dir, 'keras_cnn', 'car_detection_model.keras'))
    print("Car detection model loaded successfully.")
except:
    print("Error loading car model. Ensure it is trained.")
    raise Exception("Car detection model could not be loaded.")

def load_and_preprocess_image(image_path):
    """
    Loads an image from a path or URL.
    
    Args:
        image_path: Path to the image or URL
        
    Returns:
        image: Loaded image as a numpy array
    """
    if image_path.startswith('http'):
        response = requests.get(image_path)
        pil_image = Image.open(BytesIO(response.content))
    else:
        pil_image = Image.open(image_path)
    image = np.array(pil_image)
    return image

def generate_region_proposals(image, method='fast'):
    """
    Generates region proposals for object detection.
    
    Args:
        image: Input image
        method: 'fast' or 'quality' for the number of regions
        
    Returns:
        regions: List of proposed regions (x, y, w, h)
    """
    height, width = image.shape[:2]
    regions = []
    
    if method == 'fast':
        window_sizes = [(64, 64), (96, 96), (128, 128), (196, 196), (256, 256)]
        strides = [32, 48, 64, 98, 128]
    else:  # 'quality'
        window_sizes = [(64, 64), (96, 96), (128, 128), (160, 160), (196, 196), (224, 224), (256, 256), (320, 320)]
        strides = [16, 24, 32, 48, 64, 80, 96, 128]
    
    for window_size, stride in zip(window_sizes, strides):
        for y in range(0, height - window_size[1], stride):
            for x in range(0, width - window_size[0], stride):
                regions.append((x, y, window_size[0], window_size[1]))
    
    aspect_ratios = [0.5, 0.75, 1.0, 1.5, 2.0]
    base_sizes = [64, 96, 128, 196, 256]
    
    for base_size in base_sizes:
        for ratio in aspect_ratios:
            w = int(base_size * ratio)
            h = int(base_size / ratio)
            stride = base_size // 2
            if w <= width and h <= height:
                for y in range(0, height - h, stride):
                    for x in range(0, width - w, stride):
                        regions.append((x, y, w, h))
    
    min_area = 500
    max_area = image.shape[0] * image.shape[1] * 0.8
    filtered_regions = []
    for x, y, w, h in regions:
        area = w * h
        if min_area <= area <= max_area and 0.5 <= w/h <= 2.0:
            filtered_regions.append((x, y, w, h))
    
    filtered_regions = list(set(filtered_regions))
    max_regions = 300 if method == 'fast' else 500
    if len(filtered_regions) > max_regions:
        filtered_regions.sort(key=lambda r: r[2] * r[3], reverse=True)
        filtered_regions = filtered_regions[:max_regions]
    
    return filtered_regions

def detect_objects_with_region_proposals(image, model, object_type, confidence_threshold=0.9):
    """
    Detects objects using region proposals.
    
    Args:
        image: Input image
        model: Trained model
        object_type: 'person' or 'car'
        confidence_threshold: Confidence threshold
        
    Returns:
        detections: List of detected objects (x, y, w, h, confidence, object_type)
    """
    target_size = model.input_shape[1:3]  # Dynamically get size (32, 32) or (64, 64)
    regions = generate_region_proposals(image)
    detections = []
    
    for x, y, w, h in regions:
        if y + h <= image.shape[0] and x + w <= image.shape[1]:
            region = image[y:y+h, x:x+w]
            try:
                pil_region = Image.fromarray(region)
                region_resized = np.array(pil_region.resize(target_size))
                region_normalized = region_resized.astype('float32') / 255.0
                region_batch = np.expand_dims(region_normalized, axis=0)
                prediction = model.predict(region_batch, verbose=0)[0][0]
                if prediction > confidence_threshold:
                    detections.append((x, y, w, h, prediction, object_type))
            except:
                continue
    
    return detections

def detect_objects_multi_scale(image, model, object_type, scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5], confidence_threshold=0.9):
    """
    Detects objects using multi-scale detection.
    
    Args:
        image: Input image
        model: Trained model
        object_type: 'person' or 'car'
        scales: List of scaling factors
        confidence_threshold: Confidence threshold
        
    Returns:
        detections: List of detected objects
    """
    height, width = image.shape[:2]
    detections = detect_objects_with_region_proposals(image, model, object_type, confidence_threshold)
    
    for scale in scales:
        scaled_height = int(height * scale)
        scaled_width = int(width * scale)
        pil_image = Image.fromarray(image)
        scaled_image = np.array(pil_image.resize((scaled_width, scaled_height)))
        scaled_detections = detect_objects_with_region_proposals(scaled_image, model, object_type, confidence_threshold)
        for (x, y, w, h, conf, obj_type) in scaled_detections:
            x_orig = int(x / scale)
            y_orig = int(y / scale)
            w_orig = int(w / scale)
            h_orig = int(h / scale)
            detections.append((x_orig, y_orig, w_orig, h_orig, conf, obj_type))
    
    return detections

def detect_large_objects(image, model, object_type, confidence_threshold=0.9):
    """
    Detects large objects in the image.
    
    Args:
        image: Input image
        model: Trained model
        object_type: 'person' or 'car'
        confidence_threshold: Confidence threshold
        
    Returns:
        detections: List of detected large objects
    """
    height, width = image.shape[:2]
    target_size = model.input_shape[1:3]  # Dynamically get size
    detections = []
    coverage_ratios = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for ratio in coverage_ratios:
        region_width = int(width * ratio)
        region_height = int(height * ratio)
        x = (width - region_width) // 2
        y = (height - region_height) // 2
        region = image[y:y+region_height, x:x+region_width]
        try:
            pil_region = Image.fromarray(region)
            region_resized = np.array(pil_region.resize(target_size))
            region_normalized = region_resized.astype('float32') / 255.0
            region_batch = np.expand_dims(region_normalized, axis=0)
            prediction = model.predict(region_batch, verbose=0)[0][0]
            if prediction > confidence_threshold:
                detections.append((x, y, region_width, region_height, prediction, object_type))
        except:
            continue
    
    return detections

def advanced_non_max_suppression(boxes, overlap_threshold=0.3, score_threshold=0.9, containment_threshold=0.95):
    """
    Applies advanced non-maximum suppression to filter overlapping boxes.
    
    Args:
        boxes: List of boxes (x, y, w, h, confidence, object_type)
        overlap_threshold: IoU threshold for overlap
        score_threshold: Confidence threshold
        containment_threshold: Threshold for containment
        
    Returns:
        picked: Filtered list of boxes
    """
    if not boxes:
        return []
    
    boxes = [box for box in boxes if box[4] >= score_threshold]
    if not boxes:
        return []
    
    person_boxes = [box for box in boxes if box[5] == 'person']
    car_boxes = [box for box in boxes if box[5] == 'car']
    
    picked_persons = process_boxes_by_type(person_boxes, 'person', overlap_threshold, containment_threshold)
    picked_cars = process_boxes_by_type(car_boxes, 'car', overlap_threshold, containment_threshold)
    
    return picked_persons + picked_cars

def process_boxes_by_type(boxes, obj_type, overlap_threshold, containment_threshold):
    """
    Helper function to process boxes by object type for NMS.
    """
    if not boxes:
        return []
    
    boxes_array = np.array([(x, y, x + w, y + h, conf) for x, y, w, h, conf, _ in boxes], dtype=np.float32)
    boxes_array = boxes_array[np.argsort(boxes_array[:, 4])[::-1]]
    
    unique_boxes = []
    for box in boxes_array:
        if not any(np.array_equal(box, ub) for ub in unique_boxes):
            unique_boxes.append(box)
    
    boxes_array = np.array(unique_boxes)
    picked = []
    
    while len(boxes_array) > 0:
        current_box = boxes_array[0]
        picked.append(current_box)
        remaining_boxes = boxes_array[1:]
        if len(remaining_boxes) == 0:
            break
        
        xx1 = np.maximum(current_box[0], remaining_boxes[:, 0])
        yy1 = np.maximum(current_box[1], remaining_boxes[:, 1])
        xx2 = np.minimum(current_box[2], remaining_boxes[:, 2])
        yy2 = np.minimum(current_box[3], remaining_boxes[:, 3])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        intersection = w * h
        area1 = (current_box[2] - current_box[0] + 1) * (current_box[3] - current_box[1] + 1)
        area2 = (remaining_boxes[:, 2] - remaining_boxes[:, 0] + 1) * (remaining_boxes[:, 3] - remaining_boxes[:, 1] + 1)
        union = area1 + area2 - intersection
        iou = intersection / union
        containment_ratio2 = intersection / area2
        
        to_remove = []
        for i in range(len(remaining_boxes)):
            if iou[i] > overlap_threshold or (containment_ratio2[i] > containment_threshold and remaining_boxes[i, 4] <= current_box[4] * 1.2):
                to_remove.append(i)
        
        mask = np.ones(len(remaining_boxes), dtype=bool)
        mask[to_remove] = False
        boxes_array = remaining_boxes[mask]
    
    return [(box[0], box[1], box[2] - box[0], box[3] - box[1], box[4], obj_type) for box in picked]

def draw_boxes(image, boxes):
    """
    Draws bounding boxes on the image.
    
    Args:
        image: Input image
        boxes: List of boxes (x, y, w, h, confidence, object_type)
        
    Returns:
        result: Image with drawn boxes
    """
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    for (x, y, w, h, conf, obj_type) in boxes:
        x, y, w, h = int(x), int(y), int(w), int(h)
        color = (0, 255, 0) if obj_type == 'car' else (255, 0, 0)  # Green for cars, Red for persons
        draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=2)
        label = f"{obj_type.capitalize()}: {conf:.2f}"
        draw.text((x, y - 10), label, fill=color)
    
    return np.array(pil_image)

def detect_and_draw_objects(image_path, person_model, car_model, output_path):
    """
    Detects persons and cars in an image and draws bounding boxes.
    
    Args:
        image_path: Path to the image or URL
        person_model: Trained person detection model
        car_model: Trained car detection model
        output_path: Path to save the output image
        
    Returns:
        boxes: List of detected objects
    """
    image = load_and_preprocess_image(image_path)
    start_time = time.time()
    
    person_boxes = detect_objects_multi_scale(image, person_model, 'person', confidence_threshold=0.9)
    large_person_boxes = detect_large_objects(image, person_model, 'person', confidence_threshold=0.9)
    person_boxes.extend(large_person_boxes)
    
    car_boxes = detect_objects_multi_scale(image, car_model, 'car', confidence_threshold=0.9)
    large_car_boxes = detect_large_objects(image, car_model, 'car', confidence_threshold=0.9)
    car_boxes.extend(large_car_boxes)
    
    all_boxes = person_boxes + car_boxes
    boxes = advanced_non_max_suppression(all_boxes, overlap_threshold=0.3, score_threshold=0.6, containment_threshold=0.95)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    result = draw_boxes(image, boxes)
    pil_result = Image.fromarray(result)
    draw = ImageDraw.Draw(pil_result)
    person_count = sum(1 for box in boxes if box[5] == 'person')
    car_count = sum(1 for box in boxes if box[5] == 'car')
    info_text = f"Detected Persons: {person_count} | Detected Cars: {car_count} | Processing Time: {processing_time:.2f}s"
    draw.text((10, 30), info_text, fill=(0, 0, 255))
    result = np.array(pil_result)
    
    Image.fromarray(result).save(output_path)
    
    for i, (x, y, w, h, conf, obj_type) in enumerate(boxes):
        x, y, w, h = int(x), int(y), int(w), int(h)
        object_image = image[y:y+h, x:x+w]
        pil_object = Image.fromarray(object_image)
        draw = ImageDraw.Draw(pil_object)
        color = (0, 255, 0) if obj_type == 'car' else (255, 0, 0)
        draw.rectangle([(0, 0), (w, h)], outline=color, width=2)
        object_output_path = output_path.replace('.jpg', f'_{obj_type}_{i+1}.jpg')
        Image.fromarray(np.array(pil_object)).save(object_output_path)
    
    return boxes

# Test detection
print("Testing person and car detection on images...")
test_images = [
    "test_image_1.jpg",
    "test_image_2.jpg",
    "test_image_3.jpg"
]

for i, image_url in enumerate(test_images):
    image_path = os.path.join(images_dir, f'test_image_{i+1}.jpg')
    if not os.path.exists(image_path):
        response = requests.get(image_url)
        with open(image_path, 'wb') as f:
            f.write(response.content)
    
    output_path = os.path.join(bonus_dir, f'test_image_{i+1}_result.jpg')
    boxes = detect_and_draw_objects(image_path, person_model, car_model, output_path)
    person_count = sum(1 for box in boxes if box[5] == 'person')
    car_count = sum(1 for box in boxes if box[5] == 'car')
    print(f"Image {i+1}: {person_count} persons and {car_count} cars detected")
    plt.figure(figsize=(12, 8))
    plt.imshow(plt.imread(output_path))
    plt.title(f"Image {i+1}: {person_count} persons and {car_count} cars detected")
    plt.axis('off')
    plt.show()