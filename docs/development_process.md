# Development Process: CNNs for Car Detection

This document provides a detailed overview of the development process for the CNN-based car detection project, focusing on the challenges encountered and solutions implemented throughout the development journey.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Development Environment](#development-environment)
3. [Data Preparation Challenges](#data-preparation-challenges)
4. [Model Development Challenges](#model-development-challenges)
   - [Class Imbalance Issues](#class-imbalance-issues)
   - [Overfitting Problems](#overfitting-problems)
   - [Model Architecture Considerations](#model-architecture-considerations)
5. [Custom CNN Implementation Challenges](#custom-cnn-implementation-challenges)
6. [Object Detection Implementation](#object-detection-implementation)
7. [Lessons Learned](#lessons-learned)

## Project Overview

The project aimed to develop Convolutional Neural Networks (CNNs) for car detection in images. The development process involved several stages:

1. Preparing the CIFAR-10 dataset for car detection
2. Implementing a CNN using Keras/TensorFlow
3. Creating a custom CNN implementation without using high-level Keras APIs
4. Adapting a pre-trained CNN model for car detection
5. Implementing car detection on real-world images
6. Extending the system to detect people (bonus task)

Each stage presented unique challenges that required careful consideration and problem-solving.

## Development Environment

The project was developed using:
- Python 3.x
- TensorFlow 2.x
- Keras
- OpenCV for image processing
- NumPy and Matplotlib for data manipulation and visualization

The development environment was set up to ensure reproducibility and consistency across different stages of the project.

## Data Preparation Challenges

### CIFAR-10 Dataset Limitations

The CIFAR-10 dataset, while widely used for image classification tasks, presented several challenges for car detection:

1. **Limited Image Size**: The 32x32 pixel images in CIFAR-10 are significantly smaller than real-world images, making it difficult to capture detailed features of cars.

2. **Limited Car Variations**: The 'automobile' class in CIFAR-10 contains a limited variety of car types, angles, and lighting conditions compared to real-world scenarios.

3. **Binary Classification Conversion**: Converting the multi-class CIFAR-10 dataset to a binary classification problem (car vs. non-car) required careful handling to ensure balanced class distribution.

### Solutions Implemented

1. **Data Augmentation**: To address the limited variations, data augmentation techniques were applied, including:
   - Random horizontal flips
   - Small rotations
   - Brightness and contrast adjustments
   - Slight zooming and shifting

2. **Class Balancing**: To ensure balanced training, we:
   - Carefully sampled the 'automobile' class and other classes
   - Used class weights during training to account for any remaining imbalance
   - Monitored class distribution in training, validation, and test sets

3. **Normalization**: All images were normalized to the range [0, 1] to improve training stability and convergence.

## Model Development Challenges

### Class Imbalance Issues

One of the most significant challenges faced during development was class imbalance, particularly when converting CIFAR-10 to a binary classification problem.

#### The Problem

The original CIFAR-10 dataset contains 10 classes with equal distribution (6,000 images per class). When converting to a binary classification (car vs. non-car), we ended up with:
- 6,000 car images (10% of the dataset)
- 54,000 non-car images (90% of the dataset)

This severe imbalance led to several issues:
1. The model would achieve high accuracy (~90%) by simply predicting "non-car" for all images
2. Poor recall for the minority class (cars)
3. Difficulty in learning meaningful features for car detection

#### Solutions Tried

We experimented with multiple approaches to address class imbalance:

1. **Undersampling the Majority Class**: Reducing the number of non-car images to match the car images.
   - **Result**: This led to information loss and poor generalization.

2. **Oversampling the Minority Class**: Duplicating car images to balance the classes.
   - **Result**: This led to overfitting on the car class.

3. **Data Augmentation for Minority Class**: Applying more aggressive augmentation to car images.
   - **Result**: Improved performance but still showed signs of overfitting.

4. **Class Weights**: Assigning higher weights to the car class during training.
   - **Result**: This proved most effective, with a weight ratio of approximately 9:1 (car:non-car).

5. **Focal Loss**: Implementing a modified loss function that focuses more on difficult examples.
   - **Result**: Showed promise but required careful tuning of hyperparameters.

6. **Ensemble Methods**: Combining multiple models trained on different subsets of the data.
   - **Result**: Improved robustness but increased computational complexity.

7. **Two-Phase Training**: First training on a balanced subset, then fine-tuning on the full dataset.
   - **Result**: This approach provided a good balance between learning meaningful features and generalizing well.

After extensive experimentation, we found that a combination of class weights, data augmentation, and two-phase training yielded the best results.

### Overfitting Problems

Another major challenge was overfitting, where the model performed well on training data but poorly on validation and test data.

#### The Problem

During training, we observed:
- Training accuracy quickly reaching >95%
- Validation accuracy plateauing around 75-80%
- Increasing gap between training and validation loss

This indicated that the model was memorizing the training data rather than learning generalizable features.

#### Solutions Tried

We implemented several techniques to combat overfitting:

1. **Dropout Layers**: Adding dropout with rates between 0.25-0.5 after convolutional blocks and dense layers.
   - **Result**: Significant improvement in generalization.

2. **Batch Normalization**: Normalizing activations within the network.
   - **Result**: Improved training stability and allowed for higher learning rates.

3. **L1/L2 Regularization**: Adding weight penalties to discourage large weights.
   - **Result**: L2 regularization (weight decay) with a factor of 1e-4 showed good results.

4. **Early Stopping**: Monitoring validation loss and stopping training when it stopped improving.
   - **Result**: Prevented wasted computation and helped identify optimal training duration.

5. **Reduced Model Complexity**: Experimenting with fewer layers and fewer filters per layer.
   - **Result**: Smaller models sometimes generalized better but had limited capacity.

6. **Learning Rate Scheduling**: Reducing the learning rate when validation metrics plateaued.
   - **Result**: Helped fine-tune the model without overfitting.

7. **Data Augmentation**: Increasing the diversity of training examples.
   - **Result**: Substantial improvement in generalization.

8. **Model Ensembling**: Combining predictions from multiple training runs.
   - **Result**: Improved robustness at the cost of increased inference time.

The most effective combination proved to be dropout, batch normalization, early stopping, and data augmentation. This reduced overfitting while maintaining the model's capacity to learn complex features.

### Model Architecture Considerations

Finding the right model architecture was crucial for balancing performance and computational efficiency.

#### Architectures Explored

1. **Simple CNN**: 2-3 convolutional layers followed by dense layers.
   - **Result**: Fast training but limited performance.

2. **VGG-like Architecture**: Deep stack of 3x3 convolutional layers.
   - **Result**: Good performance but prone to overfitting on our dataset.

3. **ResNet-inspired Architecture**: Adding residual connections.
   - **Result**: Improved gradient flow and training stability.

4. **MobileNetV2**: Lightweight architecture with depthwise separable convolutions.
   - **Result**: Good balance of performance and efficiency.

5. **Custom Architecture**: Tailored specifically for our car detection task.
   - **Result**: Best performance after extensive tuning.

After multiple iterations, we settled on a custom architecture with:
- 3 convolutional blocks (each with 2 conv layers, batch normalization, and max pooling)
- Dropout after each block
- Global average pooling followed by dense layers
- Sigmoid activation for binary classification

This architecture provided the best balance between model capacity, generalization, and computational efficiency.

## Custom CNN Implementation Challenges

Implementing a CNN without using keras.models or keras.layers presented unique challenges that deepened our understanding of the underlying mechanisms.

### The Problem

Building a CNN from scratch required implementing:
1. Forward propagation for all layer types
2. Backward propagation and gradient calculation
3. Parameter updates and optimization
4. Proper initialization of weights and biases

### Solutions Implemented

1. **Modular Design**: Creating separate functions for each layer type and operation.

2. **Numerical Stability**: Implementing techniques to prevent numerical issues:
   - Adding small epsilon values to denominators
   - Using stable softmax and cross-entropy implementations
   - Gradient clipping to prevent exploding gradients

3. **Memory Management**: Carefully managing intermediate activations and gradients to avoid memory issues.

4. **Vectorization**: Using NumPy's vectorized operations for efficiency.

5. **Validation**: Comparing outputs with Keras implementations to verify correctness.

The custom implementation was significantly slower than Keras but provided valuable insights into the inner workings of CNNs.

## Object Detection Implementation

Implementing car detection on real-world images using sliding windows presented several challenges.

### The Problem

1. **Scale Variations**: Cars appear at different sizes in images.
2. **Computational Efficiency**: Naive sliding window approach is computationally expensive.
3. **Multiple Detections**: The same car often detected multiple times.
4. **False Positives**: Background elements incorrectly classified as cars.

### Solutions Implemented

1. **Multi-Scale Detection**: Processing the image at multiple scales to handle different car sizes.

2. **Optimized Sliding Window**: 
   - Using appropriate stride values to balance coverage and computation
   - Early rejection of unlikely windows
   - Parallelizing the detection process where possible

3. **Non-Maximum Suppression (NMS)**: 
   - Grouping overlapping detections
   - Keeping the detection with highest confidence
   - Using an IoU threshold of 0.3 for determining overlaps

4. **Confidence Thresholding**: 
   - Setting appropriate confidence thresholds (0.6-0.7)
   - Different thresholds for different scales

5. **Post-processing**: 
   - Filtering detections based on aspect ratio
   - Size-based filtering to eliminate unrealistic detections

These techniques significantly improved the detection quality and reduced false positives.

## Lessons Learned

Throughout the development process, several key lessons emerged:

1. **Data Quality Over Quantity**: Having well-balanced, diverse, and representative data is more important than simply having more data.

2. **Iterative Development**: Starting with simple models and gradually increasing complexity allowed for better understanding of the problem space.

3. **Systematic Experimentation**: Keeping detailed records of experiments and changing one variable at a time helped identify effective approaches.

4. **Validation Strategy**: Using appropriate validation metrics beyond accuracy (precision, recall, F1-score) was crucial for meaningful evaluation.

5. **Computational Trade-offs**: Balancing model complexity with computational constraints required careful consideration of architecture choices.

6. **Transfer Learning Value**: Pre-trained models provided significant advantages in terms of feature extraction and training efficiency.

7. **Implementation Details Matter**: Small implementation details (initialization, normalization, etc.) can have substantial impacts on model performance.

8. **Visualization is Key**: Regularly visualizing model predictions, activations, and gradients helped identify issues early.

The development process was iterative and required patience, systematic experimentation, and careful analysis of results. The challenges encountered, particularly with class imbalance and overfitting, provided valuable learning opportunities and led to a more robust final solution.
