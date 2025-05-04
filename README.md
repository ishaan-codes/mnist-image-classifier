# MNIST Digit Classification with a Convolutional Neural Network (CNN)

## Overview
This project implements a **Convolutional Neural Network (CNN)** using TensorFlow/Keras to classify handwritten digits from the MNIST dataset. The model leverages modern deep learning techniques to achieve high accuracy in recognizing digits (0-9), serving as a foundational example of image classification with CNNs.

---

## Key Components

### 1. **Dataset**
- **MNIST Dataset**: A benchmark dataset containing 70,000 grayscale images (28x28 pixels) of handwritten digits.
- **Split**: 60,000 training images and 10,000 test images.
- **Preprocessing**:
  - Pixel values normalized to [0, 1].
  - Images reshaped to include a channel dimension (for CNN compatibility).
  - Labels converted to one-hot encoded vectors.

### 2. **Model Architecture**
The CNN follows a hierarchical structure:
1. **Feature Extraction**:
   - Two convolutional layers with ReLU activation:
     - First layer: 32 filters (3x3 kernel).
     - Second layer: 64 filters (3x3 kernel).
   - Max-pooling layers (2x2 window) after each convolution to reduce spatial dimensions.
2. **Classification**:
   - Flattened features fed into a dense layer (128 units, ReLU activation).
   - Final softmax layer for 10-class probability distribution.

### 3. **Training Configuration**
- **Loss Function**: Categorical cross-entropy (suited for multi-class classification).
- **Optimizer**: Adam (adaptive learning rate).
- **Metrics**: Accuracy.
- **Batch Size**: 128 samples per batch.
- **Epochs**: 12 training cycles.
- **Validation**: 10% of training data used for validation.

---

## Key Features
- **Efficient Design**: Combines convolution, pooling, and dense layers to balance accuracy and computational cost.
- **Regularization**: Max-pooling reduces overfitting by progressively downsampling feature maps.
- **Scalability**: Architecture can be extended for more complex image recognition tasks.

---

## Performance
- **Test Accuracy**: ~99.7%.
- **Training Dynamics**:
  - Validation accuracy typically stabilizes at ~99% by epoch 12.
  - Minimal gap between training and validation accuracy, indicating low overfitting.

---

## Applications
- Handwritten digit recognition for postal services, bank check processing, or digitizing historical documents.
- Educational tool for learning CNNs and TensorFlow/Keras workflows.

---

## Requirements
- **Python Libraries**:
  - TensorFlow
  - NumPy

---

## How to Use this Model
1. **Install Dependencies**: Install TensorFlow and NumPy via pip.
2. **Run the Script**: Executing the script automatically downloads the MNIST dataset, trains the model, and reports test accuracy.
3. **Customization**: Adjust hyperparameters (epochs, batch size) or modify the architecture for experimentation.
