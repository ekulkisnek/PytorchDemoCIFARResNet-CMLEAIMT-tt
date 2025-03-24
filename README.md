
# PyTorch Deep Learning Demo: CIFAR-10 Classification

A Streamlit-based interactive demonstration of deep learning using PyTorch and the CIFAR-10 dataset. This project showcases transfer learning using ResNet18 architecture for image classification.

## Project Overview

This application provides an interactive interface to explore various aspects of deep learning, including:

### 1. Dataset Exploration
- Visualizes random samples from the CIFAR-10 dataset
- Displays images with their corresponding class labels
- CIFAR-10 consists of 60,000 32x32 color images across 10 different classes

### 2. Model Architecture
- Uses ResNet18 pre-trained on ImageNet
- Modified for CIFAR-10:
  - Adapted first convolution layer for 32x32 images
  - Removed max pooling layer
  - Modified final fully connected layer for 10 classes

### 3. Training Interface
- Interactive training with adjustable parameters:
  - Configurable number of epochs
  - Adjustable learning rate
  - Real-time training progress visualization
  - Live loss and accuracy tracking
- Uses Adam optimizer and CrossEntropy loss
- Implements data augmentation (random crops and flips)

### 4. Feature Visualization
- Displays feature maps from the first convolutional layer
- Provides insights into what the network "sees"
- Shows how input images are processed in early layers

### 5. Model Inference
- Real-time inference on test images
- Side-by-side comparison of true and predicted labels
- Demonstrates model's classification performance

## Code Structure

- `app.py`: Main Streamlit application interface and layout
- `model_utils.py`: Model loading and configuration utilities
- `training.py`: Training and evaluation functions
- `visualization.py`: Plotting and visualization functions

## Technical Details

### Model Architecture
The project uses ResNet18 with transfer learning:
- Pre-trained weights from ImageNet
- Modified input layer for 32x32 images
- Final layer adapted for 10-class classification
- Training uses Adam optimizer with CrossEntropy loss

### Data Processing
- Images are normalized using ToTensor transform
- Training augmentation includes:
  - Random horizontal flips
  - Random crops with padding
- Batched processing with DataLoader

### Visualization Features
- Real-time training metrics plotting
- Feature map visualization
- Model architecture display
- Interactive dataset exploration

## Implementation Details

The application is built using several key Python libraries:
- PyTorch for deep learning
- Streamlit for the web interface
- Matplotlib for visualization
- Torchvision for dataset and model utilities

This implementation demonstrates best practices in:
- Transfer learning
- Interactive machine learning
- Real-time visualization
- Model inference
