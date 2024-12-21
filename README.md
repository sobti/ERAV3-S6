# MNIST CNN Classifier

![Build Status](https://github.com/sobti/ERAV3/actions/workflows/ml-pipeline.yml/badge.svg)


This project implements a Convolutional Neural Network (CNN) to classify handwritten digits (0-9) from the MNIST dataset using PyTorch. The model is trained on transformed images to improve robustness and accuracy.

---

## Features

- **Convolutional Neural Network**: A simple CNN model for digit classification.
- **Data Augmentation**: Includes transformations like brightness, rotation, and normalization.
- **Training Pipeline**: Automates training, testing, and accuracy reporting.
- **Grid Visualization**: Saves a grid of transformed training images for verification.
- **GitHub Actions Support**: CI/CD pipeline for training and saving artifacts.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/mnist-cnn.git

2. # MNIST CNN Classifier

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits (0-9) from the MNIST dataset using PyTorch. The model is trained on transformed images to improve robustness and accuracy.

---

## Features

- **Convolutional Neural Network**: A simple CNN model for digit classification.
- **Data Augmentation**: Includes transformations like brightness, rotation, and normalization.
- **Training Pipeline**: Automates training, testing, and accuracy reporting.
- **Grid Visualization**: Saves a grid of transformed training images for verification.
- **GitHub Actions Support**: CI/CD pipeline for training and saving artifacts.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/mnist-cnn.git
   cd mnist-cnn
2. # MNIST CNN Classifier

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits (0-9) from the MNIST dataset using PyTorch. The model is trained on transformed images to improve robustness and accuracy.

---

## Features

- **Convolutional Neural Network**: A simple CNN model for digit classification.
- **Data Augmentation**: Includes transformations like brightness, rotation, and normalization.
- **Training Pipeline**: Automates training, testing, and accuracy reporting.
- **Grid Visualization**: Saves a grid of transformed training images for verification.
- **GitHub Actions Support**: CI/CD pipeline for training and saving artifacts.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/mnist-cnn.git
   cd mnist-cnn
2. Install dependencies: Ensure Python 3.8 or higher is installed, then run:
   pip install -r requirements.txt
3. Download the MNIST dataset: The dataset is automatically downloaded during the first training run.
4. Training the Model
   Run the training script:python train_model.py
This will:

- Train the CNN on the MNIST training dataset.
- Evaluate the model on the test dataset.
- Save the trained model to a timestamped file (e.g., model_2024-12-05_98.20.pth).
- Visualizing Transformations
- The script saves a grid of transformed training images to outputs/transformed_images_grid.png. Open this file to verify the data augmentation pipeline.
5. Model Details
   - Transformations:Brightness Adjustment: Randomly alters image brightness.
   - Rotation: Rotates images between -10° and 10°.
   - Normalization: Normalizes images using mean 0.1307 and std 0.3081.
   - Batch Norm is used
   - Drop Out is Used
   - GAP is used
   - 1*1 kernal is used
   - Parameter is below 10K
   - Test Accuracy is 99.41%
   - Early Stopping is used

## Test Logs:

- Epoch: 0 Test set: Average loss: 0.0001, Accuracy: 9795/10000 (97.95%)
- Epoch: 1 Test set: Average loss: 0.0001, Accuracy: 9799/10000 (97.99%)
- Epoch: 2 Test set: Average loss: 0.0001, Accuracy: 9841/10000 (98.41%)
- Epoch: 3 Test set: Average loss: 0.0001, Accuracy: 9842/10000 (98.42%)
- Epoch: 4 Test set: Average loss: 0.0000, Accuracy: 9928/10000 (99.28%)
- Epoch: 5 Test set: Average loss: 0.0000, Accuracy: 9937/10000 (99.37%)
- Epoch: 6 Test set: Average loss: 0.0000, Accuracy: 9939/10000 (99.39%)
- Epoch: 7 Test set: Average loss: 0.0000, Accuracy: 9941/10000 (99.41%)


