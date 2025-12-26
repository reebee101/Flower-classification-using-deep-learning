🌸 Flower Classification using Deep Learning (Oxford 102 Flowers)

This repository contains an optimized Computer Vision pipeline for flower classification using ResNet50 and the Oxford Flowers 102 dataset.
The project focuses on advanced image preprocessing, transfer learning, and fine-tuning to achieve high classification accuracy.

🚀 Project Overview

Dataset: Oxford Flowers 102

Task: Multi-class image classification (102 flower classes)

Model: ResNet50 (ImageNet pretrained)

Frameworks: TensorFlow, Keras, OpenCV

Training Strategy:

Stage 1: Train classifier head

Stage 2: Fine-tune top ResNet layers

🧠 Key Features

Advanced preprocessing using CLAHE and image sharpening

Data augmentation (flip, rotation, zoom)

Mixed precision training for performance optimization

Two-stage training strategy

Confusion matrix and accuracy/loss visualization

Custom image prediction pipeline with enhancement & segmentation

🗂️ Project Structure
📦 flower-classification
 ┣ 📜 cv project 2.py
 ┣ 📜 flower_classifier_optimized.h5
 ┣ 📜 README.md

🛠️ Technologies Used

Python

TensorFlow & Keras

TensorFlow Datasets (TFDS)

OpenCV

NumPy

Matplotlib

Scikit-learn

📊 Training Pipeline
🔹 Stage 1 – Feature Extraction

Freeze ResNet50 base

Train custom classification head

Optimizer: Adam (1e-3)

🔹 Stage 2 – Fine Tuning

Unfreeze top ResNet layers

Lower learning rate for stability

Optimizer: Adam (1e-4)

📈 Results

Final Test Accuracy: ~85%

Improved performance compared to baseline reference models

Stable convergence with early stopping

🖼️ Visualization Outputs

Training & validation accuracy curves

Training & validation loss curves

Confusion matrix

Sample predictions on test images

Accuracy comparison bar chart

🔍 Image Prediction Pipeline

Image acquisition

Color enhancement using CLAHE

Image sharpening

Segmentation (for visualization only)

Resizing & preprocessing

Model prediction

▶️ How to Run

Install dependencies:

pip install tensorflow tensorflow-datasets opencv-python matplotlib scikit-learn


Run the script:

python "cv project 2.py"


For prediction on a custom image, update:

image_path = "path_to_your_image.jpg"

💾 Model Saving

The trained model is saved as:

flower_classifier_optimized.h5

📌 Notes

GPU is recommended for training

Mixed precision is enabled for performance

Segmentation is used for visualization only (not inference)
