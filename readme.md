# This is AI capstone project IBM AI Engineering

# Satellite Imagery Classification: CNN & ViT Hybrid

## Project Overview
This project provides a deep learning pipeline for the binary classification of satellite imagery (agricultural vs. non-agricultural land). Designed for scalability and high performance, the repository demonstrates the implementation, comparison, and integration of Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) using both **TensorFlow/Keras** and **PyTorch**.

## Data Pipeline
- **Dataset:** 6,000 perfectly balanced satellite images (3,000 `class_1_agri`, 3,000 `class_0_non_agri`).
- **Data Loading:** Emphasizes memory-efficient sequential/lazy loading (`tf.data.Dataset` / PyTorch `DataLoader`) over naive bulk loading to mitigate RAM bottlenecks and optimize GPU I/O synchronization.
- **Preprocessing:** Standardized image resizing (64x64), normalization, and on-the-fly data augmentation to improve model generalization.

## Model Architectures
1. **Baseline CNNs:** Custom networks built in both Keras and PyTorch. Focuses on hierarchical spatial learning through stacked convolutional and pooling layers.
2. **Vision Transformers (ViTs):** Treats images as tokenized patch sequences, utilizing positional encodings and multi-head self-attention to model global spatial relationships.
3. **CNN-ViT Hybrid:** An optimized architecture utilizing transfer learning. A pre-trained CNN backbone (e.g., ResNet50) extracts local features and reduces dimensionality, feeding spatial feature maps into Transformer encoder blocks for global context modeling. 

## Training & Evaluation
- **Hyperparameters:** Adam optimizer, Binary Cross-Entropy loss, dynamic learning rate schedules (step decay), and dropout for regularization.
- **Metrics Evaluated:** Accuracy, Precision (minimizing false positives for resource allocation), Recall, F1-Score, ROC-AUC, and Confusion Matrices.
- **Performance:** Optimized hybrid models achieve **>95% accuracy**.

## Tech Stack
- **Frameworks:** PyTorch, TensorFlow / Keras
- **Core Concepts:** Computer Vision, Transfer Learning, Self-Attention, Scalable Data Pipelines