# Car vs Person Detection using YOLO11n

This project implements an object detection model to distinguish between cars and people using the YOLO11n architecture. The entire workflow includes data collection, annotation, augmentation, model training, evaluation, and error analysis.

---

## Project Overview

- **Objective:** Build a robust object detection model to identify cars and people in images.
- **Dataset:** Custom dataset of 149 images, manually annotated using Roboflow.
- **Augmentation:** Applied Albumentations for data augmentation, expanding dataset to over 625 samples.
- **Model:** YOLO11n (nano) trained in Google Colab with 50 epochs.
- **Results:** Achieved mean Average Precision (mAP@0.5) of 83.5% and mAP@0.5:0.95 of 47.1% on validation data.
- **Challenges:** Model performs well on clear images but struggles with crowded scenes and overlapping objects.

---

## Dataset

The dataset includes images containing cars and people with bounding box annotations in YOLO format.

- Original dataset collected manually (149 images).
- Annotated on [Roboflow](https://roboflow.com).
- Augmented dataset using Albumentations (Horizontal Flip, Rotation, Brightness/Contrast, Resize).
- [Download Dataset](https://drive.google.com/file/d/1Eu3JjGdbTSGQfmkk4PW-_JpLC6TZzSap/view?usp=sharing)

---

##Google Colab Link
 - Access the code [here](https://colab.research.google.com/drive/1qT5iCUGZN5jkqX3G4QLGLRDBLos64-Yg?usp=sharing)
