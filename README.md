# Using Data Augmentation to Improve the Performance of the ResNet-18 Model in Identifying Tuberculosis in Chest X-rays

This repository contains a PyTorch-based project for diagnosing tuberculosis (TB) from chest X-ray images using ResNet-18. Multiple data augmentation strategies were explored (horizontal flip and variations in brightness, contrast, saturation, noise, and image resolution) to improve model performance.

---

## Table of Contents
1. [Project Description](#project-description)
2. [Dataset](#dataset)
3. [Repository Structure](#repository-structure)
4. [Setup and Installation](#setup-and-installation)
5. [Usage](#usage)
6. [Files Overview](#files-overview)
7. [References](#references)
8. [License](#license)

---

## Project Description

Tuberculosis remains a major global health concern. This project leverages a Convolutional Neural Network (CNN) based on ResNet-18 to distinguish between normal and TB-positive chest X-ray images. Various data augmentations were tested to see how they affect model performance.

---

## Dataset

The dataset comes from a publicly available TB Chest Radiography Database curated by researchers from Qatar University, the University of Dhaka, and others. **We do not host the entire dataset here** due to size and licensing restrictions.

- See [`archive/TB_Chest_Radiography_Database/README.md.txt`](archive/TB_Chest_Radiography_Database/README.md.txt) for details on how the data was collected and how you can obtain it yourself.
- Once you have the dataset, place it in the same structure as described in `archive/TB_Chest_Radiography_Database/` (with `Normal/` and `Tuberculosis/` subfolders).

---

## Repository Structure

├── code/
│ ├── blur_resnet18.py
│ ├── brightness_resnet18.py
│ ├── contrast_resnet18.py
│ ├── control_resnet18.py
│ ├── enhance_resnet18.py
│ ├── horizontal_flip_resnet18.py
│ ├── noise_resnet18.py
│ └── saturation_resnet18.py
├── archive/
│ ├── TB_Chest_Radiography_Database
│ └── DATASET.md (Dataset documentation, references, etc.)
├── run.sh (Example shell script to run some scripts)
├── research_paper.pdf
└── README.md (You are here)

