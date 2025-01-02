# Using Data Augmentation to Improve the Performance of the ResNet-18 Model in Identifying Tuberculosis in Chest X-rays

This repository contains a PyTorch-based project for diagnosing tuberculosis (TB) from chest X-ray images using ResNet-18. Multiple data augmentation strategies were explored (horizontal flip and variations in brightness, contrast, saturation, noise, and image resolution) to improve model performance.

---

## Table of Contents
1. [Project Description](#project-description)
2. [Dataset](#dataset)
3. [Repository Structure](#repository-structure)
4. [Setup and Installation](#setup-and-installation)
5. [Usage](#usage)
6. [References](#references)
7. [License](#license)

---

## Project Description

Tuberculosis remains a major global health concern. This project leverages a Convolutional Neural Network (CNN) based on ResNet-18 to distinguish between normal and TB-positive chest X-ray images. Various data augmentations were tested to see how they affect model performance.

---

## Dataset

The dataset comes from a publicly available TB Chest Radiography Database curated by researchers from Qatar University, the University of Dhaka, and others. **We do not host the entire dataset here** due to size and licensing restrictions.

- See [`archive/TB_Chest_Radiography_Database/DATASET.md`](archive/TB_Chest_Radiography_Database/DATASET.md) for details on how the data was collected and how you can obtain it yourself.
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
│ ├── TB_Chest_Radiography_Database (Images and metadata)
│ └── DATASET.md (Dataset documentation, references, etc.)
├── run.sh (Example shell script to run some scripts)
├── research_paper.pdf
└── README.md (You are here)

---

## Setup and Installation

- Clone the repository:
    - `git clone https://github.com/msukhatme/research-project.git`
    - `cd research-project`
- Install dependencies:
    - Python 3.7+
    - PyTorch
    - Torchvision
    - Scikit-learn (for precision, recall, f-score)
    - TensorBoard (optional, for logs)
    - `pip install torch torchvision scikit-learn tensorboard`
- Add the dataset:
    - Download the dataset from the original source linked in `archive/TB_Chest_Radiography_Database/DATASET.md`.
    - Place the `Normal/` and `Tuberculosis/` folders under `archive/TB_Chest_Radiography_Database/`.

---

## Usage

- Training models
    - Baseline/control (no augmentation)
        - `cd code`
        - `python control_resnet18.py`
        - This trains a ResNet-18 model on the original dataset with minimal transforms.
    - Augmented models
        - Each script applies a different augmentation strategy:
            - horizontal_flip_resnet18.py (random horizontal flips)
            - brightness_resnet18.py (brightness augmentation)
            - contrast_resnet18.py (contrast augmentation)
            - saturation_resnet18.py (saturation adjustment)
            - noise_resnet18.py (random noise addition)
            - blur_resnet18.py (image blurring)
            - enhance_resnet18.py (sharpness enhancements)
        - Run them similarly to the control code.
    - Shell script
        - `./run.sh`
        - This runs a subset of experiments.
- TensorBoard logs
    - During training, logs (loss, accuracy, precision, etc.) will be written to folders like `runs/`.
    - To visualize in TensorBoard:
        - `tensorboard --logdir=runs`

---

## References

If you use this dataset or any part of this project in your work, please cite:
- Tawsifur Rahman, Amith Khandakar, Muhammad A. Kadir, Khandaker R. Islam, Khandaker F. Islam, Zaid B. Mahbub, Mohamed Arselene Ayari, Muhammad E. H. Chowdhury. (2020) "Reliable Tuberculosis Detection using Chest X-ray with Deep Learning, Segmentation and Visualization". IEEE Access, Vol. 8, pp 191586 - 191601. DOI. 10.1109/ACCESS.2020.3031384.
- Other dataset references as listed in `archive/TB_Chest_Radiography_Database/DATASET.md`.
- This repository, `https://github.com/msukhatme/research-project`.

---

## License

All rights reserved.

You are not permitted to use, copy, modify, or distribute any part of this project without prior permission. For any inquiries or requests, please contact me at `msukhatme@uchicago.edu`.
