<div align="center">

# 🧬 CNN-ALL

### Acute Lymphoblastic Leukemia Detection with Adaptive Unsharpening and Deep Learning

[![MATLAB](https://img.shields.io/badge/MATLAB-R2018%2B-orange?logo=mathworks)](https://www.mathworks.com/products/matlab.html)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20ICASSP%202021-00629B)](https://ieeexplore.ieee.org/document/9414362)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://iebil.di.unimi.it/cnnALL/index.htm)
[![Language](https://img.shields.io/badge/Language-MATLAB-yellow)](https://www.mathworks.com/products/matlab.html)

**MATLAB source code for the ICASSP 2021 paper**  
*Acute Lymphoblastic Leukemia detection based on adaptive unsharpening and Deep Learning*

</div>

---

## 🧠 Overview

**CNN-ALL** is a MATLAB implementation for **Acute Lymphoblastic Leukemia (ALL) detection** from microscopic blood-cell images.

The method combines image enhancement and deep-learning-based analysis, including:

- **Adaptive unsharpening** for image-quality enhancement
- **Deep convolutional neural network classification**
- **Focus and image-quality processing utilities**
- **Color normalization and stain-processing routines**
- **Performance evaluation for ALL detection**

The repository accompanies the 2021 IEEE ICASSP paper by Genovese, Hosseini, Piuri, Plataniotis, and Scotti.

---

## 🔬 Method at a Glance

<div align="center">

```text
Microscopic blood-cell images
        │
        ▼
Image preprocessing
        │
        ▼
Adaptive unsharpening / enhancement
        │
        ▼
CNN-based feature learning
        │
        ▼
Classification
        │
        ▼
ALL / Healthy prediction
```

</div>

The pipeline is designed to improve visual information before classification, helping the network focus on diagnostically relevant image structures.

---

## 📁 Repository Structure

```text
CNN-ALL/
│
├── launch_CNN_ALL.m          # Main launch script
├── functions/                # Core processing and evaluation functions
├── libraries/                # External and supporting libraries
├── params/                   # Experiment and dataset parameter files
├── steps/                    # Pipeline steps
├── util/                     # Utility functions
├── LICENSE                   # GPL-3.0 license
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/AngeloUNIMI/CNN-ALL.git
cd CNN-ALL
```

### 2. Download the dataset

The code expects the **ALL-IDB2** images from the ALL-IDB dataset.

Dataset page:

```text
https://homes.di.unimi.it/scotti/all/
```

Place the images in the following directory:

```text
./imgs/orig/ALL-IDB/ALL_IDB2/img/
```

Expected image filename format:

```text
Im001_1.tif
```

### 3. Configure parameters

Check and edit the parameter files in:

```text
./params/
```

These files define dataset paths, preprocessing options, classification settings, and experiment configuration.

### 4. Run the pipeline

Open MATLAB from the repository root and run:

```matlab
launch_CNN_ALL
```

---

## 📊 Expected Outputs

Depending on the selected configuration, the pipeline can generate:

| Output | Description |
|---|---|
| Classification results | ALL / non-ALL prediction performance |
| Trained models | CNN models or intermediate training artifacts |
| Intermediate features | Image descriptors and learned representations |
| Evaluation metrics | Accuracy and experiment-level statistics |
| Logs/results files | Saved outputs for reproducibility |

---

## 🧩 Included and Related Libraries

Part of this repository uses or builds upon code and concepts from:

- **PCANet**  
  T. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng, and Y. Ma,  
  *PCANet: A Simple Deep Learning Baseline for Image Classification?*  
  IEEE Transactions on Image Processing, 2015.

- **1Shot-MaxPol**  
  Mahdi S. Hosseini and Konstantinos N. Plataniotis,  
  *Convolutional Deblurring for Natural Imaging*,  
  IEEE Transactions on Image Processing, 2019.

- **FQPath**  
  Mahdi S. Hosseini et al.,  
  *Focus Quality Assessment of High-Throughput Whole Slide Imaging in Digital Pathology*,  
  IEEE Transactions on Medical Imaging, 2019.

- **Fast N-D Grayscale Image Segmentation with c-/Fuzzy c-Means**

- **Stain Deconvolution / SCD_FastICA**

- **Comprehensive Colour Image Normalization**  
  G. Finlayson, B. Schiele, and J. Crowley, ECCV 1998.

---

## 📚 Paper

If you use this code, please cite:

```bibtex
@InProceedings{icassp21,
  author    = {A. Genovese and M. S. Hosseini and V. Piuri and K. N. Plataniotis and F. Scotti},
  title     = {Acute Lymphoblastic Leukemia detection based on adaptive unsharpening and Deep Learning},
  booktitle = {Proc. of the 2021 IEEE Int. Conf. on Acoustics, Speech, and Signal Processing (ICASSP 2021)},
  address   = {Toronto, ON, Canada},
  pages     = {1205--1209},
  month     = {June},
  day       = {6--11},
  year      = {2021},
  note      = {ISBN: 978-1-7281-7605-5. DOI: 10.1109/ICASSP39728.2021.9414362}
}
```

Paper:

```text
https://ieeexplore.ieee.org/document/9414362
```

Project page:

```text
https://iebil.di.unimi.it/cnnALL/index.htm
```

---

## 👥 Authors

- **Angelo Genovese**  
  Department of Computer Science, Università degli Studi di Milano, Italy

- **Mahdi S. Hosseini**  
  Department of Electrical and Computer Engineering, University of Toronto, Canada

- **Vincenzo Piuri**  
  Department of Computer Science, Università degli Studi di Milano, Italy

- **Konstantinos N. Plataniotis**  
  Department of Electrical and Computer Engineering, University of Toronto, Canada

- **Fabio Scotti**  
  Department of Computer Science, Università degli Studi di Milano, Italy

---

## 📄 License

This project is released under the **GNU General Public License v3.0**.

See the [LICENSE](LICENSE) file for details.
