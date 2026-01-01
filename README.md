# An Operational Hybrid Approach for Vehicle Re-Id Using Composite Attention Mechanisms
Vehicle Re-Identification (V-ReID) is a fundamental yet challenging task in Intelligent Transportation Systems, particularly in scenarios involving non-overlapping camera networks and Unmanned Aerial Vehicle (UAV) surveillance. Traditional Convolutional Neural Networks (CNNs) rely predominantly on linear operations for feature extraction.

Official implementation of the proposed Vehicle Re-Identification (Re-ID) framework. This repository contains the inference code and the pre-trained model for the Vehicle Re-Identification method described in the manuscript. The model is provided in the TensorFlow `SavedModel` format, which encapsulates the architecture (including custom layers) and the trained weights.


## 📌 Overview

The proposed framework extracts discriminative embeddings by combining:

* We have used the EfficientNet B4 model instead of the ResNet model, which uses a backbone network.
* A Global-Fusion Attention Module (GFAM)
* An Operational Block Attention Module (OBAM)

All features are pooled using **Generalized Mean (GeM) pooling**, L2-normalized, and concatenated into a single embedding vector for retrieval-based vehicle re-identification.
# Vehicle Re-Identification Model Evaluation

## 📂 Repository Structure

- `exported_model/`: Contains the pre-trained model graph and weights (.pb and variables).
- `prepare_data.py`: Helper script to format the dataset list for testing.
- `inference.py`: Main script to run feature extraction and similarity analysis.
- `requirements.txt`: List of dependencies.


## 🧠 Model Architecture

* **Backbone**: EfficientNet B4
* **Pooling**: Details will be shared along with the source code once the paper is accepted.
* **GFAM**: Details will be shared along with the source code once the paper is accepted.
* **OBAM**:    Details will be shared along with the source code once the paper is accepted.

## 📂 Repository Structure

```text
project_root/
│
├── main.py
├── config.yaml
├── requirements.txt
├── README.md
│
├── models/
│   ├── __init__.py
│   ├── gem.py
│
├── data/
│   ├── __init__.py
│   └── data_utils.py
│
└── visualization/
    ├── __init__.py
    └── visualize.py
```

## ⚙️ Environment Setup

### Requirements

* Python ≥ 3.8
* PyTorch ≥ 2.0
* CUDA (A100 GPU, for GPU acceleration)

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone <REPO_URL>
   cd <REPO_NAME>

---

## 📁 Dataset Preparation

The dataset should follow the structure below:

```text
image_test/
├── 0001_xxxx_xxxxx_0
├── 0001_xxxx_xxxxx_1
├── ...
```

A text file describing the test set must be provided in the following format:

```text
image_name_without_extension  vehicle_id
```

Example:

```text
0003_c014_00018740_0 3
```

🚀 Key Contributions
ONN-CNN Hybrid Backbone: Utilizing Operational Neural Networks to capture complex, non-linear patterns that standard CNNs often overlook.

Tri-Stream Attention: Simultaneous processing of spatial, channel, and global self-attention to identify fine-grained details (e.g., logos, stickers, roof racks).

GeM Pooling: Implementation of trainable Generalized Mean Pooling for better feature saliency compared to traditional Global Average Pooling.

Metric Robustness: Optimization via a stabilized Hard Triplet Loss combined with Label Smoothing Cross-Entropy.

📊 Performance & Evaluation
To ensure transparency and reproducibility, we provide an automated evaluation script. This script allows users to verify the following metrics:

Retrieval Accuracy: mAP, Rank-1, and Rank-5.

Embedding Quality: Representation cosine distance and similarity of the Hardest 50 IDs

Discriminative Power: Inter-class to Intra-class distance ratios.

Pre-trained Weights
Download the weights from the following link and place them in the weights/ directory:

Download best_model.weights.h5

Dataset
The models are evaluated on the VeRi-776 dataset. Please download the dataset from here: https://www.kaggle.com/datasets/abhyudaya12/veri-vehicle-re-identification-dataset

The following command will load the pre-trained weights, perform inference on the test set, and generate all metrics and visual plots (t-SNE, Top-5 Rankings):
python evaluation.py --weights weights/best_model.weights.h5 --data_dir ./data/te
