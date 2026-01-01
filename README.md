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

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hidayetergn/An-Operational-Hybrid-Appr.-for-Vehicle-Re-Id-Using-Composite-Attention-Mechanisms.git
   cd An-Operational-Hybrid-Appr.-for-Vehicle-Re-Id-Using-Composite-Attention-Mechanisms.git

---

2. Install dependencies
```bash
pip install -r requirements.txt
```
🛠️ Usage
Step 1: Prepare Data. Before running the inference, ensure your label list is formatted correctly. Use the prepare_data.py script to convert your raw test list into the required format.
```bash
python prepare_data.py --input "path/to/raw_test_list.txt" --output "test_list.txt" --image_dir "path/to/images"
```
--input: Path to your original test list file.
--image_dir: (Optional) Directory of images to verify existence.

Step 2: Run Inference
Run the inference.py script to extract features and identify the hardest negative pairs (the highest similarity between different IDs).
```bash
python inference.py --image_dir "path/to/images" --test_list "test_list.txt" --model_dir "exported_model"
```
--image_dir: Path to the folder containing test images.
--model_dir: Path to the provided exported_model folder.
--output: Name of the output CSV file (default: hardest_pairs.csv).

## 📊 Model Performance & Analysis

The model has been evaluated on the dataset to identify hard-to-distinguish vehicle pairs. Below is a sample of the "Hardest Negative Pairs" where the model calculates high similarity scores for different vehicle IDs.

### Top Hardest Pairs (Sample)
The following table lists pairs of IDs with the highest cosine similarity scores, indicating difficult samples for Re-ID tasks.

| Rank | ID_1 | ID_2 | Similarity Score | Image 1 File | Image 2 File |
| :---: | :---: | :---: | :---: | :--- | :--- |
| 1 | 0032 | 0035 | **0.9124** | `0032_c001_...jpg` | `0035_c002_...jpg` |
| 2 | 0105 | 0108 | **0.8950** | `0105_c003_...jpg` | `0108_c001_...jpg` |
| 3 | 0550 | 0551 | **0.8876** | `0550_c010_...jpg` | `0551_c012_...jpg` |
| 4 | 0012 | 0019 | **0.8720** | `0012_c005_...jpg` | `0019_c005_...jpg` |
| ... | ... | ... | ... | ... | ... |
| 50 | 0772 | 0775 | 0.7540 | `0772_c001_...jpg` | `0775_c004_...jpg` |

> *Note: Full list of 50 hardest pairs is available in the generated `hardest_pairs.csv` file after inference.*

### Visual Examples
Below are examples of query images and their retrieved results using the proposed model architecture.

![Sample Output](assets/sample_result.png)


📊 Output
The script will generate a CSV file (e.g., hardest_pairs.csv) containing the top 50 pairs of vehicle IDs that the model identifies as most similar.
This shows the Rank1 and Rank-5 matches of randomly selected images from the test folder.
![Rank1 and Rank-5 Matches Output](assets/result1.png)
