Official implementation of the proposed Vehicle Re-Identification (Re-ID) framework. This repository contains the inference code and the pre-trained model for the Vehicle Re-Identification method described in the manuscript. The model is provided in the TensorFlow `SavedModel` format, which encapsulates the architecture (including custom layers) and the trained weights.


## 📌 Overview

The proposed framework extracts discriminative embeddings by combining:

* Traditional CNNs struggle to capture high-frequency spatial variations in vehicle identities due to the linear nature of standard convolutions. This repository contains the source code for the **Operational Hybrid Network**, which introduces:
1.  **Operational Neural Networks (ONN):** Replacing static weights with learnable Taylor-series nodal operations.
2.  **GCAM (Global Context Attention Module):** A non-linear self-attention mechanism for global topology.
3.  **OBAM (Operational Block Attention Module):** For suppressing environmental clutter.

All features are pooled using **Generalized Mean (GeM) pooling**, L2-normalized, and concatenated into a single embedding vector for retrieval-based vehicle re-identification.
# Vehicle Re-Identification Model Evaluation

## 📂 Repository Structure

- `saved_model/`: Contains the pre-trained model graph and weights (.pb and variables).
- `prepare_data.py`: Helper script to format the dataset list for testing.
- `evaluation.py`: Main script to run feature extraction and similarity analysis.
- `requirements.txt`: List of dependencies.
- `dataset/`: List of test images as a zip file and test_list.txt.


## 🧠 Model Architecture

* **Backbone**: EfficientNet B4
* **Pooling** : Details will be shared along with the source code once the paper is accepted.
* **GCAM**    : Details will be shared along with the source code once the paper is accepted.
* **OBAM**    : Details will be shared along with the source code once the paper is accepted.

## 📂 Repository Structure

```text
Operational-Hybrid-ReID/
├── config.yaml    
├── dataset/
│   ├── image_test.zip      <-- Downloaded raw test dataset
│   └── test_list.txt       <-- Image test list           
├── saved_model     # Pre-trained model
├── utils/
│   ├──__init__.py       
│   └── loader.py    
├── evaluation.py
├── prepare_data.py                  
├── requirements.txt
└── README.md
```

## ⚙️ Environment Setup

### Requirements

* Python ≥ 3.8
* PyTorch ≥ 2.0
* CUDA (A100 GPU, for GPU acceleration)


## 📁 Dataset Preparation

The VeRi-776 dataset should follow the structure below:

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

Tri-Stream Attention: Simultaneous processing of spatial, channel, and global self-attention with an ONNs layer to identify fine-grained details (e.g., logos, stickers, roof racks).

GeM Pooling: Implementation of trainable Generalized Mean Pooling for better feature saliency compared to traditional Global Average Pooling.

Metric Robustness: Optimization via a stabilized Hard Triplet Loss combined with Label Smoothing Cross-Entropy.

📊 Performance & Evaluation
To ensure transparency and reproducibility, we provide an automated evaluation script. This script allows users to verify the following metrics:

Retrieval Accuracy: mAP, Rank-1, and Rank-5.

Embedding Quality: Representation cosine distance and similarity of the Hardest 50 IDs

Discriminative Power: Inter-class to Intra-class distance ratios.

Dataset
The models are evaluated on the VeRi-776 dataset. Please download the dataset from here: https://www.kaggle.com/datasets/abhyudaya12/veri-vehicle-re-identification-dataset

Load the pre-trained weights, perform inference on the test set, and generate all metrics and visualizations (Hardest 50 Negatives, Top-5 Rankings).

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
python prepared_data.py
```
Step 2: Run Inference
Run the evaluation.py script to extract features and identify the hardest negative pairs (the highest similarity between different IDs).
```bash
python evaluation.py
```
When the code runs successfully, you will see a confirmation message in the terminal similar to the following: ✅✅ ALL ANALYSES COMPLETED SUCCESSFULLY ✅✅

## 📊 Model Performance & Analysis

The model has been evaluated on the dataset to identify hard-to-distinguish vehicle pairs. The model was evaluated on the dataset to identify difficult-to-distinguish vehicle pairs. The following outputs are available in these test results: "Hardest 50 Vehicle ID Table", "Hardest 50 Vehicle Failures of Model", and "Hardest 50 Vehicle Analysis". Other test results and metrics will be shared here, along with all training and test code, after our article is published.

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
