# MEDICAL-X — Chest X-Ray Disease Detection

Multi-label classification of **14 thoracic diseases** from chest X-rays using a fine-tuned **DenseNet121** on the NIH ChestX-ray14 dataset. Includes **Grad-CAM** visualization to highlight the regions the model focuses on for each predicted disease.

---
### Live Demo 
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://medical-x.streamlit.app)

## Diseases Detected

The model predicts the presence of one or more of the following conditions simultaneously:

| # | Disease | # | Disease |
|---|---------|---|---------|
| 1 | Atelectasis | 8 | Pneumothorax |
| 2 | Cardiomegaly | 9 | Consolidation |
| 3 | Effusion | 10 | Edema |
| 4 | Infiltration | 11 | Emphysema |
| 5 | Mass | 12 | Fibrosis |
| 6 | Nodule | 13 | Pleural Thickening |
| 7 | Pneumonia | 14 | Hernia |

---

## Dataset

**NIH ChestX-ray14** — Released by the National Institutes of Health.

- 112,120 frontal-view chest X-ray images
- 30,805 unique patients
- Labels extracted from radiology reports via NLP (weakly supervised)
- Each image can have multiple disease labels (multilabel)
- Images are 1024×1024 PNG, resized to 224×224 for training

Dataset source: [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC) | [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) 
Download and place files in the `data/` directory:
```
data/
├── images/
└── Data_Entry_2017.csv
```

---

## Model Architecture

**DenseNet121** pretrained on ImageNet, fine-tuned for multilabel chest X-ray classification.

```
DenseNet121
├── features          (pretrained, frozen backbone)
│   └── ...DenseBlocks + Transition Layers
│   └── norm5         ← Grad-CAM target layer
└── classifier
    └── Linear(1024 → 14)   ← custom head, trained from scratch
```

Key design decisions:
- **Sigmoid activation** (not softmax) — each of the 14 outputs is independent
- **Binary Cross-Entropy loss** — standard for multilabel classification
- **One-hot encoded labels** — multi-hot vectors per image
- **Threshold = 0.5** — a disease is predicted positive if sigmoid output ≥ 0.5

---

## Performance & Validation Metrics

The model's diagnostic capability is rigorously evaluated across all 14 classes using the Area Under the Receiver Operating Characteristic Curve (AUROC). 

<img width="3600" height="3000" alt="ROC_AUC_Curves" src="https://github.com/user-attachments/assets/22f4f117-1846-4a72-9c0a-192afd268802" />


### Class-Specific Validation Metrics (F1-Optimized)
Standard **0.50** classification thresholds are mathematically suboptimal for highly imbalanced medical data. This project implements class-specific, data-driven thresholds calculated to explicitly maximize the F1-Score for each distinct pathology. 

Below are the final convergence metrics sorted by AUROC performance, referencing the complete tabular data found in `medx_validation_metrics.xlsx`:

| Pathology Class | AUROC Score | Optimal Threshold | Max F1-Score |
| :--- | :--- | :--- | :--- |
| **Hernia** | 0.9607 | 0.85 | 0.4211 |
| **Emphysema** | 0.9450 | 0.85 | 0.5226 |
| **Cardiomegaly** | 0.8934 | 0.85 | 0.3506 |
| **Edema** | 0.8864 | 0.85 | 0.2328 |
| **Pneumothorax** | 0.8811 | 0.75 | 0.4039 |
| **Effusion** | 0.8789 | 0.70 | 0.5291 |
| **Mass** | 0.8543 | 0.80 | 0.3692 |
| **Pleural Thickening** | 0.8266 | 0.70 | 0.2362 |
| **Nodule** | 0.8214 | 0.75 | 0.3582 |
| **Fibrosis** | 0.8196 | 0.85 | 0.2011 |
| **Atelectasis** | 0.8160 | 0.65 | 0.4153 |
| **Consolidation** | 0.8038 | 0.75 | 0.2325 |
| **Pneumonia** | 0.7373 | 0.75 | 0.0853 |
| **Infiltration** | 0.7119 | 0.55 | 0.4295 |
## Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights which regions of the X-ray activated the model's prediction for a given disease. The target layer is `model.features.norm5` — the final batch norm of DenseNet121's feature extractor.

```
Input X-Ray → DenseNet121 → Sigmoid → Predicted Diseases
                  ↓
             Grad-CAM on norm5
                  ↓
         Heatmap overlaid on X-Ray
```

---

## Project Structure

```
MEDICAL-X/
├── MEDICAL-X.ipynb          # Training notebook (Kaggle, GPU)
├── predict.py               # Inference script with Grad-CAM support
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/Harsh-Prajapati54/MEDICAL-X.git
cd MEDICAL-X
```

### 2. Install dependencies
```bash
# Install PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

### 3. Download the dataset
Download the NIH ChestX-ray14 dataset from [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) and place it as:
```
data/
└── Data_Entry_2017.csv
└── images_001/images/
└── images_002/images/
...
```

---

## Inference

Run predictions on any chest X-ray image:

```bash
# Basic prediction
python predict.py --image path/to/xray.png

# With custom model path and threshold
python predict.py --image path/to/xray.png --model densenet121_model.pth --threshold 0.4

# With Grad-CAM heatmap (saves gradcam_result.jpg)
python predict.py --image path/to/xray.png --gradcam
```

### Example output
```
── Result ──────────────────────────────────────
  Image      : xray.png
  Threshold  : 0.5
  Detected   : Effusion, Infiltration

  All probabilities:
    Effusion              : 82.34% ◄
    Infiltration          : 71.12% ◄
    Consolidation         :  23.45%
    Atelectasis           :  18.90%
    ...
─────────────────────────────────────────────────
```

---

## Training Methodology & Hyperparameters

The final production model (`best121_model.pth`) was trained using an optimized deep learning workflow on advanced accelerator clusters. Given the high structural complexity and class imbalance inherent in multi-label thoracic datasets, the training pipeline incorporates rigid regularization and a dynamic reactive learning rate strategy to maximize generalization and prevent overfitting.

### Optimization & Regularization Strategy
* **Decoupled Weight Decay (`AdamW`):** Instead of standard `Adam`, the training loop utilizes `AdamW` with an elevated weight decay configuration. This mathematically penalizes exploding node weights, forcing the network's convolutional filters to analyze global structural anatomy (such as lung borders and tissue density) rather than over-focusing on high-frequency noise or isolated bright pixels.
* **Reactive Learning Rate Scaling:** The pipeline uses a `ReduceLROnPlateau` scheduler tracking validation loss. If the validation metrics stall for two consecutive epochs, the scheduler dynamically cuts the learning rate by **90% (factor=0.1)**. This enables the model to bypass local minima and perform high-precision local gradient adjustments.
* **Early Stopping Safety Net:** To conserve compute overhead and prevent catastrophic divergence, an automated early stopping tracker monitors validation loss with a **patience window of 5 epochs**. The optimal weights are captured and locked down immediately at the point of lowest validation loss before overfitting triggers.

### Hyperparameter Configuration

| Parameter / Strategy | Configuration Setting | Rationale |
| :--- | :--- | :--- |
| **Compute Hardware** | Nvidia H200 GPU | High-throughput VRAM acceleration for deep residual tracking |
| **Core Architecture** | Fine-tuned DenseNet121 | Pretrained feature extractor backbone with custom linear head |
| **Optimization Algorithm** | `AdamW` (Weight Decay = 0.05) | Decoupled regularization to mitigate multi-label memorization |
| **Learning Rate Scheduler**| `ReduceLROnPlateau` (patience=2, factor=0.1) | Reactive fine-detail optimization during convergence plateaus |
| **Regularization / Safety**| Early Stopping (patience=5) | Terminates training loop automatically; champion locked at Epoch 6 |
| **Loss Function Setup** | Binary Cross-Entropy with Logits | Standardized approach for non-mutually exclusive multi-label tasks |
| **Peak Convergence Acc.** | **~88.0% Validation Accuracy** | High-tier diagnostic metric achieved at the lowest validation loss floor |

---

> 📊 **Training Log Insight (The Breakthrough Epoch):**
> During training, the `ReduceLROnPlateau` scheduler successfully identified a validation plateau at Epoch 5 and dropped the learning rate to a microscopic fine-tuning scale. This immediately triggered a classic **"Bounce Back" in Epoch 6**, where the model broke through its previous loss floor, dropping `test_loss` to its ultimate optimal value of **0.9201** and lifting validation accuracy to **87.94%**. Subsequent epochs exhibited expanding train/validation divergence, prompting the early stopping mechanism to cleanly terminate execution at Epoch 11, preserving the Epoch 6 champion weights intact.




---

## Requirements

```
torch
torchvision
numpy
pandas
matplotlib
seaborn
opencv-python
Pillow
tqdm
```

---

## Acknowledgements

- NIH Clinical Center for the ChestX-ray14 dataset
- [CheXNet paper](https://arxiv.org/abs/1711.05225) — inspiration for DenseNet121 on chest X-rays
- PyTorch and torchvision for model building

---

## Disclaimer

This project is for educational and research purposes only. It is **not intended for clinical use or medical diagnosis**.
