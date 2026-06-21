# MEDICAL-X — Chest X-Ray Disease Detection

Multi-label classification of **14 thoracic diseases** from chest X-rays using a fine-tuned **DenseNet121** on the NIH ChestX-ray14 dataset. Includes **Grad-CAM** visualization to highlight the regions the model focuses on for each predicted disease.

---

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

## Training

Training was done on Kaggle (GPU T4 x2). Key hyperparameters:

| Parameter | Value |
|-----------|-------|
| Model | DenseNet121 (pretrained) |
| Input size | 224 × 224 |
| Batch size | 32 |
| Optimizer | Adam |
| Loss | Binary Cross-Entropy |
| Activation | Sigmoid |
| Classes | 14 |

> Prediction result<img width="1200" height="600" alt="media_images_prediction_chart_0_5ffada69efea8e00d1af" src="https://github.com/user-attachments/assets/ddbb53a7-a286-4520-9aba-ba22bc0673af" />


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
