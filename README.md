# Med-X — Chest X-Ray Disease Detection

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-DenseNet121-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://medical-x.streamlit.app)

Multi-label classification of **14 thoracic diseases** from frontal chest X-rays. A DenseNet121 pretrained on ImageNet is fine-tuned on NIH ChestX-ray14 with multi-GPU training (PyTorch DDP), per-class decision thresholds, and **Grad-CAM** heatmaps that show where the model looked.

> **Not a medical device.** This is a research and learning project. It must not be used for diagnosis or any clinical decision. See [Limitations](#limitations).


## Results at a glance

Evaluated on a held-out validation split of NIH ChestX-ray14.

| Metric (macro average over 14 classes) | Value |
| -------------------------------------- | ----- |
| AUROC                                  | **0.851** |
| AUPRC                                  | 0.275 |
| F1 (at per-class threshold)            | 0.339 |

AUROC is the headline number because it is threshold-free and comparable with published work (CheXNet reports roughly 0.84 average AUROC on this dataset's test set). AUPRC and F1 are shown because most classes are rare, and AUROC alone hides how hard they are in practice.

<p align="center">
  <img src="docs\images\best_val_roc_curve.png" width="760" alt="Validation ROC curves for all 14 classes">
</p>

### Per-class validation metrics

Sorted by AUROC. Full-precision values are in [`results/validation_metrics.csv`](results/validation_metrics.csv).

| Class              | AUROC | AUPRC | F1    | Threshold |
| ------------------ | ----- | ----- | ----- | --------- |
| Hernia             | 0.977 | 0.584 | 0.533 | 0.53      |
| Edema              | 0.917 | 0.179 | 0.267 | 0.55      |
| Emphysema          | 0.909 | 0.301 | 0.395 | 0.47      |
| Effusion           | 0.896 | 0.546 | 0.556 | 0.59      |
| Cardiomegaly       | 0.885 | 0.209 | 0.289 | 0.53      |
| Pneumothorax       | 0.866 | 0.223 | 0.336 | 0.47      |
| Mass               | 0.854 | 0.301 | 0.385 | 0.51      |
| Atelectasis        | 0.846 | 0.414 | 0.442 | 0.50      |
| Nodule             | 0.824 | 0.278 | 0.359 | 0.50      |
| Fibrosis           | 0.822 | 0.132 | 0.245 | 0.46      |
| Consolidation      | 0.806 | 0.137 | 0.212 | 0.51      |
| Pleural Thickening | 0.800 | 0.139 | 0.223 | 0.47      |
| Pneumonia          | 0.781 | 0.054 | 0.105 | 0.36      |
| Infiltration       | 0.725 | 0.356 | 0.404 | 0.54      |

How to read this honestly:

- **Pneumonia and Infiltration are the weak spots.** Infiltration has the lowest AUROC. Pneumonia has a reasonable AUROC (0.781) but an AUPRC of 0.054 and F1 of 0.105, because it is very rare in the dataset (roughly 1% of images). Ranking cases is much easier than picking a useful cut-off for them.
- **Hernia's 0.977 is the least reliable number in the table.** Hernia appears in well under 1% of images, so the validation set contains only a handful of positives. This is why its ROC curve is step-shaped.
- **F1 values are optimistic.** Thresholds were chosen per class to maximise F1 on the same validation split the F1 is reported on. An evaluation on the official NIH test list is on the [roadmap](#roadmap).

## Diseases detected

Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia.

Each image can have any number of these (multi-label), so every output is an independent sigmoid, not a softmax.

## Dataset

**NIH ChestX-ray14** (NIH Clinical Center): 112,120 frontal-view chest X-rays from 30,805 patients, with labels mined from radiology reports using NLP (weakly supervised). Images are 1024×1024 and are resized for training.

- Source: [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC) or [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- The dataset is **not** included in this repository.

## Model

```
DenseNet121 (ImageNet-pretrained)
├── features            DenseBlocks + transition layers
│   └── norm5           ← Grad-CAM target layer
└── classifier
    └── Linear(1024 → 14)   ← new head, one logit per disease
```

- **Loss:** binary cross-entropy with logits (independent per-class targets)
- **Output:** sigmoid probability per disease
- **Decision rule:** a disease is reported when its probability reaches that class's own threshold (see the table above), not a global 0.5. Rare classes like Pneumonia need a lower cut-off than common ones.

## Training

Training lives in [`notebooks/minorproject_ddp.ipynb`](notebooks/minorproject_ddp.ipynb), written to run on Kaggle's dual-GPU environment.

- **Distributed training:** PyTorch `DistributedDataParallel`, one process per GPU
- **Experiment tracking:** Weights & Biases (metrics, curves, best checkpoint)
- **Model selection:** the best checkpoint is saved as `best_model.pth`
- **Threshold tuning:** per-class thresholds are searched after training and saved with the evaluation results

To reproduce on Kaggle:

1. Create a new notebook and upload `minorproject_ddp.ipynb`.
2. In the notebook settings, enable a **GPU** accelerator with two GPUs.
3. Add the [NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data) dataset as an input.
4. Add your Weights & Biases API key as a Kaggle secret (or turn W&B logging off in the config cell).
5. Run all cells.

## Quickstart (inference)

```bash
git clone https://github.com/Harsh-Prajapati54/Medical_X.git
cd Medical_X
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Weights are included as `best_model.pth` in the repo root.

**Command line**

```bash
# Predict on one image
python predict.py --image path/to/xray.png

# Also save a Grad-CAM heatmap (writes gradcam_result.jpg)
python predict.py --image path/to/xray.png --gradcam
```

**Web app**

```bash
streamlit run app.py
```

Or try the hosted demo: **[medical-x.streamlit.app](https://medical-x.streamlit.app)**

## Explainability (Grad-CAM)

Grad-CAM weights the feature maps of `features.norm5` by the gradient of a class score, producing a heatmap of the image regions that pushed the model toward that prediction.

```
X-ray → DenseNet121 → sigmoid → predicted diseases
              ↓
     Grad-CAM on norm5
              ↓
   heatmap overlaid on the X-ray
```

A heatmap shows where the model looked, not that the model is right. Always check that attention falls on anatomically sensible regions. Models trained on this dataset can latch onto shortcuts such as text markers or devices in the image.

## Project structure

```
Medical_X/
├── app.py                      # Streamlit web app
├── predict.py                  # Inference + Grad-CAM (CLI and importable)
├── best_model.pth              # Trained weights
├── requirements.txt
├── notebooks/
│   └── minorproject_ddp.ipynb  # Multi-GPU (DDP) training on Kaggle
├── results/
│   └── validation_metrics.csv  # Per-class AUROC / AUPRC / F1 / threshold
├── docs/
│   └── images/                 # ROC curves, Grad-CAM example
├── LICENSE
└── README.md
```

## Limitations

- **Weak labels.** Labels were extracted from reports by NLP, not read from the images by radiologists. Label noise is a known issue in this dataset and puts a ceiling on achievable accuracy, especially for Infiltration and Pneumonia.
- **Single dataset, single hospital system.** There is no external validation. Performance on X-rays from other scanners, hospitals or populations is unknown.
- **Validation-set results.** Metrics come from a validation split, and thresholds were tuned on it. Expect real held-out performance to be somewhat lower.
- **Rare classes.** AUPRC and F1 are low for rare findings. At these operating points the model would produce many false alarms for Pneumonia.
- **Not clinically validated.** Never use outputs for diagnosis or treatment decisions.

## Roadmap

- [ ] Evaluate on the official NIH test list and confirm patient-level separation between splits
- [ ] Better class-imbalance handling (e.g. positive-class weighting or focal loss)
- [ ] Probability calibration (temperature scaling)
- [ ] External validation on another chest X-ray dataset
- [ ] Unit tests and CI

## Contributing

Issues and pull requests are welcome. For larger changes, please open an issue first to discuss what you'd like to change.

## References

- Wang et al., *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks*, CVPR 2017 (the NIH dataset)
- Huang et al., *Densely Connected Convolutional Networks*, CVPR 2017 (DenseNet)
- Rajpurkar et al., *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning*, 2017
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017

## License

Released under the [Apache License 2.0](LICENSE). The NIH ChestX-ray14 dataset has its own terms of use; please review them before using the data.

## Author

**Harsh Prajapati** — [GitHub](https://github.com/Harsh-Prajapati54)