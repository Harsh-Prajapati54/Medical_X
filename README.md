# MEd X: Pneumonia Detection System

**MEd X** is a deep learning initiative designed to assist in the automated detection of pneumonia from chest X-ray images. This project explores the efficacy of state-of-the-art Convolutional Neural Networks (CNNs) to classify medical imaging data with high accuracy.

## 📂 Repository Structure


This repository contains two primary implementations, each utilizing a different model architecture and dataset to evaluate performance across varied data distributions.

### 1. EfficientNet Implementation
* **File:** `Medical_X`
* **Model:** EfficientNet-B0
* **Dataset:** Normal vs. Pneumonia (Binary Classification)
* **Description:** This notebook utilizes the EfficientNet-B0 architecture, known for its efficiency and high performance on image classification tasks. It is trained on a focused dataset containing "Normal" and "Pneumonia" class images to establish a lightweight yet powerful baseline.

### 2. DenseNet Implementation
* **File:** `med-x.ipynb`
* **Model:** DenseNet121
* **Dataset:** NIH Chest X-ray Dataset
* **Description:** This notebook tackles a more complex challenge using the DenseNet121 architecture. It is trained on the NIH Chest X-ray dataset, which is significantly larger and more diverse. This approach aims to test the model's robustness and its ability to generalize on standard medical benchmarks.

## 🛠️ Tech Stack

* **Language:** Python
* **Deep Learning Framework:** PyTorch 
* **Libraries:** NumPy, Pandas, Matplotlib, Scikit-learn, Torchvision
* **Architectures:** EfficientNet-B0, DenseNet121

## 📊 Results & Performance

| Model | Dataset | Accuracy | Key Observations |
| :--- | :--- | :--- | :--- |
| **EfficientNet-B0** | Normal vs. Pneumonia | *[90%]* | *[e.g., Fast training, high precision on binary data]* |
| **DenseNet121** | NIH Dataset | *[95%]* | *[e.g., Robust feature extraction, handles class imbalance well]* |

> **Note:** Detailed confusion matrices and loss curves can be found within the respective notebooks.

## 🚀 Future Scope & Roadmap

As part of the ongoing development of MEd X, the following features and improvements are planned:

1.  **Explainable AI (XAI):** Implementing Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize exactly which regions of the X-ray the model is focusing on to make a diagnosis. This increases trust for medical professionals.
2.  **Web Deployment:** Building a user-friendly frontend (using  Streamlit) and a backend API (FastAPI or Node.js) to allow users to upload X-rays and get real-time predictions.
3.  **Multi-Class Classification:** Expanding the scope beyond Pneumonia to detect other conditions available in the NIH dataset (e.g., Atelectasis, Cardiomegaly).
4.  **Model Optimization:** Experimenting with model quantization to reduce the size of the model for mobile or edge deployment.
5.  **Localization Infected area** this will generate an square boundary across infected area on X-Ray 

## 👨‍💻 Author

**[Harsh Prajapati]**
* Computer Engineering Student
* Connect with me: [[LinkedIn Profile Link](www.linkedin.com/in/harsh-prajapati-393759296)] | [[GitHub Profile Link](https://github.com/Harsh-Prajapati54)]

---
*Disclaimer: MEd X is a research project and is not intended to replace professional medical diagnosis.*
