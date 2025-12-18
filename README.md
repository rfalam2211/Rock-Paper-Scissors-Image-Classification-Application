# 🖐️ Rock-Paper-Scissors Image Classification

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/rfalam/Graded-Challenge-7_Riko_RMT-043)

An advanced computer vision application that classifies hand gestures into **Rock**, **Paper**, or **Scissors** using Deep Learning.

---

## 🔗 Quick Links

- **Dataset:** [Kaggle Rock-Paper-Scissors](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)
- **Live Demo:** [Hugging Face Spaces](https://huggingface.co/spaces/rfalam/Rock-Paper-Scissors_Simple-Prediction)

---

## 📁 Repository Structure

| File/Folder | Description |
| :--- | :--- |
| `train_improved_model.py` | Python script for the high-performance EfficientNetB0 training pipeline. |
| `rock_paper_scissors_improved_model.h5` | The final trained model file (HDF5 format). |
| `Computer Vision Project.ipynb` | Core Jupyter Notebook containing EDA, initial experimentation, and baseline modeling. |
| `model_inference.ipynb` | Dedicated notebook for testing model predictions on new images. |
| `Deployment/` | Contains the Streamlit application code and configuration for deployment. |
| `README.md` | Project documentation. |

---

## 🌿 Project Background & Objective

The goal of this project is to build a high-accuracy **Convolutional Neural Network (CNN)** capable of distinguishing between three categorical hand gestures. 

### Key Milestones:
1.  **Exploratory Data Analysis (EDA):** Understanding image distributions and quality.
2.  **Baseline Modeling:** Creating a custom CNN architecture.
3.  **Model Improvement:** Implementing **Transfer Learning** with **EfficientNetB0**.
4.  **Fine-Tuning:** Applying a two-phase training strategy to reach state-of-the-art results.
5.  **Deployment:** Serving the model via a user-friendly Streamlit web interface.

---

## 🚀 The Improved Model: EfficientNetB0

To achieve maximum performance, we upgraded the model architecture and training strategy:

### 🧠 Architecture
- **Base Model:** `EfficientNetB0` (Pre-trained on ImageNet).
- **Custom Head:** Global Average Pooling -> Dense (256, ReLU) -> Dropout (0.4) -> Softmax Output.

### 📈 Two-Phase Training Strategy
1.  **Phase 1: Feature Extraction**
    - Base layers frozen.
    - Trained only the classification head with `Adam(lr=1e-3)`.
2.  **Phase 2: Fine-Tuning**
    - Unfrozen the top 30 layers of EfficientNetB0.
    - Re-trained with a very low learning rate `Adam(lr=1e-5)` to preserve pre-trained weights while adapting to specialized features.

---

## 🛠️ Tech Stack

- **Frameworks:** TensorFlow, Keras
- **Language:** Python
- **Visuals:** Matplotlib, Seaborn
- **Deployment:** Streamlit
- **Platform:** Hugging Face Spaces

---

## 🏃 Ready to Run?

To train the improved model locally:
```bash
python train_improved_model.py
```
*Note: Ensure you have the dataset directory path correctly set in the script.*

---

## 👤 Author
- **Riko** - [Hugging Face Profile](https://huggingface.co/rfalam)
