# 🖐️ Rock-Paper-Scissors Image Classification

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/rfalam/Rock-Paper-Scissors_Simple-Prediction)
[![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)](https://python.org)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?logo=keras)](https://keras.io)

Aplikasi **Computer Vision** berbasis Deep Learning yang mampu mengklasifikasikan gestur tangan ke dalam tiga kategori: **Batu (Rock)**, **Kertas (Paper)**, atau **Gunting (Scissors)**. Proyek ini mencakup pipeline end-to-end mulai dari eksplorasi data, pembangunan model baseline CNN, peningkatan performa melalui **Transfer Learning (EfficientNetB0)**, hingga deployment aplikasi web interaktif.

---

## 🔗 Quick Links

| Resource | Link |
| :--- | :--- |
| **📊 Dataset** | [Kaggle - Rock Paper Scissors](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors) |
| **🚀 Live Demo** | [Hugging Face Spaces](https://huggingface.co/spaces/rfalam/Rock-Paper-Scissors_Simple-Prediction) |

---

## 🎯 Latar Belakang & Tujuan

Pengenalan gestur tangan merupakan fondasi penting dalam pengembangan sistem **Human-Computer Interaction (HCI)**. Meskipun klasifikasi "Batu-Gunting-Kertas" terlihat sederhana, proyek ini menjadi langkah awal untuk membangun sistem pengenalan gerakan yang lebih canggih, seperti kontrol perangkat, interaksi virtual, atau bahkan terapi fisik.

**Tujuan utama** proyek ini adalah membangun model **Convolutional Neural Network (CNN)** yang mampu mengklasifikasikan gambar tangan ke dalam tiga kategori (Batu, Kertas, Gunting) dengan tingkat akurasi tinggi, lalu menyajikannya melalui aplikasi web yang interaktif.

---

## 📊 Exploratory Data Analysis (EDA)

Sebelum membangun model, dilakukan analisis mendalam terhadap dataset yang terdiri dari **2.188 gambar** (Rock: 726, Paper: 712, Scissors: 750):

| No | Analisis | Temuan Kunci |
| :---: | :--- | :--- |
| 1 | **Visualisasi Gambar** | Seluruh gambar menggunakan latar belakang hijau (*green screen*) untuk memudahkan model membedakan gestur tangan dari latar belakang. |
| 2 | **Distribusi Kelas** | Dataset bersifat **balanced** (seimbang), sehingga tidak memerlukan teknik oversampling/undersampling. |
| 3 | **Analisis Dimensi** | Semua gambar memiliki dimensi seragam (300x200 piksel), sehingga tidak diperlukan standarisasi ukuran. |
| 4 | **Analisis Tipe Warna** | Konversi ke channel hijau menyebabkan hilangnya detail bentuk tangan, mengindikasikan model tidak boleh terlalu bergantung pada fitur di channel warna hijau. |
| 5 | **Analisis Posisi Objek** | Posisi tangan cenderung serupa di semua gambar, sehingga **Data Augmentation** sangat diperlukan untuk meningkatkan generalisasi model. |
| 6 | **Analisis Intensitas Piksel** | Distribusi warna grayscale menunjukkan nilai kontras yang baik, memudahkan model dalam mendeteksi bentuk tangan. |

---

## 🧠 Metodologi & Arsitektur Model

### Phase 1: Baseline Model (Custom CNN)
- Arsitektur CNN sederhana dengan layer `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, dan `Dropout`.
- Framework: **TensorFlow/Keras** dengan `MobileNetV2` sebagai eksperimen awal.
- Input gambar di-*resize* ke **150x150 piksel**.

### Phase 2: Improved Model (Transfer Learning — EfficientNetB0)
Untuk mencapai performa maksimal, arsitektur ditingkatkan menggunakan **Transfer Learning**:

| Komponen | Detail |
| :--- | :--- |
| **Base Model** | `EfficientNetB0` (Pre-trained pada ImageNet) |
| **Custom Head** | Global Average Pooling → Dense(256, ReLU) → Dropout(0.4) → Softmax(3) |
| **Input Size** | 224x224 piksel |

#### Strategi Training Dua Fase:
1.  **Feature Extraction** — Seluruh layer base model dibekukan (*frozen*), hanya melatih classification head dengan `Adam(lr=1e-3)`.
2.  **Fine-Tuning** — 30 layer teratas EfficientNetB0 dibuka (*unfrozen*), lalu dilatih ulang dengan learning rate sangat rendah `Adam(lr=1e-5)` untuk menyesuaikan fitur spesifik tanpa merusak bobot pre-trained.

#### Teknik Optimasi:
- **Data Augmentation**: Rotation, shift, shear, zoom, flip, brightness adjustment.
- **Callbacks**: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint` untuk menyimpan model terbaik.

---

## 📁 Struktur Repository

```
📦 Rock-Paper-Scissors-Image-Classification-Application
├── 📓 Computer Vision Project.ipynb  # Notebook utama: EDA, eksperimen, & baseline model
├── 📓 model_inference.ipynb          # Notebook untuk pengujian prediksi model
├── 🐍 train_improved_model.py        # Script training model EfficientNetB0
├── 📂 Deployment/                    # Aplikasi Streamlit untuk deployment
│   ├── 🐍 streamlit_app.py           # Entry point aplikasi Streamlit
│   ├── 🐍 eda.py                     # Halaman EDA interaktif
│   ├── 🐍 prediction.py              # Halaman prediksi gambar
│   ├── 📂 src/                       # Model & aset gambar EDA
│   │   ├── rock_paper_scissors_model.h5           # Model baseline CNN
│   │   └── rock_paper_scissors_improved_model.h5  # Model EfficientNetB0
│   └── 📄 Requirements.txt           # Dependensi library
├── 📄 url.txt                        # Link referensi (dataset, model, deployment)
└── 📄 README.md                      # Dokumentasi proyek
```

---

## 🛠️ Tech Stack

| Kategori | Teknologi |
| :--- | :--- |
| **Language** | Python 3.9 |
| **Deep Learning** | TensorFlow, Keras |
| **Model Architecture** | EfficientNetB0 (Transfer Learning) |
| **Data Processing** | NumPy, Pillow, scikit-image |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | Streamlit |
| **Platform** | Hugging Face Spaces, Google Colab |

---

## 🏃 Cara Menjalankan

### Training Model (Lokal)
```bash
python train_improved_model.py
```
> **Catatan:** Pastikan path dataset sudah diatur dengan benar di dalam script.

### Menjalankan Aplikasi Web
```bash
cd Deployment
streamlit run streamlit_app.py
```

---

## 📈 Hasil Performa Model

### Baseline CNN (Custom Architecture)
| Metrik | Nilai |
| :--- | :--- |
| **Training Accuracy** | ~90% (epoch terakhir: 0.9017) |
| **Validation Accuracy** | ~95% (epoch terakhir: 0.9471) |
| **Best Validation Accuracy** | ~96% (epoch ke-18: 0.9639) |
| **Training Loss** | 0.2747 |
| **Validation Loss** | 0.1312 |

> Model baseline menunjukkan peningkatan akurasi yang konsisten selama 20 epoch tanpa tanda-tanda **overfitting** yang signifikan, berkat penggunaan **Dropout(0.6)** dan **Data Augmentation**.

### Improved Model (EfficientNetB0 — Transfer Learning)
- **Transfer Learning** dengan EfficientNetB0 secara signifikan meningkatkan performa dan generalisasi model dibandingkan baseline CNN.
- **Strategi fine-tuning dua fase** terbukti efektif dalam mengadaptasi fitur pre-trained ke domain spesifik (gestur tangan) tanpa *catastrophic forgetting*.
- Penggunaan **callbacks** (`EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`) memastikan model terbaik selalu tersimpan selama proses training.

---

## 💡 Insight Utama

- **Data Augmentation** yang komprehensif (rotation, shift, shear, zoom, flip, brightness) membantu model mengatasi keterbatasan variasi posisi tangan dalam dataset.
- **Green screen background** pada dataset membantu model fokus pada fitur gestur tangan, bukan latar belakang.
- Dataset yang **balanced** (~700-750 gambar per kelas) mengurangi bias dan meningkatkan kemampuan klasifikasi model.
- Aplikasi berhasil di-deploy sebagai **web app interaktif** di Hugging Face Spaces, memungkinkan pengguna mengunggah gambar dan mendapatkan prediksi secara real-time.

---

## 👤 Author
- **Riko Fadilah Alam** — [Hugging Face Profile](https://huggingface.co/rfalam)
