import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# --- Mengoptimalkan Pemuatan Model dengan Caching ---
# @st.cache_resource akan memastikan model hanya dimuat sekali saat aplikasi pertama kali dijalankan.
# Ini akan membuat aplikasi jauh lebih cepat saat pengguna berinteraksi.
@st.cache_resource
def load_keras_model():
    """
    Fungsi untuk memuat model Keras dari file rock_paper_scissors_model.h5
    """
    # Load Model Keras
    model_path = './src/rock_paper_scissors_model.h5'
    if os.path.exists(model_path):
        model = load_model(model_path)
        return model
    else:
        return None

# --- Fungsi Utama Aplikasi ---
def run():
    # --- Judul dan Header ---
    st.header("Prediksi Gambar Batu-Gunting-Kertas")
    st.write("""
    Unggah gambar tangan Anda yang membentuk isyarat batu, gunting, atau kertas.
    Model akan mencoba menebak gambar tersebut.
    """)

    # --- Memuat Model ---
    model = load_keras_model()

    # Periksa apakah model berhasil dimuat
    if model is None:
        st.error("Model tidak dapat dimuat. Pastikan file 'rock_paper_scissors_model.h5' ada di direktori yang sama dengan aplikasi Anda.")
        return

    # --- Widget untuk Mengunggah Gambar ---
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg"])

    # --- Logika Prediksi ---
    if uploaded_file is not None:
        # Tampilkan gambar yang diunggah oleh pengguna
        image_to_show = Image.open(uploaded_file)
        st.image(image_to_show, caption="Gambar yang Anda Unggah", use_container_width=True)
        st.write("") 

        # Tombol untuk memicu prediksi
        if st.button("Lakukan Prediksi"):
            with st.spinner("Model sedang berpikir..."):
                # 1. Preprocessing Gambar
                # Gambar perlu diubah menjadi format yang sama seperti saat training.
                image_for_pred = image_to_show.resize((150, 150)) # Sesuaikan ukuran
                img_array = np.array(image_for_pred)              # Ubah ke array numpy
                img_array = np.expand_dims(img_array, axis=0)     # Tambah dimensi batch
                img_array = img_array / 255.0                     # Rescale nilai piksel

                # 2. Lakukan Prediksi
                prediction = model.predict(img_array)
                score = tf.nn.softmax(prediction[0])

                # 3. Tampilkan Hasil
                class_names = ['Kertas', 'Batu', 'Gunting']
                predicted_class = class_names[np.argmax(score)]
                confidence = 100 * np.max(score)

                st.success(f"Hasil Prediksi: **{predicted_class}**")
                st.info(f"Nilai Akurasi: **{confidence:.2f}%**")

if __name__ == "__main__":
    run()