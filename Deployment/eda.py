import streamlit as st


def run():
    # Membuat title
    st.title("Aplikasi klasifikasi Gambar Batu-Gunting-Kertas")
    st.write("Aplikasi ini menggunakan model Deep Learning untuk memprediksi gambar tangan yang membentuk isyarat batu, gunting, atau kertas.")

    # Membuat link untuk melihat url dataset
    st.write("Dataset yang digunakan dalam aplikasi ini diambil dari Kaggle. Dataset ini berisi gambar tangan yang membentuk isyarat batu, gunting, atau kertas.")
    st.markdown("[View Dataset](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors/data)")

    # Menampilkan gambar hasil EDA
    st.subheader("Exploratory Data Analysis (EDA)")


    st.write("1. EDA 1: Visualisasi Gambar")
    st.image("./src/eda1.png", caption="Hasil EDA 1 Visualisasi gambar dari kelas Rock", use_container_width=True)
    st.image("./src/eda1.2.png", caption="Hasil EDA 1 Visualisasi gambar dari kelas Paper", use_container_width=True)
    st.image("./src/eda1.3.png", caption="Hasil EDA 1 Visualisasi gambar dari kelas Scissors", use_container_width=True)
    st.write("Pada dataset tiap gambar memiliki latar belakang berwarna hijau. Hal ini dilakukan karena pada tubuh manusia warna hijau sangat sedikit, sehingga dengan menggunakan latar belakang warna hijau dapat meningkatkan kinerja model dalam membedakan gerakan tangan dengan baik.")
    st.divider()

    st.write("2. EDA 2: Distribusi Tiap Kelas")
    st.image("./src/eda2.png", caption="Hasil EDA 2 Distribusi Tiap Kelas", use_container_width=True)  
    st.write("Gambar di atas menunjukkan distribusi jumlah gambar untuk setiap kelas (Batu, Gunting, Kertas) dalam dataset.")
    st.write("Jumlah gambar gunting: 750")
    st.write("Jumlah gambar kertas: 712.")
    st.write("Jumlah gambar batu: 726.")
    st.write("Dataset menggunakan jumlah gambar yang hampir seimbang atau tidak adanya data yang mendominasi. Karena data balance maka tidak perlu dilakukan oversampling atau undersampling pada tiap kelasnya.")
    st.divider()  

    st.write("3. EDA 3: Analisis Dimensi Gambar")
    st.write("Pada EDA 3 akan dilakukan analisis dimensi dari gambar yang digunakan pada dataset.")
    st.write("Berdasarkan analisis ini gambar pada tiap kelas memiliki dimensi yang sama yaitu 300x200 piksel. Hal ini penting untuk memastikan bahwa model dapat memproses gambar dengan ukuran yang konsisten." \
    "Pada dataset memiliki ukuran dimensi gambar yang sama. Berdasarkan hal ini tidak perlu dilakukan adanya standarisasi karena semua data berukuran sama.")
    st.divider()


    st.write("4. EDA 4: Analisis Jenis Warna")
    st.image("./src/eda4.png", caption="Hasil EDA 4 Analisis Jenis Warna", use_container_width=True)
    st.write("Berdasarkan analisis ini jika gambar diubah menjadi hijau maka bentuk tangan tidak terlalu terlihat. Sehingga model sebaiknya tidak terlalu memprioritaskan bobot pada fitur di kanal warna hijau.")
    st.divider()

    st.write("5. EDA 5: Analisis Letak Objek")
    st.image("./src/eda5.1.png", caption="Hasil EDA 5 Analisis Letak Objek Kelas Rock", use_container_width=True)
    st.image("./src/eda5.2.png", caption="Hasil EDA 5 Analisis Letak Objek Kelas Paper", use_container_width=True)
    st.image("./src/eda5.3.png", caption="Hasil EDA 5 Analisis Letak Objek Kelas Scissors", use_container_width=True)
    st.write("Berdasarkan bentuk gambar, posisi tangan selalu sama. Sehingga pada model harus dibuat data baru untuk mengubah posisi dari tiap tangan agar model memiliki performa yang lebih baik dalam membaca posisi tangan.")
    st.divider()

    st.write("6. EDA 6: Analisis Intensitas Piksel")
    st.image("./src/eda6.1.png", caption="Hasil EDA 6 Analisis Intensitas Piksel dari Kelas Rock", use_container_width=True)
    st.image("./src/eda6.2.png", caption="Hasil EDA 6 Analisis Intensitas Piksel dari Kelas Paper", use_container_width=True)
    st.image("./src/eda6.3.png", caption="Hasil EDA 6 Analisis Intensitas Piksel dari Kelas Scissors", use_container_width=True)
    st.write("Berdasarkan frekusensi warna yang telah diubah menjadi hitam putih dataset memiliki nilai kontras yang cukup baik dan akan memudahkan model dalam mendeteksi bentuk tangan.")
 
if __name__ == "__main__":
    run()  