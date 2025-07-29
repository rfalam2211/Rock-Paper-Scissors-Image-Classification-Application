import streamlit as st
import eda
import prediction

st.set_page_config(
    page_title='Aplikasi Prediksi Klasifikasi Gambar Batu-Gunting-Kertas',
    layout='wide',
    initial_sidebar_state='expanded'
)

page = st.sidebar.selectbox('Pilih Halaman', ('EDA', 'Prediction'))

if page == 'EDA':
    eda.run()
else:
    prediction.run()