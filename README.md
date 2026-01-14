# 🏭 Air Quality Prediction using Linear Regression

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange?style=flat&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Library-Pandas-150458?style=flat&logo=pandas)

## 📋 Tentang Proyek 

Proyek ini bertujuan untuk membangun model **Regresi Linier Berganda (Multiple Linear Regression)** guna memprediksi konsentrasi Karbon Monoksida (CO) di udara berdasarkan pembacaan sensor kimia lainnya. Dataset yang digunakan adalah dataset **Air Quality** (diduga berasal dari UCI Machine Learning Repository) yang mencakup respons sensor gas multisensor rata-rata per jam.

## 📊 Metodologi
Langkah-langkah yang dilakukan dalam proyek ini meliputi:
1.  **Akuisisi Data & Pemahaman**: Memuat data `AirQuality.xlsx` dan mengidentifikasi variabel.
2.  **Exploratory Data Analysis (EDA)**:
    * Menganalisis distribusi target (`CO(GT)`).
    * Mendeteksi outliers.
    * Analisis korelasi antar variabel (menggunakan Heatmap).
3.  **Preprocessing**:
    * Penanganan *missing values*.
    * Pemilihan fitur (Feature Selection) berdasarkan korelasi.
4.  **Pemodelan**: Melatih model Regresi Linier menggunakan `scikit-learn`.
5.  **Evaluasi**: Mengukur kinerja model menggunakan metrik statistik.

## 📈 Hasil Evaluasi Model
Berdasarkan pengujian yang dilakukan, model berhasil mencapai performa yang sangat baik dalam memprediksi konsentrasi CO:

| Metrik | Nilai |
| :--- | :--- |
| **R² Score** | **0.909** (90.9%) |
| **MAE** (Mean Absolute Error) | 0.286 |
| **RMSE** (Root Mean Squared Error) | 0.420 |

> *Model mampu menjelaskan sekitar 91% variasi dalam data target, menunjukkan akurasi yang tinggi.*

## 📂 Struktur File
* `regresi_airquality.py`: Script utama Python untuk pemrosesan data, pelatihan model, dan evaluasi.
* `Laporan regresi airquality.docx`: Laporan lengkap yang menjelaskan latar belakang, teori, dan analisis hasil.
* `AirQuality.xlsx`: Dataset mentah yang digunakan.
* **Output Visualisasi**:
    * `heatmap_correlation.png`: Matriks korelasi antar variabel sensor.
    * `scatter_actual_vs_pred.png`: Plot perbandingan nilai asli vs prediksi.
    * `boxplot_target.png` & `histogram_target.png`: Distribusi data target.
* **Output Data**:
    * `Hasil_Regresi_AirQuality.xlsx`: File Excel berisi hasil prediksi, koefisien regresi, dan metrik evaluasi.

## 🛠️ Tech Stack
* **Bahasa**: Python
* **Data Manipulation**: Pandas, NumPy
* **Visualization**: Matplotlib
* **Machine Learning**: Scikit-Learn

## 👤 Penulis
**Farid Al Farizi**
