"""
Analisis regresi linier untuk data AirQuality.

Catatan: Perapihan kode dilakukan tanpa mengubah perilaku program.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy import stats

# --------------------------------------------
# 1. PEMUATAN DATA DARI EXCEL
# --------------------------------------------

excel_path = "AirQuality.xlsx"

df = pd.read_excel(excel_path)

print("Data berhasil dimuat.")
print("Ukuran data:", df.shape)
print("\nSepuluh baris pertama:")
print(df.head(10))

# Persentase missing values untuk seluruh kolom
missing_percent = df.isna().mean() * 100
print("\nPersentase missing values per kolom (%):")
print(missing_percent)


# --------------------------------------------
# 2. PEMILIHAN VARIABEL TARGET & PREDIKTOR
# --------------------------------------------

# Variabel target (Y)
target_col = "CO(GT)"  # konsentrasi CO

# Variabel prediktor (X)
feature_cols = [
    "PT08.S1(CO)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "T",
    "RH",
    "AH",
]


# (Opsional) cek missing khusus kolom model
missing_percent_model = df[feature_cols + [target_col]].isna().mean() * 100
print("\nPersentase missing values untuk kolom model (%):")
print(missing_percent_model)

# Pastikan semua kolom ada di dataframe
missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
if missing_cols:
    raise ValueError(f"Kolom berikut tidak ditemukan di data: {missing_cols}")

# Ambil subset data untuk model
df_model = df[feature_cols + [target_col]].copy()


# --------------------------------------------
# 3. PEMBERSIHAN DATA
# --------------------------------------------

# Jika masih ada nilai -200 atau kode lain sebagai missing value,
# bisa dibersihkan di sini. Contoh:
# for col in feature_cols + [target_col]:
#     df_model.loc[df_model[col] == -200, col] = np.nan

# Hapus baris yang mengandung NaN pada kolom yang digunakan
df_model = df_model.dropna(subset=feature_cols + [target_col])

print("\nSetelah pembersihan, ukuran data untuk model:", df_model.shape)


# --------------------------------------------
# 2.5. VISUALISASI: HISTOGRAM VARIABEL TARGET
# --------------------------------------------
plt.figure(figsize=(8,5))
plt.hist(df_model[target_col], bins=30, color='skyblue', edgecolor='black')
plt.title(f'Histogram {target_col}')
plt.xlabel(target_col)
plt.ylabel('Frekuensi')
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
hist_file = "histogram_target.png"
plt.savefig(hist_file)
print(f"\nHistogram variabel target disimpan sebagai: {hist_file}")
plt.show()


# --------------------------------------------
# 2.6. PERHITUNGAN SKEWNESS VARIABEL TARGET
# --------------------------------------------

skewness = stats.skew(df_model[target_col])
kurtosis = stats.kurtosis(df_model[target_col])

print(f"\n=== STATISTIK DISTRIBUSI VARIABEL TARGET ===")
print(f"Skewness {target_col}: {skewness:.4f}")
print(f"Kurtosis {target_col}: {kurtosis:.4f}")

# Interpretasi skewness
if abs(skewness) < 0.5:
    skew_interpretation = "Distribusi hampir simetris"
elif skewness > 0.5:
    skew_interpretation = "Distribusi condong ke kanan (positif/right-skewed)"
else:
    skew_interpretation = "Distribusi condong ke kiri (negatif/left-skewed)"

print(f"Interpretasi: {skew_interpretation}")


# --------------------------------------------
# 2.7. VISUALISASI: BOX PLOT VARIABEL TARGET
# --------------------------------------------
plt.figure(figsize=(8,5))
plt.boxplot(df_model[target_col], vert=True)
plt.title(f'Box Plot {target_col}')
plt.ylabel(target_col)
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
boxplot_file = "boxplot_target.png"
plt.savefig(boxplot_file)
print(f"Box plot variabel target disimpan sebagai: {boxplot_file}")
plt.show()


# 2.1.2.1 – Deteksi outlier CO(GT) dengan metode IQR
q1 = df_model["CO(GT)"].quantile(0.25)
q3 = df_model["CO(GT)"].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print("Q1       :", q1)
print("Q3       :", q3)
print("IQR      :", iqr)
print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)

# Menandai observasi yang termasuk outlier
outliers = df_model[(df_model["CO(GT)"] < lower_bound) | (df_model["CO(GT)"] > upper_bound)]

print("\nJumlah outlier pada CO(GT):", outliers.shape[0])


# --------------------------------------------
# 2.9. MENGHITUNG MATRIKS KORELASI
# --------------------------------------------
cols_model = feature_cols + [target_col]

corr_matrix = df_model[cols_model].corr()

print("\nMatriks korelasi (Pearson):")
print(corr_matrix)


# --------------------------------------------
# 2.2.2 – Visualisasi matriks korelasi menggunakan heatmap
# --------------------------------------------
heatmap_file = "heatmap_correlation.png"
plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, interpolation="nearest", cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Heatmap Matriks Korelasi")
plt.colorbar(label="Koefisien Korelasi")

tick_marks = np.arange(len(cols_model))
plt.xticks(tick_marks, cols_model, rotation=45, ha="right")
plt.yticks(tick_marks, cols_model)

plt.tight_layout()
plt.savefig(heatmap_file)
print(f"Heatmap matriks korelasi disimpan sebagai: {heatmap_file}")
# plt.show()  # backend non-interactive; file disimpan

# Mengambil korelasi antara setiap prediktor dan target CO(GT)
corr_with_target = corr_matrix[target_col].drop(target_col)

print("\nKorelasi masing-masing prediktor dengan CO(GT):")
print(corr_with_target)

# Mengurutkan berdasarkan nilai absolut korelasi
corr_with_target_abs_sorted = corr_with_target.abs().sort_values(ascending=False)

print("\nKorelasi (berdasarkan |nilai| terbesar ke terkecil):")
print(corr_with_target_abs_sorted)

# Prediktor dengan korelasi terkuat (secara absolut)
strongest_predictor = corr_with_target_abs_sorted.index[0]
strongest_value = corr_with_target[strongest_predictor]

print(f"\nPrediktor dengan korelasi terkuat dengan CO(GT): {strongest_predictor}")
print(f"Nilai korelasinya: {strongest_value}")

# Pisahkan fitur (X) dan target (y)
X = df_model[feature_cols]
y = df_model[target_col]


# --------------------------------------------
# 3.5. EKSPOR DATASET PREDIKTOR KE CSV
# --------------------------------------------
prediktor_csv_file = "dataset_prediktor.csv"
X.to_csv(prediktor_csv_file, index=False)
print(f"\nDataset prediktor berhasil disimpan sebagai: {prediktor_csv_file}")


# --------------------------------------------
# 4. PEMBAGIAN DATA TRAIN-TEST
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,  # 20% data untuk testing
    random_state=42,  # supaya hasil bisa direplikasi
)

print("\nUkuran data train:", X_train.shape)
print("Ukuran data test :", X_test.shape)


# --------------------------------------------
# 5. PEMBANGUNAN MODEL REGRESI LINIER
# --------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)


# --------------------------------------------
# 6. PREDIKSI & EVALUASI MODEL
# --------------------------------------------
y_pred = model.predict(X_test)


# --------------------------------------------
# 6.1. VISUALISASI: SCATTER PLOT AKTUAL vs PREDIKSI
# --------------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("CO(GT) Aktual")
plt.ylabel("CO(GT) Prediksi")
plt.title("Scatter Plot Nilai Aktual vs Prediksi CO(GT)")

# Tambahkan garis diagonal y = x sebagai acuan
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linewidth=1)

plt.grid(alpha=0.3)
plt.tight_layout()
scatter_file = "scatter_actual_vs_pred.png"
plt.savefig(scatter_file)
print(f"Scatter plot disimpan sebagai: {scatter_file}")
plt.show()

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print("\n=== HASIL EVALUASI MODEL ===")
print("R²   :", r2)
print("MAE  :", mae)
print("MSE  :", mse)
print("RMSE :", rmse)


# --------------------------------------------
# 7. KOEFISIEN REGRESI
# --------------------------------------------

coef_df = pd.DataFrame({
    "Variabel": feature_cols,
    "Koefisien": model.coef_
})

print("\n=== KOEFISIEN REGRESI ===")
print(coef_df)

print("\nIntercept (β0):", model.intercept_)


# --------------------------------------------
# 8. (OPSIONAL) SIMPAN HASIL KE EXCEL
# --------------------------------------------

# Gabungkan y_test dan y_pred ke dalam satu DataFrame
hasil_prediksi = pd.DataFrame({
    "y_asli": y_test.values,
    "y_prediksi": y_pred
})

# Buat DataFrame metrik
metrik_df = pd.DataFrame({
    "Metrik": ["R2", "MAE", "RMSE"],
    "Nilai": [r2, mae, rmse]
})

# Simpan ke file Excel baru
with pd.ExcelWriter("Hasil_Regresi_AirQuality.xlsx") as writer:
    df_model.to_excel(writer, sheet_name="Data_Model", index=False)
    coef_df.to_excel(writer, sheet_name="Koefisien_Regresi", index=False)
    hasil_prediksi.to_excel(writer, sheet_name="Prediksi", index=False)
    metrik_df.to_excel(writer, sheet_name="Metrik_Model", index=False)

print("\nFile 'Hasil_Regresi_AirQuality.xlsx' berhasil dibuat.")
