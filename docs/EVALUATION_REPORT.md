# Laporan Dataset, Pelatihan, dan Evaluasi Model AI — SORTARA

## 1. Ringkasan Dataset & Pembagian Data (Bagian 9.2 PRD)
- **Sumber Dataset:** Kurasi dari *Garbage Classification Dataset* (Mostafa Mohamed / Kaggle) yang dipetakan ke 6 kategori SORTARA.
- **Total Gambar Valid (Bebas Duplikat & Corrupt):** `1500` gambar (`{'Kaca': 250, 'Kertas': 250, 'Logam': 250, 'Organik': 250, 'Plastik': 250, 'Residu': 250}`).
- **Pembagian Stratified (70% / 15% / 15%):**
  - **Training Set (70%):** `1050` gambar
  - **Validation Set (15%):** `225` gambar
  - **Testing Set (15%):** `225` gambar (tidak pernah digunakan saat training maupun pemilihan model)

## 2. Arsitektur & Konfigurasi Training (Bagian 9.1 & 9.3 PRD)
- **Base Model:** `MobileNetV2` (`weights='imagenet'`, `include_top=False`, input `224×224×3` dengan normalisasi `[-1.0, 1.0]`).
- **Classification Head:** `GlobalAveragePooling2D` -> `Dropout(0.25)` -> `Dense(6, activation='softmax')`.
- **Tahap Pelatihan:** 6 epoch *feature extraction* (`lr=1e-3`) + 4 epoch *fine-tuning* 25 layer teratas (`lr=1e-4`).

## 3. Hasil Evaluasi pada Test Set Terpisah (Bagian 9.4 PRD)
- **Overall Test Accuracy:** **92.44%** (`0.9244`)

### Metrik Per Kelas (Precision, Recall, F1-Score)
| Kategori | Precision | Recall | F1-Score | Jumlah Sampel Uji (Support) |
|---|---:|---:|---:|---:|
| **Kaca** | 100.00% | 72.97% | 84.38% | 37 |
| **Kertas** | 97.30% | 97.30% | 97.30% | 37 |
| **Logam** | 83.72% | 94.74% | 88.89% | 38 |
| **Organik** | 95.00% | 100.00% | 97.44% | 38 |
| **Plastik** | 85.71% | 94.74% | 90.00% | 38 |
| **Residu** | 97.22% | 94.59% | 95.89% | 37 |

### Analisis Ambang Keyakinan (Validation Set)
- Rata-rata skor keyakinan saat prediksi **Benar**: `91.29%`
- Rata-rata skor keyakinan saat prediksi **Salah**: `67.50%`
- Ambang keyakinan (*Confidence Threshold*) yang ditetapkan pada aplikasi: **`60%` (`0.60`)**.

### Contoh Prediksi Benar pada Test Set
- `plastik_0001.jpg` -> Label Asli: **Plastik** | Prediksi: **Plastik** (Keyakinan: `77.7%`)
- `residu_0054.jpg` -> Label Asli: **Residu** | Prediksi: **Residu** (Keyakinan: `99.9%`)
- `organik_0166.jpg` -> Label Asli: **Organik** | Prediksi: **Organik** (Keyakinan: `100.0%`)
- `plastik_0178.jpg` -> Label Asli: **Plastik** | Prediksi: **Plastik** (Keyakinan: `98.4%`)
- `organik_0098.jpg` -> Label Asli: **Organik** | Prediksi: **Organik** (Keyakinan: `100.0%`)

### Contoh Prediksi Salah (Misclassified) pada Test Set
- `kaca_0192.jpg` -> Label Asli: **Kaca** | Prediksi: **Logam** (Keyakinan: `97.2%`)
- `kaca_0204.jpg` -> Label Asli: **Kaca** | Prediksi: **Logam** (Keyakinan: `94.5%`)
- `kaca_0081.jpg` -> Label Asli: **Kaca** | Prediksi: **Logam** (Keyakinan: `59.0%`)
- `kaca_0231.jpg` -> Label Asli: **Kaca** | Prediksi: **Plastik** (Keyakinan: `87.6%`)
- `kaca_0051.jpg` -> Label Asli: **Kaca** | Prediksi: **Kertas** (Keyakinan: `26.7%`)

## 4. Keterbatasan Dataset dan Kemungkinan Bias
1. **Kemiripan Visual Material Transparan/Mengilap:** Botol kaca bening dan botol plastik PET bening memiliki fitur pantulan cahaya yang mirip sehingga sesekali dapat tertukar pada sudut pengambilan tertentu.
2. **Kondisi Pencahayaan & Latar Belakang:** Gambar dengan satu objek dominan dan pencahayaan merata menghasilkan skor keyakinan yang jauh lebih akurat dibandingkan foto dengan latar belakang yang ramai.
3. **Kategori Residu yang Beragam:** Kelas `Residu` mencakup berbagai benda komposit yang tidak seragam secara bentuk, sehingga variasi intra-kelasnya lebih tinggi dibandingkan kelas seperti `Kaca` atau `Logam`.
