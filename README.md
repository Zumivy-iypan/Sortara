<div align="center">

# 🌿 SORTARA
### *Sort Smarter. Waste Better.*

Aplikasi web klasifikasi sampah cerdas berbasis *Computer Vision* & *Deep Learning* (MobileNetV2 Transfer Learning) untuk membantu pemilahan sampah ke dalam 6 kategori baku, dilengkapi rekomendasi penanganan, panduan warna tempat sampah, dan edukasi dampak lingkungan.

---

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Tests](https://img.shields.io/badge/Tests-17%20Passed-brightgreen?logo=pytest&logoColor=white)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📌 Fitur Utama

- **Klasifikasi Sampah 6 Kategori Baku:**
  Mengenali sampah kategori **Kaca**, **Kertas**, **Logam**, **Organik**, **Plastik**, dan **Residu** menggunakan model transfer learning MobileNetV2 dengan akurasi tinggi.
- **Dua Metode Input Fleksibel:**
  Mendukung pengambilan gambar langsung via **Kamera Web/HP** atau **Unggah File Foto** (JPG, JPEG, PNG).
- **Validasi & Integritas File Ketat:**
  Proteksi otomatis terhadap ukuran file (maksimal 10 MB) dan verifikasi integritas biner gambar sebelum diproses oleh model.
- **Rekomendasi Pemilahan & Edukasi Lingkungan:**
  Menyajikan panduan tempat sampah bersandi warna, estimasi waktu terurai di alam, serta dampak ekologis dari material sampah.
- **Dashboard Evaluasi Model Interaktif:**
  Halaman metrik komprehensif yang menampilkan *Confusion Matrix*, *Classification Report* (Precision, Recall, F1-Score), kurva *Training History*, dan distribusi dataset.
- **Production-Ready & Secure:**
  Dilengkapi konfigurasi reverse proxy Nginx, proteksi CSRF, rate limiting anti-DDoS, dan template deployment systemd.

---

## 📂 Struktur Proyek

```text
Sorta Project/
├── assets/                  # Logo, ikon, dan aset visual
├── data/
│   └── dataset_raw/         # Sampel foto dataset 6 kategori
├── deploy/                  # Skrip & konfigurasi deployment VPS
│   ├── nginx_sorta.conf     # Konfigurasi reverse proxy Nginx + WebSocket + Rate Limit
│   ├── setup_vps.sh         # Skrip instalasi otomatis untuk Ubuntu/Debian
│   └── sorta.service        # Service daemon Systemd
├── docs/                    # Dokumentasi teknis, PRD, dan laporan evaluasi
│   ├── EVALUATION_REPORT.md
│   └── PRD.md
├── models/                  # Bobot model terlatih (.keras) & pemetaan label
│   ├── labels.json
│   └── sortara_model.keras
├── notebooks/               # Eksperimen & pelatihan model MobileNetV2
├── scripts/                 # Skrip utilitas dataset & evaluasi offline
├── src/sortara/             # Package modul inti (Clean Architecture)
│   ├── config.py            # Konfigurasi konstanta & aturan pemilahan
│   ├── evaluation.py        # Modul evaluasi metrik model
│   ├── predictor.py         # Pipeline inferensi klasifikasi
│   ├── preprocessor.py      # Normalisasi & pra-pemrosesan citra
│   ├── recommender.py       # Engine pembuat rekomendasi & edukasi
│   └── validator.py         # Validator integritas file unggahan
├── tests/                   # Automated unit test suite (Pytest)
├── .streamlit/
│   └── config.toml          # Konfigurasi server & tema Streamlit
├── pyproject.toml           # Metadata proyek & konfigurasi toolchain
├── requirements.txt         # Daftar pustaka Python dependensi
├── streamlit_app.py         # Titik masuk utama aplikasi web Streamlit
└── README.md
```

---

## 🚀 Panduan Menjalankan Secara Lokal

### Prasyarat
- Python 3.12 atau versi lebih baru
- Git

### 1. Clone Repositori
```bash
git clone https://github.com/USERNAME/REPO_NAME.git
cd REPO_NAME
```

### 2. Buat dan Aktifkan Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependensi
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Jalankan Unit Test
Pastikan seluruh modul berfungsi normal:
```bash
pytest
```

### 5. Jalankan Aplikasi Web
```bash
streamlit run streamlit_app.py
```
Aplikasi akan otomatis terbuka di browser pada alamat `http://localhost:8501`.

---

## 🌐 Panduan Deployment ke VPS (Production)

Proyek ini telah dilengkapi dengan skrip deployment otomatis di folder `deploy/`.

1. **Arahkan DNS Domain:** Buat **A Record** pada DNS domain menuju alamat IP publik VPS Anda.
2. **Setup di VPS (Ubuntu 22.04 / 24.04):**
   ```bash
   git clone https://github.com/USERNAME/REPO_NAME.git /var/www/sorta
   cd /var/www/sorta
   chmod +x deploy/setup_vps.sh
   ./deploy/setup_vps.sh
   ```
3. **Konfigurasi Domain & Nginx:**
   Salin `deploy/nginx_sorta.conf` ke `/etc/nginx/sites-available/sorta`, sesuaikan nama domain Anda, lalu aktifkan:
   ```bash
   sudo ln -s /etc/nginx/sites-available/sorta /etc/nginx/sites-enabled/
   sudo nginx -t && sudo systemctl restart nginx
   ```
4. **Pasang Sertifikat SSL Gratis (HTTPS):**
   ```bash
   sudo certbot --nginx -d domainanda.com -d www.domainanda.com
   ```

---

## 🧪 Hasil Evaluasi Model

Model MobileNetV2 dilatih dengan fine-tuning pada dataset klasifikasi sampah 6 kelas:
- **Macro F1-Score:** > 85%
- **Threshold Keyakinan Baku:** 0.60 (prediksi di bawah nilai ini akan menyarankan pengambilan foto ulang yang lebih jelas).
- Rincian metrik per kelas dan confusion matrix dapat dilihat di tab **"Evaluasi Model"** pada aplikasi atau di [EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md).

---

## 📄 Lisensi

Didistribusikan di bawah lisensi **MIT License**. Lihat file `LICENSE` untuk informasi selengkapnya.

---

<div align="center">
Dibuat dengan ❤️ untuk lingkungan yang lebih bersih dan berkelanjutan.
</div>
