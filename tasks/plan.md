# Technical Implementation Plan — SORTARA MVP

## 1. Major Components & Dependencies

```text
contracts-config (src/sortara/config.py, models/labels.json)
   ├──> image-validator (src/sortara/validator.py, src/sortara/preprocessor.py)
   ├──> recommendation-engine (src/sortara/recommender.py)
   ├──> ml-training-eval (notebooks/train_sortara_mobilenetv2.ipynb)
   └──> ai-inference (src/sortara/predictor.py)
            │
            ▼
       streamlit-ui (streamlit_app.py, .streamlit/config.toml)
```

## 2. Implementation Order

1. **Environment & Project Scaffolding:**
   - Buat `requirements.txt`, `pyproject.toml`, `.streamlit/config.toml`, dan `models/labels.json`.
   - Siapkan `.venv` dengan Python 3.12 dan instal dependensi inti (`streamlit`, `pillow`, `numpy`, `scikit-learn`, `matplotlib`, `pytest`, `ruff`).
2. **Module 1 — `contracts-config` (`src/sortara/config.py`):**
   - Definisikan 6 kategori (`Kaca`, `Kertas`, `Logam`, `Organik`, `Plastik`, `Residu`), batas file `10 MB`, ekstensi valid (`jpg`, `jpeg`, `png`), dimensi input `(224, 224)`, `DEFAULT_CONFIDENCE_THRESHOLD = 0.60`, dan kamus rekomendasi penanganan beserta catatan aturan lokal (`FR-01`, `FR-09`, `FR-12`).
3. **Module 2 — `image-validator` (`src/sortara/validator.py` & `src/sortara/preprocessor.py`):**
   - Ikuti siklus TDD (Red → Green → Refactor): tulis `tests/test_validator.py` dan `tests/test_preprocessor.py` terlebih dahulu, lalu implementasikan validasi format/ukuran/integritas gambar (`FR-02`, `FR-03`, `FR-04`, `AT-01..AT-04`) serta *preprocessing* gambar ke tensor `(1, 224, 224, 3)` dalam rentang `[-1.0, 1.0]` yang identik dengan `tf.keras.applications.mobilenet_v2.preprocess_input` (`FR-06`).
4. **Module 3 — `recommendation-engine` (`src/sortara/recommender.py`):**
   - Ikuti siklus TDD: tulis `tests/test_recommender.py` terlebih dahulu, lalu implementasikan pemetaan hasil prediksi ke objek rekomendasi yang menangani kondisi *confident* (`score >= threshold`) maupun *low confidence* (`score < threshold`, `FR-10`, `AT-06`).
5. **Module 4 — `ai-inference` (`src/sortara/predictor.py`):**
   - Ikuti siklus TDD: tulis `tests/test_predictor.py` terlebih dahulu, lalu implementasikan pemuat model `.keras` dengan dukungan *Dual-Mode* (otomatis memakai `models/sortara_model.keras` bila tersedia, atau menggunakan *heuristic/demo classifier* deterministik berbasis fitur visual bila file `.keras` belum tersedia, serta menangani kegagalan muat model sesuai `FR-13` / `AT-08`).
6. **Module 5 — `streamlit-ui` (`streamlit_app.py`):**
   - Ikuti panduan `developing-with-streamlit` (`width="stretch"`, Material Symbols `:material/...:`, `alt=` pada `st.image`, `@st.cache_resource`, dan tema `.streamlit/config.toml`).
   - Uji secara otomatis dengan `tests/test_app.py` menggunakan `st.testing.v1.AppTest`.
7. **Module 6 — `ml-training-eval` (`notebooks/train_sortara_mobilenetv2.ipynb` & `src/sortara/evaluation.py`):**
   - Buat modul utilitas evaluasi (`src/sortara/evaluation.py`) + unit test (`tests/test_evaluation.py`) menggunakan `scikit-learn` dan `matplotlib` untuk menghitung *Accuracy*, *Precision*, *Recall* per kelas, dan membuat grafik *Confusion Matrix* (`AT-09`).
   - Buat *notebook* Google Colab lengkap (`notebooks/train_sortara_mobilenetv2.ipynb`) untuk tugas Adit.

## 3. Risks & Mitigations

- **Belum tersedianya `sortara_model.keras` saat UI diuji:** Diatasi dengan *Dual-Mode Predictor* di `src/sortara/predictor.py` yang tetap memproses tensor `(1, 224, 224, 3)` dan mengeluarkan distribusi probabilitas 6 kelas beserta status mode model.
- **Ketidaksesuaian urutan kelas antara Colab dan Streamlit:** Diatasi dengan file `models/labels.json` yang menjadi satu-satunya sumber kebenaran (*single source of truth*).
