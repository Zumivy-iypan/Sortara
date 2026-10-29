# Task List — SORTARA MVP

- [x] Task 1: Siapkan struktur proyek, konfigurasi (`pyproject.toml`, `requirements.txt`, `.streamlit/config.toml`, `models/labels.json`), dan virtual environment `.venv`.
  - Acceptance: `.venv` terinstal dengan `streamlit`, `pillow`, `numpy`, `scikit-learn`, `matplotlib`, `pytest`, `ruff`.
  - Verify: `.\.venv\Scripts\pytest --version`
  - Files: `pyproject.toml`, `requirements.txt`, `.streamlit/config.toml`, `models/labels.json`

- [x] Task 2: Implementasikan modul `contracts-config` (`src/sortara/config.py`) dan modul `image-validator` (`src/sortara/validator.py`, `src/sortara/preprocessor.py`) menggunakan TDD.
  - Acceptance: Validasi menolak format selain JPG/JPEG/PNG (`AT-03`), menolak ukuran >10 MB (`AT-04`), menolak file corrupt, menerima JPG/PNG valid (`AT-01`, `AT-02`), dan mengubah gambar menjadi tensor `(1, 224, 224, 3)` bertipe `float32` dalam rentang `[-1.0, 1.0]` (`FR-06`).
  - Verify: `.\.venv\Scripts\pytest tests/test_validator.py tests/test_preprocessor.py -v`
  - Files: `src/sortara/__init__.py`, `src/sortara/config.py`, `src/sortara/validator.py`, `src/sortara/preprocessor.py`, `tests/test_validator.py`, `tests/test_preprocessor.py`

- [x] Task 3: Implementasikan modul `recommendation-engine` (`src/sortara/recommender.py`) menggunakan TDD.
  - Acceptance: Menghasilkan rekomendasi yang tepat untuk ke-6 kategori ketika `confidence >= threshold` (`FR-09`), menampilkan pesan belum dapat dikenali ketika `confidence < threshold` (`FR-10`, `AT-06`), dan menyertakan disclaimer aturan setempat (`FR-12`).
  - Verify: `.\.venv\Scripts\pytest tests/test_recommender.py -v`
  - Files: `src/sortara/recommender.py`, `tests/test_recommender.py`

- [x] Task 4: Implementasikan modul `ai-inference` (`src/sortara/predictor.py`) dengan Dual-Mode (Keras `.keras` + Demo Fallback) dan penanganan error model (`FR-13`, `AT-08`) menggunakan TDD.
  - Acceptance: Menghasilkan prediksi 6 kelas + skor keyakinan yang tervalidasi (`sum(probabilities) == 1.0`), mendukung pemuatan model `.keras` asli maupun mode fallback, serta menangani kegagalan muat model secara aman (`AT-08`).
  - Verify: `.\.venv\Scripts\pytest tests/test_predictor.py -v`
  - Files: `src/sortara/predictor.py`, `tests/test_predictor.py`

- [x] Task 5: Implementasikan modul evaluasi AI (`src/sortara/evaluation.py`) dan Notebook Google Colab (`notebooks/train_sortara_mobilenetv2.ipynb`) untuk tugas Adit (`AT-09`).
  - Acceptance: Menghitung Accuracy, Precision & Recall per kelas, serta menghasilkan Confusion Matrix Figure menggunakan Scikit-learn & Matplotlib.
  - Verify: `.\.venv\Scripts\pytest tests/test_evaluation.py -v`
  - Files: `src/sortara/evaluation.py`, `tests/test_evaluation.py`, `notebooks/train_sortara_mobilenetv2.ipynb`

- [x] Task 6: Implementasikan aplikasi web Streamlit (`streamlit_app.py`) sesuai best practices `developing-with-streamlit` dan uji otomatis dengan `st.testing.v1.AppTest` (`tests/test_app.py`).
  - Acceptance: Seluruh fitur `FR-01` s.d. `FR-13` dan `AT-01` s.d. `AT-10` berjalan mulus tanpa exception, semua unit & integration test lulus, dan `ruff check .` bersih.
  - Verify: `.\.venv\Scripts\pytest -v` dan `.\.venv\Scripts\ruff check .`
  - Files: `streamlit_app.py`, `tests/test_app.py`
