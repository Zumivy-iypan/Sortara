# Spec: SORTARA — Smart Waste Classification System (MVP)

## Capability Map

| Module id | Responsibility | Depends on |
|---|---|---|
| `contracts-config` | Definisi 6 kelas label, konstanta validasi file (`10 MB`), ukuran input (`224×224`), `CONFIDENCE_THRESHOLD`, aturan rekomendasi, waktu terurai, dampak lingkungan, ide *upcycling*, dan panduan warna tempat sampah | — |
| `image-validator` | Validasi format (`JPG`, `JPEG`, `PNG`), ukuran file (`<= 10 MB`), integritas gambar (anti-corrupt), dan preprocessing tensor (`224×224` MobileNetV2) | `contracts-config` |
| `recommendation-engine` | Memetakan hasil prediksi kategori dan skor keyakinan ke rekomendasi penanganan, edukasi waktu terurai & sebab-akibat, ide guna ulang (*upcycling*), warna tong sampah, serta pesan alternatif (*low-confidence guardrail*) | `contracts-config` |
| `ai-inference` | Memuat model `models/sortara_model.keras` (dengan `@st.cache_resource` & *mock/heuristic fallback mode* jika file `.keras` belum tersedia) dan menghasilkan distribusi probabilitas 6 kelas | `contracts-config`, `image-validator` |
| `streamlit-ui` | Antarmuka pengguna Streamlit (`streamlit_app.py`): halaman utama, mode input (Unggah File, Kamera Langsung, & Galeri Contoh Cepat), pratinjau, tombol aksi, kartu hasil prediksi, kartu edukasi lingkungan, disclaimer, dan reset state | `image-validator`, `ai-inference`, `recommendation-engine` |
| `ml-training-eval` | Skrip dan Notebook Google Colab untuk kurasi dataset, pembagian 70/15/15, transfer learning MobileNetV2, evaluasi Scikit-learn (Accuracy, Precision, Recall, Confusion Matrix), dan ekspor `sortara_model.keras` + `labels.json` | `contracts-config` |

Build order: `contracts-config` → `image-validator`, `recommendation-engine`, `ml-training-eval` → `ai-inference` → `streamlit-ui`

---

## Objective

Membangun prototipe aplikasi web **SORTARA** (*"Sort Smarter. Waste Better."*) yang membantu masyarakat umum dan mahasiswa mengenali 1 dari 6 kategori sampah (`Kaca`, `Kertas`, `Logam`, `Organik`, `Plastik`, `Residu`) melalui unggahan foto satu objek sampah dominan, menampilkan skor keyakinan (*confidence score*), memberikan saran awal pemilahan/penanganan, serta menangani kasus gambar tidak valid atau keyakinan model rendah secara transparan sesuai `PRD.md` (`FR-01` s.d. `FR-13` dan `AT-01` s.d. `AT-10`).

---

## Tech Stack

- **Runtime:** Python 3.12 (`Python 3.12.10`)
- **Web UI Framework:** Streamlit (`>=1.40.0`)
- **Deep Learning / Inference:** TensorFlow / Keras (`>=2.16.0`, arsitektur MobileNetV2 transfer learning)
- **Image Processing:** Pillow (`PIL >=10.0.0`), NumPy (`>=1.26.0`)
- **Evaluation & Visualization:** Scikit-learn (`>=1.5.0`), Matplotlib (`>=3.9.0`)
- **Testing & Quality:** `pytest` (termasuk `streamlit.testing.v1.AppTest`), `ruff`

---

## Commands

Semua perintah dijalankan menggunakan virtual environment lokal `.venv` di *workspace* `d:\Antigravity\Sorta Project`:

```powershell
# Setup virtual environment & install dependencies
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\pip install -r requirements.txt

# Run unit & integration tests (including Streamlit AppTest)
.\.venv\Scripts\pytest -v

# Run linter & formatter check
.\.venv\Scripts\ruff check .

# Run Streamlit web application locally
.\.venv\Scripts\streamlit run streamlit_app.py --server.port 8501
```

---

## Project Structure

```text
d:\Antigravity\Sorta Project\
├── .streamlit/
│   └── config.toml                 # Konfigurasi tema & batas upload Streamlit
├── docs/
│   └── PRD.md                      # Dokumen Product Requirements Document (PRD v1.0)
├── models/
│   ├── labels.json                 # Kontrak urutan indeks kelas (0..5)
│   └── sortara_model.keras         # File model hasil training Colab (opsional saat dev awal)
├── notebooks/
│   └── train_sortara_mobilenetv2.ipynb # Notebook Google Colab untuk training & evaluasi (Tugas Adit)
├── src/
│   └── sortara/
│       ├── __init__.py
│       ├── config.py               # [contracts-config] Konstanta, label 6 kelas, threshold, aturan rekomendasi
│       ├── validator.py            # [image-validator] Validasi ekstensi, ukuran <=10MB, anti-corrupt
│       ├── preprocessor.py         # [image-validator] Resize 224x224 RGB & MobileNetV2 preprocess_input
│       ├── recommender.py          # [recommendation-engine] Logika rekomendasi & low-confidence guardrail
│       └── predictor.py            # [ai-inference] Loader .keras + fallback mode untuk demo/testing
├── tests/
│   ├── test_validator.py           # Unit test AT-01, AT-02, AT-03, AT-04
│   ├── test_preprocessor.py        # Unit test FR-06 (dimensi tensor 1x224x224x3 & rentang nilai)
│   ├── test_recommender.py         # Unit test FR-09, FR-10, FR-12
│   ├── test_predictor.py           # Unit test FR-07, FR-08, FR-13 (termasuk gagal muat model)
│   └── test_app.py                 # Integration test UI menggunakan st.testing.v1.AppTest
├── SPEC.md                         # Spesifikasi teknis proyek (dokumen ini)
├── requirements.txt                # Daftar dependensi Python
└── streamlit_app.py                # [streamlit-ui] Entrypoint utama aplikasi web Streamlit
```

---

## Code Style

Menggunakan standar Python 3.12+ dengan *type hints* eksplisit, `dataclass(frozen=True)` untuk kontrak data *immutable*, *docstrings* ringkas, dan pemisahan tegas antara logika bisnis (`src/sortara/`) dengan tampilan UI (`streamlit_app.py`). Untuk Streamlit, mengikuti panduan `developing-with-streamlit` (menggunakan `width="stretch"` alih-alih `use_container_width`, ikon Material `:material/...:`, serta `@st.cache_resource` untuk pemuatan model).

Contoh gaya kode (`src/sortara/validator.py`):

```python
from dataclasses import dataclass
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from sortara.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    error_message: str | None = None
    image: Image.Image | None = None


def validate_uploaded_image(file_name: str, file_bytes: bytes) -> ValidationResult:
    """Memvalidasi format, ukuran, dan keterbacaan file gambar unggahan (FR-02, FR-03, FR-04)."""
    ext = file_name.rsplit(".", maxsplit=1)[-1].lower() if "." in file_name else ""
    if ext not in ALLOWED_EXTENSIONS:
        return ValidationResult(
            is_valid=False,
            error_message="Format file tidak didukung. Gunakan gambar berformat JPG, JPEG, atau PNG.",
        )

    if len(file_bytes) == 0 or len(file_bytes) > MAX_FILE_SIZE_BYTES:
        return ValidationResult(
            is_valid=False,
            error_message="Ukuran file tidak valid atau melebihi batas maksimal 10 MB.",
        )

    try:
        img = Image.open(BytesIO(file_bytes))
        img.verify()
        reopened = Image.open(BytesIO(file_bytes)).convert("RGB")
        return ValidationResult(is_valid=True, image=reopened)
    except (UnidentifiedImageError, OSError, ValueError):
        return ValidationResult(
            is_valid=False,
            error_message="File gambar rusak atau tidak dapat dibaca. Silakan coba foto lain.",
        )
```

---

## Testing Strategy

- **Framework:** `pytest` + `streamlit.testing.v1.AppTest`.
- **Unit Tests (`tests/test_*.py`):** Menguji murni logika di `src/sortara/` menggunakan gambar sintetis dalam memori (`BytesIO`) tanpa bergantung pada file eksternal.
- **UI Integration Tests (`tests/test_app.py`):** Menguji alur halaman utama, state tombol, pesan peringatan, dan hasil klasifikasi secara *headless*.
- **Target Cakupan:** Seluruh persyaratan fungsional `FR-01` s.d. `FR-13` dan kriteria penerimaan `AT-01` s.d. `AT-10` wajib memiliki pengujian terotomatisasi.

---

## Boundaries

- **Always do:**
  - Jalankan `pytest` setiap kali menyelesaikan satu modul.
  - Gunakan kontrak kelas yang konsisten antara notebook training dan aplikasi Streamlit (`labels.json` / `config.py`).
  - Proses gambar sepenuhnya di memori (`BytesIO`) tanpa menyimpan foto pengguna secara permanen ke disk (sesuai persyaratan Privasi Bagian 8 PRD).
- **Ask first:**
  - Mengubah daftar 6 kategori sampah atau menambah dependensi berat di luar `requirements.txt`.
  - Mengubah arsitektur dasar dari MobileNetV2 ke model lain.
- **Never do:**
  - Menggunakan parameter Streamlit yang sudah *deprecated* seperti `use_container_width`.
  - Menampilkan klaim bahwa suatu benda *pasti* dapat didaur ulang di semua wilayah atau mengklaim angka akurasi model sebelum eksperimen pada *test set* dijalankan.

---

## Success Criteria

1. `pytest` lulus 100% mencakup seluruh skenario penerimaan `AT-01` s.d. `AT-08`.
2. Aplikasi `streamlit_app.py` dapat dijalankan di browser, menerima unggahan `JPG/JPEG/PNG` (`<= 10 MB`), menampilkan pratinjau, memprediksi kategori beserta skor keyakinan dan rekomendasi, serta menampilkan pesan *low confidence* apabila skor `< CONFIDENCE_THRESHOLD`.
3. Tersedia *mode simulasi/fallback* yang jelas ketika file `models/sortara_model.keras` belum diletakkan di folder `models/`, dan langsung otomatis memakai model Keras asli begitu file `sortara_model.keras` tersedia.
4. Notebook `notebooks/train_sortara_mobilenetv2.ipynb` siap dijalankan di Google Colab untuk melatih MobileNetV2 (split 70/15/15), menghitung *Accuracy*, *Precision*, *Recall*, *Confusion Matrix*, dan mengekspor `sortara_model.keras` beserta `labels.json`.

---

## Open Questions

1. Apakah Anda ingin kita menginstal TensorFlow langsung di *environment* lokal `.venv` sekarang, atau membuat TensorFlow bersifat opsional (*lazy import*) di lokal agar aplikasi Streamlit tetap ringan dan instan saat dijalankan sebelum file `sortara_model.keras` dari Google Colab dimasukkan? *(Rekomendasi: dukung keduanya—jika TensorFlow & `sortara_model.keras` ada maka gunakan inferensi Keras asli; jika belum ada maka gunakan fallback/demo mode tanpa error).*
