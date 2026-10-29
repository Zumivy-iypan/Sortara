from io import BytesIO
from pathlib import Path

from PIL import Image
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def _create_test_image_bytes(fmt: str = "JPEG", color: tuple[int, int, int] = (30, 160, 75)) -> bytes:
    buf = BytesIO()
    img = Image.new("RGB", (260, 260))
    for x in range(260):
        for y in range(260):
            r = (color[0] + (x % 60)) % 256
            g = (color[1] + (y % 60)) % 256
            b = (color[2] + ((x + y) % 40)) % 256
            img.putpixel((x, y), (r, g, b))
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_app_initial_state_renders_without_exception_fr01():
    """FR-01: Menampilkan nama dan penjelasan singkat SORTARA saat halaman pertama dibuka."""
    at = AppTest.from_file(APP_PATH, default_timeout=20).run()
    assert not at.exception
    assert any("SORTARA" in t.value for t in at.title)


def test_app_end_to_end_upload_and_classify_at01_at05_at07():
    """AT-01, AT-05, AT-07: Unggah foto JPG valid di kartu kiri, lihat pratinjau, klasifikasi, dan coba foto lain."""
    at = AppTest.from_file(APP_PATH, default_timeout=20).run()
    assert not at.exception

    jpg_bytes = _create_test_image_bytes("JPEG")
    at.file_uploader(key="waste_uploader").upload("botol_plastik.jpg", jpg_bytes).run()
    assert not at.exception

    # Klik tombol "Klasifikasikan sampah"
    at.button(key="btn_classify").click().run()
    assert not at.exception
    assert at.session_state["classification_done"] is True
    assert at.session_state["last_prediction"] is not None

    # Klik tombol "Coba foto lain" (AT-07)
    at.button(key="btn_reset").click().run()
    assert not at.exception
    assert at.session_state["classification_done"] is False


def test_app_three_option_layout_camera_and_sample_tester():
    """Menguji tata letak 3 opsi: kartu utama kamera di atas serta split kartu upload & gambar tester di bawahnya."""
    at = AppTest.from_file(APP_PATH, default_timeout=20).run()
    assert not at.exception

    # Pilih gambar tester di kartu kanan bawah dan klik tombol "Gunakan gambar tester ini"
    at.selectbox(key="sample_class_selector").select("Plastik").run()
    at.button(key="btn_use_sample").click().run()
    assert not at.exception

    # Jalankan klasifikasi pada gambar tester
    at.button(key="btn_classify").click().run()
    assert not at.exception
    assert at.session_state["classification_done"] is True


def test_app_sidebar_navigation_to_categories_and_evaluation_pages():
    """Menguji navigasi sidebar ke halaman tersendiri Cakupan 6 Kategori & Tong serta Confusion Matrix & Evaluasi."""
    at = AppTest.from_file(APP_PATH, default_timeout=20).run()
    assert not at.exception
    assert at.session_state["active_page"] == "classify"

    # Buka halaman Cakupan 6 kategori & tong
    at.button(key="nav_categories").click().run()
    assert not at.exception
    assert at.session_state["active_page"] == "categories"

    # Kembali ke halaman klasifikasi dari tombol kembali
    at.button(key="btn_back_from_categories").click().run()
    assert not at.exception
    assert at.session_state["active_page"] == "classify"

    # Buka halaman Confusion Matrix & Evaluasi
    at.button(key="nav_evaluation").click().run()
    assert not at.exception
    assert at.session_state["active_page"] == "evaluation"

    # Buka halaman Tentang & Spesifikasi Model (About)
    at.button(key="nav_about").click().run()
    assert not at.exception
    assert at.session_state["active_page"] == "about"

    # Kembali ke halaman klasifikasi dari tombol kembali di halaman About
    at.button(key="btn_back_from_about").click().run()
    assert not at.exception
    assert at.session_state["active_page"] == "classify"


