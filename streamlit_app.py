import base64
import json
import sys
from html import escape
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# Pastikan paket di dalam src/ dapat diimpor ketika dijalankan langsung dengan `streamlit run streamlit_app.py`
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sortara.config import (  # noqa: E402
    APP_DESCRIPTION,
    APP_NAME,
    APP_TAGLINE,
    CATEGORY_RULES,
    CLASS_LABELS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DATASET_DIR,
    DEFAULT_LABELS_PATH,
    DEFAULT_MODEL_PATH,
    LOCAL_RULE_DISCLAIMER,
    MAX_FILE_SIZE_MB,
)
from sortara.predictor import PredictionResult, WastePredictor  # noqa: E402
from sortara.recommender import build_recommendation  # noqa: E402
from sortara.validator import ValidationResult, validate_uploaded_image  # noqa: E402

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_SVG_PATH = ASSETS_DIR / "sortara_logo.svg"
LOGO_PNG_PATH = ASSETS_DIR / "sortara_logo.png"

st.set_page_config(
    page_title=f"{APP_NAME} — {APP_TAGLINE}",
    page_icon=str(LOGO_PNG_PATH) if LOGO_PNG_PATH.exists() else ":material/recycling:",
    layout="wide",
)

# Inisialisasi state sesi secara terpusat
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "classify"  # "classify" | "categories" | "evaluation"
if "classification_done" not in st.session_state:
    st.session_state["classification_done"] = False
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None
if "last_filename" not in st.session_state:
    st.session_state["last_filename"] = None
if "active_source" not in st.session_state:
    st.session_state["active_source"] = "none"  # "camera" | "upload" | "sample" | "none"
if "camera_widget_version" not in st.session_state:
    st.session_state["camera_widget_version"] = 0



@st.cache_resource(max_entries=2)
def get_waste_predictor(model_path_str: str, labels_path_str: str) -> WastePredictor:
    """Memuat model klasifikasi sekali dan menyimpannya di cache resource Streamlit."""
    return WastePredictor(
        model_path=Path(model_path_str),
        labels_path=Path(labels_path_str),
        strict_model_loading=False,
    )


@st.cache_data(max_entries=12)
def load_sample_image_bytes(category_name: str) -> tuple[str, bytes]:
    """Mengambil 1 foto contoh untuk kategori terpilih dari dataset lokal atau fallback sintetis."""
    cls_dir = DEFAULT_DATASET_DIR / category_name
    if cls_dir.exists():
        candidates = sorted(
            p for p in cls_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if candidates:
            chosen = candidates[0]
            return chosen.name, chosen.read_bytes()

    buf = BytesIO()
    img = Image.new("RGB", (260, 260), color=(45, 160, 85))
    img.save(buf, format="JPEG")
    return f"contoh_{category_name.lower()}.jpg", buf.getvalue()


def image_to_data_uri(img: Image.Image) -> str:
    """Mengubah objek PIL Image ke base64 data URI agar dapat dirender di dalam kartu custom."""
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def file_image_to_data_uri(path: Path) -> str:
    """Membaca file PNG/JPG/SVG dari disk dan mengubahnya ke base64 data URI."""
    if not path.exists():
        return ""
    suffix = path.suffix.lower()
    if suffix == ".svg":
        mime = "image/svg+xml"
    elif suffix == ".png":
        mime = "image/png"
    else:
        mime = "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


@st.cache_data(max_entries=1)
def get_sortara_logo_uri() -> str:
    """Mengambil data URI logo resmi SORTARA (SVG prioritas utama, fallback PNG)."""
    if LOGO_SVG_PATH.exists():
        return file_image_to_data_uri(LOGO_SVG_PATH)
    if LOGO_PNG_PATH.exists():
        return file_image_to_data_uri(LOGO_PNG_PATH)
    return ""



def make_colored_bin_svg_uri(bin_color_guide: str, category_name: str) -> str:
    """Membuat ilustrasi SVG tong sampah berwarna (Hijau/Kuning/Biru/Abu-abu) sebagai data URI."""
    guide_lower = bin_color_guide.lower()
    if "hijau" in guide_lower:
        body_fill = "#16a34a"
        lid_fill = "#15803d"
        dark_shade = "#14532d"
        bin_label = "ORGANIK"
    elif "biru" in guide_lower:
        body_fill = "#2563eb"
        lid_fill = "#1d4ed8"
        dark_shade = "#1e3a8a"
        bin_label = "KERTAS"
    elif "kuning" in guide_lower:
        body_fill = "#eab308"
        lid_fill = "#ca8a04"
        dark_shade = "#713f12"
        bin_label = "ANORGANIK"
    else:
        body_fill = "#52525b"
        lid_fill = "#3f3f46"
        dark_shade = "#18181b"
        bin_label = "RESIDU"

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 200" width="100%" height="100%">
      <defs>
        <filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">
          <feDropShadow dx="0" dy="6" stdDeviation="6" flood-color="#000000" flood-opacity="0.12"/>
        </filter>
      </defs>
      <!-- Bayangan alas -->
      <ellipse cx="110" cy="182" rx="52" ry="7" fill="#d4d4d0"/>
      <g filter="url(#shadow)">
        <!-- Gagang atas tutup -->
        <rect x="92" y="18" width="36" height="10" rx="4" fill="{dark_shade}"/>
        <!-- Tutup tong sampah -->
        <rect x="52" y="25" width="116" height="16" rx="7" fill="{lid_fill}"/>
        <!-- Bibir tong -->
        <rect x="48" y="38" width="124" height="8" rx="4" fill="{dark_shade}"/>
        <!-- Badan utama tong sampah trapesium -->
        <polygon points="56,46 164,46 152,168 68,168" fill="{body_fill}"/>
        <!-- Garis-garis tulang vertikal tong -->
        <line x1="78" y1="56" x2="83" y2="156" stroke="{lid_fill}" stroke-width="4" stroke-linecap="round"/>
        <line x1="142" y1="56" x2="137" y2="156" stroke="{lid_fill}" stroke-width="4" stroke-linecap="round"/>
        <!-- Lingkaran simbol daur ulang di tengah -->
        <circle cx="110" cy="96" r="24" fill="#ffffff" fill-opacity="0.92"/>
        <path d="M110 80 L118 93 L102 93 Z" fill="{dark_shade}"/>
        <path d="M98 105 L110 95 L122 105 Z" fill="{body_fill}"/>
        <circle cx="110" cy="97" r="6" fill="{body_fill}"/>
        <!-- Label kategori pada badan tong -->
        <rect x="74" y="130" width="72" height="20" rx="5" fill="#ffffff" fill-opacity="0.92"/>
        <text x="110" y="144" font-family="Inter, sans-serif" font-size="9.5" font-weight="800" fill="{dark_shade}" text-anchor="middle">{escape(bin_label)}</text>
        <!-- Roda bawah -->
        <rect x="68" y="166" width="16" height="14" rx="4" fill="#18181b"/>
        <rect x="136" y="166" width="16" height="14" rx="4" fill="#18181b"/>
      </g>
    </svg>"""
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def make_spec_icon_svg_uri(symbol_type: str, accent_color: str = "#16a34a") -> str:
    """Membuat ilustrasi SVG minimalis bergaya editorial untuk kartu spesifikasi model & sistem."""
    if symbol_type == "layers":
        inner = f"""
        <polygon points="110,58 156,80 110,102 64,80" fill="none" stroke="{accent_color}" stroke-width="4.5" stroke-linejoin="round"/>
        <polyline points="64,100 110,122 156,100" fill="none" stroke="#18181b" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>
        <polyline points="64,120 110,142 156,120" fill="none" stroke="#71717a" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>
        """
    elif symbol_type == "training":
        inner = f"""
        <rect x="64" y="64" width="92" height="72" rx="12" fill="none" stroke="#18181b" stroke-width="4"/>
        <polyline points="78,118 98,96 114,106 142,78" fill="none" stroke="{accent_color}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="142" cy="78" r="5" fill="{accent_color}"/>
        """
    elif symbol_type == "dataset":
        inner = f"""
        <ellipse cx="110" cy="70" rx="40" ry="14" fill="none" stroke="{accent_color}" stroke-width="4.5"/>
        <path d="M70,70 v28 c0,8 18,14 40,14 s40,-6 40,-14 v-28" fill="none" stroke="#18181b" stroke-width="4.5"/>
        <path d="M70,98 v28 c0,8 18,14 40,14 s40,-6 40,-14 v-28" fill="none" stroke="#71717a" stroke-width="4.5"/>
        """
    elif symbol_type == "augment":
        inner = f"""
        <rect x="68" y="68" width="54" height="54" rx="10" fill="none" stroke="#71717a" stroke-width="4" stroke-dasharray="6 5"/>
        <rect x="94" y="82" width="54" height="54" rx="10" fill="none" stroke="{accent_color}" stroke-width="4.5"/>
        <circle cx="112" cy="100" r="6" fill="#18181b"/>
        <polyline points="96,128 114,112 130,124 146,108" fill="none" stroke="#18181b" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
        """
    elif symbol_type == "shield":
        inner = f"""
        <path d="M110,56 L148,72 V102 C148,126 131,142 110,150 C89,142 72,126 72,102 V72 Z" fill="none" stroke="#18181b" stroke-width="4.5" stroke-linejoin="round"/>
        <polyline points="96,104 106,114 126,92" fill="none" stroke="{accent_color}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
        """
    else:
        inner = f"""
        <rect x="64" y="62" width="38" height="34" rx="7" fill="none" stroke="{accent_color}" stroke-width="4.5"/>
        <rect x="118" y="62" width="38" height="34" rx="7" fill="none" stroke="#18181b" stroke-width="4.5"/>
        <rect x="91" y="110" width="38" height="34" rx="7" fill="none" stroke="#52525b" stroke-width="4.5"/>
        <line x1="83" y1="96" x2="102" y2="110" stroke="#71717a" stroke-width="3.5"/>
        <line x1="137" y1="96" x2="118" y2="110" stroke="#71717a" stroke-width="3.5"/>
        """

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 200" width="100%" height="100%">
      <circle cx="110" cy="102" r="56" fill="#ffffff" fill-opacity="0.75"/>
      {inner}
    </svg>"""
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"



def reset_classification_state() -> None:
    """Mereset hasil prediksi agar pengguna dapat mengunggah atau mencoba foto baru (FR-11)."""
    st.session_state["classification_done"] = False
    st.session_state["last_prediction"] = None
    st.session_state["last_filename"] = None


def on_smart_camera_capture() -> None:
    reset_classification_state()
    comp_state = st.session_state.get("smart_camera_widget", {})
    if isinstance(comp_state, dict) and comp_state.get("captured_jpeg"):
        st.session_state["active_source"] = "smart_camera"


_SIDEBAR_BRAND_THEME_SWITCH = st.components.v2.component(
    "sortara_brand_theme_switch",
    html="""
    <div id="sb-root" class="sb-brand-row">
      <div class="sb-left">
        <div class="sb-logo">
          <img id="sb-logo-img" src="" alt="Logo SORTARA" />
        </div>
        <span class="sb-title">SORTARA</span>
      </div>
      <button
        id="sb-theme-btn"
        type="button"
        class="sb-switch"
        aria-label="Ganti mode terang atau gelap"
        title="Ganti Mode Terang / Gelap"
      >
        <span class="sb-switch-track">
          <span class="sb-icon-sun" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="12" r="4"></circle>
              <line x1="12" y1="2" x2="12" y2="4"></line>
              <line x1="12" y1="20" x2="12" y2="22"></line>
              <line x1="4.93" y1="4.93" x2="6.34" y2="6.34"></line>
              <line x1="17.66" y1="17.66" x2="19.07" y2="19.07"></line>
              <line x1="2" y1="12" x2="4" y2="12"></line>
              <line x1="20" y1="12" x2="22" y2="12"></line>
              <line x1="4.93" y1="19.07" x2="6.34" y2="17.66"></line>
              <line x1="17.66" y1="6.34" x2="19.07" y2="4.93"></line>
            </svg>
          </span>
          <span class="sb-icon-moon" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
            </svg>
          </span>
          <span class="sb-switch-thumb"></span>
        </span>
      </button>
    </div>
    """,
    css="""
    .sb-brand-row {
      font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 4px 2px 14px 2px;
      border-bottom: 1px solid #e6e6e2;
      margin-bottom: 6px;
      transition: border-color 0.25s ease;
    }
    .sb-brand-row[data-theme="dark"] {
      border-bottom-color: #252836;
    }
    .sb-left {
      display: flex;
      align-items: center;
      gap: 11px;
      min-width: 0;
    }
    .sb-logo {
      width: 36px;
      height: 36px;
      border-radius: 9px;
      overflow: hidden;
      flex-shrink: 0;
      box-shadow: 0 2px 6px rgba(11, 51, 48, 0.22);
      background: #0c3431;
    }
    .sb-logo img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    .sb-title {
      font-weight: 800;
      font-size: 1.02rem;
      letter-spacing: -0.01em;
      color: #111111;
      transition: color 0.25s ease;
    }
    .sb-brand-row[data-theme="dark"] .sb-title {
      color: #f4f4f5;
    }
    .sb-switch {
      background: transparent;
      border: none;
      padding: 0;
      cursor: pointer;
      outline: none;
      flex-shrink: 0;
    }
    .sb-switch-track {
      position: relative;
      display: inline-flex;
      align-items: center;
      justify-content: space-between;
      width: 54px;
      height: 28px;
      padding: 0 6px;
      box-sizing: border-box;
      border-radius: 999px;
      background: #e4e4e0;
      border: 1px solid #d4d4d0;
      transition: background 0.3s ease, border-color 0.3s ease;
    }
    .sb-brand-row[data-theme="dark"] .sb-switch-track {
      background: #181b26;
      border-color: #2e3346;
    }
    .sb-icon-sun {
      color: #f59e0b;
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 1;
    }
    .sb-icon-moon {
      color: #818cf8;
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 1;
    }
    .sb-switch-thumb {
      position: absolute;
      top: 2px;
      left: 2px;
      width: 22px;
      height: 22px;
      border-radius: 50%;
      background: #ffffff;
      box-shadow: 0 2px 5px rgba(0, 0, 0, 0.18);
      transition: transform 0.32s cubic-bezier(0.22, 1, 0.36, 1), background 0.3s ease;
    }
    .sb-brand-row[data-theme="dark"] .sb-switch-thumb {
      transform: translateX(26px);
      background: #22c55e;
    }
    """,
    js="""
    export default function (component) {
      const { parentElement, data } = component;
      const root = parentElement.querySelector("#sb-root");
      const logoImg = parentElement.querySelector("#sb-logo-img");
      const btn = parentElement.querySelector("#sb-theme-btn");
      if (!root || !btn) return;

      if (logoImg && data && data.logo_uri && logoImg.getAttribute("src") !== data.logo_uri) {
        logoImg.setAttribute("src", data.logo_uri);
      }

      // Pastikan style View Transition API & override tombol/input Dark Mode terpasang di document.head
      let vtStyle = document.getElementById("sr-vt-circle-style");
      if (!vtStyle) {
        vtStyle = document.createElement("style");
        vtStyle.id = "sr-vt-circle-style";
        document.head.appendChild(vtStyle);
      }
      vtStyle.textContent = `
        ::view-transition-old(root),
        ::view-transition-new(root) {
          animation: none !important;
          mix-blend-mode: normal !important;
        }
        ::view-transition-old(root) {
          z-index: 1 !important;
        }
        ::view-transition-new(root) {
          z-index: 9999 !important;
        }
      `;

      function applyThemeToDOM(theme) {
        document.documentElement.setAttribute("data-sr-theme", theme);
        root.setAttribute("data-theme", theme);
        try {
          localStorage.setItem("sortara_theme", theme);
        } catch (e) {}
        window.dispatchEvent(new CustomEvent("sortara-theme-change", { detail: { theme } }));
      }

      // Sinkronisasi tema awal dari localStorage atau atribut dokumen
      let currentTheme = document.documentElement.getAttribute("data-sr-theme");
      if (!currentTheme) {
        try {
          currentTheme = localStorage.getItem("sortara_theme") || "light";
        } catch (e) {
          currentTheme = "light";
        }
      }
      applyThemeToDOM(currentTheme);

      // Otomatis gulir kembali ke paling atas halaman setiap kali pengguna berpindah halaman
      function scrollPageToTop() {
        try {
          window.scrollTo({ top: 0, left: 0, behavior: "instant" });
        } catch (e) {
          window.scrollTo(0, 0);
        }
        if (document.documentElement) document.documentElement.scrollTop = 0;
        if (document.body) document.body.scrollTop = 0;
        const scrollTargets = document.querySelectorAll(
          '[data-testid="stMain"], [data-testid="stAppScrollToBottomContainer"], [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"], main.stMain, .main, .stApp'
        );
        scrollTargets.forEach((el) => {
          try {
            el.scrollTo({ top: 0, left: 0, behavior: "instant" });
          } catch (e) {}
          el.scrollTop = 0;
        });
      }

      if (data && data.active_page) {
        if (window.__sortaraActivePage && window.__sortaraActivePage !== data.active_page) {
          scrollPageToTop();
          requestAnimationFrame(() => {
            scrollPageToTop();
            setTimeout(scrollPageToTop, 50);
            setTimeout(scrollPageToTop, 150);
          });
        }
        window.__sortaraActivePage = data.active_page;
      }

      if (btn.dataset.bound === "1") return;
      btn.dataset.bound = "1";

      btn.onclick = () => {
        const active = document.documentElement.getAttribute("data-sr-theme") || "light";
        const nextTheme = active === "dark" ? "light" : "dark";

        const rect = btn.getBoundingClientRect();
        const cx = Math.round(rect.left + rect.width / 2);
        const cy = Math.round(rect.top + rect.height / 2);
        const maxRadius = Math.ceil(
          Math.hypot(
            Math.max(cx, window.innerWidth - cx),
            Math.max(cy, window.innerHeight - cy)
          )
        );

        // Gunakan View Transitions API dengan animasi circle membesar jika didukung browser
        if (document.startViewTransition) {
          try {
            const transition = document.startViewTransition(() => {
              applyThemeToDOM(nextTheme);
            });
            transition.ready.then(() => {
              document.documentElement.animate(
                {
                  clipPath: [
                    "circle(0px at " + cx + "px " + cy + "px)",
                    "circle(" + maxRadius + "px at " + cx + "px " + cy + "px)"
                  ]
                },
                {
                  duration: 680,
                  easing: "cubic-bezier(0.22, 1, 0.36, 1)",
                  pseudoElement: "::view-transition-new(root)"
                }
              );
            }).catch(() => {});
            return;
          } catch (e) {}
        }

        // Fallback circle membesar halus untuk browser mobile/lama tanpa View Transitions API
        const overlay = document.createElement("div");
        overlay.style.position = "fixed";
        overlay.style.inset = "0";
        overlay.style.zIndex = "999999";
        overlay.style.pointerEvents = "none";
        overlay.style.background = nextTheme === "dark" ? "#090a0f" : "#f7f6f2";
        overlay.style.clipPath = "circle(0px at " + cx + "px " + cy + "px)";
        overlay.style.transition = "clip-path 0.62s cubic-bezier(0.22, 1, 0.36, 1), opacity 0.25s ease";
        document.body.appendChild(overlay);

        requestAnimationFrame(() => {
          overlay.style.clipPath = "circle(" + maxRadius + "px at " + cx + "px " + cy + "px)";
        });

        setTimeout(() => {
          applyThemeToDOM(nextTheme);
          overlay.style.opacity = "0";
          setTimeout(() => {
            overlay.remove();
          }, 260);
        }, 420);
      };
    }
    """,
)


_SMART_CAMERA = st.components.v2.component(
    "sortara_smart_camera",
    html="""
    <div id="sc-root" class="sc-wrap">
      <input id="sc-mobile-rear" type="file" accept="image/*" capture="environment" style="display:none;" />

      <div class="sc-main-row">
        <button id="sc-open-btn" type="button" class="sc-btn-primary">
          Buka Kamera
        </button>
        <select id="sc-device-select" class="sc-select" style="display:none;" aria-label="Pilih Kamera"></select>
      </div>

      <div id="sc-status" class="sc-status" style="display:none;"></div>

      <div id="sc-stage" class="sc-stage" style="display:none;">
        <video id="sc-video" autoplay playsinline muted class="sc-video"></video>
        <img id="sc-preview" class="sc-preview" style="display:none;" alt="Pratinjau foto kamera" />
        <canvas id="sc-canvas" style="display:none;"></canvas>
        <div class="sc-actions">
          <button id="sc-snap-btn" type="button" class="sc-btn-snap">Ambil Foto</button>
          <button id="sc-retake-btn" type="button" class="sc-btn-secondary" style="display:none;">Ambil Ulang</button>
        </div>
      </div>
    </div>
    """,
    css="""
    .sc-wrap {
      font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
      margin: 4px 0 6px 0;
    }
    .sc-main-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }
    .sc-btn-primary {
      background: #111111;
      color: #ffffff;
      border: 1px solid #111111;
      border-radius: 11px;
      padding: 12px 20px;
      min-height: 44px;
      font-size: 0.90rem;
      font-weight: 600;
      cursor: pointer;
      width: 100%;
      transition: background 0.2s ease, color 0.2s ease, border-color 0.2s ease, opacity 0.15s ease;
    }
    .sc-wrap[data-theme="dark"] .sc-btn-primary {
      background: #22c55e;
      color: #052e16;
      border-color: #22c55e;
      font-weight: 700;
    }
    .sc-btn-primary:hover {
      opacity: 0.90;
    }
    .sc-btn-secondary {
      background: #f4f4f5;
      color: #18181b;
      border: 1px solid #d4d4d8;
      border-radius: 10px;
      padding: 10px 18px;
      min-height: 42px;
      font-size: 0.84rem;
      font-weight: 600;
      cursor: pointer;
    }
    .sc-wrap[data-theme="dark"] .sc-btn-secondary {
      background: #1e2230;
      color: #f4f4f5;
      border-color: #2e3346;
    }
    .sc-select {
      background: #ffffff;
      color: #18181b;
      border: 1px solid #d4d4d8;
      border-radius: 10px;
      padding: 11px 12px;
      font-size: 0.84rem;
      font-weight: 500;
      width: 100%;
    }
    .sc-wrap[data-theme="dark"] .sc-select {
      background: #181b26;
      color: #f4f4f5;
      border-color: #2e3346;
    }
    .sc-status {
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 10px;
      padding: 9px 13px;
      font-size: 0.82rem;
      color: #3f3f46;
      line-height: 1.45;
      margin-top: 10px;
    }
    .sc-status.ok {
      background: #f0fdf4;
      border-color: #bbf7d0;
      color: #14532d;
    }
    .sc-status.err {
      background: #fef2f2;
      border-color: #fecaca;
      color: #7f1d1d;
    }
    .sc-wrap[data-theme="dark"] .sc-status.ok {
      background: #0c2918;
      border-color: #166534;
      color: #86efac;
    }
    .sc-wrap[data-theme="dark"] .sc-status.err {
      background: #2c1215;
      border-color: #7f1d1d;
      color: #fca5a5;
    }
    .sc-stage {
      background: #0b0b0c;
      border-radius: 14px;
      padding: 14px;
      text-align: center;
      border: 1px solid #27272a;
      margin-top: 12px;
    }
    .sc-video, .sc-preview {
      width: 100%;
      max-height: 340px;
      border-radius: 10px;
      object-fit: contain;
      background: #000000;
      display: block;
      margin: 0 auto 12px auto;
    }
    .sc-actions {
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      gap: 10px;
    }
    .sc-btn-snap {
      background: #16a34a;
      color: #ffffff;
      border: none;
      border-radius: 10px;
      padding: 11px 24px;
      min-height: 44px;
      font-size: 0.90rem;
      font-weight: 700;
      cursor: pointer;
      width: 100%;
      max-width: 320px;
    }
    .sc-btn-snap:hover {
      background: #15803d;
    }
    """,
    js="""
    export default function (component) {
      const { parentElement, setStateValue } = component;
      const root = parentElement.querySelector("#sc-root");
      const openBtn = parentElement.querySelector("#sc-open-btn");
      const rearInput = parentElement.querySelector("#sc-mobile-rear");
      const devSelect = parentElement.querySelector("#sc-device-select");
      const statusBox = parentElement.querySelector("#sc-status");
      const stage = parentElement.querySelector("#sc-stage");
      const video = parentElement.querySelector("#sc-video");
      const preview = parentElement.querySelector("#sc-preview");
      const canvas = parentElement.querySelector("#sc-canvas");
      const snapBtn = parentElement.querySelector("#sc-snap-btn");
      const retakeBtn = parentElement.querySelector("#sc-retake-btn");

      if (!openBtn || !statusBox || !root) return;

      function syncTheme() {
        const t = document.documentElement.getAttribute("data-sr-theme") || "light";
        root.setAttribute("data-theme", t);
      }
      syncTheme();

      if (openBtn.dataset.bound === "1") return;
      openBtn.dataset.bound = "1";

      window.addEventListener("sortara-theme-change", syncTheme);

      const isMobileUA = /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent || "");
      let activeStream = null;

      function stopStream() {
        if (activeStream) {
          activeStream.getTracks().forEach(t => t.stop());
          activeStream = null;
        }
      }

      function showStatus(text, kind) {
        statusBox.style.display = "block";
        statusBox.className = kind ? ("sc-status " + kind) : "sc-status";
        statusBox.textContent = text;
      }

      function handleCapturedFile(file) {
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (ev) => {
          const img = new Image();
          img.onload = () => {
            const maxDim = 1024;
            let w = img.width;
            let h = img.height;
            if (w > maxDim || h > maxDim) {
              if (w >= h) {
                h = Math.round((h * maxDim) / w);
                w = maxDim;
              } else {
                w = Math.round((w * maxDim) / h);
                h = maxDim;
              }
            }
            canvas.width = w;
            canvas.height = h;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0, w, h);
            const dataUrl = canvas.toDataURL("image/jpeg", 0.90);
            stopStream();
            stage.style.display = "block";
            video.style.display = "none";
            preview.src = dataUrl;
            preview.style.display = "block";
            snapBtn.style.display = "none";
            retakeBtn.style.display = "inline-block";
            showStatus("Foto berhasil diambil. Silakan lanjutkan ke tombol Klasifikasikan sampah di bawah.", "ok");
            setStateValue("captured_jpeg", dataUrl);
          };
          img.src = ev.target.result;
        };
        reader.readAsDataURL(file);
      }

      rearInput.onchange = (e) => {
        if (e.target.files && e.target.files[0]) {
          handleCapturedFile(e.target.files[0]);
        }
      };

      async function startCameraWithDevice(deviceId) {
        stopStream();
        const constraints = deviceId
          ? { video: { deviceId: { exact: deviceId } }, audio: false }
          : { video: { facingMode: isMobileUA ? "environment" : "user" }, audio: false };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        activeStream = stream;
        video.srcObject = stream;
        video.style.display = "block";
        preview.style.display = "none";
        snapBtn.style.display = "inline-block";
        retakeBtn.style.display = "none";
        stage.style.display = "block";
        statusBox.style.display = "none";
        await video.play().catch(() => {});
      }

      async function populateCameraDevices() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return null;
        const devices = await navigator.mediaDevices.enumerateDevices();
        const cams = devices.filter(d => d.kind === "videoinput");
        if (cams.length === 0) return null;

        devSelect.innerHTML = "";
        let preferredId = cams[0].deviceId;
        cams.forEach((cam, idx) => {
          const opt = document.createElement("option");
          opt.value = cam.deviceId;
          const label = cam.label || ("Kamera " + (idx + 1));
          opt.textContent = label;
          devSelect.appendChild(opt);
          const low = label.toLowerCase();
          if (low.includes("usb") || low.includes("webcam") || low.includes("integrated") || low.includes("back") || low.includes("rear")) {
            if (!low.includes("obs") && !low.includes("virtual") && !low.includes("vtube")) {
              preferredId = cam.deviceId;
            }
          }
        });
        devSelect.value = preferredId;
        devSelect.style.display = cams.length > 1 ? "inline-block" : "none";
        return preferredId;
      }

      devSelect.onchange = async () => {
        try {
          await startCameraWithDevice(devSelect.value);
        } catch (err) {
          showStatus("Tidak dapat beralih ke kamera tersebut.", "err");
        }
      };

      openBtn.onclick = async () => {
        stopStream();
        if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          rearInput.value = "";
          rearInput.click();
          return;
        }

        try {
          await startCameraWithDevice(null);
          const prefId = await populateCameraDevices();
          if (prefId && activeStream) {
            const currentTrack = activeStream.getVideoTracks()[0];
            const curLabel = (currentTrack && currentTrack.label ? currentTrack.label : "").toLowerCase();
            if ((curLabel.includes("obs") || curLabel.includes("virtual") || curLabel.includes("vtube")) && devSelect.options.length > 1) {
              await startCameraWithDevice(prefId);
            }
          }
        } catch (err) {
          if (isMobileUA) {
            rearInput.value = "";
            rearInput.click();
            return;
          }
          showStatus("Pastikan izin akses kamera telah diaktifkan pada pengaturan browser Anda, lalu klik Buka Kamera kembali.", "err");
        }
      };

      snapBtn.onclick = () => {
        if (!video.videoWidth || !video.videoHeight) return;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
        preview.src = dataUrl;
        video.style.display = "none";
        preview.style.display = "block";
        snapBtn.style.display = "none";
        retakeBtn.style.display = "inline-block";
        stopStream();
        showStatus("Foto berhasil diambil. Silakan lanjutkan ke tombol Klasifikasikan sampah di bawah.", "ok");
        setStateValue("captured_jpeg", dataUrl);
      };

      retakeBtn.onclick = () => {
        openBtn.click();
      };
    }
    """,
)




def on_upload_change() -> None:
    reset_classification_state()
    if st.session_state.get("waste_uploader") is not None:
        st.session_state["active_source"] = "upload"


def on_sample_change() -> None:
    reset_classification_state()
    st.session_state["active_source"] = "sample"


predictor = get_waste_predictor(str(DEFAULT_MODEL_PATH), str(DEFAULT_LABELS_PATH))

# Styling CSS khusus mengikuti seluruh referensi tata letak pengguna
st.html(
    """
<style>
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1160px !important;
}

div[data-testid="stHeadingWithActionElements"] h1 {
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #71717a !important;
    margin-bottom: 0.25rem !important;
}

/* === REFERENSI 4: SIDEBAR MINIMALIS (ACME STUDIO STYLE) === */
.sr-side-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 4px 4px 16px 4px;
    border-bottom: 1px solid #e6e6e2;
    margin-bottom: 12px;
}
.sr-side-logo {
    width: 38px;
    height: 38px;
    border-radius: 9px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 6px rgba(11, 51, 48, 0.18);
    flex-shrink: 0;
}
.sr-side-logo img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}
.sr-side-title {
    font-weight: 800;
    font-size: 1.04rem;
    letter-spacing: -0.01em;
    color: #111111;
}
.sr-side-section {
    font-size: 0.70rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #8e8e93;
    margin: 16px 0 6px 4px;
}

/* Tombol navigasi sidebar bergaya Acme Studio */
section[data-testid="stSidebar"] .stButton button {
    justify-content: flex-start !important;
    text-align: left !important;
    border-radius: 10px !important;
    font-size: 0.88rem !important;
    padding: 8px 12px !important;
    border: 1px solid transparent !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] .stButton button[kind="secondary"] {
    background: transparent !important;
    color: #3a3a3c !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {
    background: #f1f1ef !important;
    color: #111111 !important;
}
section[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background: #f1f1ef !important;
    color: #111111 !important;
    font-weight: 700 !important;
    border-left: 3.5px solid #111111 !important;
    border-top-left-radius: 4px !important;
    border-bottom-left-radius: 4px !important;
}

.sr-nav-stat-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 7px 12px;
    font-size: 0.84rem;
    color: #52525b;
}
.sr-nav-badge {
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 999px;
    background: #ececea;
    color: #3a3a3c;
}
.sr-nav-badge.alert {
    background: #fde8e8;
    color: #c81e1e;
}
.sr-nav-badge.green {
    background: #def7ec;
    color: #03543f;
}
.sr-side-footer {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 20px;
    padding-top: 14px;
    border-top: 1px solid #e6e6e2;
}
.sr-avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: #e5e5ea;
    color: #1c1c1e;
    font-weight: 700;
    font-size: 0.78rem;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* === REFERENSI EDITORIAL BARU (HALAMAN CAKUPAN 6 KATEGORI & HALAMAN CONFUSION MATRIX) === */
.sr-ed-hero {
    padding: 18px 4px 30px 4px;
}
.sr-ed-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #ffffff;
    border: 1px solid #e4e4e0;
    border-radius: 999px;
    padding: 6px 15px;
    font-size: 0.80rem;
    font-weight: 500;
    color: #27272a;
    margin-bottom: 22px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
}
.sr-ed-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #16a34a;
    display: inline-block;
}
.sr-ed-headline {
    font-size: 3.15rem;
    font-weight: 700;
    line-height: 1.08;
    letter-spacing: -0.035em;
    color: #111111;
    max-width: 820px;
    margin: 0 0 20px 0;
}
.sr-ed-lead {
    font-size: 1.08rem;
    line-height: 1.6;
    color: #52525b;
    max-width: 650px;
    margin: 0 0 26px 0;
}
.sr-ed-cta-row {
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 14px;
}
.sr-ed-btn-black {
    display: inline-block;
    background: #111111;
    color: #ffffff;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 11px 22px;
    border-radius: 12px;
}
.sr-ed-link {
    font-size: 0.90rem;
    font-weight: 500;
    color: #18181b;
    text-decoration: underline;
    text-underline-offset: 4px;
}
.sr-ed-grid-3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
    margin-top: 12px;
    margin-bottom: 18px;
}
.sr-ed-grid-2 {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 18px;
    margin-top: 12px;
    margin-bottom: 22px;
}
@media (max-width: 960px) {
    .sr-ed-grid-3,
    .sr-ed-grid-2 {
        grid-template-columns: 1fr;
    }
    .sr-ed-headline {
        font-size: 2.25rem;
    }
}
.sr-ed-card {
    background: #ffffff;
    border: 1px solid #e5e5e0;
    border-radius: 18px;
    padding: 12px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: border-color 0.15s ease;
}
.sr-ed-card:hover {
    border-color: #c4c4be;
}
.sr-ed-thumb {
    background: #efeee9;
    border-radius: 12px;
    position: relative;
    height: 195px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 16px;
    margin-bottom: 14px;
    overflow: hidden;
}
.sr-ed-thumb.tall {
    height: 310px;
    background: #f5f5f2;
}
.sr-ed-thumb img {
    max-height: 100%;
    max-width: 100%;
    object-fit: contain;
}
.sr-ed-year {
    position: absolute;
    top: 12px;
    right: 14px;
    font-size: 0.72rem;
    font-weight: 600;
    color: #8e8e93;
    letter-spacing: 0.03em;
}
.sr-ed-body {
    padding: 2px 6px 6px 6px;
}
.sr-ed-card-title {
    font-size: 1.02rem;
    font-weight: 700;
    color: #111111;
    margin: 0 0 3px 0;
}
.sr-ed-card-sub {
    font-size: 0.86rem;
    color: #71717a;
    margin: 0 0 10px 0;
    line-height: 1.45;
}
.sr-ed-card-detail {
    font-size: 0.80rem;
    color: #52525b;
    background: #f8f8f6;
    border-radius: 9px;
    padding: 9px 11px;
    margin-bottom: 12px;
    line-height: 1.45;
}
.sr-ed-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}
.sr-ed-tag {
    background: #f1f1ed;
    color: #3f3f46;
    font-size: 0.73rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 7px;
}
.sr-ed-foot-note {
    font-size: 0.80rem;
    color: #8e8e93;
    margin-top: 8px;
}

/* === REFERENSI 2: HERO BANNER GELAP DENGAN GRADIENT HEADLINE === */
.sr-hero {
    background: radial-gradient(circle at 50% 22%, #231d47 0%, #0d0c18 52%, #07070b 100%);
    border-radius: 22px;
    padding: 48px 32px 42px 32px;
    text-align: center;
    color: #ffffff;
    margin-bottom: 30px;
    border: 1px solid #1f1f2e;
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.14);
}
.sr-hero-pill {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 999px;
    font-size: 0.80rem;
    font-weight: 500;
    color: #d4d4d8;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.14);
    margin-bottom: 18px;
}
.sr-hero-title {
    font-size: 3.0rem;
    font-weight: 800;
    line-height: 1.12;
    letter-spacing: -0.03em;
    margin: 0 auto 14px auto;
    color: #ffffff;
}
.sr-hero-gradient {
    background: linear-gradient(90deg, #818cf8 0%, #c084fc 52%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sr-hero-sub {
    font-size: 1.0rem;
    color: #a1a1aa;
    max-width: 620px;
    margin: 0 auto 22px auto;
    line-height: 1.55;
}
.sr-hero-cta {
    display: inline-block;
    background: #ffffff;
    color: #09090b;
    font-weight: 700;
    font-size: 0.86rem;
    padding: 10px 22px;
    border-radius: 12px;
    box-shadow: 0 0 25px rgba(129, 140, 248, 0.35);
}

/* === REFERENSI STEP 1 OF 3: 1 KARTU UTAMA KAMERA + SPLIT 2 KARTU DI BAWAHNYA === */
.sr-step-header-wrap {
    max-width: 780px;
    margin: 8px auto 18px auto;
}
.sr-step-eyebrow {
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #8e8e93;
    margin-bottom: 6px;
}
.sr-step-main-title {
    font-size: 1.85rem;
    font-weight: 700;
    color: #111111;
    letter-spacing: -0.02em;
    margin: 0 0 4px 0;
}
.sr-step-main-sub {
    font-size: 0.94rem;
    color: #71717a;
    margin: 0;
}

.st-key-card_camera_primary {
    background: #ffffff !important;
    border: 1.5px solid #18181b !important;
    border-radius: 18px !important;
    padding: 24px 26px !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.04) !important;
    margin-bottom: 6px !important;
}
.st-key-card_upload_left,
.st-key-card_tester_right {
    background: #ffffff !important;
    border: 1px solid #e4e4e7 !important;
    border-radius: 18px !important;
    padding: 22px 24px !important;
    height: 100% !important;
}
.sr-card-head-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
}
.sr-card-title {
    font-size: 1.02rem;
    font-weight: 700;
    color: #111111;
}
.sr-badge-fastest {
    background: #111111;
    color: #ffffff;
    font-size: 0.73rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 999px;
}
.sr-card-foot {
    font-size: 0.80rem;
    color: #71717a;
    line-height: 1.45;
    margin-top: 8px;
}
.sr-or-line {
    display: flex;
    align-items: center;
    text-align: center;
    margin: 18px 0;
    color: #a1a1aa;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
}
.sr-or-line::before,
.sr-or-line::after {
    content: "";
    flex: 1;
    border-bottom: 1px solid #e5e5e0;
}
.sr-or-line:not(:empty)::before {
    margin-right: 14px;
}
.sr-or-line:not(:empty)::after {
    margin-left: 14px;
}
.sr-trust-strip {
    border-top: 1px solid #e5e5e0;
    margin-top: 22px;
    padding-top: 16px;
    margin-bottom: 26px;
}
.sr-trust-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    font-size: 0.80rem;
    color: #52525b;
    margin-bottom: 12px;
}
@media (max-width: 768px) {
    .sr-trust-grid {
        grid-template-columns: 1fr;
    }
}
.sr-trust-legal {
    text-align: center;
    font-size: 0.76rem;
    color: #71717a;
}

/* === REFERENSI 1: WORKSPACE GELAP (KARTU PUTIH DI KIRI + 3 STEP DI KANAN) === */
.sr-work-shell {
    background: #0b0b0c;
    border-radius: 24px;
    padding: 32px 32px 28px 32px;
    color: #ffffff;
    border: 1px solid #1f1f22;
    margin-bottom: 22px;
}
.sr-work-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
}
.sr-pill-green {
    display: inline-block;
    background: #0d2818;
    color: #4ade80;
    border: 1px solid #164e2e;
    padding: 4px 13px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
}
.sr-pill-mode {
    display: inline-block;
    background: #141416;
    color: #e4e4e7;
    border: 1px solid #27272a;
    padding: 6px 14px;
    border-radius: 10px;
    font-size: 0.82rem;
    font-weight: 500;
}
.sr-work-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 6px 0;
    letter-spacing: -0.02em;
}
.sr-work-sub {
    font-size: 0.94rem;
    color: #a1a1aa;
    margin: 0 0 24px 0;
}
.sr-work-grid {
    display: grid;
    grid-template-columns: 1.08fr 1fr;
    gap: 20px;
    align-items: stretch;
}
@media (max-width: 900px) {
    .sr-work-grid {
        grid-template-columns: 1fr;
    }
}
.sr-white-card {
    background: #ffffff;
    color: #111111;
    border-radius: 18px;
    padding: 22px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.sr-white-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 14px;
}
.sr-file-name {
    font-size: 0.84rem;
    color: #71717a;
    font-weight: 500;
}
.sr-sync-status {
    font-size: 0.83rem;
    color: #16a34a;
    font-weight: 600;
    margin-top: 2px;
}
.sr-black-chip {
    background: #111111;
    color: #ffffff;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 7px 14px;
    border-radius: 9px;
}
.sr-bar-dark {
    height: 11px;
    width: 66%;
    background: #18181b;
    border-radius: 99px;
    margin-bottom: 9px;
}
.sr-bar-light {
    height: 7px;
    width: 92%;
    background: #f1f1f4;
    border-radius: 99px;
    margin-bottom: 7px;
}
.sr-preview-well {
    background: #f3f1ff;
    border-radius: 13px;
    padding: 12px;
    margin-top: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 195px;
    overflow: hidden;
}
.sr-preview-well img {
    max-height: 210px;
    width: auto;
    max-width: 100%;
    border-radius: 10px;
    object-fit: contain;
}
.sr-step-stack {
    display: flex;
    flex-direction: column;
    gap: 12px;
    justify-content: space-between;
}
.sr-step-card {
    background: #111113;
    border: 1px solid #232329;
    border-radius: 15px;
    padding: 16px 18px;
}
.sr-step-card.emerald {
    background: #0c2318;
    border: 1px solid #1d4d32;
}
.sr-step-card.amber {
    background: #271c09;
    border: 1px solid #6b4c1b;
}
.sr-step-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 4px;
}
.sr-step-title {
    font-size: 0.94rem;
    font-weight: 700;
    color: #ffffff;
}
.sr-step-meta {
    font-size: 0.78rem;
    color: #a1a1aa;
}
.sr-step-desc {
    font-size: 0.84rem;
    color: #9ca3af;
    line-height: 1.45;
}

/* === REFERENSI 3: KARTU BENTO 4 METRIK HORIZONTAL (TIMER STYLE) === */
.sr-bento-shell {
    background: #0e0e10;
    border-radius: 24px;
    padding: 38px 28px 32px 28px;
    text-align: center;
    color: #ffffff;
    border: 1px solid #222226;
    margin: 22px 0;
}
.sr-bento-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #18181b;
    border: 1px solid #27272a;
    color: #e4e4e7;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 600;
    margin-bottom: 14px;
}
.sr-dot-yellow {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #eab308;
    display: inline-block;
}
.sr-bento-heading {
    font-size: 1.85rem;
    font-weight: 700;
    color: #ffffff;
    max-width: 560px;
    margin: 0 auto 24px auto;
    line-height: 1.22;
}
.sr-bento-boxes {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    max-width: 760px;
    margin: 0 auto 22px auto;
}
@media (max-width: 768px) {
    .sr-bento-boxes {
        grid-template-columns: repeat(2, 1fr);
    }
}
.sr-stat-tile {
    background: #18181b;
    border: 1px solid #26262b;
    border-radius: 15px;
    padding: 18px 12px;
}
.sr-stat-val {
    font-size: 1.45rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 4px;
}
.sr-stat-lbl {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #71717a;
}
.sr-bento-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    max-width: 660px;
    margin: 0 auto 10px auto;
    background: #161619;
    border: 1px solid #27272a;
    border-radius: 12px;
    padding: 10px 14px;
}
.sr-bento-bar-text {
    font-size: 0.84rem;
    color: #a1a1aa;
    text-align: left;
}
.sr-bento-bar-btn {
    background: #ffffff;
    color: #111111;
    font-size: 0.80rem;
    font-weight: 700;
    padding: 8px 16px;
    border-radius: 9px;
    white-space: nowrap;
}
.sr-bento-foot {
    font-size: 0.76rem;
    color: #71717a;
}

/* === REFERENSI 5: SPLIT CARD 50/50 (KIRI PUTIH FORM/TINDAKAN — KANAN HITAM QUOTE DAMPAK) === */
.sr-split-shell {
    display: grid;
    grid-template-columns: 1fr 1fr;
    border-radius: 22px;
    overflow: hidden;
    border: 1px solid #e4e4e0;
    margin-top: 22px;
    background: #ffffff;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
}
@media (max-width: 900px) {
    .sr-split-shell {
        grid-template-columns: 1fr;
    }
}
.sr-split-left {
    background: #ffffff;
    color: #111111;
    padding: 36px 34px;
}
.sr-split-logo {
    width: 38px;
    height: 38px;
    border-radius: 9px;
    overflow: hidden;
    margin-bottom: 20px;
    box-shadow: 0 2px 6px rgba(11, 51, 48, 0.18);
}
.sr-split-logo img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}
.sr-split-title {
    font-size: 1.45rem;
    font-weight: 700;
    color: #111111;
    margin: 0 0 4px 0;
}
.sr-split-sub {
    font-size: 0.86rem;
    color: #71717a;
    margin-bottom: 20px;
}
.sr-field-lbl {
    font-size: 0.78rem;
    font-weight: 600;
    color: #27272a;
    margin-bottom: 6px;
}
.sr-field-box {
    border: 1px solid #e4e4e7;
    border-radius: 10px;
    padding: 11px 14px;
    font-size: 0.85rem;
    color: #27272a;
    background: #fafafa;
    margin-bottom: 14px;
    line-height: 1.5;
}
.sr-btn-black-full {
    display: block;
    width: 100%;
    text-align: center;
    background: #111111;
    color: #ffffff;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 11px 16px;
    border-radius: 10px;
    margin: 14px 0 12px 0;
}
.sr-or-divider {
    text-align: center;
    font-size: 0.72rem;
    color: #a1a1aa;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 10px 0;
}
.sr-btn-outline-full {
    display: block;
    width: 100%;
    text-align: center;
    background: #ffffff;
    color: #111111;
    border: 1px solid #e4e4e7;
    font-weight: 600;
    font-size: 0.84rem;
    padding: 10px 16px;
    border-radius: 10px;
}
.sr-split-right {
    background: #0b0b0c;
    color: #ffffff;
    padding: 40px 36px;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    min-height: 380px;
}
.sr-quote-text {
    font-size: 1.32rem;
    font-weight: 700;
    line-height: 1.38;
    color: #ffffff;
    margin-bottom: 14px;
    letter-spacing: -0.01em;
}
.sr-quote-author {
    font-size: 0.83rem;
    color: #a1a1aa;
}
.sr-side-footer-name {
    font-weight: 700;
    font-size: 0.86rem;
    color: #111111;
}
.sr-side-footer-sub {
    font-size: 0.75rem;
    color: #71717a;
}

/* ============================================================================
   GAYA TOMBOL & HOVER KONSISTEN DI MODE TERANG (DEFAULT)
   ============================================================================ */
[data-testid="stMain"] .stButton button[kind="primary"],
[data-testid="stMain"] button[data-testid="stBaseButton-primary"] {
    background-color: #111111 !important;
    color: #ffffff !important;
    border: 1px solid #111111 !important;
    font-weight: 700 !important;
    transition: background-color 0.18s ease, border-color 0.18s ease, color 0.18s ease !important;
}
[data-testid="stMain"] .stButton button[kind="primary"] *,
[data-testid="stMain"] button[data-testid="stBaseButton-primary"] * {
    color: #ffffff !important;
}
[data-testid="stMain"] .stButton button[kind="primary"]:hover,
[data-testid="stMain"] .stButton button[kind="primary"]:focus,
[data-testid="stMain"] .stButton button[kind="primary"]:active,
[data-testid="stMain"] button[data-testid="stBaseButton-primary"]:hover,
[data-testid="stMain"] button[data-testid="stBaseButton-primary"]:focus {
    background-color: #16a34a !important;
    border-color: #15803d !important;
    color: #ffffff !important;
}
[data-testid="stMain"] .stButton button[kind="primary"]:hover *,
[data-testid="stMain"] .stButton button[kind="primary"]:focus *,
[data-testid="stMain"] button[data-testid="stBaseButton-primary"]:hover * {
    color: #ffffff !important;
}
[data-testid="stMain"] .stButton button[kind="primary"]:disabled,
[data-testid="stMain"] button[data-testid="stBaseButton-primary"]:disabled {
    background-color: #e4e4e7 !important;
    border-color: #d4d4d8 !important;
    color: #71717a !important;
    opacity: 1 !important;
}
[data-testid="stMain"] .stButton button[kind="primary"]:disabled *,
[data-testid="stMain"] button[data-testid="stBaseButton-primary"]:disabled * {
    color: #71717a !important;
}

[data-testid="stMain"] .stButton button[kind="secondary"],
[data-testid="stMain"] button[data-testid="stBaseButton-secondary"],
[data-testid="stFileUploaderDropzone"] button {
    background-color: #ffffff !important;
    color: #111111 !important;
    border: 1px solid #d4d4d8 !important;
    font-weight: 600 !important;
    transition: background-color 0.18s ease, border-color 0.18s ease, color 0.18s ease !important;
}
[data-testid="stMain"] .stButton button[kind="secondary"] *,
[data-testid="stMain"] button[data-testid="stBaseButton-secondary"] *,
[data-testid="stFileUploaderDropzone"] button * {
    color: #111111 !important;
}
[data-testid="stMain"] .stButton button[kind="secondary"]:hover,
[data-testid="stMain"] .stButton button[kind="secondary"]:focus,
[data-testid="stMain"] .stButton button[kind="secondary"]:active,
[data-testid="stMain"] button[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploaderDropzone"] button:focus {
    background-color: #f4f4f5 !important;
    border-color: #18181b !important;
    color: #111111 !important;
}
[data-testid="stMain"] .stButton button[kind="secondary"]:hover *,
[data-testid="stMain"] .stButton button[kind="secondary"]:focus *,
[data-testid="stFileUploaderDropzone"] button:hover * {
    color: #111111 !important;
}

/* ============================================================================
   FITUR DARK MODE & ANIMASI TRANSISI EXPANDING CIRCLE (VIEW TRANSITION API)
   ============================================================================ */
::view-transition-old(root),
::view-transition-new(root) {
    animation: none;
    mix-blend-mode: normal;
}
::view-transition-old(root) {
    z-index: 1;
}
::view-transition-new(root) {
    z-index: 9999;
}

:root[data-sr-theme="dark"],
:root[data-sr-theme="dark"] body,
:root[data-sr-theme="dark"] .stApp {
    background-color: #09090b !important;
    color: #f4f4f5 !important;
}
:root[data-sr-theme="dark"] header[data-testid="stHeader"] {
    background-color: rgba(9, 9, 11, 0.82) !important;
}
:root[data-sr-theme="dark"] section[data-testid="stSidebar"] {
    background-color: #111114 !important;
    border-right: 1px solid #232329 !important;
}
:root[data-sr-theme="dark"] section[data-testid="stSidebar"] * {
    border-color: #27272a;
}
:root[data-sr-theme="dark"] .sr-side-brand,
:root[data-sr-theme="dark"] .sr-side-footer {
    border-color: #27272a !important;
}
:root[data-sr-theme="dark"] .sr-side-title,
:root[data-sr-theme="dark"] .sr-side-footer-name {
    color: #f4f4f5 !important;
}
:root[data-sr-theme="dark"] .sr-side-footer-sub,
:root[data-sr-theme="dark"] .sr-side-section {
    color: #a1a1aa !important;
}
:root[data-sr-theme="dark"] .sr-avatar {
    background: #27272a !important;
    color: #4ade80 !important;
}

/* Tombol Sidebar di Mode Gelap + Hover Kontras Tinggi */
:root[data-sr-theme="dark"] section[data-testid="stSidebar"] .stButton button[kind="secondary"] {
    background-color: transparent !important;
    color: #d4d4d8 !important;
    border: 1px solid transparent !important;
}
:root[data-sr-theme="dark"] section[data-testid="stSidebar"] .stButton button[kind="secondary"] * {
    color: #d4d4d8 !important;
}
:root[data-sr-theme="dark"] section[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover,
:root[data-sr-theme="dark"] section[data-testid="stSidebar"] .stButton button[kind="secondary"]:focus,
:root[data-sr-theme="dark"] section[data-testid="stSidebar"] .stButton button[kind="secondary"]:active {
    background-color: #1f222a !important;
    color: #ffffff !important;
    border-color: #323644 !important;
}
:root[data-sr-theme="dark"] section[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover *,
:root[data-sr-theme="dark"] section[data-testid="stSidebar"] .stButton button[kind="secondary"]:focus *,
:root[data-sr-theme="dark"] section[data-testid="stSidebar"] .stButton button[kind="secondary"]:active * {
    color: #ffffff !important;
}
:root[data-sr-theme="dark"] section[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background-color: #162a1f !important;
    color: #4ade80 !important;
    border-left: 3.5px solid #4ade80 !important;
}
:root[data-sr-theme="dark"] section[data-testid="stSidebar"] .stButton button[kind="primary"] * {
    color: #4ade80 !important;
}
:root[data-sr-theme="dark"] section[data-testid="stSidebar"] .stButton button[kind="primary"]:hover,
:root[data-sr-theme="dark"] section[data-testid="stSidebar"] .stButton button[kind="primary"]:focus {
    background-color: #1d3829 !important;
    color: #86efac !important;
    border-left: 3.5px solid #86efac !important;
}
:root[data-sr-theme="dark"] section[data-testid="stSidebar"] .stButton button[kind="primary"]:hover *,
:root[data-sr-theme="dark"] section[data-testid="stSidebar"] .stButton button[kind="primary"]:focus * {
    color: #86efac !important;
}

/* Tombol Area Utama (Primary, Secondary & Tombol Upload) di Mode Gelap + Hover Kontras Tinggi */
:root[data-sr-theme="dark"] [data-testid="stMain"] .stButton button[kind="primary"],
:root[data-sr-theme="dark"] [data-testid="stMain"] button[data-testid="stBaseButton-primary"] {
    background-color: #22c55e !important;
    color: #052e16 !important;
    border: 1px solid #16a34a !important;
    font-weight: 700 !important;
}
:root[data-sr-theme="dark"] [data-testid="stMain"] .stButton button[kind="primary"] *,
:root[data-sr-theme="dark"] [data-testid="stMain"] button[data-testid="stBaseButton-primary"] * {
    color: #052e16 !important;
}
:root[data-sr-theme="dark"] [data-testid="stMain"] .stButton button[kind="primary"]:hover,
:root[data-sr-theme="dark"] [data-testid="stMain"] .stButton button[kind="primary"]:focus,
:root[data-sr-theme="dark"] [data-testid="stMain"] .stButton button[kind="primary"]:active,
:root[data-sr-theme="dark"] [data-testid="stMain"] button[data-testid="stBaseButton-primary"]:hover,
:root[data-sr-theme="dark"] [data-testid="stMain"] button[data-testid="stBaseButton-primary"]:focus {
    background-color: #4ade80 !important;
    border-color: #86efac !important;
    color: #052e16 !important;
    box-shadow: 0 4px 16px rgba(74, 222, 128, 0.25) !important;
}
:root[data-sr-theme="dark"] [data-testid="stMain"] .stButton button[kind="primary"]:hover *,
:root[data-sr-theme="dark"] [data-testid="stMain"] .stButton button[kind="primary"]:focus *,
:root[data-sr-theme="dark"] [data-testid="stMain"] button[data-testid="stBaseButton-primary"]:hover * {
    color: #052e16 !important;
}
:root[data-sr-theme="dark"] [data-testid="stMain"] .stButton button[kind="primary"]:disabled,
:root[data-sr-theme="dark"] [data-testid="stMain"] button[data-testid="stBaseButton-primary"]:disabled {
    background-color: #18181f !important;
    border-color: #2e2e38 !important;
    color: #71717a !important;
    opacity: 1 !important;
}
:root[data-sr-theme="dark"] [data-testid="stMain"] .stButton button[kind="primary"]:disabled *,
:root[data-sr-theme="dark"] [data-testid="stMain"] button[data-testid="stBaseButton-primary"]:disabled * {
    color: #71717a !important;
}

:root[data-sr-theme="dark"] [data-testid="stMain"] .stButton button[kind="secondary"],
:root[data-sr-theme="dark"] [data-testid="stMain"] button[data-testid="stBaseButton-secondary"],
:root[data-sr-theme="dark"] [data-testid="stFileUploaderDropzone"] button {
    background-color: #1e2028 !important;
    color: #f4f4f5 !important;
    border: 1px solid #3f3f46 !important;
    font-weight: 600 !important;
}
:root[data-sr-theme="dark"] [data-testid="stMain"] .stButton button[kind="secondary"] *,
:root[data-sr-theme="dark"] [data-testid="stMain"] button[data-testid="stBaseButton-secondary"] *,
:root[data-sr-theme="dark"] [data-testid="stFileUploaderDropzone"] button * {
    color: #f4f4f5 !important;
}
:root[data-sr-theme="dark"] [data-testid="stMain"] .stButton button[kind="secondary"]:hover,
:root[data-sr-theme="dark"] [data-testid="stMain"] .stButton button[kind="secondary"]:focus,
:root[data-sr-theme="dark"] [data-testid="stMain"] .stButton button[kind="secondary"]:active,
:root[data-sr-theme="dark"] [data-testid="stMain"] button[data-testid="stBaseButton-secondary"]:hover,
:root[data-sr-theme="dark"] [data-testid="stMain"] button[data-testid="stBaseButton-secondary"]:focus,
:root[data-sr-theme="dark"] [data-testid="stFileUploaderDropzone"] button:hover,
:root[data-sr-theme="dark"] [data-testid="stFileUploaderDropzone"] button:focus,
:root[data-sr-theme="dark"] [data-testid="stFileUploaderDropzone"] button:active {
    background-color: #292d38 !important;
    border-color: #4ade80 !important;
    color: #4ade80 !important;
}
:root[data-sr-theme="dark"] [data-testid="stMain"] .stButton button[kind="secondary"]:hover *,
:root[data-sr-theme="dark"] [data-testid="stMain"] .stButton button[kind="secondary"]:focus *,
:root[data-sr-theme="dark"] [data-testid="stMain"] .stButton button[kind="secondary"]:active *,
:root[data-sr-theme="dark"] [data-testid="stMain"] button[data-testid="stBaseButton-secondary"]:hover *,
:root[data-sr-theme="dark"] [data-testid="stFileUploaderDropzone"] button:hover *,
:root[data-sr-theme="dark"] [data-testid="stFileUploaderDropzone"] button:focus * {
    color: #4ade80 !important;
}
:root[data-sr-theme="dark"] [data-testid="stMain"] .stButton button[kind="secondary"]:disabled,
:root[data-sr-theme="dark"] [data-testid="stMain"] button[data-testid="stBaseButton-secondary"]:disabled {
    background-color: #141419 !important;
    border-color: #27272a !important;
    color: #52525b !important;
}
:root[data-sr-theme="dark"] [data-testid="stMain"] .stButton button[kind="secondary"]:disabled *,
:root[data-sr-theme="dark"] [data-testid="stMain"] button[data-testid="stBaseButton-secondary"]:disabled * {
    color: #52525b !important;
}

/* Dropzone File Uploader & Selectbox (React Aria Streamlit 1.66) di Mode Gelap */
:root[data-sr-theme="dark"] [data-testid="stFileUploaderDropzone"] {
    background-color: #16171d !important;
    border: 1.5px dashed #3f3f46 !important;
    color: #e4e4e7 !important;
}
:root[data-sr-theme="dark"] [data-testid="stFileUploaderDropzone"]:hover {
    border-color: #4ade80 !important;
    background-color: #1a1c24 !important;
}
:root[data-sr-theme="dark"] [data-testid="stFileUploaderDropzoneInstructions"],
:root[data-sr-theme="dark"] [data-testid="stFileUploaderDropzoneInstructions"] *,
:root[data-sr-theme="dark"] [data-testid="stFileUploaderDropzone"] small {
    color: #a1a1aa !important;
}
:root[data-sr-theme="dark"] [data-testid="stFileUploaderFile"],
:root[data-sr-theme="dark"] [data-testid="stFileUploaderFile"] * {
    color: #f4f4f5 !important;
}
:root[data-sr-theme="dark"] [data-testid="stSelectbox"] div:has(input),
:root[data-sr-theme="dark"] [data-testid="stSelectbox"] .e1oobldh0 {
    background-color: #1e2028 !important;
    border-color: #3f3f46 !important;
    color: #f4f4f5 !important;
}
:root[data-sr-theme="dark"] [data-testid="stSelectbox"] div:has(input):hover,
:root[data-sr-theme="dark"] [data-testid="stSelectbox"] .e1oobldh0:hover {
    border-color: #4ade80 !important;
}
:root[data-sr-theme="dark"] [data-testid="stSelectbox"] input,
:root[data-sr-theme="dark"] [data-testid="stSelectbox"] button,
:root[data-sr-theme="dark"] [data-testid="stSelectbox"] svg {
    color: #f4f4f5 !important;
}
:root[data-sr-theme="dark"] [data-testid="stSelectboxVirtualDropdown"],
:root[data-sr-theme="dark"] [data-testid="stSelectboxVirtualDropdown"] ul,
:root[data-sr-theme="dark"] [data-testid="stSelectboxVirtualDropdown"] [role="option"] {
    background-color: #181a22 !important;
    color: #f4f4f5 !important;
    border-color: #2e3240 !important;
}
:root[data-sr-theme="dark"] [data-testid="stSelectboxVirtualDropdown"] [role="option"][data-hovered] [data-item-hl],
:root[data-sr-theme="dark"] [data-testid="stSelectboxVirtualDropdown"] [role="option"][data-focused] [data-item-hl],
:root[data-sr-theme="dark"] [data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover [data-item-hl] {
    background-color: #272b38 !important;
    color: #4ade80 !important;
}

:root[data-sr-theme="dark"] div[data-testid="stHeadingWithActionElements"] h1 {
    color: #a1a1aa !important;
}

/* Kartu & Tipografi Halaman di Mode Gelap */
:root[data-sr-theme="dark"] .sr-ed-headline,
:root[data-sr-theme="dark"] .sr-step-main-title,
:root[data-sr-theme="dark"] .sr-card-title,
:root[data-sr-theme="dark"] .sr-ed-card-title,
:root[data-sr-theme="dark"] .sr-split-title {
    color: #f4f4f5 !important;
}
:root[data-sr-theme="dark"] .sr-ed-lead,
:root[data-sr-theme="dark"] .sr-step-main-sub,
:root[data-sr-theme="dark"] .sr-card-foot,
:root[data-sr-theme="dark"] .sr-ed-card-sub,
:root[data-sr-theme="dark"] .sr-split-sub,
:root[data-sr-theme="dark"] .sr-trust-grid,
:root[data-sr-theme="dark"] .sr-trust-legal {
    color: #a1a1aa !important;
}
:root[data-sr-theme="dark"] .sr-ed-pill {
    background: #141418 !important;
    border-color: #27272a !important;
    color: #e4e4e7 !important;
}
:root[data-sr-theme="dark"] .sr-ed-btn-black,
:root[data-sr-theme="dark"] .sr-badge-fastest,
:root[data-sr-theme="dark"] .sr-btn-black-full {
    background: #22c55e !important;
    color: #052e16 !important;
}
:root[data-sr-theme="dark"] .sr-ed-link {
    color: #4ade80 !important;
}
:root[data-sr-theme="dark"] .st-key-card_camera_primary {
    background: #121215 !important;
    border-color: #3f3f46 !important;
    box-shadow: 0 8px 26px rgba(0, 0, 0, 0.45) !important;
}
:root[data-sr-theme="dark"] .st-key-card_upload_left,
:root[data-sr-theme="dark"] .st-key-card_tester_right,
:root[data-sr-theme="dark"] .sr-ed-card {
    background: #121215 !important;
    border-color: #27272a !important;
}
:root[data-sr-theme="dark"] .sr-ed-card:hover {
    border-color: #52525b !important;
}
:root[data-sr-theme="dark"] .sr-ed-thumb {
    background: #1a1a1f !important;
}
:root[data-sr-theme="dark"] .sr-ed-thumb.tall {
    background: #18181c !important;
}
:root[data-sr-theme="dark"] .sr-ed-card-detail {
    background: #1a1a20 !important;
    color: #d4d4d8 !important;
}
:root[data-sr-theme="dark"] .sr-ed-tag {
    background: #232329 !important;
    color: #e4e4e7 !important;
}
:root[data-sr-theme="dark"] .sr-or-line::before,
:root[data-sr-theme="dark"] .sr-or-line::after,
:root[data-sr-theme="dark"] .sr-trust-strip {
    border-color: #27272a !important;
}
:root[data-sr-theme="dark"] .sr-white-card {
    background: #16161a !important;
    color: #f4f4f5 !important;
    border: 1px solid #27272a !important;
}
:root[data-sr-theme="dark"] .sr-bar-dark {
    background: #e4e4e7 !important;
}
:root[data-sr-theme="dark"] .sr-bar-light {
    background: #27272a !important;
}
:root[data-sr-theme="dark"] .sr-preview-well {
    background: #1f1f26 !important;
}
:root[data-sr-theme="dark"] .sr-split-shell {
    background: #121215 !important;
    border-color: #27272a !important;
}
:root[data-sr-theme="dark"] .sr-split-left {
    background: #141418 !important;
    color: #f4f4f5 !important;
}
:root[data-sr-theme="dark"] .sr-field-lbl {
    color: #d4d4d8 !important;
}
:root[data-sr-theme="dark"] .sr-field-box {
    background: #1c1c22 !important;
    border-color: #2e2e36 !important;
    color: #e4e4e7 !important;
}
:root[data-sr-theme="dark"] .sr-btn-outline-full {
    background: #1c1c22 !important;
    border-color: #3f3f46 !important;
    color: #f4f4f5 !important;
}
:root[data-sr-theme="dark"] [data-testid="stExpander"] {
    background-color: #121215 !important;
    border-color: #27272a !important;
}
:root[data-sr-theme="dark"] [data-testid="stExpander"] summary,
:root[data-sr-theme="dark"] [data-testid="stExpander"] summary * {
    color: #f4f4f5 !important;
}
:root[data-sr-theme="dark"] label[data-testid="stWidgetLabel"] p {
    color: #e4e4e7 !important;
}

/* ============================================================================
   OPTIMASI RESPONSIVE MULTI-DEVICE (DESKTOP, TABLET, DAN MOBILE)
   ============================================================================ */
html, body, .stApp {
    overflow-x: hidden !important;
}

@media (max-width: 1024px) {
    .block-container {
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
        padding-top: 1.2rem !important;
    }
    .sr-ed-grid-3 {
        grid-template-columns: repeat(2, 1fr);
        gap: 14px;
    }
    .sr-hero {
        padding: 36px 24px 32px 24px;
    }
    .sr-hero-title {
        font-size: clamp(1.85rem, 4.5vw, 2.55rem);
    }
    .sr-ed-headline {
        font-size: clamp(1.9rem, 4.5vw, 2.6rem);
    }
    .sr-work-shell {
        padding: 24px 20px;
    }
    .sr-split-left,
    .sr-split-right {
        padding: 28px 24px;
    }
}

@media (max-width: 768px) {
    .block-container {
        padding-left: 0.9rem !important;
        padding-right: 0.9rem !important;
        padding-top: 0.85rem !important;
        padding-bottom: 2.5rem !important;
    }
    .sr-hero {
        padding: 28px 18px 24px 18px;
        border-radius: 18px;
        margin-bottom: 20px;
    }
    .sr-hero-pill {
        font-size: 0.72rem;
        padding: 5px 12px;
        margin-bottom: 12px;
    }
    .sr-hero-title {
        font-size: clamp(1.55rem, 6vw, 2.05rem);
        line-height: 1.16;
    }
    .sr-hero-sub {
        font-size: 0.88rem;
        margin-bottom: 18px;
    }
    .sr-step-main-title {
        font-size: clamp(1.32rem, 5vw, 1.6rem);
    }
    .sr-step-main-sub {
        font-size: 0.85rem;
    }
    .st-key-card_camera_primary,
    .st-key-card_upload_left,
    .st-key-card_tester_right {
        padding: 16px 15px !important;
        border-radius: 15px !important;
    }
    .sr-card-head-row {
        flex-wrap: wrap;
        gap: 8px;
    }
    .sr-ed-hero {
        padding: 10px 2px 20px 2px;
    }
    .sr-ed-headline {
        font-size: clamp(1.6rem, 6vw, 2.1rem);
        line-height: 1.14;
    }
    .sr-ed-lead {
        font-size: 0.92rem;
        margin-bottom: 18px;
    }
    .sr-ed-cta-row {
        flex-wrap: wrap;
        gap: 12px;
    }
    .sr-ed-grid-3,
    .sr-ed-grid-2 {
        grid-template-columns: 1fr;
        gap: 14px;
    }
    .sr-ed-thumb.tall {
        height: 220px;
    }
    .sr-work-shell {
        padding: 20px 16px;
        border-radius: 18px;
    }
    .sr-work-top {
        flex-wrap: wrap;
        gap: 8px;
    }
    .sr-work-title {
        font-size: 1.35rem;
    }
    .sr-white-card {
        padding: 16px;
    }
    .sr-white-head {
        flex-wrap: wrap;
        gap: 8px;
    }
    .sr-bento-shell {
        padding: 26px 16px 22px 16px;
        border-radius: 18px;
    }
    .sr-bento-heading {
        font-size: 1.38rem;
    }
    .sr-bento-boxes {
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
    }
    .sr-bento-bar {
        flex-direction: column;
        align-items: stretch;
        text-align: center;
    }
    .sr-bento-bar-text {
        text-align: center;
    }
    .sr-split-left,
    .sr-split-right {
        padding: 22px 18px;
        min-height: auto;
    }
    .sr-quote-text {
        font-size: 1.1rem;
    }
}

@media (max-width: 480px) {
    .block-container {
        padding-left: 0.65rem !important;
        padding-right: 0.65rem !important;
    }
    .sr-bento-boxes {
        grid-template-columns: 1fr 1fr;
        gap: 8px;
    }
    .sr-stat-val {
        font-size: 1.2rem;
    }
    .sr-ed-thumb {
        height: 165px;
    }
}
</style>
"""
)

# Muat data evaluasi untuk ditampilkan di Sidebar & Halaman Evaluasi
eval_report_path = DEFAULT_MODEL_PATH.parent / "evaluation_report.json"
cm_image_path = DEFAULT_MODEL_PATH.parent / "confusion_matrix.png"
history_image_path = DEFAULT_MODEL_PATH.parent / "training_history.png"
test_acc_pct = "92.4%"
eval_data_obj: dict | None = None
if eval_report_path.exists():
    try:
        eval_data_obj = json.loads(eval_report_path.read_text(encoding="utf-8"))
        test_acc_pct = f"{eval_data_obj['test_metrics']['overall_accuracy'] * 100:.1f}%"
    except Exception:
        eval_data_obj = None

# === REFERENSI 4: SIDEBAR STUDIO MINIMALIS DENGAN NAVIGASI MULTI-HALAMAN ===
sortara_logo_uri = get_sortara_logo_uri()
logo_img_html = (
    f'<img src="{sortara_logo_uri}" alt="Logo SORTARA" />'
    if sortara_logo_uri
    else "S"
)

with st.sidebar:
    _SIDEBAR_BRAND_THEME_SWITCH(
        key="sidebar_brand_switch",
        data={
            "logo_uri": sortara_logo_uri or "",
            "active_page": st.session_state["active_page"],
        },
    )
    st.html('<div class="sr-side-section">Workspace</div>')

    if st.button(
        "Klasifikasi foto",
        key="nav_classify",
        icon=":material/folder_open:",
        type="primary" if st.session_state["active_page"] == "classify" else "secondary",
        width="stretch",
    ):
        st.session_state["active_page"] = "classify"
        st.rerun()

    if st.button(
        "Cakupan 6 kategori & tong",
        key="nav_categories",
        icon=":material/delete_sweep:",
        type="primary" if st.session_state["active_page"] == "categories" else "secondary",
        width="stretch",
    ):
        st.session_state["active_page"] = "categories"
        st.rerun()

    st.html('<div class="sr-side-section">Insights</div>')

    if st.button(
        "Confusion Matrix & Evaluasi",
        key="nav_evaluation",
        icon=":material/bar_chart:",
        type="primary" if st.session_state["active_page"] == "evaluation" else "secondary",
        width="stretch",
    ):
        st.session_state["active_page"] = "evaluation"
        st.rerun()

    if st.button(
        "Tentang & Spesifikasi Model",
        key="nav_about",
        icon=":material/info:",
        type="primary" if st.session_state["active_page"] == "about" else "secondary",
        width="stretch",
    ):
        st.session_state["active_page"] = "about"
        st.rerun()

    st.html('<div class="sr-side-section">Pengaturan Model</div>')

    confidence_threshold = st.slider(
        "Ambang keyakinan minimum",
        min_value=0.30,
        max_value=0.90,
        value=DEFAULT_CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Jika skor keyakinan prediksi berada di bawah nilai ini, sistem akan meminta foto ulang (FR-10).",
        key="confidence_threshold_slider",
    )

    st.html(
        """
        <div class="sr-side-footer">
            <div class="sr-avatar">AI</div>
            <div>
                <div class="sr-side-footer-name">Adit &amp; Irfan</div>
                <div class="sr-side-footer-sub">Smart Waste AI · v1.0</div>
            </div>
        </div>
        """
    )

# Judul terstruktur untuk aksesibilitas dan pengujian otomatis (FR-01)
st.title(f"{APP_NAME} — {APP_TAGLINE}")

current_page = st.session_state["active_page"]

# ==============================================================================
# HALAMAN 2: CAKUPAN 6 KATEGORI & TONG SAMPAH BERWARNA (REFERENSI EDITORIAL BARU)
# ==============================================================================
if current_page == "categories":
    st.html(
        """
        <div class="sr-ed-hero">
            <div class="sr-ed-pill">
                <span class="sr-ed-dot"></span>
                <span>Standar pemilahan 6 kategori sampah · Panduan tong berwarna</span>
            </div>
            <div class="sr-ed-headline">
                Enam kategori pemilahan untuk sampah yang kita temui setiap hari
            </div>
            <div class="sr-ed-lead">
                Setiap kategori memiliki karakteristik visual, warna tempat sampah nasional, estimasi waktu
                terurai di alam, serta langkah persiapan khusus sebelum dapat didaur ulang atau dikomposkan.
            </div>
            <div class="sr-ed-cta-row">
                <span class="sr-ed-btn-black">6 Kategori Didukung</span>
                <span class="sr-ed-link">Tong Hijau · Tong Kuning · Tong Biru · Tong Abu-abu</span>
            </div>
        </div>
        """
    )

    if st.button(
        "Kembali ke halaman klasifikasi foto",
        key="btn_back_from_categories",
        icon=":material/arrow_back:",
    ):
        st.session_state["active_page"] = "classify"
        st.rerun()

    cards_html_list: list[str] = []
    for cat_name in CLASS_LABELS:
        rule = CATEGORY_RULES[cat_name]
        bin_svg_uri = make_colored_bin_svg_uri(rule.bin_color_guide, cat_name)
        short_bin = rule.bin_color_guide.split("(")[0].strip()
        short_decomp = rule.decomposition_time.split("(")[0].strip()

        cards_html_list.append(
            f"""
            <div class="sr-ed-card">
                <div>
                    <div class="sr-ed-thumb">
                        <span class="sr-ed-year">{escape(short_bin.upper())}</span>
                        <img src="{bin_svg_uri}" alt="Ilustrasi {escape(rule.bin_color_guide)} untuk kategori {escape(cat_name)}" />
                    </div>
                    <div class="sr-ed-body">
                        <div class="sr-ed-card-title">{escape(cat_name)} — {escape(short_bin)}</div>
                        <div class="sr-ed-card-sub">{escape(rule.examples)}</div>
                        <div class="sr-ed-card-detail">
                            <strong>Saran penanganan:</strong> {escape(rule.handling_advice)}<br/>
                            <strong>Dampak jika dibuang sembarangan:</strong> {escape(rule.littering_impact)}
                        </div>
                    </div>
                </div>
                <div class="sr-ed-body">
                    <div class="sr-ed-tags">
                        <span class="sr-ed-tag">{escape(short_bin)}</span>
                        <span class="sr-ed-tag">Terurai: {escape(short_decomp)}</span>
                    </div>
                </div>
            </div>
            """
        )

    st.html(
        f"""
        <div class="sr-ed-grid-3">
            {"".join(cards_html_list)}
        </div>
        <div class="sr-ed-foot-note">
            Menampilkan seluruh enam kategori klasifikasi SORTARA beserta ilustrasi warna tong sampah dan estimasi waktu dekomposisi.
        </div>
        """
    )

# ==============================================================================
# HALAMAN 3: CONFUSION MATRIX & EVALUASI MODEL AI (REFERENSI EDITORIAL BARU)
# ==============================================================================
elif current_page == "evaluation":
    st.html(
        f"""
        <div class="sr-ed-hero">
            <div class="sr-ed-pill">
                <span class="sr-ed-dot"></span>
                <span>Evaluasi Test Set terpisah · Akurasi keseluruhan {escape(test_acc_pct)}</span>
            </div>
            <div class="sr-ed-headline">
                Evaluasi transparan performa AI pada dua ratus dua puluh lima data uji
            </div>
            <div class="sr-ed-lead">
                Model CNN MobileNetV2 dilatih menggunakan pendekatan transfer learning (70% training, 15% validation)
                dan diuji secara jujur pada 15% test set yang tidak pernah terlihat selama proses pelatihan.
            </div>
            <div class="sr-ed-cta-row">
                <span class="sr-ed-btn-black">Overall Test Accuracy: {escape(test_acc_pct)}</span>
                <span class="sr-ed-link">models/sortara_model.keras · 224×224 RGB</span>
            </div>
        </div>
        """
    )

    if st.button(
        "Kembali ke halaman klasifikasi foto",
        key="btn_back_from_evaluation",
        icon=":material/arrow_back:",
    ):
        st.session_state["active_page"] = "classify"
        st.rerun()

    cm_uri = file_image_to_data_uri(cm_image_path)
    hist_uri = file_image_to_data_uri(history_image_path)

    cm_img_tag = (
        f'<img src="{cm_uri}" alt="Confusion Matrix pada Test Set" />'
        if cm_uri
        else "<div>Grafik Confusion Matrix belum tersedia.</div>"
    )
    hist_img_tag = (
        f'<img src="{hist_uri}" alt="Kurva Training dan Validation" />'
        if hist_uri
        else "<div>Grafik kurva training belum tersedia.</div>"
    )

    # 2 Kartu Grafik Utama (Confusion Matrix & Kurva Training) dengan gaya referensi editorial
    st.html(
        f"""
        <div class="sr-ed-grid-2">
            <div class="sr-ed-card">
                <div>
                    <div class="sr-ed-thumb tall">
                        <span class="sr-ed-year">TEST SET (N=225)</span>
                        {cm_img_tag}
                    </div>
                    <div class="sr-ed-body">
                        <div class="sr-ed-card-title">Confusion Matrix — 6 Kategori Sampah</div>
                        <div class="sr-ed-card-sub">
                            Memetakan perbandingan antara label sebenarnya (Ground Truth) dan prediksi model pada 225 foto uji.
                        </div>
                    </div>
                </div>
                <div class="sr-ed-body">
                    <div class="sr-ed-tags">
                        <span class="sr-ed-tag">Akurasi {escape(test_acc_pct)}</span>
                        <span class="sr-ed-tag">208 / 225 Benar</span>
                        <span class="sr-ed-tag">Scikit-learn</span>
                    </div>
                </div>
            </div>
            <div class="sr-ed-card">
                <div>
                    <div class="sr-ed-thumb tall">
                        <span class="sr-ed-year">10 EPOCHS</span>
                        {hist_img_tag}
                    </div>
                    <div class="sr-ed-body">
                        <div class="sr-ed-card-title">Kurva Training vs Validation (Accuracy &amp; Loss)</div>
                        <div class="sr-ed-card-sub">
                            6 epoch tahap feature extraction (lr=1e-3) dilanjutkan 4 epoch fine-tuning 25 layer teratas (lr=1e-4).
                        </div>
                    </div>
                </div>
                <div class="sr-ed-body">
                    <div class="sr-ed-tags">
                        <span class="sr-ed-tag">MobileNetV2</span>
                        <span class="sr-ed-tag">Transfer Learning</span>
                        <span class="sr-ed-tag">EarlyStopping</span>
                    </div>
                </div>
            </div>
        </div>
        """
    )

    # Kartu Metrik Precision, Recall, F1-Score per Kelas dengan gaya kartu referensi yang sama
    if eval_data_obj is not None and "test_metrics" in eval_data_obj:
        per_class_metrics = eval_data_obj["test_metrics"].get("per_class", {})
        metric_cards_html: list[str] = []
        for cls_name in CLASS_LABELS:
            m = per_class_metrics.get(cls_name, {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "support": 0})
            rule = CATEGORY_RULES[cls_name]
            bin_svg_uri = make_colored_bin_svg_uri(rule.bin_color_guide, cls_name)
            p_pct = f"{m['precision'] * 100:.1f}%"
            r_pct = f"{m['recall'] * 100:.1f}%"
            f1_pct = f"{m['f1_score'] * 100:.1f}%"
            sup_cnt = int(m["support"])

            metric_cards_html.append(
                f"""
                <div class="sr-ed-card">
                    <div>
                        <div class="sr-ed-thumb" style="height:145px;">
                            <span class="sr-ed-year">N = {sup_cnt} FOTO</span>
                            <img src="{bin_svg_uri}" alt="Ikon tong {escape(cls_name)}" />
                        </div>
                        <div class="sr-ed-body">
                            <div class="sr-ed-card-title"> Evaluasi Kelas {escape(cls_name)}</div>
                            <div class="sr-ed-card-sub">F1-Score: <strong>{f1_pct}</strong> pada data uji terpisah.</div>
                        </div>
                    </div>
                    <div class="sr-ed-body">
                        <div class="sr-ed-tags">
                            <span class="sr-ed-tag">Precision {p_pct}</span>
                            <span class="sr-ed-tag">Recall {r_pct}</span>
                        </div>
                    </div>
                </div>
                """
            )

        st.html(
            f"""
            <div class="sr-ed-grid-3">
                {"".join(metric_cards_html)}
            </div>
            <div class="sr-ed-foot-note">
                Metrik dihitung pada 15% Test Set (225 gambar dari total 1.500 gambar) yang tidak pernah digunakan selama pelatihan model.
            </div>
            """
        )

# ==============================================================================
# HALAMAN 4: TENTANG & SPESIFIKASI MODEL AI (ABOUT PAGE — SEDERHANA & MUDAH DIPAHAMI)
# ==============================================================================
elif current_page == "about":
    st.html(
        f"""
        <div class="sr-ed-hero">
            <div class="sr-ed-pill">
                <span class="sr-ed-dot"></span>
                <span>Tentang SORTARA · Cara Kerja &amp; Kemampuan AI</span>
            </div>
            <div class="sr-ed-headline">
                Mengenal cara kerja SORTARA dalam membantu memilah sampah Anda
            </div>
            <div class="sr-ed-lead">
                SORTARA adalah asisten pintar yang mengenali jenis sampah hanya dari satu foto, lalu langsung
                memberitahu warna tong sampah dan cara pembuangan yang benar agar mudah diterapkan sehari-hari di rumah.
            </div>
            <div class="sr-ed-cta-row">
                <span class="sr-ed-btn-black">Ketepatan Uji {escape(test_acc_pct)} · 6 Jenis Sampah</span>
                <span class="sr-ed-link">Cepat, Gratis &amp; Privasi Aman (Maks {MAX_FILE_SIZE_MB} MB)</span>
            </div>
        </div>
        """
    )

    if st.button(
        "Kembali ke halaman klasifikasi foto",
        key="btn_back_from_about",
        icon=":material/arrow_back:",
    ):
        st.session_state["active_page"] = "classify"
        st.rerun()

    # Ringkasan Bento 4 Angka Utama dengan bahasa yang ramah pengguna
    st.html(
        f"""
        <div class="sr-bento-shell" style="margin-top:14px; margin-bottom:26px;">
            <div class="sr-bento-pill">
                <span class="sr-dot-yellow"></span>
                <span>Sekilas Kemampuan Sistem SORTARA</span>
            </div>
            <div class="sr-bento-heading">
                Seberapa akurat dan praktis sistem ini saat Anda gunakan?
            </div>
            <div class="sr-bento-boxes">
                <div class="sr-stat-tile">
                    <div class="sr-stat-val">{len(CLASS_LABELS)} Jenis</div>
                    <div class="sr-stat-lbl">Kategori Sampah</div>
                </div>
                <div class="sr-stat-tile">
                    <div class="sr-stat-val">{escape(test_acc_pct)}</div>
                    <div class="sr-stat-lbl">Tingkat Ketepatan</div>
                </div>
                <div class="sr-stat-tile">
                    <div class="sr-stat-val">&lt; 2 Detik</div>
                    <div class="sr-stat-lbl">Waktu Analisis</div>
                </div>
                <div class="sr-stat-tile">
                    <div class="sr-stat-val">{MAX_FILE_SIZE_MB} MB</div>
                    <div class="sr-stat-lbl">Maks Ukuran Foto</div>
                </div>
            </div>
            <div class="sr-bento-bar">
                <div class="sr-bento-bar-text">
                    Dilatih dari 1.500 foto sampah asli (250 foto per jenis sampah) dan telah diuji pada 225 foto baru.
                </div>
                <div class="sr-bento-bar-btn">Bisa di HP &amp; PC</div>
            </div>
            <div class="sr-bento-foot">
                Batas keyakinan saat ini: {confidence_threshold * 100:.0f}% — jika foto terlalu buram, sistem akan meminta Anda memotret ulang agar tidak salah buang.
            </div>
        </div>
        """
    )

    spec_items = [
        {
            "year": "OTAK AI",
            "symbol": "layers",
            "accent": "#16a34a",
            "title": "Otak Pengenal Gambar Pintar",
            "sub": "Mengenali bentuk, tekstur, dan warna sampah seperti mata manusia",
            "detail": (
                "Menggunakan model AI SORTARA yang ringan dan cepat untuk membedakan "
                "botol plastik, pecahan kaca, kertas/kardus, kaleng logam, sisa makanan organik, "
                "hingga sampah residu hanya dalam hitungan detik."
            ),
            "tags": ["SORTARA AI", "Cepat & Ringan", "6 Jenis Sampah"],
        },
        {
            "year": "1.500 FOTO",
            "symbol": "dataset",
            "accent": "#ca8a04",
            "title": "Belajar dari 1.500 Contoh Foto Asli",
            "sub": "Jumlah contoh merata (250 foto per kategori) agar AI adil dan akurat",
            "detail": (
                "AI mempelajari 1.050 foto saat belajar, 225 foto saat latihan, dan diuji pada "
                "225 foto baru yang belum pernah dilihat sebelumnya dengan hasil ketepatan 92,4%."
            ),
            "tags": ["Data Seimbang", "250 Foto / Jenis", "Ketepatan 92,4%"],
        },
        {
            "year": "FLEKSIBEL",
            "symbol": "augment",
            "accent": "#16a34a",
            "title": "Tetap Akurat dari Berbagai Sudut Foto",
            "sub": "Siap mengenali foto dari kamera HP maupun webcam komputer",
            "detail": (
                "Selama pelatihan, AI dilatih mengenali foto yang miring, jarak dekat maupun agak jauh, "
                "serta kondisi cahaya ruangan yang berbeda-beda. Cukup potret 1 benda sampah dengan jelas."
            ),
            "tags": ["Kamera HP & PC", "Berbagai Sudut", "Cahaya Ruangan"],
        },
        {
            "year": "PENGAMAN",
            "symbol": "shield",
            "accent": "#2563eb",
            "title": "Deteksi Otomatis Jika Foto Kurang Jelas",
            "sub": "Jujur memberi tahu bila AI ragu agar Anda tidak salah membuang sampah",
            "detail": (
                f"Menerima foto format JPG, JPEG, atau PNG (maksimal {MAX_FILE_SIZE_MB} MB). "
                "Jika foto terlalu gelap, buram, atau skor keyakinan AI rendah, sistem otomatis "
                "menyarankan foto ulang."
            ),
            "tags": ["Cek Foto Buram", "JPG & PNG", f"Maks {MAX_FILE_SIZE_MB} MB"],
        },
        {
            "year": "PANDUAN",
            "symbol": "training",
            "accent": "#2563eb",
            "title": "Saran Warna Tong & Langkah Daur Ulang",
            "sub": "Tidak hanya menebak jenis sampah, tapi memandu tindakan nyata",
            "detail": (
                "Setiap hasil foto langsung menampilkan panduan warna tong sampah (Hijau, Kuning, "
                "Biru, Merah, atau Abu-abu) beserta tips singkat cara menyiapkan sampah sebelum dibuang."
            ),
            "tags": ["5 Warna Tong", "Tips Praktis", "Edukasi Lingkungan"],
        },
        {
            "year": "TIM & PRIVASI",
            "symbol": "arch",
            "accent": "#18181b",
            "title": "Privasi Aman & Tim Pengembang",
            "sub": "Adit (Model AI & Data) · Irfan (Sistem Web & Antarmuka Pengguna)",
            "detail": (
                "Foto sampah yang Anda ambil atau unggah hanya diproses sementara di memori saat itu juga "
                "dan tidak disimpan permanen di server. Aman digunakan kapan saja tanpa perlu registrasi akun."
            ),
            "tags": ["Privasi Terjaga", "Tanpa Akun", "Adit & Irfan"],
        },
    ]

    spec_cards_html: list[str] = []
    for item in spec_items:
        icon_uri = make_spec_icon_svg_uri(item["symbol"], item["accent"])
        tags_html = "".join(f'<span class="sr-ed-tag">{escape(t)}</span>' for t in item["tags"])
        spec_cards_html.append(
            f"""
            <div class="sr-ed-card">
                <div>
                    <div class="sr-ed-thumb" style="height:155px;">
                        <span class="sr-ed-year">{escape(item["year"])}</span>
                        <img src="{icon_uri}" alt="Ilustrasi {escape(item["title"])}" />
                    </div>
                    <div class="sr-ed-body">
                        <div class="sr-ed-card-title">{escape(item["title"])}</div>
                        <div class="sr-ed-card-sub">{escape(item["sub"])}</div>
                        <div class="sr-ed-card-detail">{escape(item["detail"])}</div>
                    </div>
                </div>
                <div class="sr-ed-body">
                    <div class="sr-ed-tags">
                        {tags_html}
                    </div>
                </div>
            </div>
            """
        )

    st.html(
        f"""
        <div class="sr-ed-grid-3">
            {"".join(spec_cards_html)}
        </div>
        <div class="sr-ed-foot-note">
            Ringkasan cara kerja dan spesifikasi SORTARA — Sistem Pintar Pemilahan Sampah Rumah Tangga.
        </div>
        """
    )

# ==============================================================================
# HALAMAN 1 (UTAMA): KLASIFIKASI FOTO SAMPAH
# ==============================================================================
else:
    # === REFERENSI 2: HERO BANNER GELAP ===
    model_badge_text = (
        "SORTARA · Akurasi Uji 92,44%"
        if predictor.model_mode == "keras_mobilenetv2"
        else "SORTARA · Mode Simulasi Visual"
    )
    st.html(
        f"""
        <div class="sr-hero">
            <div class="sr-hero-pill">{escape(model_badge_text)}</div>
            <div class="sr-hero-title">
                Sort smarter. <span class="sr-hero-gradient">waste better.</span>
            </div>
            <div class="sr-hero-sub">
                {escape(APP_DESCRIPTION)}
            </div>
            <div class="sr-hero-cta">Pilih metode input foto di bawah</div>
        </div>
        """
    )

    # === REFERENSI STEP 1 OF 3: 3 OPSI INPUT GAMBAR (KAMERA UTAMA DI ATAS + SPLIT UPLOAD & TESTER DI BAWAH) ===
    st.html(
        """
        <div class="sr-step-header-wrap">
            <div class="sr-step-eyebrow">Step 1 of 3</div>
            <div class="sr-step-main-title">Bagaimana Anda ingin memasukkan foto sampah?</div>
            <div class="sr-step-main-sub">Tanpa perlu membuat akun — gunakan kamera langsung sebagai opsi utama atau pilih alternatif di bawahnya.</div>
        </div>
        """
    )

    # OPSI UTAMA (ATAS): KARTU KAMERA DENGAN BORDER TEGAS & BADGE "TERCEPAT"
    with st.container(border=True, key="card_camera_primary"):
        st.html(
            """
            <div class="sr-card-head-row">
                <span class="sr-card-title">Ambil foto menggunakan kamera</span>
                <span class="sr-badge-fastest">Tercepat</span>
            </div>
            """
        )

        # Satu fitur utama kamera terpadu (otomatis mereset sesi kamera & mendukung HP maupun Laptop)
        smart_cam_res = _SMART_CAMERA(
            key="smart_camera_widget",
            data={},
            on_captured_jpeg_change=on_smart_camera_capture,
        )

        st.html(
            """
            <div class="sr-card-foot">
                Anda dapat memotret satu objek sampah secara langsung dari kamera perangkat Anda — foto diproses di memori tanpa disimpan permanen.
            </div>
            """
        )

    # PEMBATAS HORIZONTAL "ATAU" ("OR")
    st.html('<div class="sr-or-line">ATAU</div>')

    # DUA OPSI DI BAWAHNYA DENGAN SPLIT SCREEN (KIRI: UNGGAH FILE | KANAN: GAMBAR TESTER)
    split_left_col, split_right_col = st.columns(2, gap="medium")

    with split_left_col:
        with st.container(border=True, key="card_upload_left"):
            st.html(
                """
                <div class="sr-card-head-row">
                    <span class="sr-card-title">Unggah atau pilih file</span>
                </div>
                """
            )
            uploaded_file = st.file_uploader(
                f"Format JPG, JPEG, atau PNG (Maksimal {MAX_FILE_SIZE_MB} MB)",
                type=None,
                accept_multiple_files=False,
                key="waste_uploader",
                on_change=on_upload_change,
            )
            st.html(
                """
                <div class="sr-card-foot">
                    Pilih satu foto benda sampah yang sudah tersimpan di galeri komputer atau perangkat Anda.
                </div>
                """
            )

    with split_right_col:
        with st.container(border=True, key="card_tester_right"):
            st.html(
                """
                <div class="sr-card-head-row">
                    <span class="sr-card-title">Gunakan gambar tester</span>
                </div>
                """
            )
            selected_sample_class = st.selectbox(
                "Pilih kategori sampel dari Test Set",
                options=list(CLASS_LABELS),
                index=0,
                key="sample_class_selector",
                on_change=on_sample_change,
            )
            use_sample_clicked = st.button(
                "Gunakan gambar tester ini",
                key="btn_use_sample",
                icon=":material/science:",
                width="stretch",
            )
            if use_sample_clicked:
                reset_classification_state()
                st.session_state["active_source"] = "sample"

            st.html(
                """
                <div class="sr-card-foot">
                    Kami memuat sampel foto asli dari 225 gambar Test Set untuk menguji model secara instan.
                </div>
                """
            )

    # STRIP KEPERCAYAAN DI BAWAH 3 KARTU
    st.html(
        f"""
        <div class="sr-trust-strip">
            <div class="sr-trust-grid">
                <div>&#10003; Diproses lokal di memori sesi berjalan</div>
                <div>&#9993; Validasi otomatis JPG, JPEG, PNG (&le; {MAX_FILE_SIZE_MB} MB)</div>
                <div>&#10003; Tanpa registrasi akun atau penyimpanan data</div>
            </div>
            <div class="sr-trust-legal">
                {escape(LOCAL_RULE_DISCLAIMER)}
            </div>
        </div>
        """
    )

    # Ekstrak foto dari komponen Kamera Pintar CCv2 jika tersedia
    smart_cam_bytes: bytes | None = None
    captured_data_url = getattr(smart_cam_res, "captured_jpeg", None)
    if isinstance(captured_data_url, str) and captured_data_url.startswith("data:image"):
        try:
            _, b64_part = captured_data_url.split(",", 1)
            smart_cam_bytes = base64.b64decode(b64_part)
        except Exception:
            smart_cam_bytes = None

    # Tentukan sumber gambar yang sedang aktif berdasarkan interaksi pengguna
    active_filename: str | None = None
    active_bytes: bytes | None = None
    active_mode_label: str = "Menunggu input"

    if st.session_state["active_source"] == "smart_camera" and smart_cam_bytes is not None:
        active_filename = "kamera_sortara.jpg"
        active_bytes = smart_cam_bytes
        active_mode_label = "Kamera langsung"
    elif st.session_state["active_source"] == "upload" and uploaded_file is not None:
        active_filename = uploaded_file.name
        active_bytes = uploaded_file.getvalue()
        active_mode_label = "Unggah file"
    elif st.session_state["active_source"] == "sample":
        chosen_cls = selected_sample_class or CLASS_LABELS[0]
        active_filename, active_bytes = load_sample_image_bytes(chosen_cls)
        active_mode_label = f"Gambar tester ({chosen_cls})"
    elif smart_cam_bytes is not None:
        active_filename = "kamera_sortara.jpg"
        active_bytes = smart_cam_bytes
        active_mode_label = "Kamera langsung"
    elif uploaded_file is not None:
        active_filename = uploaded_file.name
        active_bytes = uploaded_file.getvalue()
        active_mode_label = "Unggah file"

    # Validasi gambar jika sudah ada input
    validation: ValidationResult | None = None
    if active_filename is not None and active_bytes is not None:
        validation = validate_uploaded_image(active_filename, active_bytes)
        if not validation.is_valid:
            st.error(
                validation.error_message or "File gambar tidak valid.",
                title="File tidak dapat diproses",
                icon=":material/error:",
            )

    # Tombol aksi utama ("Klasifikasikan sampah" & "Coba foto lain")
    if validation is not None and validation.is_valid and validation.image is not None:
        with st.container(horizontal=True):
            classify_clicked = st.button(
                "Klasifikasikan sampah",
                key="btn_classify",
                type="primary",
                icon=":material/psychology:",
                width="stretch",
            )
            reset_clicked = st.button(
                "Coba foto lain",
                key="btn_reset",
                icon=":material/refresh:",
                width="stretch",
            )

        if reset_clicked:
            reset_classification_state()
            st.session_state["active_source"] = "none"
            st.rerun()

        if classify_clicked:
            with st.spinner("Menganalisis citra dengan SORTARA..."):
                pred_result = predictor.predict(
                    validation.image,
                    filename_hint=active_filename,
                )
                st.session_state["last_prediction"] = pred_result
                st.session_state["last_filename"] = active_filename
                st.session_state["classification_done"] = True

    # Siapkan variabel tampilan untuk Kartu Workspace (Referensi 1), Bento (Referensi 3), dan Split Card (Referensi 5)
    prediction: PredictionResult | None = (
        st.session_state["last_prediction"] if st.session_state["classification_done"] else None
    )

    if validation is not None and validation.is_valid and validation.image is not None:
        preview_uri = image_to_data_uri(validation.image)
        file_label = escape(active_filename or "sampah.jpg")
        dims_text = (
            f"{validation.image_dimensions[0]}×{validation.image_dimensions[1]} px · {validation.file_size_kb} KB"
            if validation.image_dimensions
            else f"{validation.file_size_kb} KB"
        )
        step1_status = "Tervalidasi · Siap dianalisis"
        step1_desc = f"Format RGB valid ({dims_text}). Ukuran sesuai batas <= {MAX_FILE_SIZE_MB} MB."
        preview_inner_html = f'<img src="{preview_uri}" alt="Pratinjau foto sampah" />'
    else:
        file_label = "belum-ada-foto.jpg"
        step1_status = "Menunggu unggahan foto"
        step1_desc = "Ambil foto dari kamera di atas, unggah file JPG/PNG, atau klik 'Gunakan gambar tester ini'."
        preview_inner_html = (
            '<div style="color:#6b7280; font-size:0.86rem; text-align:center; padding:32px 16px;">'
            "Pratinjau foto akan tampil di area kanvas ini setelah Anda memilih salah satu dari 3 opsi di atas."
            "</div>"
        )

    if prediction is not None and prediction.is_success and prediction.predicted_category is not None:
        rec = build_recommendation(
            predicted_category=prediction.predicted_category,
            confidence_score=prediction.confidence_score,
            threshold=confidence_threshold,
        )
        conf_pct_str = f"{prediction.confidence_score * 100:.1f}%"
        if rec.is_confident:
            chip_text = f"Terprediksi: {rec.category}"
            step2_desc = (
                f"Model memprediksi kategori {rec.category} dengan keyakinan {conf_pct_str} "
                f"(di atas ambang {confidence_threshold * 100:.0f}%)."
            )
            step3_class = "sr-step-card emerald"
            step3_title = f"3 · Arahkan ke {rec.bin_color_guide}"
            step3_desc = rec.handling_advice
        else:
            chip_text = "Keyakinan rendah"
            step2_desc = (
                f"Skor tertinggi ({conf_pct_str}) masih di bawah ambang {confidence_threshold * 100:.0f}%."
            )
            step3_class = "sr-step-card amber"
            step3_title = "3 · Belum dapat dikenali — ambil foto ulang"
            step3_desc = rec.low_confidence_message
    else:
        rec = None
        conf_pct_str = "—"
        chip_text = "Siap dianalisis"
        step2_desc = "Menunggu tombol 'Klasifikasikan sampah' ditekan untuk menjalankan inferensi."
        step3_class = "sr-step-card"
        step3_title = "3 · Rekomendasi & panduan tong sampah"
        step3_desc = "Saran penanganan, warna tong sampah, dan waktu terurai akan muncul setelah analisis selesai."

    # === REFERENSI 1: WORKSPACE GELAP (KARTU PUTIH DI KIRI + 3 STEP DI KANAN) ===
    st.html(
        f"""
        <div class="sr-work-shell">
            <div class="sr-work-top">
                <span class="sr-pill-green">Step 2 of 3 · Pratinjau &amp; Status</span>
                <span class="sr-pill-mode">Sumber: {escape(active_mode_label)} ▾</span>
            </div>
            <div class="sr-work-title">Analisis satu objek sampah secara instan.</div>
            <div class="sr-work-sub">
                Foto diproses langsung di memori perangkat Anda lalu diklasifikasikan ke 6 kategori sampah.
            </div>
            <div class="sr-work-grid">
                <div class="sr-white-card">
                    <div>
                        <div class="sr-white-head">
                            <div>
                                <div class="sr-file-name">{file_label}</div>
                                <div class="sr-sync-status">{escape(step1_status)}</div>
                            </div>
                            <div class="sr-black-chip">{escape(chip_text)}</div>
                        </div>
                        <div class="sr-bar-dark"></div>
                        <div class="sr-bar-light"></div>
                        <div class="sr-bar-light" style="width: 76%;"></div>
                    </div>
                    <div class="sr-preview-well">
                        {preview_inner_html}
                    </div>
                </div>
                <div class="sr-step-stack">
                    <div class="sr-step-card">
                        <div class="sr-step-row">
                            <span class="sr-step-title">1 · Validasi &amp; preprocessing citra</span>
                            <span class="sr-step-meta">224×224 RGB</span>
                        </div>
                        <div class="sr-step-desc">{escape(step1_desc)}</div>
                    </div>
                    <div class="sr-step-card">
                        <div class="sr-step-row">
                            <span class="sr-step-title">2 · Inferensi AI SORTARA</span>
                            <span class="sr-step-meta">Skor: {escape(conf_pct_str)}</span>
                        </div>
                        <div class="sr-step-desc">{escape(step2_desc)}</div>
                    </div>
                    <div class="{step3_class}">
                        <div class="sr-step-row">
                            <span class="sr-step-title">{escape(step3_title)}</span>
                        </div>
                        <div class="sr-step-desc">{escape(step3_desc)}</div>
                    </div>
                </div>
            </div>
        </div>
        """
    )

    # Jika hasil prediksi gagal karena error model (FR-13 / AT-08)
    if prediction is not None and not prediction.is_success:
        st.error(
            prediction.error_message or "Terjadi kegagalan saat menjalankan model AI.",
            title="Gagal memproses prediksi",
            icon=":material/warning:",
        )

    # Jika hasil prediksi berhasil, tampilkan REFERENSI 3 (4 Bento Boxes) & REFERENSI 5 (50/50 Split Card)
    if prediction is not None and prediction.is_success and rec is not None:
        if not rec.is_confident:
            st.warning(
                rec.low_confidence_message,
                title="Prediksi belum cukup meyakinkan",
                icon=":material/help_center:",
            )
            if prediction.quality_warning:
                st.caption(f"Catatan kualitas citra: {prediction.quality_warning}")
        else:
            short_decomp = rec.decomposition_time.split("(")[0].strip()
            short_bin = rec.bin_color_guide.split("(")[0].strip()

            # === REFERENSI 3: 4-BOX METRIC BENTO CARD (TIMER LAYOUT STYLE) ===
            st.html(
                f"""
                <div class="sr-bento-shell">
                    <div class="sr-bento-pill">
                        <span class="sr-dot-yellow"></span>
                        <span>Step 3 of 3 · Ringkasan hasil &amp; estimasi dekomposisi</span>
                    </div>
                    <div class="sr-bento-heading">
                        Sampah {escape(rec.category or "")} membutuhkan waktu {escape(short_decomp)} untuk terurai di alam
                    </div>
                    <div class="sr-bento-boxes">
                        <div class="sr-stat-tile">
                            <div class="sr-stat-val">{escape(rec.category or "-")}</div>
                            <div class="sr-stat-lbl">Kategori AI</div>
                        </div>
                        <div class="sr-stat-tile">
                            <div class="sr-stat-val">{escape(conf_pct_str)}</div>
                            <div class="sr-stat-lbl">Keyakinan</div>
                        </div>
                        <div class="sr-stat-tile">
                            <div class="sr-stat-val">{escape(short_bin)}</div>
                            <div class="sr-stat-lbl">Target Tong</div>
                        </div>
                        <div class="sr-stat-tile">
                            <div class="sr-stat-val">{escape(short_decomp)}</div>
                            <div class="sr-stat-lbl">Waktu Terurai</div>
                        </div>
                    </div>
                    <div class="sr-bento-bar">
                        <div class="sr-bento-bar-text">{escape(rec.examples)}</div>
                        <div class="sr-bento-bar-btn">{escape(rec.bin_color_guide)}</div>
                    </div>
                    <div class="sr-bento-foot">
                        Ambang batas keyakinan aktif: {confidence_threshold * 100:.0f}% · {escape(LOCAL_RULE_DISCLAIMER)}
                    </div>
                </div>
                """
            )

            # === REFERENSI 5: 50/50 SPLIT CARD (KIRI PUTIH LANGKAH PEMILAHAN — KANAN HITAM DAMPAK LINGKUNGAN) ===
            steps_html = "".join(
                f"<div style='margin-bottom:6px;'>{idx}. {escape(step)}</div>"
                for idx, step in enumerate(rec.preparation_steps, start=1)
            )
            upcycle_html = "".join(
                f"<div style='margin-bottom:6px;'>• {escape(idea)}</div>"
                for idea in rec.upcycling_ideas
            )

            st.html(
                f"""
                <div class="sr-split-shell">
                    <div class="sr-split-left">
                        <div class="sr-split-logo">{logo_img_html}</div>
                        <div class="sr-split-title">Panduan pemilahan &amp; guna ulang</div>
                        <div class="sr-split-sub">Langkah praktis menangani sampah {escape(rec.category or "")} sebelum dibuang.</div>

                        <div class="sr-field-lbl">Langkah persiapan pemilahan</div>
                        <div class="sr-field-box">{steps_html}</div>

                        <div class="sr-field-lbl">Ide guna ulang (Upcycling / DIY)</div>
                        <div class="sr-field-box">{upcycle_html}</div>

                        <div class="sr-btn-black-full">Target pembuangan: {escape(rec.bin_color_guide)}</div>
                        <div class="sr-or-divider">Manfaat daur ulang</div>
                        <div class="sr-btn-outline-full">{escape(rec.recycling_benefit)}</div>
                    </div>
                    <div class="sr-split-right">
                        <div>
                            <div style="font-size:0.75rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#ef4444; margin-bottom:10px;">
                                Sebab &amp; akibat jika dibuang sembarangan
                            </div>
                            <div class="sr-quote-text">
                                &ldquo;{escape(rec.littering_impact)}&rdquo;
                            </div>
                            <div class="sr-quote-author">
                                Edukasi Dampak Lingkungan SORTARA — Kategori {escape(rec.category or "")} ({escape(rec.decomposition_time)})
                            </div>
                        </div>
                    </div>
                </div>
                """
            )

        # Tabel distribusi probabilitas 6 kelas tetap tersedia di bagian bawah
        with st.expander("Lihat distribusi skor keyakinan 6 kelas", icon=":material/bar_chart:"):
            prob_df = pd.DataFrame(
                [
                    {
                        "Kategori": cls_name,
                        "Probabilitas (%)": round(prob * 100.0, 2),
                    }
                    for cls_name, prob in prediction.class_probabilities.items()
                ]
            ).sort_values(by="Probabilitas (%)", ascending=False)
            st.dataframe(
                prob_df,
                hide_index=True,
                alt="Tabel distribusi skor keyakinan model untuk enam kategori sampah",
            )
