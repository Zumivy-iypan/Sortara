"""Generator logo resmi SORTARA (SVG & High-Res PNG) sesuai kisi geometri 60 derajat simbol daur ulang & 3 segitiga."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"

CX = 128.0
CY = 138.0


def _rotate_point(x: float, y: float, cx: float, cy: float, deg: float) -> tuple[float, float]:
    rad = math.radians(deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    dx = x - cx
    dy = y - cy
    return (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)


def _sample_bezier_quad(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    steps: int = 24,
) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for i in range(steps + 1):
        t = i / steps
        mt = 1.0 - t
        x = mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0]
        y = mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1]
        pts.append((x, y))
    return pts


def _build_single_arm_polygon() -> list[tuple[float, float]]:
    """
    Membangun poligon satu lengan panah daur ulang di atas kisi segitiga sama sisi (60 derajat)
    sehingga ketika diputar 0, 120, dan 240 derajat terhadap (128, 138):
    - Batang dalam setiap panah 100% sejajar dengan ketiga sisi piramida segitiga putih di tengah.
    - Sayap panah kiri benar-benar horizontal, dan batang panah bawah benar-benar horizontal.
    """
    # 1. Lengkungan luar puncak atas: dari ekor horizontal atas (y=44) berbelok 60 derajat turun-kanan
    outer_top_curve = _sample_bezier_quad(
        (117.0, 44.0),
        (139.0, 44.0),
        (150.0, 63.0),
        steps=24,
    )

    # 2. Lengkungan ujung luar sayap panah (chevron)
    wing_tip_curve = _sample_bezier_quad(
        (191.0, 88.0),
        (202.0, 91.5),
        (205.5, 103.5),
        steps=12,
    )

    # 3. Lengkungan dalam puncak atas: dari batang dalam 60 derajat kembali ke ekor horizontal (y=63)
    inner_top_curve = _sample_bezier_quad(
        (134.5, 73.2),
        (127.0, 60.2),
        (112.0, 62.5),
        steps=20,
    )

    pts: list[tuple[float, float]] = []
    # Potongan pangkal ekor miring 60 derajat
    pts.append((102.0, 62.5))
    pts.append((112.5, 44.0))
    # Lengkungan luar atas
    pts.extend(outer_top_curve)
    # Batang luar turun-kanan tepat 60 derajat menuju pangkal sayap panah
    pts.append((178.0, 111.5))
    # Sayap panah naik-kanan tepat -60 derajat
    pts.extend(wing_tip_curve)
    # Sisi luar sayap panah turun-kiri tepat -60 derajat menuju ujung lancip panah (tip)
    pts.append((182.0, 144.2))
    # Batang dalam naik-kiri tepat 60 derajat (dx=-47.5, dy=-71.0 -> sejajar segitiga tengah)
    pts.extend(inner_top_curve)
    return pts


def _build_center_triangles() -> list[list[tuple[float, float]]]:
    """Tiga segitiga putih sama sisi di tengah (piramida pemilahan sampah) berpusat di (CX, CY)."""
    return [
        # Segitiga atas
        [(128.0, 97.5), (114.2, 121.4), (141.8, 121.4)],
        # Segitiga kiri bawah
        [(110.2, 129.8), (96.4, 153.7), (124.0, 153.7)],
        # Segitiga kanan bawah
        [(145.8, 129.8), (132.0, 153.7), (159.6, 153.7)],
    ]


def build_sortara_logo_svg() -> str:
    """Menghasilkan SVG logo SORTARA vektor murni yang tajam di semua ukuran layar."""
    arm_pts = _build_single_arm_polygon()
    arm_points_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in arm_pts)

    tri_elems: list[str] = []
    for tri in _build_center_triangles():
        pts_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in tri)
        tri_elems.append(f'    <polygon points="{pts_str}" />')
    tris_svg = "\n".join(tri_elems)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="100%" height="100%">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#11423e"/>
      <stop offset="55%" stop-color="#0c3431"/>
      <stop offset="100%" stop-color="#072220"/>
    </linearGradient>
    <linearGradient id="emeraldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#22e362"/>
      <stop offset="100%" stop-color="#14c14c"/>
    </linearGradient>
    <filter id="softShadow" x="-12%" y="-12%" width="124%" height="124%">
      <feDropShadow dx="0" dy="3" stdDeviation="3.5" flood-color="#000000" flood-opacity="0.24"/>
    </filter>
    <g id="recycleArm">
      <polygon points="{arm_points_str}" fill="url(#emeraldGrad)" stroke="url(#emeraldGrad)" stroke-width="2.5" stroke-linejoin="round"/>
    </g>
  </defs>

  <!-- Latar kotak membulat (squircle) warna Deep Forest Teal sesuai referensi -->
  <rect x="0" y="0" width="256" height="256" rx="56" fill="url(#bgGrad)"/>
  <rect x="5" y="5" width="246" height="246" rx="51" fill="none" stroke="#1f6a63" stroke-width="2" stroke-opacity="0.45"/>

  <!-- 3 panah daur ulang simetris (0, 120, 240 derajat terhadap pusat {CX}, {CY}) -->
  <g filter="url(#softShadow)">
    <use href="#recycleArm" transform="rotate(0 {CX} {CY})"/>
    <use href="#recycleArm" transform="rotate(120 {CX} {CY})"/>
    <use href="#recycleArm" transform="rotate(240 {CX} {CY})"/>
  </g>

  <!-- 3 segitiga putih piramida di tengah -->
  <g fill="#ffffff" stroke="#ffffff" stroke-width="3.0" stroke-linejoin="round" filter="url(#softShadow)">
{tris_svg}
  </g>
</svg>"""


def build_sortara_logo_png(output_path: Path, size: int = 512) -> None:
    """Merender PNG beresolusi tinggi dengan supersampling 4x agar anti-alias sangat halus."""
    scale = 4
    canvas_size = size * scale
    s = canvas_size / 256.0

    img = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background rounded rectangle (Deep Forest Teal #0c3633)
    draw.rounded_rectangle(
        [(0, 0), (canvas_size - 1, canvas_size - 1)],
        radius=int(56 * s),
        fill=(12, 54, 51, 255),
        outline=(28, 98, 91, 255),
        width=int(3 * s),
    )

    base_arm_pts = _build_single_arm_polygon()
    for angle in (0.0, 120.0, 240.0):
        rot_pts = [
            (
                _rotate_point(px, py, CX, CY, angle)[0] * s,
                _rotate_point(px, py, CX, CY, angle)[1] * s,
            )
            for px, py in base_arm_pts
        ]
        draw.polygon(rot_pts, fill=(22, 204, 82, 255))

    for tri in _build_center_triangles():
        scaled_tri = [(px * s, py * s) for px, py in tri]
        draw.polygon(scaled_tri, fill=(255, 255, 255, 255))

    final_img = img.resize((size, size), Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_img.save(output_path, format="PNG")


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = ASSETS_DIR / "sortara_logo.svg"
    png_path = ASSETS_DIR / "sortara_logo.png"
    svg_path.write_text(build_sortara_logo_svg(), encoding="utf-8")
    build_sortara_logo_png(png_path, size=512)
    print(f"Saved {svg_path} and {png_path}")


if __name__ == "__main__":
    main()
