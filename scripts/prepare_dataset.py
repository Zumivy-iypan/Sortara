"""Skrip otomatisasi pengumpulan, kurasi, dan penataan dataset 6 kelas SORTARA (Tugas Adit).

Mengunduh dataset publik Garbage Classification (12/10 kelas yang mencakup Organic/Biological,
Glass, Paper/Cardboard, Metal, Plastic, dan Trash/Residual) melalui `kagglehub`, memvalidasi
setiap gambar dengan PIL (anti-corrupt/duplikat), lalu menyusunnya ke dalam:
`data/dataset_raw/{Kaca,Kertas,Logam,Organik,Plastik,Residu}/` dengan target 250 gambar per kelas
(total 1.500 gambar sesuai target perencanaan PRD Bagian 9.2: 200-500 gambar/kelas).
"""

import hashlib
import random
import shutil
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_DATASET_DIR = PROJECT_ROOT / "data" / "dataset_raw"
TARGET_IMAGES_PER_CLASS = 250
RANDOM_SEED = 42

# Pemetaan nama folder sumber Kaggle -> 6 Kategori Baku SORTARA
FOLDER_TO_SORTARA_CLASS: dict[str, str] = {
    "glass": "Kaca",
    "brown-glass": "Kaca",
    "green-glass": "Kaca",
    "white-glass": "Kaca",
    "paper": "Kertas",
    "cardboard": "Kertas",
    "metal": "Logam",
    "biological": "Organik",
    "organic": "Organik",
    "plastic": "Plastik",
    "trash": "Residu",
}


def is_valid_rgb_image(path: Path) -> bool:
    """Memastikan file gambar tidak rusak dan memiliki dimensi yang layak."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img2:
            w, h = img2.size
            return w >= 48 and h >= 48
    except Exception:
        return False


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def prepare_sortara_dataset(max_per_class: int = TARGET_IMAGES_PER_CLASS) -> dict[str, int]:
    import kagglehub  # type: ignore[import-untyped]

    print("Mengunduh dataset Garbage Classification via kagglehub...")
    downloaded_root = Path(kagglehub.dataset_download("mostafaabla/garbage-classification"))
    print(f"Dataset berhasil diunduh di: {downloaded_root}")

    candidates_by_class: dict[str, list[Path]] = {
        "Kaca": [],
        "Kertas": [],
        "Logam": [],
        "Organik": [],
        "Plastik": [],
        "Residu": [],
    }

    for folder in downloaded_root.rglob("*"):
        if folder.is_dir():
            folder_key = folder.name.lower().strip()
            if folder_key in FOLDER_TO_SORTARA_CLASS:
                target_cls = FOLDER_TO_SORTARA_CLASS[folder_key]
                for img_file in folder.iterdir():
                    if img_file.is_file() and img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        candidates_by_class[target_cls].append(img_file)

    rng = random.Random(RANDOM_SEED)
    if TARGET_DATASET_DIR.exists():
        shutil.rmtree(TARGET_DATASET_DIR)

    summary_counts: dict[str, int] = {}
    seen_hashes: set[str] = set()

    for cls_name, file_list in candidates_by_class.items():
        cls_out_dir = TARGET_DATASET_DIR / cls_name
        cls_out_dir.mkdir(parents=True, exist_ok=True)

        rng.shuffle(file_list)
        copied = 0
        for src_path in file_list:
            if copied >= max_per_class:
                break
            if not is_valid_rgb_image(src_path):
                continue
            digest = file_md5(src_path)
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)

            dst_name = f"{cls_name.lower()}_{copied + 1:04d}{src_path.suffix.lower()}"
            shutil.copy2(src_path, cls_out_dir / dst_name)
            copied += 1

        summary_counts[cls_name] = copied
        print(f"Kelas {cls_name:8s}: {copied} gambar valid disiapkan.")

    return summary_counts


if __name__ == "__main__":
    counts = prepare_sortara_dataset()
    print("Total gambar siap latih:", sum(counts.values()))
