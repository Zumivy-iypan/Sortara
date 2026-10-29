from dataclasses import dataclass
from pathlib import Path

APP_NAME: str = "SORTARA"
APP_TAGLINE: str = "Sort Smarter. Waste Better."
APP_DESCRIPTION: str = (
    "Aplikasi berbasis web yang membantu mengenali kategori sampah melalui foto satu benda dominan, "
    "menampilkan skor keyakinan prediksi AI, rekomendasi pemilahan, panduan warna tempat sampah, "
    "serta edukasi waktu terurai dan dampak lingkungannya."
)

# FR-02 & FR-04: Validasi format dan ukuran maksimal file (10 MB)
ALLOWED_EXTENSIONS: tuple[str, ...] = ("jpg", "jpeg", "png")
MAX_FILE_SIZE_MB: int = 10
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024

# FR-06: Ukuran input model MobileNetV2
INPUT_IMAGE_SIZE: tuple[int, int] = (224, 224)

# FR-07: Enam kategori klasifikasi sesuai urutan alfabetis dataset Keras
CLASS_LABELS: tuple[str, ...] = (
    "Kaca",
    "Kertas",
    "Logam",
    "Organik",
    "Plastik",
    "Residu",
)

# FR-10: Ambang batas keyakinan default (dapat disesuaikan dari hasil evaluasi validation set)
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.60

# Lokasi default file model, kontrak label, dan dataset contoh
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH: Path = PROJECT_ROOT / "models" / "sortara_model.keras"
DEFAULT_LABELS_PATH: Path = PROJECT_ROOT / "models" / "labels.json"
DEFAULT_DATASET_DIR: Path = PROJECT_ROOT / "data" / "dataset_raw"

# FR-10: Pesan standar ketika prediksi di bawah ambang keyakinan
LOW_CONFIDENCE_MESSAGE: str = (
    "SORTARA belum dapat mengenali gambar ini. Coba gunakan foto dengan satu benda, "
    "pencahayaan cukup, dan objek terlihat jelas."
)

# FR-12 & Bagian 8: Disclaimer transparansi AI & aturan pengelolaan setempat
LOCAL_RULE_DISCLAIMER: str = (
    "Rekomendasi ini bersifat saran awal secara umum. Prediksi AI dapat keliru dan kategori visual "
    "tidak selalu menjamin kelayakan daur ulang di setiap fasilitas. Selalu ikuti aturan pengelolaan "
    "sampah setempat di daerah Anda."
)


@dataclass(frozen=True)
class CategoryRule:
    category: str
    icon: str
    badge_color: str
    examples: str
    handling_advice: str
    preparation_steps: tuple[str, ...]
    bin_color_guide: str
    bin_badge_color: str
    decomposition_time: str
    littering_impact: str
    recycling_benefit: str
    upcycling_ideas: tuple[str, ...]


# Bagian 5 & Bagian 10 PRD + Fitur Edukasi Lingkungan, Upcycling, & Warna Tong Sampah
CATEGORY_RULES: dict[str, CategoryRule] = {
    "Plastik": CategoryRule(
        category="Plastik",
        icon=":material/water_bottle:",
        badge_color="blue",
        examples="Botol minuman plastik (PET), gelas plastik, dan wadah kemasan plastik",
        handling_advice=(
            "Kosongkan dan bersihkan jika memungkinkan; periksa apakah fasilitas setempat "
            "menerima jenis plastik tersebut."
        ),
        preparation_steps=(
            "Kosongkan sisa cairan atau makanan dari dalam wadah plastik.",
            "Bilas singkat dan keringkan agar tidak mengotori sampah daur ulang lain.",
            "Lepaskan tutup atau label dan pipihkan botol untuk menghemat ruang wadah.",
        ),
        bin_color_guide="Tong Kuning (Sampah Anorganik / Daur Ulang)",
        bin_badge_color="orange",
        decomposition_time="100 – 500 tahun (terpecah menjadi mikroplastik)",
        littering_impact=(
            "Jika dibuang sembarangan, plastik menyumbat saluran drainase pemicu banjir dan terfragmentasi "
            "menjadi mikroplastik yang mencemari tanah, sungai, laut, serta rantai makanan."
        ),
        recycling_benefit=(
            "Daur ulang 1 ton plastik menghemat hingga ~5.700 kWh energi listrik dan mengurangi ketergantungan "
            "pada bahan baku minyak bumi baru."
        ),
        upcycling_ideas=(
            "Potong botol plastik bekas sebagai wadah semai tanaman atau pot gantung vertikal.",
            "Manfaatkan wadah plastik tebal sebagai organizer alat tulis atau perkakas kecil di meja.",
        ),
    ),
    "Kertas": CategoryRule(
        category="Kertas",
        icon=":material/description:",
        badge_color="orange",
        examples="Kertas HVS, koran, majalah, dan kardus kemasan",
        handling_advice=(
            "Jaga agar tetap kering dan tidak terlalu kotor; pisahkan dari bahan basah atau berminyak "
            "jika memungkinkan karena penerimaan bergantung pada fasilitas daur ulang."
        ),
        preparation_steps=(
            "Pastikan kertas atau kardus tidak basah dan bebas dari noda minyak/makanan.",
            "Lipat atau pipihkan kardus untuk menghemat ruang penyimpanan.",
            "Lepaskan selotip plastik atau staples besar jika memungkinkan.",
        ),
        bin_color_guide="Tong Biru (Khusus Kertas & Kardus)",
        bin_badge_color="blue",
        decomposition_time="2 – 6 bulan di alam terbuka",
        littering_impact=(
            "Jika tercampur sampah basah atau dibuang di TPA tanpa dipilah, pembusukan kertas secara anaerob "
            "menghasilkan gas metana (CH4) serta merusak serat selulosa yang seharusnya bisa didaur ulang."
        ),
        recycling_benefit=(
            "Mendaur ulang 1 ton kertas/kardus menyelamatkan sekitar 17 pohon dewasa dan menghemat hingga "
            "26.000 liter air dalam proses produksi bubur kertas."
        ),
        upcycling_ideas=(
            "Gunakan sisi kosong kertas HVS bekas sebagai kertas catatan/buram sebelum didaur ulang.",
            "Manfaatkan kardus kokoh sebagai kotak penyimpanan arsip, buku, atau pelindung paket.",
        ),
    ),
    "Kaca": CategoryRule(
        category="Kaca",
        icon=":material/wine_bar:",
        badge_color="violet",
        examples="Botol kaca minuman, botol kecap/saus, atau stoples kaca",
        handling_advice=(
            "Tangani dengan hati-hati; pisahkan pecahan kaca dan ikuti petunjuk keselamatan "
            "serta aturan pengumpulan setempat."
        ),
        preparation_steps=(
            "Kosongkan isi botol atau stoples kaca dan bilas seperlunya.",
            "Jika kaca pecah, bungkus rapat dengan kardus tebal dan beri tulisan peringatan.",
            "Pisahkan dari sampah rumah tangga biasa agar petugas pengumpul tetap aman.",
        ),
        bin_color_guide="Tong Kuning (Sampah Anorganik Guna Ulang / Daur Ulang)",
        bin_badge_color="orange",
        decomposition_time="> 1.000 tahun (hampir tidak terurai secara biologis)",
        littering_impact=(
            "Kaca yang dibuang sembarangan tidak dapat diurai mikroorganisme, berisiko tinggi melukai "
            "manusia maupun hewan, serta dapat memicu kebakaran lahan kering akibat efek lensa cahaya matahari."
        ),
        recycling_benefit=(
            "Kaca bersifat 100% dapat didaur ulang berulang kali tanpa penurunan kualitas murni dan menghemat "
            "hingga 30% energi peleburan pasir silika baru."
        ),
        upcycling_ideas=(
            "Cuci bersih stoples kaca bekas selai/saus untuk dijadikan wadah bumbu dapur kedap udara.",
            "Gunakan botol kaca utuh sebagai vas tanaman air (hidroponik meja) atau wadah sabun cair isi ulang.",
        ),
    ),
    "Logam": CategoryRule(
        category="Logam",
        icon=":material/takeout_dining:",
        badge_color="gray",
        examples="Kaleng minuman aluminium, kaleng susu/makanan logam",
        handling_advice=(
            "Kosongkan wadah kemasan dan periksa ketentuan fasilitas pengumpulan logam setempat."
        ),
        preparation_steps=(
            "Kosongkan dan bilas sisa makanan atau minuman dari dalam kaleng.",
            "Hati-hati terhadap bagian tepi tutup kaleng yang tajam.",
            "Simpan di tempat kering dan pisahkan menuju wadah daur ulang anorganik/logam.",
        ),
        bin_color_guide="Tong Kuning (Sampah Anorganik / Daur Ulang)",
        bin_badge_color="orange",
        decomposition_time="50 – 200 tahun (aluminium hingga 200+ tahun)",
        littering_impact=(
            "Kaleng logam yang dibuang sembarangan dapat berkarat dan mencemari tanah, menjadi tempat "
            "genangan air sarang nyamuk, serta melukai kaki akibat tepian logam yang tajam."
        ),
        recycling_benefit=(
            "Mendaur ulang kaleng aluminium menghemat hingga 95% energi dibandingkan menambang dan memurnikan "
            "bijih bauksit baru dari alam."
        ),
        upcycling_ideas=(
            "Haluskan tepi kaleng logam bekas dan hias sebagai wadah pensil/kuas di meja belajar.",
            "Lubangi bagian bawah kaleng makanan untuk dijadikan pot tanaman sukulen atau kaktus.",
        ),
    ),
    "Organik": CategoryRule(
        category="Organik",
        icon=":material/eco:",
        badge_color="green",
        examples="Kulit buah, potongan sisa sayuran, daun kering, dan sisa makanan alami",
        handling_advice=(
            "Dapat diarahkan untuk pengomposan rumah tangga atau fasilitas pengolahan organik "
            "jika sesuai dan tersedia."
        ),
        preparation_steps=(
            "Pisahkan segera dari kemasan plastik, karet, atau tusuk gigi sebelum dikomposkan.",
            "Potong sisa organik berukuran besar agar proses pengomposan lebih cepat.",
            "Masukkan ke wadah kompos tertutup atau tempat sampah khusus organik.",
        ),
        bin_color_guide="Tong Hijau (Sampah Organik Mudah Terurai)",
        bin_badge_color="green",
        decomposition_time="2 minggu – 6 bulan (tergantung jenis dan kelembapan)",
        littering_impact=(
            "Jika dibuang bercampur sampah plastik di TPA tertutup tanpa oksigen, sampah organik menghasilkan "
            "gas metana (gas rumah kaca 25x lebih kuat dari CO2), cairan lindi pencemar air tanah, dan bau menyengat."
        ),
        recycling_benefit=(
            "Pengomposan sampah organik mengurangi hingga 50% beban volume sampah rumah tangga ke TPA dan "
            "menghasilkan humus alami yang menyuburkan tanah tanpa pupuk kimia."
        ),
        upcycling_ideas=(
            "Olah kulit buah segar (jeruk/nanas/apel) bersama gula merah dan air menjadi cairan pembersih alami Eco-Enzyme.",
            "Jadikan kompos padat di pot/komposter rumah tangga untuk memupuk tanaman pekarangan.",
        ),
    ),
    "Residu": CategoryRule(
        category="Residu",
        icon=":material/delete:",
        badge_color="red",
        examples="Barang komposit berlapis (sachet), tisu kotor, popok, atau kemasan sulit didaur ulang",
        handling_advice=(
            "Ikuti aturan pembuangan daerah setempat; jangan menganggap seluruh barang residu "
            "memiliki penanganan yang sama."
        ),
        preparation_steps=(
            "Jangan mencampurkan sampah residu ke dalam tempat sampah daur ulang (kertas/plastik bersih).",
            "Pastikan tidak mengandung bahan berbahaya/B3 (seperti baterai, obat kedaluwarsa, atau lampu neon).",
            "Kemas dengan rapi dan buang sesuai jadwal pengangkutan sampah residu setempat.",
        ),
        bin_color_guide="Tong Abu-abu (Sampah Residu / Sisa Akhir)",
        bin_badge_color="gray",
        decomposition_time="Puluhan hingga ratusan tahun (material komposit berlapis)",
        littering_impact=(
            "Sampah residu seperti kemasan sachet berlapis aluminium-plastik atau tisu kotor mencemari "
            "lingkungan dalam jangka panjang dan merusak kualitas batch daur ulang jika tercampur."
        ),
        recycling_benefit=(
            "Memisahkan residu dengan disiplin menjaga agar sampah kertas, plastik, kaca, dan logam tetap "
            "bersih sehingga tingkat keberhasilan daur ulang di bank sampah meningkat drastis."
        ),
        upcycling_ideas=(
            "Kurangi timbulan residu dari hulu dengan mengganti tisu sekali pakai ke kain lap/saputangan cuci ulang.",
            "Untuk kemasan plastik/sachet bersih dan kering, dapat dipadatkan ke dalam botol menjadi Ecobrick.",
        ),
    ),
}
