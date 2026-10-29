# PRD — SORTARA

## Smart Waste Classification System

**Dokumen:** Product Requirements Document (PRD)  
**Versi:** 1.0  
**Status:** Draft untuk prototipe tugas Kecerdasan Buatan  
**Tim:** Adit (Aditya Rafaul) dan Irfan Dwiki  
**Tanggal:** 28 Oktober 2026

---

## 1. Ringkasan Produk

SORTARA adalah aplikasi berbasis web yang membantu pengguna mengenali kategori sampah melalui foto. Pengguna mengunggah foto satu benda, kemudian model AI menganalisis gambar dan memprediksi kategori sampah. Setelah klasifikasi, SORTARA menampilkan hasil prediksi, skor keyakinan model, serta saran awal untuk menangani atau memilah barang tersebut.

SORTARA pada versi awal merupakan sistem **klasifikasi gambar**. Aplikasi mengategorikan objek yang tampak dominan pada satu foto; aplikasi belum mendeteksi banyak objek dalam satu gambar dan tidak melakukan pemilahan sampah secara fisik.

### 1.1 Masalah yang Ingin Diselesaikan

Pengguna terkadang tidak mengetahui kategori tempat sampah yang sesuai untuk barang yang akan dibuang. Informasi mengenai pemilahan dan daur ulang juga dapat berbeda menurut material, kondisi barang, dan fasilitas setempat. SORTARA membantu memberikan petunjuk awal melalui antarmuka yang sederhana.

### 1.2 Solusi

Pengguna mengunggah foto sampah, AI mengklasifikasikan gambar ke salah satu kategori yang didukung, lalu aplikasi menampilkan hasil dan rekomendasi penanganan. Apabila foto tidak dapat diprediksi dengan keyakinan yang memadai, aplikasi meminta pengguna mencoba foto lain.

---

## 2. Tujuan dan Indikator Keberhasilan

### 2.1 Tujuan

1. Membuat prototipe web yang dapat menerima unggahan gambar sampah.
2. Menggunakan model klasifikasi gambar berbasis AI untuk memprediksi kategori sampah.
3. Menampilkan hasil prediksi dengan cara yang mudah dipahami.
4. Memberikan rekomendasi awal untuk pemilahan atau penanganan sampah.
5. Menunjukkan proses pengembangan AI, mulai dari dataset, pelatihan, evaluasi, hingga integrasi ke aplikasi.

### 2.2 Indikator Keberhasilan

- Pengguna dapat mengunggah gambar dengan format yang didukung.
- Aplikasi dapat menampilkan pratinjau foto sebelum klasifikasi.
- Aplikasi menghasilkan satu kategori prediksi untuk foto yang sesuai dengan cakupan model.
- Aplikasi menampilkan rekomendasi yang berhubungan dengan kategori prediksi.
- Aplikasi menampilkan pesan yang jelas ketika file tidak valid atau prediksi belum meyakinkan.
- Performa model dilaporkan menggunakan accuracy, precision, recall, dan confusion matrix pada data uji yang tidak digunakan saat training.
- Prototipe dapat dijalankan dan didemonstrasikan melalui komputer atau browser.

Target numerik akurasi ditetapkan setelah baseline model diuji. Nilai target tidak boleh diklaim sebagai hasil sebelum pengujian dilakukan.

---

## 3. Target Pengguna

### Pengguna utama

Masyarakat umum atau mahasiswa yang ingin mengetahui kategori sampah dan memperoleh saran awal mengenai pemilahannya.

### Kebutuhan pengguna

- Cara sederhana untuk mengunggah foto.
- Hasil klasifikasi yang mudah dibaca.
- Penjelasan singkat dan rekomendasi praktis.
- Pesan yang jelas ketika foto tidak dapat dikenali.

---

## 4. Cakupan Produk

### 4.1 Termasuk dalam MVP

- Halaman utama SORTARA.
- Unggah satu gambar untuk satu objek sampah yang dominan.
- Pratinjau gambar yang diunggah.
- Validasi format dan ukuran file.
- Preprocessing gambar sebelum inferensi model.
- Prediksi kategori dengan model klasifikasi gambar.
- Tampilan kategori dan skor keyakinan prediksi.
- Rekomendasi awal berdasarkan kategori hasil prediksi.
- Penanganan kesalahan unggahan dan prediksi dengan keyakinan rendah.
- Evaluasi model dan dokumentasi hasil pengujian.

### 4.2 Tidak Termasuk dalam MVP

- Deteksi dan klasifikasi banyak benda dalam satu foto.
- Segmentasi area sampah per piksel.
- Pemilahan sampah secara fisik menggunakan perangkat keras atau robot.
- Pelacakan lokasi tempat daur ulang secara langsung.
- Akun pengguna, riwayat klasifikasi, dan basis data pengguna.
- Klaim bahwa suatu benda pasti dapat didaur ulang di semua wilayah.

Fitur-fitur tersebut dapat dipertimbangkan sebagai pengembangan berikutnya.

---

## 5. Kategori Klasifikasi

Model awal mendukung enam kategori berikut:

| Kategori | Contoh | Catatan rekomendasi |
|---|---|---|
| Plastik | Botol dan wadah plastik | Kosongkan dan bersihkan jika memungkinkan; periksa apakah fasilitas setempat menerima jenis plastik tersebut. |
| Kertas | Kertas, koran, kardus | Jaga agar tetap kering dan tidak terlalu kotor; penerimaan bergantung pada fasilitas daur ulang. |
| Kaca | Botol atau stoples kaca | Tangani dengan hati-hati; pisahkan pecahan kaca dan ikuti aturan pengumpulan setempat. |
| Logam | Kaleng aluminium atau kaleng logam | Kosongkan wadah dan periksa ketentuan pengumpulan setempat. |
| Organik | Kulit buah dan sisa sayuran | Dapat diarahkan untuk pengomposan jika sesuai dan tersedia. |
| Residu | Barang yang tidak termasuk kategori lain atau sulit didaur ulang | Ikuti aturan pembuangan lokal; jangan menganggap seluruh barang residu memiliki penanganan yang sama. |

Kategori ditentukan berdasarkan label dataset. Kategori visual tidak selalu cukup untuk memastikan komposisi material atau kelayakan daur ulang suatu barang.

---

## 6. Alur Kerja Pengguna dan Sistem

1. **Pengguna membuka SORTARA.** Halaman menjelaskan fungsi aplikasi dan cara mengambil foto yang sesuai.
2. **Pengguna mengunggah foto.** Foto dipilih dari perangkat dalam format yang didukung.
3. **Sistem memvalidasi file.** Sistem memeriksa format, ukuran file, serta apakah gambar dapat dibaca.
4. **Sistem menampilkan pratinjau.** Pengguna dapat memastikan foto yang dipilih sudah benar.
5. **Pengguna menjalankan klasifikasi.** Pengguna menekan tombol untuk memulai analisis.
6. **Sistem melakukan preprocessing.** Gambar diubah ke ukuran dan format numerik yang sesuai dengan model.
7. **Model AI melakukan inferensi.** Model memprediksi kategori untuk objek yang dominan pada foto.
8. **Sistem memeriksa hasil prediksi.** Jika keyakinan model rendah atau hasil belum dapat digunakan, aplikasi menampilkan pesan untuk mencoba foto lain. Jika tidak, aplikasi melanjutkan ke hasil.
9. **Sistem menampilkan hasil.** Hasil memuat gambar, kategori, skor keyakinan prediksi, dan rekomendasi penanganan.
10. **Pengguna membaca rekomendasi.** Pengguna dapat mengunggah foto lain untuk melakukan klasifikasi berikutnya.

### Diagram alur ringkas

```text
Mulai
  ↓
Buka SORTARA
  ↓
Unggah foto satu benda sampah
  ↓
Validasi file
  ├── Tidak valid → Tampilkan pesan kesalahan → Unggah ulang
  └── Valid
        ↓
   Tampilkan pratinjau
        ↓
   Pengguna menekan "Klasifikasikan"
        ↓
   Preprocessing gambar
        ↓
   Model AI memprediksi kategori
        ↓
   Apakah hasil cukup meyakinkan?
        ├── Tidak → Tampilkan "Belum dapat dikenali" dan saran foto ulang
        └── Ya → Tampilkan kategori, skor, dan rekomendasi
                         ↓
                   Selesai / unggah foto lain
```

**Catatan:** ambang keyakinan perlu ditentukan melalui pengujian pada validation set. Skor keyakinan merupakan keluaran model, bukan jaminan kebenaran.

---

## 7. Persyaratan Fungsional

| ID | Persyaratan | Prioritas |
|---|---|---|
| FR-01 | Sistem menampilkan nama dan penjelasan singkat SORTARA. | Must |
| FR-02 | Sistem menerima unggahan gambar JPG, JPEG, dan PNG. | Must |
| FR-03 | Sistem menolak file yang tidak didukung atau tidak dapat dibaca dan menampilkan pesan yang jelas. | Must |
| FR-04 | Sistem membatasi ukuran file unggahan; batas awal yang disarankan adalah 10 MB dan dapat disesuaikan. | Must |
| FR-05 | Sistem menampilkan pratinjau gambar yang dipilih. | Must |
| FR-06 | Sistem memproses gambar dengan preprocessing yang konsisten dengan proses training. | Must |
| FR-07 | Sistem memprediksi salah satu dari enam kategori yang didukung model. | Must |
| FR-08 | Sistem menampilkan nama kategori dan skor keyakinan prediksi. | Must |
| FR-09 | Sistem menampilkan rekomendasi berdasarkan kategori yang diprediksi. | Must |
| FR-10 | Sistem menampilkan pesan alternatif apabila skor prediksi di bawah ambang yang ditetapkan atau hasil tidak dapat digunakan. | Must |
| FR-11 | Pengguna dapat menghapus atau mengganti gambar dan melakukan klasifikasi baru. | Must |
| FR-12 | Sistem menyediakan penjelasan bahwa rekomendasi bersifat umum dan pengguna perlu mengikuti aturan pengelolaan sampah setempat. | Must |
| FR-13 | Sistem dapat menampilkan pesan ketika model gagal dimuat atau proses prediksi mengalami kesalahan. | Should |

---

## 8. Persyaratan Non-Fungsional

- **Kemudahan penggunaan:** alur inti dapat digunakan tanpa membuat akun.
- **Responsivitas:** halaman dapat digunakan pada layar desktop dan ponsel.
- **Kinerja:** untuk prototipe lokal, hasil diharapkan muncul dalam beberapa detik pada perangkat yang mampu menjalankan model; waktu sebenarnya diukur ketika pengujian.
- **Keandalan:** kesalahan file dan kegagalan model tidak boleh membuat aplikasi berhenti tanpa pesan yang dapat dipahami.
- **Privasi:** gambar hanya diproses untuk klasifikasi pada sesi berjalan dan tidak disimpan permanen dalam MVP, kecuali tim kemudian menetapkan kebijakan yang jelas dan memberi tahu pengguna.
- **Pemeliharaan:** kode preprocessing, prediksi, dan rekomendasi dipisahkan agar mudah diuji dan diperbaiki.
- **Transparansi:** antarmuka menjelaskan bahwa prediksi AI dapat keliru.

---

## 9. Rancangan AI

### 9.1 Pendekatan Model

Gunakan **Convolutional Neural Network (CNN)** melalui pendekatan **transfer learning**, dengan MobileNetV2 sebagai model awal yang direkomendasikan. Model pretrained digunakan sebagai ekstraktor fitur, kemudian lapisan klasifikasi disesuaikan dengan enam kategori SORTARA.

MobileNetV2 merupakan pilihan awal karena relatif ringan dibandingkan beberapa model CNN yang lebih besar. Kinerja dan kelayakannya tetap harus diukur pada dataset yang dipilih.

### 9.2 Dataset

- Kumpulkan gambar berlabel untuk enam kategori.
- Pastikan definisi kelas konsisten dan contoh gambar cukup beragam.
- Periksa gambar duplikat, gambar rusak, dan label yang keliru.
- Target awal yang dapat dipertimbangkan adalah 200–500 gambar per kelas, jika tersedia. Jumlah ini merupakan target perencanaan, bukan jaminan akurasi.
- Pisahkan dataset menjadi training, validation, dan testing. Contoh pembagian awal adalah 70% / 15% / 15%.
- Hindari menempatkan foto yang sama atau hampir identik pada bagian training dan testing.
- Pastikan hak penggunaan dataset sesuai dengan ketentuan sumbernya.

### 9.3 Preprocessing dan Training

1. Muat gambar dan label.
2. Ubah gambar ke ukuran input yang ditetapkan model, misalnya 224 × 224 piksel.
3. Terapkan fungsi preprocessing yang sesuai dengan MobileNetV2.
4. Gunakan augmentasi ringan hanya pada data training jika sesuai.
5. Latih lapisan klasifikasi awal.
6. Jika diperlukan, lakukan fine-tuning secara terbatas.
7. Pilih model berdasarkan performa validation set.
8. Evaluasi model terpilih pada test set yang tidak digunakan untuk memilih model.
9. Simpan model untuk dipakai aplikasi, misalnya `sortara_model.keras`.

### 9.4 Evaluasi

Laporan evaluasi sekurang-kurangnya memuat:

- Accuracy keseluruhan.
- Precision dan recall per kelas.
- Confusion matrix.
- Contoh prediksi benar dan salah.
- Keterbatasan dataset dan kemungkinan bias.

Jangan menuliskan angka performa sebagai hasil final sebelum eksperimen dijalankan.

### 9.5 Penanganan Keyakinan Rendah

Sistem dapat menampilkan pesan seperti **“SORTARA belum dapat mengenali gambar ini. Coba gunakan foto dengan satu benda, pencahayaan cukup, dan objek terlihat jelas.”** Ambang keyakinan ditentukan melalui validation set. Skor tinggi pun tidak menjamin bahwa prediksi selalu benar, khususnya untuk gambar di luar kategori training.

---

## 10. Rekomendasi Hasil

Rekomendasi dibuat berdasarkan kategori prediksi menggunakan aturan sederhana yang dapat diperiksa dan diperbarui oleh tim. Contohnya:

- **Plastik:** kosongkan dan bersihkan jika memungkinkan; periksa jenis plastik dan aturan fasilitas setempat.
- **Kertas:** pisahkan dari bahan basah atau berminyak jika memungkinkan.
- **Kaca:** tangani dengan hati-hati; pecahan kaca perlu mengikuti petunjuk keselamatan dan aturan pengumpulan lokal.
- **Logam:** kosongkan kemasan dan periksa fasilitas penerima logam.
- **Organik:** pertimbangkan pengomposan untuk sisa organik yang sesuai.
- **Residu:** ikuti aturan pembuangan daerah setempat.

Rekomendasi ini merupakan saran umum. SORTARA tidak boleh mengklaim bahwa suatu barang pasti dapat didaur ulang hanya dari tampilan foto.

---

## 11. Rancangan Antarmuka

### Halaman utama

- Logo atau nama SORTARA.
- Tagline opsional: **“Sort Smarter. Waste Better.”**
- Deskripsi singkat fungsi aplikasi.
- Area upload foto.
- Petunjuk: unggah satu benda dan pastikan objek terlihat jelas.

### Area pratinjau

- Menampilkan foto yang dipilih.
- Tombol untuk mengganti atau menghapus foto.
- Tombol **“Klasifikasikan Sampah”**.

### Area hasil

- Foto yang dianalisis.
- Nama kategori hasil prediksi.
- Skor keyakinan model.
- Rekomendasi penanganan.
- Catatan bahwa prediksi dapat keliru dan aturan setempat tetap berlaku.
- Tombol **“Coba Foto Lain”**.

### Keadaan yang perlu ditangani

- Belum ada foto yang diunggah.
- Format atau ukuran file tidak sesuai.
- Gambar gagal dibaca.
- Model sedang memproses.
- Prediksi berhasil.
- Prediksi belum meyakinkan.
- Model gagal dimuat atau terjadi kesalahan pemrosesan.

---

## 12. Arsitektur dan Teknologi

### Teknologi yang direkomendasikan

| Komponen | Teknologi |
|---|---|
| Bahasa pemrograman | Python |
| Eksperimen dan training | Google Colab |
| Pengolahan dan training model | TensorFlow / Keras |
| Model awal | MobileNetV2 dengan transfer learning |
| Antarmuka web | Streamlit |
| Penyimpanan model | File model `.keras` |
| Evaluasi | Scikit-learn dan visualisasi confusion matrix |

### Aliran komponen

```text
Pengguna
   ↓
Antarmuka Streamlit
   ↓
Validasi dan preprocessing gambar
   ↓
Model CNN / MobileNetV2
   ↓
Kategori prediksi + skor keyakinan
   ↓
Aturan rekomendasi berdasarkan kategori
   ↓
Hasil ditampilkan kepada pengguna
```

Model dilatih di Google Colab, lalu file model yang telah dievaluasi digunakan oleh aplikasi Streamlit. Model tidak dilatih ulang setiap kali pengguna mengunggah gambar.

---

## 13. Pengujian dan Kriteria Penerimaan

| ID | Skenario | Kriteria lulus |
|---|---|---|
| AT-01 | Mengunggah gambar JPG yang valid | Pratinjau ditampilkan tanpa error. |
| AT-02 | Mengunggah gambar PNG yang valid | Gambar dapat dibaca dan dipratinjau. |
| AT-03 | Mengunggah format yang tidak didukung | Sistem menampilkan pesan kesalahan dan tidak menjalankan klasifikasi. |
| AT-04 | Mengunggah file yang melebihi batas | Sistem menolak file dan memberi tahu pengguna. |
| AT-05 | Mengklasifikasikan foto sampah dari salah satu kelas | Sistem menampilkan kategori prediksi, skor, dan rekomendasi. |
| AT-06 | Menguji gambar buram atau tidak jelas | Sistem memberi pesan sesuai hasil validasi dan mekanisme keyakinan yang diterapkan. |
| AT-07 | Memilih foto baru setelah hasil keluar | Sistem dapat menjalankan klasifikasi baru. |
| AT-08 | Model tidak dapat dimuat | Aplikasi menampilkan pesan kegagalan yang dapat dipahami. |
| AT-09 | Mengukur performa model | Metrik evaluasi dilaporkan dari test set terpisah. |
| AT-10 | Menggunakan halaman di desktop dan ponsel | Elemen utama dapat dibaca dan dioperasikan. |

---

## 14. Pembagian Tugas Tim

Pembagian awal untuk dua anggota adalah sebagai berikut. Kedua anggota tetap perlu saling meninjau dan memahami hasil kerja masing-masing.

### Adit (Aditya Rafaul)

- Mengumpulkan, meninjau, dan merapikan dataset.
- Menyiapkan preprocessing dan eksperimen training model.
- Mengevaluasi model dan mencatat hasilnya.
- Menyusun bagian laporan terkait dataset dan evaluasi AI.

### Irfan Dwiki

- Mendesain halaman web SORTARA.
- Membuat alur upload, validasi, dan pratinjau foto.
- Mengintegrasikan model ke aplikasi dan menampilkan hasil prediksi.
- Menguji alur penggunaan serta mendokumentasikan antarmuka.

### Pekerjaan bersama

- Menentukan label dan aturan rekomendasi.
- Menguji aplikasi dari awal hingga akhir.
- Meninjau hasil klasifikasi dan keterbatasan sistem.
- Menyiapkan laporan dan demonstrasi presentasi.

---

## 15. Risiko dan Mitigasi

| Risiko | Dampak | Mitigasi |
|---|---|---|
| Dataset tidak seimbang atau label kurang tepat | Model bias atau sering salah pada kelas tertentu | Periksa distribusi kelas, label, dan kualitas foto. |
| Foto nyata berbeda dari dataset | Performa menurun pada penggunaan sebenarnya | Gunakan variasi latar, pencahayaan, sudut, dan foto uji yang berbeda. |
| Kategori visual tertukar | Saran pengelolaan bisa tidak sesuai | Tampilkan catatan keterbatasan dan evaluasi metrik per kelas. |
| Model selalu memilih satu kelas untuk gambar di luar dataset | Pengguna menerima hasil yang menyesatkan | Tentukan ambang keyakinan, uji foto di luar kategori, dan pertimbangkan kelas tambahan “lainnya/tidak dikenali”. |
| Rekomendasi tidak sesuai aturan daerah | Pengguna bisa salah memilah barang | Gunakan bahasa sebagai saran umum dan arahkan pengguna memeriksa aturan setempat. |
| Waktu pengerjaan terbatas | Fitur inti tidak selesai | Prioritaskan alur MVP satu benda per foto sebelum fitur lanjutan. |

---

## 16. Rencana Implementasi

Rencana awal untuk prototipe adalah tujuh hari dan dapat disesuaikan dengan jadwal perkuliahan.

| Hari | Pekerjaan utama |
|---|---|
| 1 | Menetapkan kategori dan mengumpulkan dataset. |
| 2 | Memeriksa label, membersihkan data, dan membuat pembagian dataset. |
| 3 | Melatih baseline model MobileNetV2. |
| 4 | Mengevaluasi model, melakukan perbaikan seperlunya, dan menyimpan model. |
| 5 | Membuat halaman upload dan pratinjau dengan Streamlit. |
| 6 | Mengintegrasikan model, menampilkan prediksi, dan menambahkan rekomendasi. |
| 7 | Pengujian akhir, perbaikan antarmuka, laporan, dan persiapan demo. |

Durasi tersebut adalah target perencanaan, bukan jaminan bahwa semua pekerjaan akan selesai dalam tujuh hari.

---

## 17. Pengembangan Berikutnya

Setelah MVP berfungsi, pengembangan dapat mencakup:

- Object detection untuk mengenali beberapa benda dalam satu foto.
- Segmentasi untuk menunjukkan area setiap benda.
- Kelas “lainnya/tidak dikenali” yang lebih kuat untuk mengurangi klasifikasi paksa.
- Riwayat klasifikasi yang disimpan secara sukarela.
- Informasi lokasi fasilitas pengumpulan atau daur ulang.
- Pengenalan kode material pada kemasan jika layak secara teknis.

Fitur-fitur ini tidak menjadi syarat agar MVP SORTARA dapat didemonstrasikan.

---

## 18. Definisi Selesai (Definition of Done)

MVP SORTARA dianggap selesai untuk demonstrasi tugas apabila:

1. Model dapat dilatih dan dievaluasi menggunakan dataset terdokumentasi.
2. Metrik evaluasi dihitung dan dilaporkan secara jujur.
3. Aplikasi dapat menerima foto dan menampilkan pratinjau.
4. Model menghasilkan prediksi dan aplikasi menampilkan kategori beserta rekomendasi.
5. File tidak valid dan hasil yang belum meyakinkan ditangani dengan pesan yang jelas.
6. Tim dapat mendemonstrasikan alur lengkap dan menjelaskan keterbatasan model.

---

## 19. Glosarium

- **AI (Artificial Intelligence):** teknik untuk membangun sistem yang melakukan tugas yang membutuhkan kemampuan seperti pengenalan pola atau pengambilan keputusan.
- **Klasifikasi gambar:** proses memprediksi label kategori untuk sebuah gambar.
- **CNN (Convolutional Neural Network):** jenis jaringan saraf yang banyak digunakan untuk pemrosesan gambar.
- **Transfer learning:** penggunaan model yang telah dilatih sebelumnya sebagai titik awal untuk tugas baru.
- **MobileNetV2:** arsitektur CNN yang dirancang relatif ringan untuk pemrosesan gambar.
- **Preprocessing:** proses menyiapkan gambar agar sesuai dengan masukan model.
- **Inferensi:** proses menggunakan model terlatih untuk menghasilkan prediksi.
- **Confidence score / skor keyakinan:** skor keluaran model yang membantu menggambarkan tingkat keyakinan prediksi; bukan jaminan kebenaran.
- **MVP (Minimum Viable Product):** versi awal dengan fitur penting untuk menguji konsep.

---

**Akhir dokumen PRD SORTARA**
