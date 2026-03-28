# Phase 0 — Dataset Audit & EDA

Sebelum melatih model apapun, kita perlu memastikan dataset dalam kondisi yang layak. Dokumen ini merangkum audit dataset dan exploratory data analysis yang dilakukan di Phase 0 — mulai dari integritas file, distribusi kelas, geometri bounding box, sampai pengecekan kebocoran antar split.

Keputusan Phase 0 dalam konteks eksperimen penuh ada di [phase0_summary.md](phase0_summary.md). Peta baca repo di [README.md](../../README.md).

## Sumber data

- [dataset_audit.json](dataset_audit.json) — hasil audit otomatis
- [class_distribution.csv](class_distribution.csv) — distribusi kelas
- [bbox_stats.csv](bbox_stats.csv) — statistik bounding box per kelas
- [leakage_report.json](leakage_report.json) — laporan group overlap antar split
- [Dataset-YOLO/data.yaml](../../Dataset-YOLO/data.yaml) — konfigurasi dataset

## 1. Ringkasan audit

| Item | Nilai |
|---|---|
| Data YAML | [Dataset-YOLO/data.yaml](../../Dataset-YOLO/data.yaml) |
| Dataset root | `/workspace/Dataset-Sawit-YOLO` |
| Status audit | **ok** |
| Total images | **3,992** |
| Total labels | **3,992** |
| Total instances | **17,987** |
| Split images | train **2,764** / val **604** / test **624** |
| Empty-label images | **83** |
| Missing label images | **0** |
| Orphan labels | **0** |
| Invalid label issues | **0** |
| Group overlap antar split | **0** |

Audit otomatis tidak menemukan blocker teknis: tidak ada file yang hilang, tidak ada label yang invalid, dan tidak ada indikasi leakage antar split. Dataset siap dipakai untuk training.

83 image tanpa label perlu dicatat — ini bisa jadi image background (tanpa objek sawit) yang sengaja disertakan untuk mengurangi false positive, atau image yang terlewat di-annotate. Jumlahnya ~2% dari total, sehingga dampaknya terhadap training minimal.

## 2. Distribusi kelas

| Class | Count | Share |
|---|---:|---:|
| `B1` | 2,177 | 12.1% |
| `B2` | 4,073 | 22.6% |
| `B3` | 8,295 | 46.1% |
| `B4` | 3,442 | 19.1% |

![Distribusi kelas dalam dataset](figures/eda_class_distribution.png)

Distribusi ini sangat tidak merata — B3 mendominasi dengan hampir separuh dari seluruh instance, sementara B1 hanya 12%. Ketimpangan ini bukan kebetulan; secara biologis, buah pada tahap B3 (hitam, berduri) memang paling banyak ditemukan di tandan karena periode waktu di tahap ini relatif panjang.

Implikasi untuk training cukup signifikan. Model akan melihat B3 hampir 4× lebih sering dari B1, yang bisa membuat model bias ke arah B3 — terutama saat objek ambigu. Ini juga sebagian menjelaskan kenapa B3 recall cenderung tinggi di fase-fase selanjutnya (model "default" ke B3 saat ragu), sementara B1 yang jumlahnya sedikit justru performanya paling baik karena visual signature-nya paling distinct.

## 3. Statistik bounding box

| Class | Count | Median Width (px) | Median Height (px) | Median Area (norm) |
|---|---:|---:|---:|---:|
| `B1` | 2,177 | 125.0 | 136.8 | 0.0140 |
| `B2` | 4,073 | 109.1 | 120.7 | 0.0107 |
| `B3` | 8,295 | 105.3 | 114.4 | 0.0096 |
| `B4` | 3,442 | 93.6 | 96.1 | 0.0072 |

![Perbandingan ukuran bounding box per kelas](figures/eda_bbox_size_comparison.png)

Ada gradasi ukuran yang sangat konsisten dari B1 ke B4 — dan ini langsung mencerminkan tahap kematangan biologis. B1 (buah matang) adalah yang terbesar karena sudah membengkak penuh, sementara B4 (buah paling muda) masih kecil dan compact di dalam tandan.

Perbedaan area antara B1 dan B4 hampir 2× (0.014 vs 0.007 normalized). Pada resolusi kerja 640, B4 dengan median width ~94px masih cukup besar untuk dideteksi secara teori, tapi dalam praktiknya B4 sering tertutup oleh buah lain atau berada di posisi dalam tandan yang sulit dijangkau kamera. Kombinasi ukuran kecil + oklusi inilah yang membuat B4 konsisten menjadi kelas tersulit di sepanjang eksperimen.

Perlu juga dicatat bahwa B2 dan B3 punya ukuran yang berdekatan (median area 0.0107 vs 0.0096) — model tidak bisa mengandalkan ukuran saja untuk membedakan keduanya, dan harus bergantung pada fitur warna/tekstur yang, seperti terlihat di fase selanjutnya, memang sulit dipisahkan.

## 4. Kebocoran split (leakage check)

| Pasangan Split | Overlap |
|---|---:|
| train ↔ val | 0 |
| train ↔ test | 0 |
| val ↔ test | 0 |

Data dari [leakage_report.json](leakage_report.json). Group overlap dihitung berdasarkan prefix nama file (asumsi: image dari tandan yang sama punya prefix yang sama). Hasilnya bersih — tidak ada grup yang bocor antar split.

Ini penting karena pada dataset sawit, image dari tandan yang sama bisa terlihat sangat mirip (hanya beda sudut pandang). Kalau image dari satu tandan tersebar ke train dan test, performa test set akan terlalu optimistik. Audit ini mengonfirmasi bahwa evaluasi yang dilakukan di fase-fase berikutnya memang jujur.

## 5. Kesimpulan

Audit dataset Phase 0 memberikan tiga kesimpulan utama:

1. **Dataset cukup bersih untuk baseline.** Tidak ada file yang hilang, tidak ada label yang invalid, dan split sudah aman dari leakage. Bukan dataset sempurna (ada 83 empty-label images), tapi tidak ada alasan teknis untuk menunda eksperimen.

2. **Ketimpangan distribusi kelas perlu diwaspadai.** B3 mendominasi hampir 4× lipat dari B1. Ini akan mempengaruhi learning dynamics — model cenderung bias ke majority class, dan metrik overall bisa "tersembunyi" di balik performa B3 yang inflated.

3. **B4 punya beban ganda: minority class + small object.** Ukuran fisik terkecil ditambah posisi yang sering tersembunyi membuat B4 secara inherent menjadi kelas tersulit. Ini bukan masalah yang bisa diselesaikan hanya dengan menambah epoch atau mengubah loss function — perlu pendekatan yang lebih targeted.

Untuk keputusan resolusi kerja dan learning curve, lanjut ke [phase0_summary.md](phase0_summary.md).
