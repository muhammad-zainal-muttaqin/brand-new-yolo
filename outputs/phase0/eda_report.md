# Phase 0 Dataset Audit

Dokumen ini merangkum audit dataset yang dilakukan pada Phase 0. Untuk melihat keputusan Phase 0 dalam konteks eksperimen penuh, lanjut ke [outputs/phase0/phase0_summary.md](phase0_summary.md). Untuk peta baca repo, buka [README.md](../../README.md).

## Sumber utama

Dokumen ini ditulis dari artefak berikut:

- [outputs/phase0/dataset_audit.json](dataset_audit.json)
- [outputs/phase0/class_distribution.csv](class_distribution.csv)
- [outputs/phase0/bbox_stats.csv](bbox_stats.csv)
- [outputs/phase0/leakage_report.json](leakage_report.json)
- [Dataset-YOLO/data.yaml](../../Dataset-YOLO/data.yaml)

## Ringkasan audit

- data YAML: [Dataset-YOLO/data.yaml](../../Dataset-YOLO/data.yaml)
- dataset root: `/workspace/Dataset-Sawit-YOLO`
- status audit: **ok**
- total images: **3992**
- total labels: **3992**
- total instances: **17987**
- split images: train **2764**, val **604**, test **624**
- empty-label images: **83**
- missing label images: **0**
- orphan labels: **0**
- invalid label issues: **0**
- group overlap antar split: **0**

Kesimpulan awal dari audit otomatis ini sederhana: **tidak ada blocker teknis** yang mencegah eksperimen dilanjutkan ke tahap training.

## Distribusi kelas

Distribusi kelas diambil dari [outputs/phase0/class_distribution.csv](class_distribution.csv):

| Class | Count | Share |
|---|---:|---:|
| `B1` | 2177 | 0.1210 |
| `B2` | 4073 | 0.2264 |
| `B3` | 8295 | 0.4612 |
| `B4` | 3442 | 0.1914 |

Pembacaan singkat:

- `B3` adalah kelas dominan
- `B1` adalah kelas minor dibanding kelas lain
- distribusi ini perlu diingat saat membaca metrik per kelas di fase-fase berikutnya

## Statistik bounding box

Statistik bbox diambil dari [outputs/phase0/bbox_stats.csv](bbox_stats.csv):

| Class | Count | Median width px | Median height px | Median area norm |
|---|---:|---:|---:|---:|
| `B1` | 2177 | 125.00 | 136.79 | 0.014006 |
| `B2` | 4073 | 109.09 | 120.66 | 0.010655 |
| `B3` | 8295 | 105.31 | 114.37 | 0.009645 |
| `B4` | 3442 | 93.61 | 96.12 | 0.007221 |

Pembacaan singkat:

- `B4` adalah objek paling kecil secara median
- `B1` adalah objek terbesar secara median
- pola ini membantu menjelaskan mengapa `B4` sering lebih sulit dideteksi

## Kebocoran split

Laporan kebocoran berasal dari [outputs/phase0/leakage_report.json](leakage_report.json). Hasilnya menunjukkan:

- `train__val = 0`
- `train__test = 0`
- `val__test = 0`

Artinya, audit group overlap tidak menemukan indikasi leakage antar split pada dataset aktif repo ini.

## Kesimpulan Phase 0 untuk audit dataset

Audit otomatis Phase 0 memberi tiga keputusan praktis:

1. dataset aktif **cukup bersih** untuk dipakai sebagai baseline
2. split aktif **aman** dari group overlap yang terdeteksi
3. `B4` memang memiliki beban small-object yang lebih berat daripada kelas lain

Untuk keputusan resolusi dan learning curve, lanjut ke [outputs/phase0/phase0_summary.md](phase0_summary.md).
