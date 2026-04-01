# Phase 1 Summary

Phase 1 menjawab dua pertanyaan besar: pipeline mana yang paling realistis untuk task 4-kelas ini (one-stage vs two-stage), dan arsitektur mana yang paling stabil di pipeline yang menang. Keduanya dijalankan dalam kondisi terkontrol — resolusi, batch, augmentation, dan seed sudah di-lock dari Phase 0.

Dasar keputusan resolusi dan dataset ada di [phase0_summary.md](../phase0/phase0_summary.md). Hasil tuning di [phase2_summary.md](../phase2/phase2_summary.md).

## Sumber data

- [one_stage_results.csv](one_stage_results.csv) — hasil one-stage baseline
- [two_stage_results.csv](two_stage_results.csv) — hasil two-stage per komponen
- [architecture_benchmark.csv](architecture_benchmark.csv) — benchmark 11 arsitektur
- [phase1b_top3.csv](phase1b_top3.csv) — top-3 model
- [locked_setup.yaml](locked_setup.yaml) — lock file Phase 1

## 1. Input dari Phase 0

Phase 1 membawa dua lock dari Phase 0:
- Resolusi kerja: **640**
- Dataset aktif: [Dataset-YOLO/data.yaml](../../Dataset-YOLO/data.yaml)

Semua perbandingan di Phase 1 menggunakan konfigurasi yang identik supaya hasilnya apple-to-apple.

## 2. Phase 1A — Keputusan pipeline

### One-stage baseline

Dari [one_stage_results.csv](one_stage_results.csv), one-stage detector (yolo11n, 4-class) menghasilkan:

- Mean mAP50: **0.5241**
- Mean mAP50-95: **0.2526**
- Variance antar seed sangat kecil (±0.001)

### Two-stage feasibility

Pada benchmark final Phase 3, cabang two-stage yang dibangun ulang menghasilkan:

- **Stage-1** (`last`, `test`): single-class detector → **mAP50 0.8130**, **mAP50-95 0.3860**
- **Stage-2 GT-crop** (`last`, `test`): classifier pada ground-truth crops → **top-1 64.85%**, **weighted F1 0.6337**
- **Two-stage end-to-end** (`last`, `test`): detector + classifier → **precision 0.4840**, **recall 0.5053**, **weighted F1 0.4802**

Penting: GT-crop classifier tetap hanya *upper bound*. Hasil operasional sebenarnya ada di jalur **end-to-end**, karena di situ error detector dan classifier bertemu dalam pipeline yang sama.

![Perbandingan one-stage vs two-stage](figures/p1_one_vs_two_stage.png)

### Kenapa two-stage tidak dipilih

Catatan ini sekarang mengacu ke hasil final Phase 3, bukan confusion matrix lama 2 kelas. Pada evaluasi **GT-crop classifier** `last/test`, confusion matrix penuhnya adalah:

| Ground truth \\ Prediksi | B1 | B2 | B3 | B4 |
|---|---:|---:|---:|---:|
| B1 | 297 | 36 | 3 | 0 |
| B2 | 57 | 251 | 306 | 4 |
| B3 | 2 | 152 | 1,072 | 117 |
| B4 | 1 | 15 | 306 | 223 |

Poin utamanya:

- `B2` benar hanya `40.6%`, dan paling sering salah ke `B3` (`49.5%`).
- `B4` benar hanya `40.9%`, dan paling sering salah ke `B3` (`56.1%`).
- `B3` masih bocor ke `B2` dan `B4`, walaupun diagonalnya lebih kuat.
- Pada jalur end-to-end, error detector memperparah hasil akhir, jadi cabang two-stage tidak memberi keuntungan operasional yang cukup.

Jadi alasan utama tidak berubah: bukan sekadar karena pipeline two-stage lebih panjang, tetapi karena pemisahan kelas sulitnya masih lemah bahkan pada crop ground truth. One-stage tetap lebih layak sebagai pipeline utama.

> **Keputusan: pipeline `one-stage`.**

## 3. Phase 1B — Benchmark arsitektur

Setelah pipeline di-lock, 11 arsitektur YOLO di-benchmark dalam kondisi identik: resolusi 640, `lr0=0.001`, `batch=16`, augmentasi medium, 2 seed per model.

### Ranking lengkap

![Architecture benchmark — ranking by mAP50](figures/p1_architecture_benchmark.png)

### Top-3

Dari [phase1b_top3.csv](phase1b_top3.csv):

| Rank | Model | Mean mAP50 | Mean mAP50-95 | Mean B4 Recall |
|---:|---|---:|---:|---:|
| 1 | `yolo11m.pt` | 0.5298 | 0.2570 | 0.367 |
| 2 | `yolov9c.pt` | 0.5292 | 0.2518 | 0.352 |
| 3 | `yolov8s.pt` | 0.5256 | 0.2521 | 0.411 |

Ada beberapa hal menarik dari benchmark ini:

**Gap antar model teratas sangat kecil.** Selisih yolo11m dan yolov9c hanya 0.0006 mAP50 — nyaris dalam margin of error. Ini menandakan bahwa di task dan dataset ini, bottleneck performa bukan di pilihan arsitektur model, tapi di task difficulty dan data quality itu sendiri. Ganti model family dari YOLOv8 ke YOLO11 ke YOLOv9 tidak menghasilkan lompatan performa.

**yolov8s punya B4 recall tertinggi** (0.411) meskipun overall mAP50-nya lebih rendah. Ini menarik — model yang lebih kecil (s-variant) kadang lebih baik mendeteksi objek kecil karena feature map-nya tidak terlalu ter-downsample. Tapi keunggulan ini tidak cukup untuk mengimbangi kelemahannya di kelas lain.

**Model-model besar belum tentu lebih baik.** yolov10m (0.505) dan yolo26m (0.516) kalah dari yolov8s (0.526). Ini lagi-lagi menunjukkan bahwa capacity model bukan bottleneck — data dan task yang membatasi.

### Per-class heatmap

![Per-class heatmap across architectures](figures/p1_per_class_heatmap.png)

Heatmap ini memperlihatkan mAP50 per kelas di semua arsitektur. Pola yang muncul sangat konsisten: B1 selalu hijau (tinggi), B4 selalu merah (rendah), terlepas dari model yang dipakai. Ini mengonfirmasi bahwa difficulty ranking antar kelas — B1 > B3 > B2 > B4 — adalah sifat inherent dari task dan dataset, bukan artefak dari arsitektur tertentu.

### Gate canonical dan override

Dari [locked_setup.yaml](locked_setup.yaml):
- Gate canonical `mAP50 >= 0.70`: **False** — tidak ada model yang melewati threshold ini
- Local override continue: **True**

Secara protokol E0, fase ini seharusnya berhenti karena gate tidak lolos. Tapi repo ini menggunakan override operasional agar pipeline end-to-end tetap berjalan sampai Phase 3 — keputusan ini disengaja untuk menghasilkan satu baseline lengkap yang bisa dijadikan referensi, meskipun performanya belum ideal.

## 4. Model yang di-lock

Model yang di-lock ke Phase 2: **`yolo11m.pt`**.

Lock ini artinya Phase 2 tidak membuka architecture search baru — hanya melakukan hyperparameter tuning pada satu model yang sudah dipilih.

Bukti resmi:
- [phase1b_top3.csv](phase1b_top3.csv)
- [locked_setup.yaml](locked_setup.yaml)

## 5. Keputusan akhir Phase 1

Phase 1 menghasilkan dua keputusan yang dibawa ke fase selanjutnya:

1. **Pipeline: `one-stage`** — two-stage gagal menunjukkan keunggulan, bahkan di kondisi ideal (GT crops)
2. **Model: `yolo11m.pt`** — menang tipis tapi konsisten, dan menunjukkan bahwa bottleneck bukan di arsitektur

## 6. Langkah berikutnya

Setelah pipeline dan model di-lock, eksperimen lanjut ke Phase 2 untuk menjawab pertanyaan: apakah tuning hyperparameter bisa mendorong performa melewati ceiling yang terlihat di Phase 1? Buka [phase2_summary.md](../phase2/phase2_summary.md).
