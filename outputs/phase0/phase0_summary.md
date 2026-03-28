# Phase 0 Summary

Phase 0 menjawab tiga pertanyaan fundamental sebelum training dimulai: apakah dataset cukup bersih, resolusi kerja mana yang paling masuk akal, dan apakah volume data yang ada sudah cukup atau masih bisa ditambah.

Audit dataset mentah ada di [eda_report.md](eda_report.md). Untuk keputusan fase selanjutnya, lanjut ke [phase1_summary.md](../phase1/phase1_summary.md).

## Sumber data

- [dataset_audit.json](dataset_audit.json) — hasil audit otomatis
- [eda_report.md](eda_report.md) — EDA lengkap
- [resolution_sweep.csv](resolution_sweep.csv) — perbandingan resolusi 640 vs 1024
- [learning_curve.csv](learning_curve.csv) — kurva belajar pada fraksi data 25%-100%
- [locked_setup.yaml](../phase1/locked_setup.yaml) — lock file yang membawa keputusan Phase 0 ke fase selanjutnya

## 1. Validasi dataset

| Item | Nilai |
|---|---|
| Total images | **3,992** |
| Total labels | **3,992** |
| Total instances | **17,987** |
| Split | train **2,764** / val **604** / test **624** |
| Empty-label images | **83** |
| Invalid issues | **0** |
| Group overlap antar split | **0** |

Dataset lolos audit dasar tanpa blocker teknis. Detail distribusi kelas dan geometri bounding box dibahas di [eda_report.md](eda_report.md) — intinya, B3 mendominasi (46%) dan B4 punya ukuran terkecil, dua fakta yang akan terus relevan di sepanjang eksperimen.

## 2. Resolution sweep

Pertanyaan ini penting karena resolusi langsung mempengaruhi dua hal: kemampuan model mendeteksi objek kecil (terutama B4), dan biaya komputasi per run. Kita membandingkan 640 vs 1024 pada model yolo11n dengan 2 seed.

| imgsz | seed | mAP50 | mAP50-95 | precision | recall |
|---:|---:|---:|---:|---:|---:|
| 640 | 1 | 0.5237 | 0.2538 | 0.4906 | 0.5864 |
| 1024 | 1 | 0.5363 | 0.2571 | 0.4888 | 0.6016 |
| 640 | 2 | 0.5245 | 0.2514 | 0.4923 | 0.5838 |
| 1024 | 2 | 0.5276 | 0.2589 | 0.4952 | 0.6004 |

**Mean per resolusi:**
- `640`: mAP50 = 0.5241, mAP50-95 = 0.2526
- `1024`: mAP50 = 0.5320, mAP50-95 = 0.2580
- Relative gain 1024 vs 640 pada mAP50-95: **+2.15%**

![Perbandingan resolusi 640 vs 1024](figures/p0_resolution_comparison.png)

Gain 2.15% itu memang ada, tapi konteksnya perlu dilihat: setiap run di 1024 memakan hampir 2.5× lebih banyak VRAM dan waktu training dibanding 640. Dalam pipeline E0 yang menjalankan puluhan run (benchmark 11 arsitektur × 2 seed, tuning sweeps, dsb.), pilihan 1024 akan menggandakan total compute budget tanpa jaminan bahwa gain kecil ini akan bertahan saat arsitektur dan hyperparameter berubah.

Sesuai aturan di [E0.md](../../E0.md), gain 2-5% tidak otomatis mengunci resolusi yang lebih tinggi — keputusan harus mempertimbangkan efisiensi keseluruhan pipeline. Karena itu, **resolusi kerja di-lock pada 640** dan dibawa ke semua fase selanjutnya melalui [locked_setup.yaml](../phase1/locked_setup.yaml).

## 3. Learning curve @ 640

Learning curve dijalankan untuk melihat apakah volume data saat ini sudah cukup, atau menambah data masih bisa memberikan gain yang signifikan.

| Fraction | mAP50 | mAP50-95 | Precision | Recall |
|---:|---:|---:|---:|---:|
| 25% | 0.4444 | 0.1984 | 0.4187 | 0.5758 |
| 50% | 0.4637 | 0.2202 | 0.4410 | 0.5791 |
| 75% | 0.5033 | 0.2444 | 0.4683 | 0.5906 |
| 100% | 0.5237 | 0.2538 | 0.4906 | 0.5864 |

![Learning curve — mAP vs fraksi data](figures/p0_learning_curve.png)

Kenaikan mAP50-95 antar step:
- 25% → 50%: **+0.0217**
- 50% → 75%: **+0.0243**
- 75% → 100%: **+0.0093**

Polanya menarik. Dari 25% ke 75%, gain per step relatif konsisten (~0.02). Tapi dari 75% ke 100%, gain tiba-tiba mengecil ke kurang dari setengahnya (0.009). Ini menunjukkan awal dari diminishing returns — model masih belajar sesuatu dari data tambahan, tapi rate-nya sudah melambat.

Apakah ini berarti menambah data tidak berguna? Tidak juga. Kurva belum benar-benar plateau (masih naik), jadi menambah data berkualitas — terutama untuk kelas underrepresented seperti B1 dan B4 — kemungkinan masih bisa membantu. Tapi menambah data secara acak tanpa memperhatikan distribusi kelas mungkin hanya memberi diminishing returns yang semakin kecil.

## 4. Keputusan akhir Phase 0

Phase 0 menutup tiga hal penting:

1. **Dataset cukup bersih untuk baseline.** Tidak ada leakage, tidak ada label invalid, split sudah terisolasi dengan benar. Bukan dataset sempurna, tapi cukup untuk membangun baseline yang jujur.

2. **Resolusi kerja = 640.** Gain dari 1024 terlalu kecil (2.15%) relatif terhadap peningkatan compute cost. Di skala pipeline E0 dengan puluhan run, 640 adalah pilihan yang paling realistis.

3. **Data belum saturasi, tapi diminishing returns sudah mulai terlihat.** Menambah data secara targeted (bukan random) masih bisa membantu, terutama untuk kelas B1 dan B4 yang underrepresented.

## 5. Langkah berikutnya

Setelah Phase 0 mengonfirmasi bahwa dataset dan resolusi sudah ter-lock, eksperimen lanjut ke Phase 1A untuk memilih pipeline (one-stage vs two-stage). Buka [phase1_summary.md](../phase1/phase1_summary.md).
