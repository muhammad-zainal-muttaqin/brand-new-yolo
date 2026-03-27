# Phase 0 Summary

Ringkasan keputusan Phase 0: validasi dataset, pilihan resolusi kerja, dan pembacaan learning curve. Audit dataset mentah ada di [eda_report.md](eda_report.md), keputusan pipeline di [phase1_summary.md](../phase1/phase1_summary.md).

## Sumber utama

Artefak yang dipakai:

- [dataset_audit.json](dataset_audit.json)
- [eda_report.md](eda_report.md)
- [resolution_sweep.csv](resolution_sweep.csv)
- [learning_curve.csv](learning_curve.csv)
- [locked_setup.yaml](../phase1/locked_setup.yaml)

## 1. Validasi dataset

Berdasarkan [dataset_audit.json](dataset_audit.json), status dataset **ok**.

Ringkasan:

- total images: **3992**
- total labels: **3992**
- total instances: **17987**
- split: train **2764**, val **604**, test **624**
- empty-label images: **83**
- invalid label issues setelah self-healing: **0**
- group overlap antar split: **0**

Dataset cukup bersih buat baseline E0, jadi Phase 0 bisa lanjut ke pemilihan resolusi kerja.

## 2. Resolution sweep

Resolution sweep pakai run valid `>=30` epoch aktual dengan `patience=10`, sesuai [resolution_sweep.csv](resolution_sweep.csv).

| imgsz | seed | mAP50 | mAP50-95 | precision | recall |
|---:|---:|---:|---:|---:|---:|
| 640 | 1 | 0.5237 | 0.2538 | 0.4906 | 0.5864 |
| 1024 | 1 | 0.5363 | 0.2571 | 0.4888 | 0.6016 |
| 640 | 2 | 0.5245 | 0.2514 | 0.4923 | 0.5838 |
| 1024 | 2 | 0.5276 | 0.2589 | 0.4952 | 0.6004 |

### Mean per resolution

- `640`: mAP50 **0.5241**, mAP50-95 **0.2526**
- `1024`: mAP50 **0.5320**, mAP50-95 **0.2580**
- relative gain `1024` vs `640` pada mAP50-95: **2.15%**

### Keputusan resolusi

Sesuai aturan di [E0.md](../../E0.md), gain `2–5%` nggak otomatis ngunci `1024`. Keputusan akhir tetap harus ngimbangin biaya komputasi.

Karena kenaikan `1024` kecil sementara biaya compute dan inference jauh lebih berat, **resolusi kerja Phase 0 dipilih = `640`**.

Keputusan ini dibawa ke fase berikutnya dan tercermin di [locked_setup.yaml](../phase1/locked_setup.yaml).

## 3. Learning curve @ 640

Learning curve dibaca dari [outputs/phase0/learning_curve.csv](learning_curve.csv):

| fraction | mAP50 | mAP50-95 | precision | recall |
|---:|---:|---:|---:|---:|
| 0.25 | 0.4444 | 0.1984 | 0.4187 | 0.5758 |
| 0.50 | 0.4637 | 0.2202 | 0.4410 | 0.5791 |
| 0.75 | 0.5033 | 0.2444 | 0.4683 | 0.5906 |
| 1.00 | 0.5237 | 0.2538 | 0.4906 | 0.5864 |

### Cara membacanya

- `25% -> 50%`: **+0.0217** mAP50-95
- `50% -> 75%`: **+0.0243** mAP50-95
- `75% -> 100%`: **+0.0093** mAP50-95

Interpretasinya:

- performa masih naik saat data bertambah
- kenaikan itu mulai mengecil mendekati 100%
- dataset belum benar-benar saturasi, tetapi tanda **diminishing returns** sudah mulai terlihat

## 4. Keputusan akhir Phase 0

Phase 0 menutup tiga hal penting:

1. dataset aktif lolos audit dasar dan bebas leakage yang terdeteksi
2. resolusi kerja terbaik yang realistis untuk repo ini adalah **`640`**
3. learning curve menunjukkan masih ada manfaat dari penambahan data, tetapi tidak besar secara linear

## 5. Langkah berikutnya

Setelah Phase 0, eksperimen lanjut ke Phase 1A untuk memilih pipeline. Buka [outputs/phase1/phase1_summary.md](../phase1/phase1_summary.md) untuk melihat keputusan one-stage vs two-stage.
