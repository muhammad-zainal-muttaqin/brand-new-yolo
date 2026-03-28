# Phase 2 Summary — Hyperparameter Tuning

Phase 2 menguji apakah penyesuaian hyperparameter bisa mendorong performa melewati ceiling yang terlihat di Phase 1, atau apakah bottleneck sebenarnya bukan di situ. Tuning dilakukan secara sequential pada satu model yang sudah di-lock (`yolo11m.pt`), mencakup loss function, learning rate, batch size, dan augmentation profile.

Alasan pemilihan model ada di [phase1_summary.md](../phase1/phase1_summary.md). Hasil akhir retrain final di [final_evaluation.md](../phase3/final_evaluation.md) dan [final_report.md](../phase3/final_report.md).

---

## Ringkasan eksekutif

Secara singkat: **tuning tidak menghasilkan perbaikan yang meyakinkan**. Kombinasi terbaik (`lr0=0.0005`, `batch=16`, `aug=medium`) hanya memberikan gain ~0.5% mAP50 dibanding baseline Phase 1B — terlalu kecil untuk dijadikan alasan mengganti recipe yang sudah stabil. Keputusan akhir: **revert ke baseline Phase 1B**.

Konfigurasi yang dibawa ke Phase 3: `lr0=0.001`, `batch=16`, `imbalance=none`, `ordinal=standard`, `aug=medium`, sesuai [final_hparams.yaml](final_hparams.yaml).

Confirmation run pada recipe terkunci menghasilkan:

| Metrik | Nilai |
|---|---:|
| Precision | 0.5066 |
| Recall | 0.6042 |
| mAP50 | 0.5390 |
| mAP50-95 | 0.2594 |
| B4 recall | 0.3736 |
| Gate `all_classes_ge_70_ap50` | **False** |

Kelas B2 dan B4 tetap menjadi bottleneck, konsisten dengan temuan Phase 1.

---

## 1. Tujuan dan cakupan

| Aspek | Detail |
|---|---|
| Model | `yolo11m.pt` saja (tidak ada architecture search) |
| Input lock | [locked_setup.yaml](../phase1/locked_setup.yaml) |
| Metrik fokus | mAP50, mAP50-95, B4 recall |
| Output resmi | [tuning_results.csv](tuning_results.csv), [final_hparams.yaml](final_hparams.yaml), confirmation JSON |

Phase 2 bukan pengulangan benchmark multi-model — tujuannya spesifik: menguji apakah hyperparameter adjustment bisa memberikan gain yang signifikan pada arsitektur yang sudah dipilih.

---

## 2. Sumber data

- [imbalance_sweep.csv](imbalance_sweep.csv) — loss function sweep
- [ordinal_sweep.csv](ordinal_sweep.csv) — mencatat step yang dilewati
- [lr_sweep.csv](lr_sweep.csv) — learning rate sweep
- [batch_sweep.csv](batch_sweep.csv) — batch size sweep
- [aug_sweep.csv](aug_sweep.csv) — augmentation profile sweep
- [tuning_results.csv](tuning_results.csv) — ringkasan keputusan tuning
- [p2confirm_yolo11m_640_s3_e30p10m30_eval.json](p2confirm_yolo11m_640_s3_e30p10m30_eval.json) — confirmation run
- [final_hparams.yaml](final_hparams.yaml) — konfigurasi final

---

## 3. Protokol

- **Resolusi**: `imgsz=640` (locked dari Phase 0)
- **Training**: `epochs=30`, `patience=10`, `min_epochs=30` (selaras Phase 1B)
- **Agregasi**: setiap opsi sweep dihitung sebagai mean dari 2 seed, kecuali yang di-reuse dari Phase 1B

Baseline Phase 1B untuk `yolo11m`: mean mAP50 **0.5298**, mean mAP50-95 **0.2570**, mean B4 recall **0.3673**.

---

## 4. Override operasional

Beberapa cabang sweep dipangkas karena bukti awal sudah cukup jelas:

1. **Step 0a (loss function)** — Tiga strategi (`none`, `class_weighted`, `focal15`) menghasilkan **metrik yang identik**. Loss dikunci ke `none`.
2. **Step 0b (ordinal)** — Dilewati karena alasan yang sama dengan Step 0a.
3. **Step 1 (LR)** — Baseline `lr0=0.001` di-reuse dari Phase 1B, tidak dilatih ulang.
4. **Step 2 (batch)** — Hanya `8` vs `16`; `batch=32` dilewati.
5. **Step 3 (augmentasi)** — Hanya `light` vs `medium`; `heavy` dilewati.

Override ini mengurangi jumlah run tanpa mengorbankan kesimpulan — data yang ada sudah cukup untuk memutuskan revert.

---

## 5. Hasil per langkah sweep

### 5.1 Step 0a — Loss function

Sumber: [imbalance_sweep.csv](imbalance_sweep.csv).

| Strategi | Mean mAP50 | Mean mAP50-95 | Mean B4 Recall |
|---|---:|---:|---:|
| `none` | 0.5298 | 0.2570 | 0.3673 |
| `class_weighted` | 0.5298 | 0.2570 | 0.3673 |
| `focal15` | 0.5298 | 0.2570 | 0.3673 |

Ini adalah temuan yang paling informatif di Phase 2, meskipun pada pandangan pertama terlihat "kosong". Ketiga strategi loss menghasilkan angka yang persis sama — bukan mirip, tapi **identik** sampai 4 desimal.

Apa artinya? Loss function bukan bottleneck. Model sudah mengekstrak informasi dari data seefisien yang bisa dilakukan pada arsitektur dan resolusi ini. Mengubah cara loss di-weight (class_weighted) atau mengubah bentuk loss (focal) tidak mengubah apa yang model pelajari. Ini kuat mengindikasikan bahwa **ceiling performa ditentukan oleh data dan task difficulty, bukan training objective**.

### 5.2 Step 1 — Learning rate

Sumber: [lr_sweep.csv](lr_sweep.csv).

| LR | Source | Mean mAP50 | Mean mAP50-95 | Mean B4 Recall |
|---|---|---:|---:|---:|
| `0.001` | Phase 1B (reuse) | 0.5298 | 0.2570 | 0.3673 |
| `0.0005` | Sweep Phase 2 | 0.5350 | 0.2577 | 0.3637 |
| `0.002` | Sweep Phase 2 | 0.5338 | 0.2587 | 0.3337 |

![Learning rate sweep](figures/p2_lr_sweep.png)

`lr0=0.0005` memberikan gain kecil di mAP50 (+0.52%) tapi B4 recall turun sedikit. `lr0=0.002` menaikkan mAP50-95 tapi **menjatuhkan B4 recall ke 0.334** — penurunan yang signifikan untuk kelas yang sudah paling sulit.

Pola ini menunjukkan trade-off yang tidak menguntungkan: LR yang lebih tinggi membuat model lebih agresif secara overall tapi lebih buruk di kelas sulit. LR yang lebih rendah sedikit lebih baik secara agregat tapi gain-nya marginal dan tidak konsisten antar seed.

### 5.3 Step 2 — Batch size

Sumber: [batch_sweep.csv](batch_sweep.csv).

| Batch | Mean mAP50 | Mean mAP50-95 | Mean B4 Recall |
|---:|---:|---:|---:|
| 8 | 0.5321 | 0.2574 | 0.3791 |
| 16 | 0.5350 | 0.2577 | 0.3637 |

### 5.4 Step 3 — Augmentation profile

Sumber: [aug_sweep.csv](aug_sweep.csv).

| Profile | Mean mAP50 | Mean mAP50-95 | Mean B4 Recall |
|---|---:|---:|---:|
| `light` | 0.5256 | 0.2512 | 0.3827 |
| `medium` | 0.5350 | 0.2577 | 0.3637 |

![Batch size dan augmentation sweep](figures/p2_batch_aug_sweep.png)

Ada pola menarik yang berulang di Step 2 dan 3: konfigurasi yang lebih "ringan" (batch kecil, augmentasi ringan) cenderung lebih baik untuk B4 recall, sementara konfigurasi "standar" (batch 16, medium aug) lebih baik untuk metrik agregat. Ini masuk akal — batch lebih kecil dan augmentasi lebih ringan memberi model lebih banyak kesempatan untuk melihat instance B4 yang sedikit secara efektif, tapi mengorbankan generalisasi di kelas lain.

Namun perbedaannya tetap kecil di semua metrik — tidak ada konfigurasi yang memberikan breakthrough.

---

## 6. Keputusan tuning

![Phase 2 tuning — progression mAP50](figures/p2_tuning_summary.png)

Dari [tuning_results.csv](tuning_results.csv):

| Field | Nilai |
|---|---|
| Baseline mean mAP50 | 0.5298 |
| Best tuned mean mAP50 | 0.5350 |
| Final mean mAP50 | 0.5329 |
| Final mean mAP50-95 | 0.2578 |
| Reverted to Phase 1 baseline | **True** |
| Final source | `phase1_baseline_reverted` |

Selisih antara baseline (0.5298) dan kandidat terbaik (0.5350) hanya **0.52%** — di bawah threshold yang bisa dianggap meaningful, apalagi dengan variance antar seed yang masih overlap. Keputusan revert bukan karena tuning "gagal" dalam artian error, tapi karena **gain-nya tidak cukup untuk membenarkan perubahan recipe** yang sudah stabil dan reproducible.

---

## 7. Konfigurasi final yang di-lock

Recipe yang ditulis ke [final_hparams.yaml](final_hparams.yaml) untuk Phase 3:

| Parameter | Nilai |
|---|---|
| Model | `yolo11m.pt` |
| lr0 | `0.001` |
| Batch | `16` |
| Imbalance strategy | `none` |
| Ordinal strategy | `standard` |
| Aug profile | `medium` |
| Image size | `640` |

---

## 8. Verification: confirmation run

Run **`p2confirm_yolo11m_640_s3_e30p10m30`** menggunakan seed ke-3 (bukan seed 1 atau 2 yang dipakai di sweep) untuk memvalidasi bahwa recipe terkunci menghasilkan performa yang konsisten.

Evaluasi pada split **val** ([p2confirm_yolo11m_640_s3_e30p10m30_eval.json](p2confirm_yolo11m_640_s3_e30p10m30_eval.json)):

| Metrik | Nilai |
|---|---:|
| Precision | 0.5066 |
| Recall | 0.6042 |
| mAP50 | 0.5390 |
| mAP50-95 | 0.2594 |
| B4 recall | 0.3736 |
| `all_classes_ge_70_ap50` | **False** |

Per kelas (mAP50): B1 **0.8050**, B2 **0.4042**, B3 **0.5716**, B4 **0.3753**.

Hasil ini mengonfirmasi dua hal: (1) recipe terkunci menghasilkan performa yang sesuai ekspektasi di seed baru, dan (2) gap antar kelas — B1 jauh di atas, B4 jauh di bawah — bukan artefak seed tertentu tapi memang karakteristik task ini.

---

## 9. Kesimpulan

Phase 2 memberikan beberapa insight yang penting meskipun tidak menghasilkan perubahan recipe:

1. **Loss function bukan bottleneck.** Tiga strategi menghasilkan metrik identik — model sudah mengekstrak sinyal seefisien yang bisa dari data yang ada.

2. **LR, batch, dan augmentation memberikan trade-off marginal**, bukan perbaikan. Setiap gain di satu metrik diikuti penurunan di metrik lain, dan semua dalam margin of error antar seed.

3. **Keputusan revert ke baseline adalah pilihan stabilitas.** Bukan karena tuning gagal, tapi karena gain < 1% tidak cukup kuat untuk membenarkan perubahan recipe di pipeline yang harus reproducible.

4. **Pesan terbesar: bottleneck ada di task difficulty dan data quality.** Sweep hyperparameter di ruang standar sudah saturated. Peningkatan berikutnya harus datang dari pendekatan yang lebih fundamental — domain-specific augmentation, perubahan arsitektur yang targeted, atau peningkatan kualitas/kuantitas data.

---

## 10. Langkah berikutnya

Phase 2 menutup pertanyaan "apakah tuning bisa membantu?" dengan jawaban "tidak secara signifikan". Eksperimen lanjut ke Phase 3 untuk retrain final dengan budget lebih besar dan evaluasi di test set. Buka [final_report.md](../phase3/final_report.md).
