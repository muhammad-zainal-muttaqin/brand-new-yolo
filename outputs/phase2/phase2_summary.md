# Phase 2 Summary — Tuning Hyperparameter

Ringkasan Phase 2: tuning terkontrol di satu model yang udah dikunci (`yolo11m.pt`), protokolnya dua seed per konfigurasi (kecuali catatan reuse). Alasan pemilihan model ada di [phase1_summary.md](../phase1/phase1_summary.md). Hasil akhir test set dan retrain final di [final_evaluation.md](../phase3/final_evaluation.md) dan [final_report.md](../phase3/final_report.md).

---

## Ringkasan eksekutif

Phase 2 ngeksplor variasi **strategi loss**, **learning rate**, **batch size**, dan **profil augmentasi** di setup yang selain itu sama dengan baseline Phase 1B. Beberapa cabang sweep **disingkat atau dilewati** soalnya Step 0a nunjukin perilaku plateau — metrik identik antar varian loss.

Secara agregat, kombinasi terbaik dalam sweep (kayak `lr0=0.0005` + `batch=16` + `aug=medium`) kasih **kenaikan kecil** di mean mAP50, tapi kenaikan itu **nggak cukup kuat** buat ganti resep baseline yang udah stabil. Makanya kita catat **`reverted_to_phase1_baseline = True`**: konfigurasi buat Phase 3 ngikutin **baseline Phase 1B** (`lr0=0.001`, `batch=16`, `imbalance=none`, `ordinal=standard`, `aug=medium`), sesuai [final_hparams.yaml](final_hparams.yaml).

Run konfirmasi dengan resep terkunci:

- precision: **0.5066**
- recall: **0.6042**
- mAP50: **0.5390**
- mAP50-95: **0.2594**
- split: **val**
- gate `all_classes_ge_70_ap50`: **False**

Kelas **B2** dan **B4** tetep paling lemah, konsisten sama bottleneck yang muncul di Phase 1.

---

## 1. Tujuan dan cakupan

| Aspek | Isi |
|--------|-----|
| Model | `yolo11m.pt` aja (nggak ada architecture search) |
| Input lock | [locked_setup.yaml](../phase1/locked_setup.yaml) |
| Metrik fokus | mAP50, mAP50-95, plus recall B4 buat lihat kelas minor |
| Output resmi | [tuning_results.csv](tuning_results.csv), [final_hparams.yaml](final_hparams.yaml), eval konfirmasi JSON |

Phase 2 **bukan** pengulangan benchmark multi-model Phase 1B — tujuannya ngetes apakah penyesuaian hparams bisa kasih lompatan jelas tanpa ngorbankan stabilitas resep.

---

## 2. Sumber data

File yang dipakai buat laporan ini:

- [imbalance_sweep.csv](imbalance_sweep.csv)
- [ordinal_sweep.csv](ordinal_sweep.csv) — nyatet langkah yang dilewati
- [lr_sweep.csv](lr_sweep.csv)
- [batch_sweep.csv](batch_sweep.csv)
- [aug_sweep.csv](aug_sweep.csv)
- [tuning_results.csv](tuning_results.csv)
- [p2confirm_yolo11m_640_s3_e30p10m30_eval.json](p2confirm_yolo11m_640_s3_e30p10m30_eval.json)
- [final_hparams.yaml](final_hparams.yaml)

---

## 3. Protokol singkat

- **Resolusi**: `imgsz = 640`
- **Pelatihan**: `epochs = 30`, `patience = 10`, `min_epochs = 30` (selaras dengan eksperimen Phase 1B)
- **Agregasi**: untuk setiap opsi sweep, metrik dihitung sebagai **rata-rata dua run** (seed 1 dan 2), kecuali baris yang secara eksplisit berstatus **reused reference** dari Phase 1B

Baseline numerik Phase 1B untuk `yolo11m` pada dua seed referensi: mean mAP50 **0.5298**, mean mAP50-95 **0.2570**, mean B4 recall **0.3673** (sumber: baris `lr001_phase1_reference` di `lr_sweep.csv`).

---

## 4. Override operasional dan ruang pencarian

Kita terapin **override operasional** biar compute nggak kebuang di cabang yang udah ketahuan datar atau redundan. Intinya:

1. **Step 0a (imbalance / loss)** — Tiga opsi (`none`, `class_weighted`, `focal15`) hasilin **metrik identik** di agregat dua seed. Perilaku ini plateau; setup loss selanjutnya **dikunci** ke `imbalance=none` dan `ordinal=standard`.
2. **Step 0b (ordinal)** — **Nggak dijalankan**; statusnya di [ordinal_sweep.csv](ordinal_sweep.csv) sama kayak alasan Step 0a.
3. **Step 1 (LR)** — Kandidat `lr0=0.001` **nggak dilatih ulang** di Phase 2; nilainya **direuse** dari baseline Phase 1B (`status=reused_reference` di `lr_sweep.csv`).
4. **Step 2 (batch)** — `batch=32` **dilewati**; cuma bandingin **8 vs 16**.
5. **Step 3 (augmentasi)** — Profil `heavy` **dilewati**; cuma bandingin **light vs medium**.

Alasan teknis lengkap ada di [GUIDE.md](../../GUIDE.md). Override ini ngurangin cakupan laporan dibanding “sweep penuh”, tapi **nggak ngubah fakta** bahwa data yang ada udah cukup buat keputusan revert baseline.

---

## 5. Hasil per langkah sweep

Angka di bawah adalah **mean** dari dua seed (atau referensi Phase 1B buat `lr0=0.001`), dibulatkan ke empat desimal biar konsisten sama artefak CSV.

### 5.1 Step 0a — Imbalance handling

Sumber: [imbalance_sweep.csv](imbalance_sweep.csv).

| Opsi | Mean mAP50 | Mean mAP50-95 | Mean B4 recall |
|------|-----------:|---------------:|---------------:|
| `none` | 0.5298 | 0.2570 | 0.3673 |
| `class_weighted` | 0.5298 | 0.2570 | 0.3673 |
| `focal15` | 0.5298 | 0.2570 | 0.3673 |

Nggak ada pemisahan sinyal antar strategi loss di setup ini. Ngelanjutin eksplorasi loss yang lebih eksotis **nggak masuk akal** berdasarkan bukti numerik yang ada.

### 5.2 Step 1 — Learning rate

Sumber: [lr_sweep.csv](lr_sweep.csv).

| LR | Peran | Mean mAP50 | Mean mAP50-95 | Mean B4 recall |
|----|--------|-----------:|---------------:|---------------:|
| `0.001` | Reuse Phase 1B | 0.5298 | 0.2570 | 0.3673 |
| `0.0005` | Sweep Phase 2 | 0.5350 | 0.2577 | 0.3637 |
| `0.002` | Sweep Phase 2 | 0.5338 | 0.2587 | 0.3337 |

`0.0005` dan `0.002` sedikit ng geser mAP50 dan mAP50-95, tapi **nggak bikin lompatan besar**. `0.002` nurunin B4 recall jadi **0.3337**, yang merugikan kelas yang udah sulit.

### 5.3 Step 2 — Batch size

Sumber: [batch_sweep.csv](batch_sweep.csv).

| Batch | Mean mAP50 | Mean mAP50-95 | Mean B4 recall |
|------:|-----------:|---------------:|---------------:|
| 8 | 0.5321 | 0.2574 | 0.3791 |
| 16 | 0.5350 | 0.2577 | 0.3637 |

Trade-off jelas: **batch 8** sedikit ngebantu B4 recall, **batch 16** sedikit unggul di mAP50 agregat. Perbedaan mAP50-95 di antara keduanya **kecil**.

### 5.4 Step 3 — Profil augmentasi

Sumber: [aug_sweep.csv](aug_sweep.csv).

| Profil | Mean mAP50 | Mean mAP50-95 | Mean B4 recall |
|--------|-----------:|---------------:|---------------:|
| `light` | 0.5256 | 0.2512 | 0.3827 |
| `medium` | 0.5350 | 0.2577 | 0.3637 |

`medium` unggul di metrik utama agregat; `light` sedikit ngebantu B4 tapi **nurunin** mAP50-95 secara berarti dibanding `medium`.

---

## 6. Agregat keputusan tuning

Baris tunggal di [tuning_results.csv](tuning_results.csv) merangkum perbandingan baseline versus kandidat terbaik dalam sweep:

| Field | Nilai |
|--------|--------|
| `baseline_mean_map50` | 0.5298 |
| `tuned_mean_map50` (kandidat terbaik dalam sweep) | 0.5350 |
| `mean_map50` (nilai agregat yang dicatat untuk baris final) | 0.5329 |
| `mean_map50_95` | 0.2578 |
| `reverted_to_phase1_baseline` | **True** |
| `final_source` | `phase1_baseline_reverted` |

Meski **0.5350** sedikit di atas **0.5298**, selisihnya **kecil banget** buat ngunci resep baru sebagai pengganti baseline Phase 1B — apalagi kalo udah ngimbangin stabilitas lintas fase dan trade-off di B4 di beberapa cabang.

---

## 7. Konfigurasi final yang dikunci

Resep yang ditulis ke [final_hparams.yaml](final_hparams.yaml) dan dibawa ke Phase 3:

- **model:** `yolo11m.pt`
- **lr0:** `0.001`
- **batch:** `16`
- **imbalance_strategy:** `none`
- **ordinal_strategy:** `standard`
- **aug_profile:** `medium`
- **imgsz:** `640` (dengan `epochs` / `patience` / `min_epochs` sebagaimana di file)

---

## 8. Verifikasi: confirm run Phase 2

Evaluasi formal pada run **`p2confirm_yolo11m_640_s3_e30p10m30`** ([p2confirm_yolo11m_640_s3_e30p10m30_eval.json](p2confirm_yolo11m_640_s3_e30p10m30_eval.json)), split **val**:

| Metrik | Nilai |
|--------|------:|
| Precision | 0.5066 |
| Recall | 0.6042 |
| mAP50 | 0.5390 |
| mAP50-95 | 0.2594 |
| B4 recall | 0.3736 |
| `all_classes_ge_70_ap50` | **False** |

**Per kelas (mAP50):** B1 **0.8050**, B2 **0.4042**, B3 **0.5716**, B4 **0.3753**. Pola ini makin nguatin bahwa tuning hparams di model yang sama **nggak ngehapus** kesenjangan antar kelas yang udah mapan sejak benchmark arsitektur.

---

## 9. Kesimpulan

1. **Varian loss di Step 0a** nggak ngasih variasi terukur; baseline loss tetep `none` / `standard`.
2. **LR, batch, dan aug** nunjukin **penyesuaian marjinal**; nggak ada kombinasi yang ngasih bukti kuat buat ganti resep Phase 1B.
3. **Keputusan revert** ngecerminin preferensi buat **baseline stabil** dibanding kenaikan kecil yang nggak konsisten sama tujuan robustness lintas fase.
4. **Confirm run** memvalidasi bahwa resep terkunci tetep di rentang performa yang diharapkan, tanpa nutup gap kelas lemah.

---

## 10. Langkah berikutnya

Lanjut ke Phase 3 buat retrain final, evaluasi test set, dan laporan penutup: [final_report.md](../phase3/final_report.md).
