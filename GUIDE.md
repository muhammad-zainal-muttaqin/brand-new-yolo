# Panduan Operasional E0 di Repo Ini

`GUIDE.md` adalah runbook operasional untuk menjalankan E0 di repo `brand-new-yolo`.

- **Protokol canonical**: [E0.md](E0.md)
- **Flowchart sumber**: `https://github.com/muhammad-zainal-muttaqin/YOLOBench/blob/main/E0_Protocol_Flowchart.html`
- **Konteks keputusan**: [CONTEXT.md](CONTEXT.md)

Dokumen ini menjelaskan bagaimana protokol E0 dijalankan **secara nyata** di repo ini: status fase, override operasional, policy Git, policy artefak, dan lock file yang dipakai sampai akhir.

> Catatan label: flowchart memakai `M1–M4`, tetapi dataset aktif repo ini memakai `B1–B4`. Source of truth operasional repo ini adalah:
>
> - `B1 = buah paling matang / ripe`
> - `B2 = transisi setelah B1`
> - `B3 = lebih mentah dari B2`
> - `B4 = paling belum matang`
>
> Panduan visual yang dipakai konsisten:
> - `B1`: merah, besar, bulat, paling bawah pada tandan
> - `B2`: hitam → mulai merah, besar, bulat, di atas `B1`
> - `B3`: full hitam, berduri, lonjong, di atas `B2`
> - `B4`: paling kecil, paling dalam di tandan, sulit terlihat, hitam sampai hijau

## 1. Prinsip sinkronisasi dokumen

Aturan sinkronisasi yang dipakai di repo ini:

1. [E0.md](E0.md) = **referensi protocol canonical**
2. [GUIDE.md](GUIDE.md) = **runbook eksekusi repo ini**
3. `scripts/e0_master_autonomous.py` = orchestrator yang harus mengikuti `GUIDE.md`
4. Jika ada perbedaan antara canonical protocol dan kebutuhan operasional repo ini, tuliskan perbedaan itu **secara eksplisit** di `GUIDE.md`

## 2. Fakta repo dan status nyata saat ini

### Dataset aktif

- source of truth dataset: `/workspace/Dataset-Sawit-YOLO`
- [Dataset-YOLO/data.yaml](Dataset-YOLO/data.yaml) menunjuk ke dataset aktif tersebut
- split aktual terverifikasi:
  - train `2764`
  - val `604`
  - test `624`
- total images `3992`
- total labels `3992`
- total instances `17987`
- empty-label images `83`
- invalid label issues setelah self-heal: `0`
- group overlap antar split: `0`

Sumber audit utamanya ada di:

- [outputs/phase0/dataset_audit.json](outputs/phase0/dataset_audit.json)
- [outputs/phase0/eda_report.md](outputs/phase0/eda_report.md)

### Environment aktif

- Python `3.11.15`
- GPU `NVIDIA A40`
- `torch`, `ultralytics`, `pandas`, `matplotlib`, `seaborn` tersedia pada environment eksperimen yang sah
- `git lfs` dibutuhkan untuk restore dataset dari Hugging Face

### Status fase yang benar-benar selesai

- [x] Bootstrap environment
- [x] Phase 0 — Validation & Calibration
- [x] Phase 1A — Pipeline Decision
- [x] Phase 1B — Canonical flowchart-synced
- [x] Phase 2 — Canonical flowchart-synced
- [x] Phase 3 — Canonical flowchart-synced
- [x] Final report canonical

### Status operasional singkat

- Phase 0 selesai. Resolusi kerja final: **`640`**
- Phase 1A selesai. Pipeline final: **`one-stage`**
- Phase 1B selesai. Model yang di-lock: **`yolo11m.pt`**
- Phase 2 selesai. Recipe final kembali ke baseline stabil
- Phase 3 lama sudah ada, tetapi kontraknya sekarang diganti dan perlu dijalankan ulang

## 3. Override operasional repo ini

Bagian ini berisi aturan lokal yang perlu dipatuhi orchestrator di repo ini.

### 3.1 Ukuran model maksimum

- Semua eksperimen E0 di repo ini dibatasi sampai ukuran **medium**
- Varian `large`, `xlarge`, atau lebih besar **tidak dipakai**
- Roster yang dipakai konsisten dengan `n/s/m` atau ekuivalennya, plus `yolov9c`

### 3.2 Semua weight penting harus aman

- Karena model dibatasi sampai medium, weight run penting dianggap masih layak disimpan
- Minimal `best.pt` dan `last.pt` harus dipertahankan jika tersedia
- Policy default: **commit + push setiap akhir run**

### 3.3 GitHub sync policy

- remote target: `https://github.com/muhammad-zainal-muttaqin/brand-new-yolo`
- autentikasi memakai `GITHUB_TOKEN`
- token **tidak boleh** ditulis ke file, log, atau commit
- jika push gagal:
  1. hasil tetap disimpan lokal
  2. kegagalan dicatat di [outputs/reports/git_sync_log.md](outputs/reports/git_sync_log.md)
  3. retry otomatis dilakukan
  4. status boleh lanjut dengan label **pending sync**

### 3.4 Run yang sah untuk pengambilan keputusan

- run keputusan harus **minimal 30 epoch aktual**
- smoke test di bawah itu tidak boleh dipakai untuk keputusan akhir
- repo ini memakai **`patience=10`** sebagai default operasional, kecuali ada override baru yang ditulis eksplisit
- parameter penting seperti `patience`, `lr0`, `batch`, loss strategy, dan augmentation profile harus tercatat jelas pada nama run atau summary file

### 3.5 Override gate Phase 1B

Flowchart canonical menyatakan:

- jika **best mAP50 < 70%**, seharusnya stop sebelum lanjut ke Phase 2/3

Override repo ini:

- gate itu **diabaikan untuk sesi ini**
- eksperimen **tetap lanjut** ke Phase 2 dan Phase 3 walaupun `mAP50 < 70%`
- status gate tetap **harus dicatat apa adanya** di artefak dan report

Alasannya: target `70% mAP50` belum realistis untuk baseline yang sedang dibangun, tetapi pipeline lengkap tetap perlu dijalankan agar repo punya baseline end-to-end yang utuh.

## 4. Aturan lock yang berlaku di repo ini

### Setelah Phase 0

Yang dikunci:

- split final
- resolution final
- data sufficiency assessment

### Setelah Phase 1A

Yang dikunci:

- pipeline final (`one-stage` atau `two-stage`)

### Setelah Phase 1B

Yang dikunci di [outputs/phase1/locked_setup.yaml](outputs/phase1/locked_setup.yaml):

- pipeline final
- resolution final
- **1 model terbaik tunggal untuk Phase 2**

Override repo ini: untuk mempercepat eksekusi, **Phase 2 hanya dijalankan pada 1 model terbaik**, bukan top-3.

### Setelah Phase 2

File [outputs/phase1/locked_setup.yaml](outputs/phase1/locked_setup.yaml) harus memuat:

- model terpilih tunggal Phase 2
- hyperparameter final Phase 2
- kontrak kandidat Phase 3
- recipe one-stage Phase 3
- recipe cabang two-stage Phase 3

Hyperparameter final juga harus ditulis di [outputs/phase2/final_hparams.yaml](outputs/phase2/final_hparams.yaml).

### Pada Phase 3

- wajib membaca [outputs/phase1/locked_setup.yaml](outputs/phase1/locked_setup.yaml)
- wajib memakai pipeline dan resolution yang sudah di-lock
- kandidat one-stage Phase 3 boleh lebih dari satu jika itu ditulis eksplisit pada lock file
- tidak boleh membuka architecture search lagi
- training one-stage final memakai split `train` saja
- evaluasi kandidat utama wajib tersedia pada `val` dan `test`
- deploy/TFLite/INT8 check boleh ditunda setelah `best.pt` aman
- jika konversi deploy dilakukan nanti, akurasi, ukuran, latency, dan kompatibilitas device wajib divalidasi ulang

## 5. Artefak wajib per fase

### 5.1 Phase 0

```text
outputs/phase0/
├── dataset_audit.json
├── eda_report.md
├── class_distribution.csv
├── bbox_stats.csv
├── leakage_report.json
├── resolution_sweep.csv
├── learning_curve.csv
└── phase0_summary.md
```

### 5.2 Phase 1

```text
outputs/phase1/
├── one_stage_results.csv
├── two_stage_results.csv
├── architecture_benchmark.csv
├── per_class_metrics.csv
├── locked_setup.yaml
├── phase1_summary.md
├── phase1b_top3.csv
└── phase1b_error_stratification.csv
```

### 5.3 Phase 2

```text
outputs/phase2/
├── imbalance_sweep.csv
├── ordinal_sweep.csv
├── lr_sweep.csv
├── batch_sweep.csv
├── aug_sweep.csv
├── tuning_results.csv
├── final_hparams.yaml
└── phase2_summary.md
```

### 5.4 Phase 3

```text
outputs/phase3/
├── final_metrics.csv
├── per_class_metrics.csv
├── confusion_matrix.csv
├── threshold_sweep.csv
├── error_stratification.csv
├── error_analysis.md
├── deploy_check.md
├── final_report.md
├── final_evaluation.md
├── final_data.yaml
├── detail/
└── figures/
```

### 5.5 Ledger dan status

```text
outputs/reports/
├── run_ledger.csv
├── latest_status.md
├── git_sync_log.md
├── master_state.json
├── autopilot.log
├── autopilot_monitor.log
└── master_autopilot_console.log
```

## 6. Checklist operasional

### 6.1 Readiness

- [x] Repo diaudit
- [x] Dataset aktif ditemukan
- [x] `data.yaml` tervalidasi
- [x] Split train/val/test tervalidasi
- [x] Pairing image-label tervalidasi
- [x] Invalid label dibersihkan
- [x] Dependency Python tersedia
- [x] GPU tersedia
- [x] Struktur output ada
- [x] Ledger eksperimen ada
- [x] `GITHUB_TOKEN` tersedia
- [x] Remote Git valid

### 6.2 Phase 0 — Done

- [x] EDA
- [x] Resolution sweep
- [x] Learning curve
- [x] Resolusi final dipilih (`640`)
- [x] Split bebas leakage
- [x] Label audit selesai
- [ ] Visual sample / hard cases
- [ ] Plot learning curve final

### 6.3 Phase 1A — Done

- [x] One-stage baseline
- [x] Two-stage baseline komponen
- [x] Analisis confusion kelas berdekatan
- [x] Pipeline final dipilih: `one-stage`

### 6.4 Phase 1B — Done

Roster canonical yang dijalankan:

- [x] `yolov8n.pt`
- [x] `yolov8s.pt`
- [x] `yolov8m.pt`
- [x] `yolov9c.pt`
- [x] `yolov10n.pt`
- [x] `yolov10s.pt`
- [x] `yolov10m.pt`
- [x] `yolo26n.pt`
- [x] `yolo26s.pt`
- [x] `yolo26m.pt`
- [x] `yolo11m.pt`

Aturan yang dijalankan:

- [x] Semua model dijalankan pada pipeline `one-stage`
- [x] Semua model dijalankan pada resolusi `640`
- [x] `lr0=0.001`
- [x] `batch=16`
- [x] medium augmentation
- [x] 2 seeds per model
- [x] tracking per-class `B1/B2/B3/B4`
- [x] top-3 referensi diidentifikasi
- [x] error stratification top-3 dibuat
- [x] `M4/B4` feasibility dicatat
- [x] [outputs/phase1/locked_setup.yaml](outputs/phase1/locked_setup.yaml) dibuat untuk 1 model terbaik

### 6.5 Phase 2 — Done

- [x] hanya memakai model yang di-lock dari Phase 1B
- [x] Step 0a: imbalance dijalankan
- [x] override plateau-aware diterapkan
- [x] sweep dilanjutkan untuk LR, batch, dan augmentation
- [x] final confirm run dijalankan
- [x] model dan config final ditulis ke lock file

Override operasional yang dipakai di fase ini:

- hasil Step 0a (`none`, `class_weighted`, `focal15`) identik per-seed dan diperlakukan sebagai plateau
- repo mengunci baseline loss setup: `imbalance=none`, `ordinal=standard`
- kandidat `lr0=0.001` direuse dari baseline Phase 1B
- `batch=32` dan `aug=heavy` dilewati untuk efisiensi

Semua ini harus dibaca bersama [outputs/phase2/phase2_summary.md](outputs/phase2/phase2_summary.md).

### 6.6 Phase 3 — Redefined

- [x] membaca [outputs/phase1/locked_setup.yaml](outputs/phase1/locked_setup.yaml) versi kontrak baru
- [x] restore dataset aktif ke `/workspace/Dataset-Sawit-YOLO`
- [x] benchmark one-stage `yolo11m.pt` dan `yolov8s.pt`
- [x] training one-stage pada gabungan `train+val`
- [x] `30 epoch`, `patience=0`, dan `val=False` saat training
- [x] evaluasi `last.pt` pada `val` dan `test`
- [x] cabang two-stage dikeluarkan dari kontrak aktif rerun final
- [x] confusion matrix penuh 4 kelas digenerate
- [x] confidence evaluasi dikunci tetap di `0.10`
- [x] tracking error utama
- [ ] deploy check ditandai **deferred**
- [ ] final report dan final evaluation diregenerasi setelah rerun

## 7. Policy otomatisasi

Perilaku default agent untuk setiap run:

1. jalankan eksperimen
2. simpan artefak run
3. tulis summary JSON
4. append ke [outputs/reports/run_ledger.csv](outputs/reports/run_ledger.csv)
5. update [outputs/reports/latest_status.md](outputs/reports/latest_status.md)
6. commit Git
7. push ke GitHub
8. catat status sinkronisasi di [outputs/reports/git_sync_log.md](outputs/reports/git_sync_log.md)

Isi minimal summary run:

- nama run
- phase
- model
- resolution
- seed
- split evaluasi
- path `best` dan `last`
- precision / recall / `mAP50` / `mAP50-95`
- timestamp
- status

## 8. Kapan agent boleh stop

Agent perlu mencoba recovery lebih dulu. Stop hanya jika:

- dataset korup atau hilang
- split sah tidak bisa dibangun
- label rusak tidak bisa dipulihkan aman
- dependency inti tidak bisa dipasang
- disk habis
- training crash berulang tanpa recovery
- push terus gagal dan status pending sync tidak bisa dijaga aman
- perubahan yang dibutuhkan akan merusak validitas ilmiah secara spekulatif

Override khusus: gate `best mAP50 < 70%` **bukan alasan stop** untuk repo ini.

## 9. Immediate next actions

Urutan aksi setelah clone ulang atau audit cepat:

1. baca [README.md](README.md)
2. cek [outputs/phase3/final_report.md](outputs/phase3/final_report.md)
3. cek [outputs/phase3/final_evaluation.md](outputs/phase3/final_evaluation.md)
4. buka [outputs/phase1/locked_setup.yaml](outputs/phase1/locked_setup.yaml)
5. buka [outputs/phase2/final_hparams.yaml](outputs/phase2/final_hparams.yaml)
6. pakai [outputs/reports/reproducibility_and_termination.md](outputs/reports/reproducibility_and_termination.md) jika ingin mereproduksi atau melanjutkan dari mesin lain

## 10. Regenerasi figure dokumentasi

### Research progress charts

Script: [`scripts/generate_e0_research_progress_charts.py`](scripts/generate_e0_research_progress_charts.py)
Output: `outputs/figures/e0_research_progress_*.png` (4 file: map50, map50_95, precision, recall)

Jalankan ulang setelah ada run baru di ledger:

```bash
python scripts/generate_e0_research_progress_charts.py
```

**Parameter yang bisa diubah langsung di script:**

| Konstanta | Default | Fungsi |
|---|---|---|
| `LABEL_THRESHOLD` | `0.015` | Minimum improvement mAP50 agar titik dapat label. Naikkan jika label masih terlalu padat. |
| `PHASE3_NOTABLE` | `{p3tv_yolo11m..., p3tv_yolov8s...}` | Run Phase 3 yang selalu dilabeli meskipun bukan new-best. Tambah nama run baru jika ada kandidat Phase 3 tambahan. |

**Perilaku single-class runs:**
- Run dengan `single_cls=True` di ledger (misal Stage-1 detector) tetap muncul sebagai titik di chart
- Tapi **running-best line** (garis biru) tidak naik ke nilai mereka — dihitung dari 4-class runs saja
- Y-axis otomatis mencakup nilai single-cls agar titik terlihat

**Troubleshooting:**
- Label masih tumpang tindih → naikkan `LABEL_THRESHOLD` (misal ke `0.02`)
- Run baru Phase 3 tidak terlabeli → tambahkan nama run ke `PHASE3_NOTABLE`
- Chart tidak update → pastikan run ada di `outputs/reports/run_ledger.csv` dengan `status=completed`

### Figures lain

| Script | Output |
|---|---|
| `scripts/generate_doc_figures.py` | Semua figure per-fase (`outputs/phaseN/figures/`) |

---

## 11. Konvensi penulisan README.md

Ketika menulis ulang atau memperbarui `README.md`, ikuti konvensi berikut. Format acuan adalah `README-Old.md` — struktur dan cara penjelasannya dipakai sebagai template.

### 11.1 Struktur dokumen (urutan wajib)

1. `# Brand New YOLO — E0 End-to-End Report` — judul tunggal, level 1
2. `## Overview` — canonical protocol source + Class Semantics table 3 kolom
3. `## Executive Summary` — tabel ringkasan + gambar progress map
4. Collapsible ToC (lihat §11.2)
5. `## Phase 0` ... `## Phase 3` — masing-masing diawali `---`
6. `## Appendix` — diawali `---`

### 11.2 Table of Contents

Pakai blockquote + heading untuk header agar terlihat saat scroll cepat:

```markdown
> ## 📋 Table of Contents — Expand Here 👇

<details>
<summary><strong>📋 Klik untuk buka / tutup Table of Contents</strong></summary>

- [link-link section...]

</details>
```

### 11.3 Path file dan gambar

- **Semua** path gambar pakai `outputs/phaseN/figures/...` — tidak ada path root `figures/`
- **Semua** path file teks pakai `outputs/phaseN/filename`
- Confusion matrices masuk subfolder: `outputs/phase3/figures/confusion/`

### 11.4 Heading hierarchy

- `#` hanya untuk judul dokumen (baris pertama)
- `##` untuk section besar (Phase N, Appendix, Overview, Executive Summary)
- `###` untuk subsection dalam phase
- `####` untuk subsection dalam subsection
- **Tidak boleh** ada `# Final Report`, `# Error Analysis`, dsb. di tengah dokumen

### 11.5 Tampilan confusion matrix

Tampilkan langsung sebagai tabel 2-kolom Markdown (val | test per model), **bukan** di dalam `<details>`:

```markdown
| Model — val | Model — test |
|---|---|
| ![...](path/cm_model_last_val.png) | ![...](path/cm_model_last_test.png) |
```

### 11.6 Phase 3 — konvensi konten khusus

- Overview table wajib memuat: protocol source, training split, training mode (`val=False`, `patience=0`), primary checkpoint (`last.pt` saja), eval splits, eval confidence
- Tabel metrics hanya `last.pt` (tidak ada `best.pt` karena protokol `val=False`)
- Tambahkan catatan interpretasi: metrik `val` bersifat **in-sample** karena `train+val` dipakai saat training; metrik `test` yang relevan untuk generalisasi
- Gap val→test wajib disebut dan dijelaskan wajar
- Referensi two-stage tetap ada tapi diberi keterangan eksplisit "tidak dirun ulang"

---

## 12. Status sinkronisasi otomatis

<!-- AUTOSTATUS:START -->
- Canonical source synced: `E0.md` mengikuti flowchart YOLOBench.
- Phase 3 sekarang mengikuti protokol final dosen: train `train+val`, `val=False` saat fit, lalu evaluasi `last.pt` pada `val` dan `test`.
- Kandidat utama one-stage yang dijalankan: `yolo11m.pt` dan `yolov8s.pt`.
- Cabang two-stage tidak lagi termasuk kontrak aktif rerun final Phase 3.
- Threshold evaluasi one-stage dikunci di `conf=0.10` tanpa threshold sweep tambahan.
<!-- AUTOSTATUS:END -->
