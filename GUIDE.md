# GUIDE.md

# Panduan Operasional E0 di Repo Ini

Dokumen ini adalah **runbook operasional** untuk menjalankan E0 di repo `brand-new-yolo`.

- **Sumber protocol canonical**: `E0.md`
- **Sumber asli flowchart**: `https://github.com/muhammad-zainal-muttaqin/YOLOBench/blob/main/E0_Protocol_Flowchart.html`
- **Peran dokumen ini**: menjabarkan bagaimana protocol canonical dijalankan **secara nyata** di workspace ini, termasuk policy Git, policy artefak, lock file, dan override operasional yang sudah diputuskan selama eksekusi.

> Catatan label: flowchart memakai `M1–M4`, sedangkan dataset aktif di workspace ini memakai `B1–B4`. **Source of truth operasional di repo ini adalah:** `B1 = buah paling matang / ripe`, lalu berurutan turun tingkat kematangannya sampai `B4 = buah paling mengkal / belum matang`. Jadi urutan biologis `B1 → B4` bergerak dari **paling matang** ke **paling belum matang**, dan interpretasi ini **tidak boleh tertukar**.
>
> Panduan domain visual/posisional yang wajib dipakai konsisten:
> - `B1`: buah **merah**, **besar**, **bulat**, posisi **paling bawah** pada tandan → kelas paling matang.
> - `B2`: buah masih **hitam** tetapi mulai **transisi ke merah**, sudah **besar** dan **bulat**, posisinya **di atas B1**.
> - `B3`: buah **full hitam**, masih ada **duri**, bentuk masih **lonjong**, posisinya **di atas B2**.
> - `B4`: buah **paling kecil**, **paling dalam di batang/tandan**, sulit terlihat, masih banyak **duri**, warna **hitam sampai hijau**, dan buahnya masih bisa berkembang lebih besar → kelas paling belum matang.

---

## 1. Prinsip Sinkronisasi Dokumen

Aturan sinkronisasi yang dipakai mulai sekarang:

1. `E0.md` = **referensi protocol canonical** yang harus mencerminkan flowchart YOLOBench.
2. `GUIDE.md` = **runbook eksekusi** untuk repo ini.
3. `scripts/e0_master_autonomous.py` = **orchestrator** yang harus mengikuti `GUIDE.md`.
4. Jika ada perbedaan antara canonical protocol dan kebutuhan operasional repo ini, perbedaannya **harus ditulis eksplisit** di `GUIDE.md`, bukan disembunyikan.

---

## 2. Fakta Repo dan Status Nyata Saat Ini

### Dataset aktif
- source of truth dataset: `/workspace/Dataset-Sawit-YOLO`
- `Dataset-YOLO/data.yaml` menunjuk ke dataset aktif tersebut
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

### Environment aktif
- Python `3.11.15`
- GPU `NVIDIA A40`
- `torch`, `ultralytics`, `pandas`, `matplotlib`, `seaborn` sudah tersedia
- `git lfs` **belum** tersedia

### Status phase yang sudah benar-benar selesai
- [x] Bootstrap environment
- [x] Phase 0 — Validation & Calibration
- [x] Phase 1A — Pipeline Decision
- [ ] Phase 1B canonical flowchart-synced
- [ ] Phase 2 canonical flowchart-synced
- [ ] Phase 3 canonical flowchart-synced
- [ ] Final report canonical

### Catatan penting status saat ini
- Phase 0 sudah selesai dan resolusi kerja saat ini adalah **`640`**.
- Phase 1A sudah selesai dan pipeline yang dipilih adalah **`one-stage`**.
- Ada **partial legacy runs** untuk Phase 1B yang dijalankan sebelum sinkronisasi ulang ke flowchart canonical.
- Legacy runs itu **tidak otomatis cukup** untuk menyatakan Phase 1B canonical selesai, karena roster model dan aturan orchestration sebelumnya belum sepenuhnya sinkron dengan flowchart canonical.

---

## 3. Override Operasional Repo Ini

Bagian ini adalah aturan lokal yang **wajib** dipatuhi orchestrator di repo ini.

### 3.1 Ukuran model maksimum
- Semua eksperimen E0 di repo ini **dibatasi sampai ukuran medium**.
- Varian `large`, `xlarge`, atau lebih besar **dilarang**.
- Ini konsisten dengan roster canonical yang dipakai di repo ini (`n/s/m` atau ekuivalennya, plus `yolov9c`).

### 3.2 Semua weight wajib tersimpan dan ter-push
- Karena model dibatasi sampai medium, semua weight run dianggap cukup aman untuk ikut disimpan.
- Setiap akhir run, minimal `best.pt` dan `last.pt` harus ikut dipertahankan jika tersedia.
- Default policy: **commit + push setiap akhir run**.

### 3.3 GitHub sync policy
- remote target: `https://github.com/muhammad-zainal-muttaqin/brand-new-yolo`
- autentikasi memakai `GITHUB_TOKEN`
- token **tidak boleh** ditulis ke file, log, atau commit
- jika push gagal:
  1. hasil tetap disimpan lokal,
  2. kegagalan dicatat ke `outputs/reports/git_sync_log.md`,
  3. retry otomatis dilakukan,
  4. status boleh lanjut dengan label **pending sync**

### 3.4 Run sah untuk pengambilan keputusan
- run keputusan harus **minimal 30 epoch aktual**
- smoke test / bootstrap di bawah itu tidak boleh jadi dasar keputusan akhir
- untuk eksekusi repo ini, policy operasional run sah memakai **`patience=10`** kecuali nanti ada override eksplisit baru
- parameter penting seperti `patience`, `lr0`, `batch`, loss strategy, dan augmentation profile harus tercatat eksplisit di nama run atau file summary

### 3.5 Override gate Phase 1B
Canonical flowchart menyatakan:
- jika **best mAP50 < 70%**, seharusnya stop sebelum Phase 2/3

**Override eksplisit untuk repo ini:**
- gate ini **diabaikan untuk eksekusi saat ini**
- eksperimen **tetap lanjut** ke Phase 2 dan Phase 3 walaupun `mAP50 < 70%`
- namun status gate tetap harus **dicatat apa adanya** di artefak dan report

Alasan override:
- kondisi baseline saat ini membuat target `70% mAP50` sangat mungkin belum realistis,
- tetapi user secara eksplisit meminta pipeline **tetap lanjut** untuk memperoleh baseline lengkap end-to-end.

---

## 4. Aturan Lock yang Berlaku di Repo Ini

Agar tetap selaras dengan flowchart canonical **dan** kebutuhan operasional repo ini, lock diberlakukan sebagai berikut:

### Setelah Phase 0
yang dikunci:
- split final
- resolution final
- data sufficiency assessment

### Setelah Phase 1A
yang dikunci:
- pipeline final (`one-stage` atau `two-stage`)

### Setelah Phase 1B
yang dikunci di `outputs/phase1/locked_setup.yaml`:
- pipeline final
- resolution final
- **1 model terbaik tunggal untuk Phase 2**

> Override operasional repo ini: untuk mempercepat eksekusi, **Phase 2 hanya dijalankan pada 1 model terbaik dari Phase 1B**, bukan top-3. Setelah akhir Phase 1B, Phase 2 **tidak boleh** memasukkan arsitektur baru di luar model tunggal yang sudah di-lock.

### Setelah Phase 2
file `outputs/phase1/locked_setup.yaml` **harus diperbarui** sehingga memuat:
- model final tunggal untuk Phase 3
- hyperparameter final
- loss strategy final
- batch final
- LR final
- augmentation final

### Phase 3
- wajib **membaca** `outputs/phase1/locked_setup.yaml`
- wajib memakai pipeline, resolution, dan model final yang ada di file lock itu
- tidak boleh membuka architecture search lagi

---

## 5. Artefak Wajib per Fase

## 5.1 Phase 0
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

## 5.2 Phase 1
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

## 5.3 Phase 2
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

## 5.4 Phase 3
```text
outputs/phase3/
├── final_metrics.csv
├── confusion_matrix.csv
├── threshold_sweep.csv
├── error_stratification.csv
├── error_analysis.md
├── deploy_check.md
├── final_report.md
├── final_data.yaml
└── trainval.txt
```

## 5.5 Ledger dan status
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

---

## 6. Checklist Operasional

## 6.1 Readiness
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

## 6.2 Phase 0 — Done
- [x] EDA
- [x] Resolution sweep
- [x] Learning curve
- [x] Resolusi final dipilih (`640`)
- [x] Split bebas leakage
- [x] Label audit selesai
- [ ] Visual sample / hard cases
- [ ] Plot learning curve final

## 6.3 Phase 1A — Done
- [x] One-stage baseline
- [x] Two-stage baseline komponen
- [x] Analisis confusion kelas berdekatan
- [x] Pipeline final dipilih: `one-stage`

## 6.4 Phase 1B — Canonical flowchart-synced (belum selesai)

### Roster model canonical yang harus dipakai
- [ ] `yolov8n.pt`
- [ ] `yolov8s.pt`
- [ ] `yolov8m.pt`
- [ ] `yolov9c.pt`
- [ ] `yolov10n.pt`
- [ ] `yolov10s.pt`
- [ ] `yolov10m.pt`
- [ ] `yolo26n.pt`
- [ ] `yolo26s.pt`
- [ ] `yolo26m.pt`
- [ ] `yolo11m.pt`

### Aturan Phase 1B
- [ ] Semua model dijalankan dalam pipeline `one-stage`
- [ ] Semua model dijalankan pada resolution `640`
- [ ] `lr0=0.001`
- [ ] `batch=16`
- [ ] medium augmentation
- [ ] 2 seeds per model
- [ ] tracking per-class `B1/B2/B3/B4`
- [ ] ranking top-3 referensi diidentifikasi
- [ ] error stratification worst-20 untuk top-3 dibuat
- [ ] `M4/B4` feasibility dicatat
- [ ] `outputs/phase1/locked_setup.yaml` dibuat untuk **1 model terbaik** yang dikunci ke Phase 2

### Catatan legacy
- run Phase 1B lama yang berada **di luar roster canonical** atau memakai orchestrator lama harus diperlakukan sebagai **legacy exploratory evidence**, bukan bukti final Phase 1B canonical.

## 6.5 Phase 2 — Canonical flowchart-synced (belum selesai)
- [ ] hanya memakai **1 model terbaik yang di-lock** dari `locked_setup.yaml`
- [ ] Step 0a: imbalance (`none`, `class_weighted`, `focal_gamma_1.5`)
- [ ] Step 0b: ordinal (`standard`, `ordinal_weighted`; `CORAL` hanya jika two-stage)
- [ ] Step 1: LR (`0.0005`, `0.001`, `0.002`)
- [ ] Step 2: batch (`8`, `16`, `32`)
- [ ] Step 3: augmentation (`light`, `medium`, `heavy`)
- [ ] parameter dikunci step-by-step
- [ ] 2 seeds untuk model tunggal yang dituning
- [ ] 3+ seeds pada final config
- [ ] jika improvement `<1%`, revert ke baseline terbaik Phase 1
- [ ] model final tunggal + config final ditulis ke `locked_setup.yaml`

## 6.6 Phase 3 — Canonical flowchart-synced (belum selesai)
- [ ] membaca `outputs/phase1/locked_setup.yaml`
- [ ] retrain final pada `train+val`
- [ ] no early stopping
- [ ] evaluasi final pada `test`
- [ ] threshold sweep `0.1–0.5`
- [ ] tracking confusion `B2/B3` dan `B3/B4`
- [ ] tracking recall `B4`
- [ ] export TFLite
- [ ] cek ukuran `<100MB`
- [ ] optional INT8 export dicoba
- [ ] error stratification worst-20 dibuat
- [ ] final report ditulis

---

## 7. Policy Otomatisasi

Default perilaku agent untuk setiap run:
1. jalankan eksperimen,
2. simpan artefak run,
3. tulis summary JSON,
4. append ke `outputs/reports/run_ledger.csv`,
5. update `outputs/reports/latest_status.md`,
6. commit Git,
7. push ke GitHub,
8. catat status sinkronisasi di `outputs/reports/git_sync_log.md`.

### Isi minimal hasil run
- nama run
- phase
- model
- resolution
- seed
- split evaluasi
- weight `best` dan `last`
- precision / recall / `mAP50` / `mAP50-95`
- timestamp
- status

---

## 8. Kapan Agent Boleh Stop

Agent wajib mencoba memperbaiki sendiri dulu. Stop hanya jika:
- dataset korup / hilang,
- split sah tidak bisa dibangun,
- label rusak tidak bisa dipulihkan aman,
- dependency inti tidak bisa dipasang,
- disk habis,
- training crash berulang tanpa recovery,
- push terus gagal dan status pending sync tidak bisa dijaga aman,
- perubahan yang dibutuhkan akan merusak validitas ilmiah secara spekulatif.

> Override khusus: gate `best mAP50 < 70%` **bukan alasan stop** untuk repo ini. Tetap lanjut.

---

## 9. Immediate Next Actions

Urutan aksi yang sekarang paling benar:
1. hentikan orchestrator lama yang belum sinkron dengan flowchart canonical,
2. pakai master orchestrator baru yang flowchart-synced,
3. jalankan ulang / lanjutkan **Phase 1B canonical** dengan roster 11 model,
4. kunci **1 model terbaik** ke `outputs/phase1/locked_setup.yaml`,
5. lanjut ke Phase 2 hanya pada model tunggal itu,
6. lanjut ke Phase 3 dengan final model yang di-lock.

---

## 10. Status Sinkronisasi Otomatis

<!-- AUTOSTATUS:START -->
- Canonical source synced: `E0.md` mengikuti flowchart YOLOBench.
- Phase 1B canonical flowchart-synced selesai untuk roster 11 model × 2 seeds.
- Model tunggal untuk Phase 2 dikunci di `outputs/phase1/locked_setup.yaml`: `yolo11m.pt`.
- Ranking referensi top-3 tetap disimpan di `outputs/phase1/phase1b_top3.csv`: `yolo11m.pt, yolov9c.pt, yolov8s.pt`.
- Gate canonical `mAP50 >= 70%` tercatat sebagai `False`, tetapi override lokal repo tetap lanjut = `True`.
<!-- AUTOSTATUS:END -->
