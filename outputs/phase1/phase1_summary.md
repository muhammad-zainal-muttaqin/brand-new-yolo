# Phase 1 Summary

Dua keputusan Phase 1:

1. **Phase 1A** — milih pipeline `one-stage` atau `two-stage`
2. **Phase 1B** — milih arsitektur terbaik di pipeline yang menang

Dasar keputusan resolusi dan dataset ada di [phase0_summary.md](../phase0/phase0_summary.md), hasil tuning di [phase2_summary.md](../phase2/phase2_summary.md).

## Sumber utama

Artefak yang dipakai:

- [one_stage_results.csv](one_stage_results.csv)
- [two_stage_results.csv](two_stage_results.csv)
- [architecture_benchmark.csv](architecture_benchmark.csv)
- [phase1b_top3.csv](phase1b_top3.csv)
- [locked_setup.yaml](locked_setup.yaml)

## 1. Input dari Phase 0

Phase 1 pakai hasil lock dari Phase 0:

- resolusi kerja: **`640`**
- dataset aktif: [Dataset-YOLO/data.yaml](../../Dataset-YOLO/data.yaml)
- tujuan perbandingan: jaga setup tetap apple-to-apple antar pipeline

## 2. Phase 1A — keputusan pipeline

### One-stage baseline

Hasil one-stage dari [one_stage_results.csv](one_stage_results.csv):

- mean mAP50-95: **0.2526**
- seed 1 mAP50-95: **0.2538**
- seed 2 mAP50-95: **0.2514**

### Two-stage feasibility

Hasil two-stage komponen dari [two_stage_results.csv](two_stage_results.csv):

- stage-1 single-class detector mean mAP50-95: **0.3850**
- stage-2 GT-crop classifier mean top-1 accuracy: **0.6380**

Catatan: stage-2 diukur di **ground-truth crops**, jadi angka itu **upper-bound komponen klasifikasi**, bukan hasil pipeline end-to-end penuh.

### Confusion di stage-2 classifier

Dari evaluasi seed 1 di GT crops:

- `B2 correct`: **211**
- `B2 -> B3`: **94**
- `B3 correct`: **1112**
- `B3 -> B2`: **334**

Jadi, bahkan pas objek udah dipotong pake bounding box ground truth, confusion `B2/B3` masih besar.

### Keputusan Phase 1A

> **Pipeline yang dipilih: `one-stage`.**

Alasannya:

- one-stage langsung ngukur task akhir 4 kelas
- stage-1 two-stage emang kuat buat deteksi 1 kelas, tapi stage-2 classifier belum nunjukin bukti kuat bisa nyelesaiin confusion `B2/B3`
- belum ada evidence cukup buat bilang two-stage bakal ngunggulin one-stage secara end-to-end

## 3. Phase 1B — Benchmark arsitektur

Phase 1B jalanin benchmark arsitektur di pipeline `one-stage` dengan resolusi `640`, `lr0=0.001`, `batch=16`, `medium augmentation`, dan `2 seeds` per model. Hasil agregasi ada di [architecture_benchmark.csv](architecture_benchmark.csv).

### Top-3 arsitektur

Referensi resmi top-3 ada di [outputs/phase1/phase1b_top3.csv](phase1b_top3.csv):

| Rank | Model | mean mAP50 | mean mAP50-95 |
|---:|---|---:|---:|
| 1 | `yolo11m.pt` | 0.5298 | 0.2570 |
| 2 | `yolov9c.pt` | 0.5292 | 0.2518 |
| 3 | `yolov8s.pt` | 0.5256 | 0.2521 |

### Pembacaan hasil benchmark

- `yolo11m.pt` menang tipis, tetapi konsisten
- gap antar model teratas tidak besar
- tidak ada model yang menembus gate canonical `mAP50 >= 0.70`

### Gate canonical dan override repo

Dari [outputs/phase1/locked_setup.yaml](locked_setup.yaml):

- gate canonical `mAP50 >= 0.70`: **False**
- local override continue: **True**

Jadi, secara canonical fase ini seharusnya berhenti. Tapi repo ini emang pakai override operasional biar baseline end-to-end tetep selesai sampe Phase 3.

## 4. Model yang dikunci ke Phase 2

Model yang di-lock: **`yolo11m.pt`**.

Bukti resmi:

- [phase1b_top3.csv](phase1b_top3.csv)
- [locked_setup.yaml](locked_setup.yaml)

Lock ini artinya Phase 2 **nggak** buka architecture search baru. Phase 2 cuma nge-tune satu model yang udah dipilih di Phase 1B.

## 5. Keputusan akhir Phase 1

Phase 1 berakhir dengan dua keputusan yang jelas:

1. pipeline final: **`one-stage`**
2. model final untuk tuning: **`yolo11m.pt`**

## 6. Langkah berikutnya

Setelah model dikunci, eksperimen lanjut ke Phase 2. Buka [outputs/phase2/phase2_summary.md](../phase2/phase2_summary.md) untuk melihat apakah tuning memberi perbaikan nyata atau tidak.
