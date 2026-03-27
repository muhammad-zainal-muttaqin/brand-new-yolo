# Phase 1A Summary

## Navigasi mini

- Ringkasan repo: [README.md](../../README.md)
- Kembali ke resolusi/dataset: [outputs/phase0/phase0_summary.md](../phase0/phase0_summary.md)
- Hasil one-stage: [outputs/phase1/one_stage_results.csv](one_stage_results.csv)
- Hasil two-stage: [outputs/phase1/two_stage_results.csv](two_stage_results.csv)
- Benchmark arsitektur Phase 1B: [outputs/phase1/architecture_benchmark.csv](architecture_benchmark.csv)
- Top-3 Phase 1B: [outputs/phase1/phase1b_top3.csv](phase1b_top3.csv)
- Lock setup final: [outputs/phase1/locked_setup.yaml](locked_setup.yaml)
- Lanjut ke tuning: [outputs/phase2/phase2_summary.md](../phase2/phase2_summary.md)

## Inputs dari Phase 0

- Resolusi kerja yang dipakai: `640`
- Pipeline dibandingkan pada family ringan yang sama untuk menjaga apple-to-apple.

## One-stage baseline (4 kelas langsung)

- Mean mAP50-95: **0.2526**
- Seed 1 mAP50-95: **0.2538**
- Seed 2 mAP50-95: **0.2514**

## Two-stage feasibility (component-level)

- Stage-1 single-class detector mean mAP50-95: **0.3850**
- Stage-2 GT-crop classifier mean top-1 accuracy: **0.6380**
- Stage-2 diukur pada crop ground-truth, sehingga ini adalah upper-bound untuk bagian klasifikasinya, bukan metrik end-to-end penuh.

## Confusion penting stage-2 classifier (seed 1, val GT-crops)

- B2 correct: **211.0**
- B2 -> B3: **94.0**
- B3 correct: **1112.0**
- B3 -> B2: **334.0**
- Confusion B2/B3 tetap besar bahkan saat objek sudah di-crop dengan ground-truth box.

## Keputusan Phase 1A

> **Pilih pipeline one-stage untuk Phase 1B.**

Alasan:

- one-stage sudah memberi baseline yang jujur dan langsung terukur pada task akhir 4 kelas,
- stage-1 detector two-stage memang kuat untuk lokalisasi 1 kelas, tetapi stage-2 classifier pada GT crops masih mentok di sekitar 0.638 top-1,
- karena classifier pada crop bersih saja masih menunjukkan confusion B2/B3 yang besar, belum ada evidence kuat bahwa two-stage akan mengungguli one-stage secara end-to-end.

## Phase 1B — Canonical Flowchart-Synced Sweep

- Phase 2 locked single best model: `yolo11m.pt`
- Reference top-3 ranking saved to `outputs/phase1/phase1b_top3.csv`: `yolo11m.pt, yolov9c.pt, yolov8s.pt`
- Best Phase 1B mean mAP50: `0.5298`
- Best Phase 1B mean mAP50-95: `0.2570`
- Gate canonical `mAP50 >= 0.70`: `False`
- Local override continue despite gate: `True`

## Phase 1B — Canonical Flowchart-Synced Sweep

- Phase 2 locked single best model: `yolo11m.pt`
- Reference top-3 ranking saved to `outputs/phase1/phase1b_top3.csv`: `yolo11m.pt, yolov9c.pt, yolov8s.pt`
- Best Phase 1B mean mAP50: `0.5298`
- Best Phase 1B mean mAP50-95: `0.2570`
- Gate canonical `mAP50 >= 0.70`: `False`
- Local override continue despite gate: `True`

## Phase 1B — Canonical Flowchart-Synced Sweep

- Phase 2 locked single best model: `yolo11m.pt`
- Reference top-3 ranking saved to `outputs/phase1/phase1b_top3.csv`: `yolo11m.pt, yolov9c.pt, yolov8s.pt`
- Best Phase 1B mean mAP50: `0.5298`
- Best Phase 1B mean mAP50-95: `0.2570`
- Gate canonical `mAP50 >= 0.70`: `False`
- Local override continue despite gate: `True`

## Phase 1B — Canonical Flowchart-Synced Sweep

- Phase 2 locked single best model: `yolo11m.pt`
- Reference top-3 ranking saved to `outputs/phase1/phase1b_top3.csv`: `yolo11m.pt, yolov9c.pt, yolov8s.pt`
- Best Phase 1B mean mAP50: `0.5298`
- Best Phase 1B mean mAP50-95: `0.2570`
- Gate canonical `mAP50 >= 0.70`: `False`
- Local override continue despite gate: `True`

## Phase 1B — Canonical Flowchart-Synced Sweep

- Phase 2 locked single best model: `yolo11m.pt`
- Reference top-3 ranking saved to `outputs/phase1/phase1b_top3.csv`: `yolo11m.pt, yolov9c.pt, yolov8s.pt`
- Best Phase 1B mean mAP50: `0.5298`
- Best Phase 1B mean mAP50-95: `0.2570`
- Gate canonical `mAP50 >= 0.70`: `False`
- Local override continue despite gate: `True`

## Phase 1B — Canonical Flowchart-Synced Sweep

- Phase 2 locked single best model: `yolo11m.pt`
- Reference top-3 ranking saved to `outputs/phase1/phase1b_top3.csv`: `yolo11m.pt, yolov9c.pt, yolov8s.pt`
- Best Phase 1B mean mAP50: `0.5298`
- Best Phase 1B mean mAP50-95: `0.2570`
- Gate canonical `mAP50 >= 0.70`: `False`
- Local override continue despite gate: `True`
