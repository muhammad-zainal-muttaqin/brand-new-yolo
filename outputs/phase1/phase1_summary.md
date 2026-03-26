# Phase 1A Summary

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
