# Final Report - Phase 3 Multi-Candidate Benchmark

- Canonical protocol source: `https://github.com/muhammad-zainal-muttaqin/YOLOBench/blob/main/E0_Protocol_Flowchart.html`
- Phase 3 ini menimpa definisi lama dan sekarang mengikuti split adil: training hanya `train`, evaluasi pada `val` dan `test`.
- Kandidat utama one-stage: `yolo11m.pt` dan `yolov8s.pt`.
- Checkpoint utama untuk pelaporan final: `last.pt`; `best.pt` tetap ikut dievaluasi.
- Cabang two-stage di-run ulang sebagai pembanding pendukung: Stage-1 single-class detector, Stage-2 GT-crop classifier, dan evaluasi end-to-end.

## One-Stage Test Set — `last.pt`

- `yolo11m` | mAP50 `0.4840` | mAP50-95 `0.2470` | precision `0.4944` | recall `0.5563` | conf `0.1`
- `yolov8s` | mAP50 `0.4621` | mAP50-95 `0.2378` | precision `0.4626` | recall `0.5431` | conf `0.2`

## Gap Val vs Test — `last.pt`

- `yolo11m` | mAP50 val `0.4996` -> test `0.4840` | precision val `0.5011` -> test `0.4944` | recall val `0.5630` -> test `0.5563`
- `yolov8s` | mAP50 val `0.4776` -> test `0.4621` | precision val `0.5054` -> test `0.4626` | recall val `0.5352` -> test `0.5431`

## Two-Stage GT-Crop (`last.pt`, test)

- top1 `0.6485` | weighted F1 `0.6337` | macro F1 `0.6337`

## Two-Stage End-to-End (`last.pt`, test)

- precision `0.4840` | recall `0.5053` | F1 `0.4944` | accuracy `0.5053`
