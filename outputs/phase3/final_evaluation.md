# Final Evaluation — Phase 3

> Status: file ini masih berisi ringkasan evaluasi Phase 3 lama. Protokol aktif sekarang adalah **train `train+val`, `val=False`, lalu evaluasi `last.pt` pada `val` dan `test`**. Metrik di bawah adalah **arsip historis** sampai rerun baru dijalankan.

Dokumen ini memuat ringkasan evaluasi teknis hasil otomasi ulang Phase 3 sesuai `GUIDE.md`: train-only benchmark, dual-candidate one-stage, dan cabang two-stage yang dibangun ulang.

## Source of truth

- `outputs/phase3/final_metrics.csv`
- `outputs/phase3/per_class_metrics.csv`
- `outputs/phase3/confusion_matrix.csv`
- `outputs/phase3/threshold_sweep.csv`
- `outputs/phase3/error_stratification.csv`

## Kandidat One-Stage

- `yolo11m` (`last`, test): mAP50 `0.4840`, mAP50-95 `0.2470`, precision `0.4944`, recall `0.5563`, weighted F1 `0.4705`
- `yolov8s` (`last`, test): mAP50 `0.4621`, mAP50-95 `0.2378`, precision `0.4626`, recall `0.5431`, weighted F1 `0.4857`

## Cabang Two-Stage

- GT-crop classifier (`last`, test): top1 `0.6485`, weighted F1 `0.6337`
- End-to-end (`last`, test): precision `0.4840`, recall `0.5053`, F1 `0.4944`
