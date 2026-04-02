# Final Evaluation — Phase 3

Dokumen ini memuat ringkasan evaluasi teknis hasil rerun Phase 3 dengan protokol final dosen: train `train+val`, tanpa validasi saat training, lalu evaluasi `last.pt` pada `val` dan `test`.

## Source of truth

- `outputs/phase3/final_metrics.csv`
- `outputs/phase3/per_class_metrics.csv`
- `outputs/phase3/confusion_matrix.csv`
- `outputs/phase3/error_stratification.csv`

## Kandidat One-Stage

- `yolo11m` (`last`, test): mAP50 `0.5081`, mAP50-95 `0.2693`, precision `0.5038`, recall `0.5991`, weighted F1 `0.4717`
- `yolov8s` (`last`, test): mAP50 `0.4962`, mAP50-95 `0.2600`, precision `0.4940`, recall `0.5757`, weighted F1 `0.4685`
