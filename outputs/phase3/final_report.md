# Final Report - Phase 3 Multi-Candidate Benchmark

- Canonical protocol source: `https://github.com/muhammad-zainal-muttaqin/YOLOBench/blob/main/E0_Protocol_Flowchart.html`
- Phase 3 aktif mengikuti protokol final dosen: training memakai gabungan `train+val`, tanpa validasi saat training (`val=False`), lalu evaluasi post-fit pada `val` dan `test`.
- Kandidat utama one-stage: `yolo11m.pt` dan `yolov8s.pt`.
- Checkpoint utama untuk pelaporan final: `last.pt`.
- Confidence evaluasi dikunci tetap di `0.10`.

## One-Stage Test Set — `last.pt`

- `yolo11m` | mAP50 `0.5081` | mAP50-95 `0.2693` | precision `0.5038` | recall `0.5991` | conf `0.1`
- `yolov8s` | mAP50 `0.4962` | mAP50-95 `0.2600` | precision `0.4940` | recall `0.5757` | conf `0.1`

## Gap Val vs Test — `last.pt`

- `yolo11m` | mAP50 val `0.6334` -> test `0.5081` | precision val `0.5889` -> test `0.5038` | recall val `0.6627` -> test `0.5991`
- `yolov8s` | mAP50 val `0.6650` -> test `0.4962` | precision val `0.6220` -> test `0.4940` | recall val `0.6904` -> test `0.5757`
