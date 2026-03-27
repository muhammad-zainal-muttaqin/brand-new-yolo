# Latest Execution Status

Dokumen ini adalah snapshot status eksekusi terakhir. Untuk narasi lengkap lintas fase, buka [outputs/phase3/final_report.md](../phase3/final_report.md). Untuk metrik teknis run final, buka [outputs/phase3/final_evaluation.md](../phase3/final_evaluation.md).

## Sumber utama

Status ini merujuk langsung ke:

- [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json](../phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json)
- [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](../phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json)

## Snapshot terakhir

- Timestamp UTC: `2026-03-27T13:28:14.265947+00:00`
- Phase: `phase3`
- Run name: `p3_final_yolo11m_640_s42_e60p15m60`
- Model: `yolo11m.pt`
- Image size: `640`
- Epochs: `60`
- Batch: `16`
- Seed: `42`
- Eval split: `test`
- Status: **completed**

## Lokasi artefak utama

- Save dir: `/workspace/brand-new-yolo/runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60`
- Best weight: `/workspace/brand-new-yolo/runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt`
- Last weight: `/workspace/brand-new-yolo/runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/last.pt`

Untuk file yang bisa dicek langsung di repo, buka:

- [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json](../phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json)
- [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](../phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json)
- [runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt](../../runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt)

## Metrik ringkas

- Precision: `0.47629844652241093`
- Recall: `0.5537874150529214`
- mAP50: `0.4677363979544993`
- mAP50-95: `0.2214749388595758`

## Cara lanjut membaca

- Ringkasan akhir: [outputs/phase3/final_report.md](../phase3/final_report.md)
- Evaluasi teknis: [outputs/phase3/final_evaluation.md](../phase3/final_evaluation.md)
- Reproduksi dan terminasi: [outputs/reports/reproducibility_and_termination.md](reproducibility_and_termination.md)
