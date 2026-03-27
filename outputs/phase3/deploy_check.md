# Deploy Check

## Navigasi mini

- Ringkasan repo: [README.md](../../README.md)
- Final report: [outputs/phase3/final_report.md](final_report.md)
- Final evaluation: [outputs/phase3/final_evaluation.md](final_evaluation.md)
- Eval JSON resmi: [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json)
- Weight final: [runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt](../../runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt)

- Status: **deferred by repo override**.
- TFLite export: `skipped for now`
- TFLite INT8 export: `skipped for now`
- Rationale: amankan final `best.pt` dan validasi metrik eksperimen lebih dulu; konversi deploy boleh dilakukan belakangan sebagai langkah engineering terpisah.
- Important: jika nanti dikonversi ke TFLite / INT8 / format lain, akurasi, ukuran, latency, dan kompatibilitas hardware **wajib divalidasi ulang** pada artefak hasil konversi itu.
- Best weight size MB: `38.63`
- Inference viability nyata di tablet tetap perlu pengujian hardware terpisah bila device tersedia.
