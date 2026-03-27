# Deploy Check

Dokumen ini menjelaskan status deploy untuk run final `p3_final_yolo11m_640_s42_e60p15m60`. Untuk melihat alasan model final dipilih, buka [outputs/phase3/final_report.md](final_report.md). Untuk melihat metrik teknis final, buka [outputs/phase3/final_evaluation.md](final_evaluation.md).

## Sumber utama

Dokumen ini merujuk langsung ke:

- [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json)
- [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json](p3_final_yolo11m_640_s42_e60p15m60_summary.json)
- [runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt](../../runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt)

## Status saat ini

- status deploy: **deferred by repo override**
- TFLite export: `skipped for now`
- TFLite INT8 export: `skipped for now`
- best weight size: **`38.63 MB`**

## Alasan deploy ditunda

Repo ini memprioritaskan dua hal lebih dulu:

1. memastikan `best.pt` final aman dan tersimpan
2. memastikan metrik eksperimen final sudah terdokumentasi dengan jelas

Karena itu, konversi deploy diperlakukan sebagai langkah engineering terpisah, bukan bagian yang harus selesai pada sesi final training ini.

## Konsekuensi praktis

Penundaan deploy berarti:

- weight final sudah siap sebagai artefak utama
- tetapi artefak deploy seperti TFLite atau INT8 **belum** menjadi keluaran resmi sesi ini
- inference viability nyata di tablet atau device edge **masih perlu diuji** di hardware yang sesuai

## Aturan jika deploy dilakukan nanti

Jika model ini nanti dikonversi ke TFLite, INT8, atau format lain, maka empat hal ini **wajib** diuji ulang pada artefak hasil konversi:

- akurasi
- ukuran file
- latency
- kompatibilitas hardware target

Dengan kata lain, jangan menganggap hasil evaluasi pada file `.pt` otomatis berlaku untuk artefak deploy hasil konversi.

## File yang harus dibuka jika ingin lanjut ke deploy

- [outputs/phase3/final_evaluation.md](final_evaluation.md)
- [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json)
- [outputs/phase2/final_hparams.yaml](../phase2/final_hparams.yaml)
- [runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt](../../runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt)
