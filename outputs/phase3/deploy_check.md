# Deploy Check

Dokumen ini menjelaskan status deployment untuk run final `p3_final_yolo11m_640_s42_e60p15m60`. Untuk narasi kenapa model ini dipilih, buka [final_report.md](final_report.md). Untuk metrik teknis lengkap, buka [final_evaluation.md](final_evaluation.md).

## Sumber data

- [p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json)
- [p3_final_yolo11m_640_s42_e60p15m60_summary.json](p3_final_yolo11m_640_s42_e60p15m60_summary.json)
- [best.pt](../../runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt)

## Status saat ini

| Item | Status |
|---|---|
| Deploy status | **deferred by repo override** |
| TFLite export | skipped |
| TFLite INT8 export | skipped |
| Best weight size | **38.63 MB** |

## Kenapa deploy ditunda

Keputusan ini disengaja, bukan karena keterbatasan teknis. Dua hal yang diprioritaskan lebih dulu:

1. **Weight final harus aman dan tersimpan** — `best.pt` sudah ada di Git dan terverifikasi.
2. **Evaluasi eksperimen harus terdokumentasi** — metrik, error analysis, dan threshold sweep sudah selesai.

Konversi ke format deployment (TFLite, ONNX, INT8 quantization) diperlakukan sebagai tahap engineering terpisah. Menjalankannya di atas model yang belum memenuhi target `AP50 >= 0.70` di semua kelas berarti menghasilkan artefak deployment yang kemungkinan besar perlu diulang setelah iterasi model berikutnya — effort yang tidak efisien.

## Konsekuensi praktis

- Weight `.pt` final sudah siap sebagai artefak utama untuk inference di GPU
- Artefak deployment (TFLite, INT8) **belum** menjadi keluaran sesi ini
- Viabilitas inference di edge device (tablet, embedded) **masih perlu divalidasi** di hardware target

## Jika deploy dilanjutkan nanti

Saat model dikonversi ke format deployment, empat hal berikut **wajib** divalidasi ulang pada artefak hasil konversi:

1. **Akurasi** — quantization bisa menurunkan performa, terutama di kelas yang sudah lemah (B2, B4)
2. **Ukuran file** — memastikan fit di storage target
3. **Latency** — mengukur inference time di hardware sebenarnya
4. **Kompatibilitas** — memastikan format berjalan di runtime target (TFLite Interpreter, ONNX Runtime, dsb.)

Jangan mengasumsikan bahwa metrik dari evaluasi `.pt` otomatis berlaku untuk artefak deployment — konversi format selalu berpotensi mengubah perilaku model.

## File yang diperlukan untuk melanjutkan ke deploy

- [final_evaluation.md](final_evaluation.md) — untuk baseline metrik pre-conversion
- [p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json) — angka resmi
- [final_hparams.yaml](../phase2/final_hparams.yaml) — konfigurasi model
- [best.pt](../../runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt) — weight sumber
