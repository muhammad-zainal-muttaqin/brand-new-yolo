# Local Cleanup Report

Dokumen ini mencatat pembersihan artefak lokal yang dilakukan untuk menghemat ruang disk di environment RunPod. Pembersihan ini **tidak** dimaksudkan untuk menghapus source of truth eksperimen dari remote. Untuk memastikan artefak final tetap aman, baca juga [outputs/reports/reproducibility_and_termination.md](reproducibility_and_termination.md).

## Prinsip cleanup yang dipakai

Cleanup lokal dilakukan dengan aturan berikut:

- artefak aktif yang masih dibutuhkan tidak dihapus
- artefak penting yang sudah aman di Git tetap dipertahankan secara logis sebagai source of truth
- penghapusan lokal tidak boleh diartikan sebagai penghapusan artefak resmi dari repo remote

## Event cleanup 1 — 2026-03-27T02:03:28Z

- active run yang dipertahankan: `p1bfc_yolov10s_640_s2_e30p10m30`
- jumlah path yang dibersihkan: **35**
- estimasi ruang yang dibebaskan: **968,223,126 bytes**

Contoh artefak yang dibersihkan pada event ini:

- `runs/detect/runs/e0/p0_yolo11n_640_s1_e3`
- `runs/detect/runs/e0/p0_yolo11n_1024_s1_e3`
- `runs/detect/runs/e0/p0_lc25_yolo11n_640_s1_e30p10m30`
- `runs/detect/runs/e0/p1b_yolov8m_640_s1_e30p10m30`
- `runs/detect/runs/e0/p1bfc_yolov9c_640_s2_e30p10m30`
- `yolov10s.pt`
- `yolo26n.pt`

## Event cleanup 2 — 2026-03-27T09:57:26Z

- remote diverifikasi: `origin/main @ 1d14971`
- active run yang dipertahankan: `p2s2_bs8_yolo11m_640_s2_e30p10m30`
- jumlah path yang dibersihkan: **58**
- estimasi ruang yang dibebaskan: **2,796,780,532 bytes**
- ukuran repo sebelum cleanup: **5,809,762,784 bytes**
- ukuran repo sesudah cleanup: **3,003,069,692 bytes**

Artefak kecil yang diunggah saat cleanup ini:

- `outputs/phase2/p2s2_bs8_yolo11m_640_s1_e30p10m30_eval.json`

Contoh path yang dibersihkan pada event ini:

- beberapa run Phase 0 (`p0_*`)
- run komponen Phase 1A (`p1a_*`)
- benchmark Phase 1B untuk berbagai model (`p1bfc_*`)
- beberapa run tuning Phase 2 (`p2s0a_*`, `p2s1_*`, `p2s2_*`)
- `yolo11m.pt`
- `yolo26n.pt`

## Cara membaca cleanup ini

Dua hal paling penting:

1. cleanup ini adalah **operasi lokal untuk menghemat disk**
2. cleanup ini **bukan** keputusan untuk membatalkan artefak yang sudah menjadi source of truth di repo

Karena itu, jika sebuah run lokal dibersihkan tetapi metrik dan artefak pentingnya sudah tersimpan di Git, maka referensi resmi tetap ada pada file summary, eval JSON, CSV agregat, dan dokumen phase summary.

## Dokumen yang tetap menjadi acuan resmi

Setelah cleanup lokal, source of truth tetap berada di:

- [outputs/phase0/phase0_summary.md](../phase0/phase0_summary.md)
- [outputs/phase1/phase1_summary.md](../phase1/phase1_summary.md)
- [outputs/phase2/phase2_summary.md](../phase2/phase2_summary.md)
- [outputs/phase3/final_report.md](../phase3/final_report.md)
- [outputs/phase3/final_evaluation.md](../phase3/final_evaluation.md)
- [outputs/reports/reproducibility_and_termination.md](reproducibility_and_termination.md)

## Kesimpulan

Cleanup lokal berhasil mengurangi penggunaan disk tanpa mengubah keputusan ilmiah repo. Jika ada keraguan tentang artefak mana yang harus dipercaya, prioritaskan dokumen dan file run-specific yang tercantum pada [README.md](../../README.md) dan [outputs/reports/reproducibility_and_termination.md](reproducibility_and_termination.md).
