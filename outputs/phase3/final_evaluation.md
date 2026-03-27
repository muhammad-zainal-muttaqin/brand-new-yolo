# Evaluasi Final — Phase 3 Test Set

Dokumen ini adalah evaluasi teknis untuk run final `p3_final_yolo11m_640_s42_e60p15m60`. Jika Anda ingin ringkasan lintas fase terlebih dahulu, baca [README.md](../../README.md). Jika Anda ingin narasi keputusan end-to-end, baca [outputs/phase3/final_report.md](final_report.md).

## Navigasi mini

- Ringkasan repo: [README.md](../../README.md)
- Final report lintas fase: [outputs/phase3/final_report.md](final_report.md)
- Eval JSON resmi: [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json)
- Metadata run final: [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json](p3_final_yolo11m_640_s42_e60p15m60_summary.json)
- Threshold sweep: [outputs/phase3/threshold_sweep.csv](threshold_sweep.csv)
- Error analysis: [outputs/phase3/error_analysis.md](error_analysis.md)
- Deploy check: [outputs/phase3/deploy_check.md](deploy_check.md)
- Weight final: [runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt](../../runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt)

## 1. File sumber evaluasi yang dipakai

Evaluasi pada dokumen ini dibaca langsung dari file-file berikut:

- [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json) sebagai sumber utama metrik test-set resmi
- [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json](p3_final_yolo11m_640_s42_e60p15m60_summary.json) sebagai sumber metadata run, path weight, dan konfigurasi eksekusi
- [outputs/phase3/final_metrics.csv](final_metrics.csv) sebagai ringkasan cepat tambahan
- [outputs/phase3/threshold_sweep.csv](threshold_sweep.csv) sebagai analisis threshold operasi
- [outputs/phase3/error_analysis.md](error_analysis.md) dan [outputs/phase3/error_stratification.csv](error_stratification.csv) sebagai analisis error
- [outputs/phase3/deploy_check.md](deploy_check.md) sebagai status deploy/TFLite/INT8

Ketika pembaca membutuhkan **angka resmi evaluasi final**, prioritaskan [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json) dan [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json](p3_final_yolo11m_640_s42_e60p15m60_summary.json), karena kedua file itu melekat langsung ke run final.

## 2. Identitas run yang dievaluasi

Metadata run final dapat dibaca langsung di [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json](p3_final_yolo11m_640_s42_e60p15m60_summary.json):

| Item | Nilai |
|---|---|
| Run name | `p3_final_yolo11m_640_s42_e60p15m60` |
| Phase | `phase3` |
| Task | `detect` |
| Model | `yolo11m.pt` |
| Split evaluasi | `test` |
| Seed | `42` |
| Epoch | `60` |
| Patience | `15` |
| Image size | `640` |
| Batch | `16` |
| Optimizer | `AdamW` |
| `lr0` | `0.001` |
| Imbalance strategy | `none` |
| Ordinal strategy | `standard` |
| Data file | [outputs/phase3/final_data.yaml](final_data.yaml) |
| Best weight | [runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt](../../runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt) |
| Last weight | [runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/last.pt](../../runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/last.pt) |

## 3. Metrik resmi test-set

Angka pada tabel berikut diambil dari [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json):

| Metrik | Nilai |
|---|---:|
| Precision | 0.4763 |
| Recall | 0.5538 |
| mAP50 | 0.4677 |
| mAP50-95 | 0.2215 |
| B4 recall | 0.3798 |
| All classes `AP50 >= 0.70` | False |

Interpretasi langsung dari [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json):

- model final **sudah usable sebagai baseline eksperimen akhir**, tetapi belum kuat secara merata pada semua kelas
- kelas `B1` jauh lebih mudah dikenali daripada `B2` dan `B4`
- target gate keras `all classes >= 70% AP50` **belum tercapai**

## 4. Metrik per kelas

Tabel berikut juga berasal langsung dari [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json):

| Kelas | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| `B1` | 0.7237 | 0.7262 | 0.7821 | 0.4246 |
| `B2` | 0.3932 | 0.4182 | 0.3266 | 0.1481 |
| `B3` | 0.4603 | 0.6910 | 0.4880 | 0.2138 |
| `B4` | 0.3280 | 0.3798 | 0.2742 | 0.0993 |

Pembacaan singkat terhadap tabel per kelas di [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json):

- `B1` adalah kelas paling stabil dan paling mudah dikenali
- `B3` sudah cukup terdeteksi, tetapi presisinya masih sedang
- `B2` masih tertahan oleh confusion dengan kelas tetangga tingkat kematangan
- `B4` tetap menjadi kelas paling sulit, baik dari sisi ukuran, visibilitas, maupun recall

## 5. Threshold operasi

Threshold operasi tidak diambil dari tebakan manual, tetapi dari [outputs/phase3/threshold_sweep.csv](threshold_sweep.csv). File itu menunjukkan kandidat threshold terbaik pada artefak ini berada di `conf=0.1`.

Ringkasan kandidat `conf=0.1` dari [outputs/phase3/threshold_sweep.csv](threshold_sweep.csv):

| conf | precision | recall | map50 | map50-95 | b4_recall |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.7032 | 0.6995 | 0.7395 | 0.4499 | 0.5415 |

Catatan penting: angka pada [outputs/phase3/threshold_sweep.csv](threshold_sweep.csv) dipakai untuk **membandingkan operating point antar-threshold**, sedangkan angka resmi skor model final tetap harus dibaca dari [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json).

## 6. Error dominan yang perlu diperhatikan

Ringkasan error praktis dapat dibaca di [outputs/phase3/error_analysis.md](error_analysis.md). Dokumen itu menunjukkan pola error dominan berikut:

- `false_positive`: `20` image
- `B2_B3_confusion`: `13` image
- `B4_missed`: `11` image
- `B3_B4_confusion`: `10` image

Daftar contoh image tersulit dapat dilihat langsung di [outputs/phase3/error_stratification.csv](error_stratification.csv). Contoh baris teratas di file itu menunjukkan bahwa image seperti `/workspace/Dataset-Sawit-YOLO/images/test/DAMIMAS_A21B_0838_5.jpg`, `/workspace/Dataset-Sawit-YOLO/images/test/DAMIMAS_A21B_0075_2.jpg`, dan `/workspace/Dataset-Sawit-YOLO/images/test/DAMIMAS_A21B_0292_2.jpg` termasuk kasus yang paling berat untuk run final ini.

File [outputs/phase3/confusion_matrix.csv](confusion_matrix.csv) tetap tersedia sebagai artefak ekspor, tetapi untuk pembacaan run ini file tersebut tidak lebih informatif daripada [outputs/phase3/error_analysis.md](error_analysis.md) dan [outputs/phase3/error_stratification.csv](error_stratification.csv).

## 7. Hubungan evaluasi final dengan keputusan Phase 2

Recipe yang dievaluasi pada Phase 3 ini tetap mengikuti hasil lock yang dapat dibaca pada [outputs/phase1/locked_setup.yaml](../phase1/locked_setup.yaml) dan [outputs/phase2/final_hparams.yaml](../phase2/final_hparams.yaml). Validasi stabilitas sebelum masuk Phase 3 dapat dilihat di [outputs/phase2/p2confirm_yolo11m_640_s3_e30p10m30_eval.json](../phase2/p2confirm_yolo11m_640_s3_e30p10m30_eval.json).

Dengan kata lain, evaluasi final ini **bukan** hasil eksperimen model baru; evaluasi ini adalah pembuktian terakhir untuk setup yang sudah dipilih dan di-lock sebelumnya.

## 8. Status deploy dan artefak keluaran utama

Status deploy saat ini tetap **deferred** sesuai [outputs/phase3/deploy_check.md](deploy_check.md). Jadi keluaran utama yang benar-benar siap dipakai dari sesi ini adalah:

- laporan akhir lintas fase di [outputs/phase3/final_report.md](final_report.md)
- evaluasi final teknis di [outputs/phase3/final_evaluation.md](final_evaluation.md)
- weight final di [runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt](../../runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt)
- recipe final di [outputs/phase2/final_hparams.yaml](../phase2/final_hparams.yaml)

## 9. Kesimpulan evaluasi

Evaluasi final pada [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json) menunjukkan bahwa eksperimen E0 berhasil menghasilkan **satu setup final yang konsisten dan terdokumentasi**, tetapi kualitas model belum merata di semua kelas. Jalur perbaikan yang paling jelas setelah membaca [outputs/phase3/error_analysis.md](error_analysis.md) dan [outputs/phase3/error_stratification.csv](error_stratification.csv) adalah:

1. mengurangi confusion `B2/B3`
2. meningkatkan recall `B4`
3. menekan `false_positive`
4. baru setelah itu melanjutkan ke export deploy yang tervalidasi ulang
