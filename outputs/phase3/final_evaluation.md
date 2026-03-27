# Evaluasi Final — Phase 3 Test Set

Dokumen ini jelasin evaluasi teknis run final `p3_final_yolo11m_640_s42_e60p15m60`. Narasi keputusan lintas fase ada di [final_report.md](final_report.md), peta baca repo di [README.md](../../README.md).

## Sumber utama

File yang dirujuk langsung:

- [p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json)
- [p3_final_yolo11m_640_s42_e60p15m60_summary.json](p3_final_yolo11m_640_s42_e60p15m60_summary.json)
- [final_metrics.csv](final_metrics.csv)
- [threshold_sweep.csv](threshold_sweep.csv)
- [error_analysis.md](error_analysis.md)
- [error_stratification.csv](error_stratification.csv)
- [deploy_check.md](deploy_check.md)

Kalau butuh angka resmi, prioritaskan **eval JSON** dan **summary JSON** di atas.

## 1. Identitas run yang dievaluasi

Metadata run final berasal dari [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json](p3_final_yolo11m_640_s42_e60p15m60_summary.json):

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

## 2. Metrik resmi test set

Angka berikut diambil langsung dari [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json):

| Metrik | Nilai |
|---|---:|
| Precision | 0.4763 |
| Recall | 0.5538 |
| mAP50 | 0.4677 |
| mAP50-95 | 0.2215 |
| B4 recall | 0.3798 |
| All classes `AP50 >= 0.70` | False |

Interpretasi langsung:

- model final ini **cukup layak sebagai baseline akhir repo**
- tetapi kualitasnya belum merata di semua kelas
- target keras `all classes >= 70% AP50` **belum tercapai**

## 3. Metrik per kelas

Tabel ini juga berasal dari eval JSON final:

| Kelas | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| `B1` | 0.7237 | 0.7262 | 0.7821 | 0.4246 |
| `B2` | 0.3932 | 0.4182 | 0.3266 | 0.1481 |
| `B3` | 0.4603 | 0.6910 | 0.4880 | 0.2138 |
| `B4` | 0.3280 | 0.3798 | 0.2742 | 0.0993 |

Pembacaan singkat:

- `B1` jelas paling stabil
- `B3` sudah cukup baik pada recall, tetapi presisinya belum tinggi
- `B2` masih tertahan confusion kelas tetangga
- `B4` tetap paling sulit, baik dari sisi ukuran maupun recall

## 4. Threshold operasi

Threshold operasi tidak dipilih secara manual. Kita memakai [outputs/phase3/threshold_sweep.csv](threshold_sweep.csv).

Kandidat threshold terbaik pada artefak ini berada di **`conf=0.1`**.

Ringkasan kandidat `conf=0.1`:

| conf | precision | recall | map50 | map50-95 | b4_recall |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.7032 | 0.6995 | 0.7395 | 0.4499 | 0.5415 |

Catatan penting: angka threshold sweep dipakai untuk memilih **operating point**, bukan untuk mengganti skor resmi run final di eval JSON.

## 5. Error dominan

Error praktis dirangkum di [outputs/phase3/error_analysis.md](error_analysis.md). Berdasarkan [outputs/phase3/error_stratification.csv](error_stratification.csv), pola error dominan adalah:

- `false_positive`: **20** image
- `B2_B3_confusion`: **13** image
- `B4_missed`: **11** image
- `B3_B4_confusion`: **10** image

Daftar gambar tersulit bisa dicek langsung di [outputs/phase3/error_stratification.csv](error_stratification.csv). Beberapa contoh awalnya adalah:

- `/workspace/Dataset-Sawit-YOLO/images/test/DAMIMAS_A21B_0838_5.jpg`
- `/workspace/Dataset-Sawit-YOLO/images/test/DAMIMAS_A21B_0075_2.jpg`
- `/workspace/Dataset-Sawit-YOLO/images/test/DAMIMAS_A21B_0292_2.jpg`

## 6. Hubungan hasil final dengan Phase 2

Run final ini **bukan model baru di luar lock file**. Ini adalah evaluasi terakhir dari setup yang sudah dipilih sebelumnya.

Bukti keterkaitannya ada di:

- [outputs/phase1/locked_setup.yaml](../phase1/locked_setup.yaml)
- [outputs/phase2/final_hparams.yaml](../phase2/final_hparams.yaml)
- [outputs/phase2/p2confirm_yolo11m_640_s3_e30p10m30_eval.json](../phase2/p2confirm_yolo11m_640_s3_e30p10m30_eval.json)

## 7. Status deploy

Status deploy saat ini tetap **deferred**, sesuai [outputs/phase3/deploy_check.md](deploy_check.md).

Artinya, keluaran utama yang siap dipakai dari sesi ini adalah:

- [outputs/phase3/final_report.md](final_report.md)
- [outputs/phase3/final_evaluation.md](final_evaluation.md)
- [runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt](../../runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt)
- [outputs/phase2/final_hparams.yaml](../phase2/final_hparams.yaml)

## 8. Kesimpulan evaluasi

Evaluasi final ini menunjukkan bahwa repo berhasil menghasilkan **satu setup final yang konsisten dan bisa dilacak**, tetapi model masih belum kuat di semua kelas.

Jalur perbaikan yang paling jelas setelah membaca hasil final ini adalah:

1. mengurangi confusion `B2/B3`
2. meningkatkan recall `B4`
3. menekan `false_positive`
4. baru setelah itu melanjutkan ke deploy yang tervalidasi ulang
