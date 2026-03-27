# Final Report — E0 End-to-End

Dokumen ini adalah ringkasan keputusan akhir eksperimen E0 dari **Phase 0** sampai **Phase 3**. Jika Anda baru membuka repo ini, mulai dari [README.md](../../README.md). Jika Anda ingin evaluasi teknis run final secara lebih rinci, lanjut ke [outputs/phase3/final_evaluation.md](final_evaluation.md).

## Navigasi mini

- Ringkasan repo: [README.md](../../README.md)
- Phase 0 summary: [outputs/phase0/phase0_summary.md](../phase0/phase0_summary.md)
- Phase 1 summary: [outputs/phase1/phase1_summary.md](../phase1/phase1_summary.md)
- Phase 2 summary: [outputs/phase2/phase2_summary.md](../phase2/phase2_summary.md)
- Evaluasi final teknis: [outputs/phase3/final_evaluation.md](final_evaluation.md)
- Eval JSON resmi: [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json)
- Weight final: [runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt](../../runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt)

## 1. Dokumen acuan yang dipakai

Protokol canonical eksperimen repo ini diringkas di [E0.md](../../E0.md) dan runbook operasional repo ini dicatat di [GUIDE.md](../../GUIDE.md). Konteks keputusan tambahan ada di [CONTEXT.md](../../CONTEXT.md). Lock setup yang benar-benar dipakai lintas Phase 1B sampai Phase 3 dapat dicek langsung di [outputs/phase1/locked_setup.yaml](../phase1/locked_setup.yaml).

Dataset aktif yang dipakai tetap mengikuti [Dataset-YOLO/data.yaml](../../Dataset-YOLO/data.yaml), sedangkan audit dataset aktualnya tercatat di [outputs/phase0/dataset_audit.json](../phase0/dataset_audit.json).

## 2. Keputusan akhir yang dikunci

Keputusan akhir eksperimen ini adalah:

- model final: `yolo11m.pt`, sesuai [outputs/phase1/locked_setup.yaml](../phase1/locked_setup.yaml)
- resolusi kerja final: `640`, sesuai [outputs/phase0/phase0_summary.md](../phase0/phase0_summary.md)
- pipeline final: **one-stage**, sesuai [outputs/phase1/one_stage_results.csv](../phase1/one_stage_results.csv) dan [outputs/phase1/two_stage_results.csv](../phase1/two_stage_results.csv)
- hyperparameter final: `lr0=0.001`, `batch=16`, `imbalance_strategy=none`, `ordinal_strategy=standard`, `aug_profile=medium`, sesuai [outputs/phase2/final_hparams.yaml](../phase2/final_hparams.yaml)
- final test run: `p3_final_yolo11m_640_s42_e60p15m60`, sesuai [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json](p3_final_yolo11m_640_s42_e60p15m60_summary.json)
- final best weight: [runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt](../../runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt)

## 3. Narasi keputusan per fase

### Phase 0 — validasi dataset dan pemilihan resolusi

Validasi dataset untuk eksperimen ini dapat dilihat pada [outputs/phase0/dataset_audit.json](../phase0/dataset_audit.json). File itu menunjukkan dataset aktif berasal dari `/workspace/Dataset-Sawit-YOLO` dengan total **3992** image, **17987** instance, split **train 2764 / val 604 / test 624**, serta **0 invalid label issue** setelah self-healing.

Keputusan resolusi kerja diambil dari [outputs/phase0/phase0_summary.md](../phase0/phase0_summary.md) dan angka mentah learning curve dapat ditelusuri di [outputs/phase0/learning_curve.csv](../phase0/learning_curve.csv). Di Phase 0, `1024` memang sedikit lebih tinggi daripada `640`, tetapi kenaikannya tidak cukup besar dibanding ongkos compute, sehingga **imgsz `640` dipilih dan dikunci**.

### Phase 1A — pemilihan pipeline

Perbandingan one-stage baseline dapat dilihat langsung di [outputs/phase1/one_stage_results.csv](../phase1/one_stage_results.csv), sedangkan komponen two-stage dapat dilihat di [outputs/phase1/two_stage_results.csv](../phase1/two_stage_results.csv). Ringkasan bacanya juga ada di [outputs/phase1/phase1_summary.md](../phase1/phase1_summary.md).

Keputusan yang diambil di Phase 1A adalah **memilih pipeline one-stage**. Alasan utamanya: one-stage memberi baseline end-to-end yang langsung terukur pada task 4 kelas, sedangkan two-stage masih menyisakan confusion yang besar pada klasifikasi `B2/B3` walaupun dievaluasi pada GT crops.

### Phase 1B — benchmark arsitektur dan lock model

Benchmark arsitektur lengkap ada di [outputs/phase1/architecture_benchmark.csv](../phase1/architecture_benchmark.csv), dan shortlist terbaiknya ada di [outputs/phase1/phase1b_top3.csv](../phase1/phase1b_top3.csv). File [outputs/phase1/locked_setup.yaml](../phase1/locked_setup.yaml) menjadi bukti lock akhir bahwa model yang lolos ke fase berikutnya adalah **`yolo11m.pt`**.

Tiga model terbaik yang tercatat di [outputs/phase1/phase1b_top3.csv](../phase1/phase1b_top3.csv) adalah:

| Rank | Model | mean mAP50 | mean mAP50-95 |
|---:|---|---:|---:|
| 1 | `yolo11m.pt` | 0.5298 | 0.2570 |
| 2 | `yolov9c.pt` | 0.5292 | 0.2518 |
| 3 | `yolov8s.pt` | 0.5256 | 0.2521 |

Gate canonical `mAP50 >= 0.70` memang tidak lolos, dan status itu tetap tercatat di [outputs/phase1/locked_setup.yaml](../phase1/locked_setup.yaml). Namun pipeline dilanjutkan sesuai override repo yang tertulis di [GUIDE.md](../../GUIDE.md).

### Phase 2 — tuning hyperparameter pada model yang sudah di-lock

Ringkasan Phase 2 tersimpan di [outputs/phase2/phase2_summary.md](../phase2/phase2_summary.md), hasil agregasi tuning ada di [outputs/phase2/tuning_results.csv](../phase2/tuning_results.csv), dan konfigurasi final yang dibawa ke Phase 3 tersimpan di [outputs/phase2/final_hparams.yaml](../phase2/final_hparams.yaml).

File [outputs/phase2/tuning_results.csv](../phase2/tuning_results.csv) mencatat bahwa hasil akhir memakai `final_source=phase1_baseline_reverted` dan `reverted_to_phase1_baseline=True`. Artinya, walaupun sweep Phase 2 tetap dilakukan, recipe final yang dipertahankan tetap sama dengan baseline yang paling stabil: `lr0=0.001`, `batch=16`, `imbalance=none`, `ordinal=standard`, dan `aug=medium`.

Run konfirmasi Phase 2 dapat dicek langsung di [outputs/phase2/p2confirm_yolo11m_640_s3_e30p10m30_eval.json](../phase2/p2confirm_yolo11m_640_s3_e30p10m30_eval.json) dan [outputs/phase2/p2confirm_yolo11m_640_s3_e30p10m30_summary.json](../phase2/p2confirm_yolo11m_640_s3_e30p10m30_summary.json). Confirm run tersebut menutup Phase 2 dengan `mAP50=0.5390` dan `mAP50-95=0.2594` pada split `val`.

### Phase 3 — final retrain, evaluasi test, dan artefak akhir

Run final Phase 3 didokumentasikan di [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json](p3_final_yolo11m_640_s42_e60p15m60_summary.json), sedangkan metrik test resminya dapat dibaca di [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json). File [outputs/phase3/final_data.yaml](final_data.yaml) menunjukkan data split yang dipakai untuk final test protocol repo ini.

Untuk weight final yang harus dipakai sebagai keluaran utama eksperimen ini, buka [runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt](../../runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt).

## 4. Hasil akhir yang harus dianggap resmi

Untuk angka resmi final, prioritaskan [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json) dan [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json](p3_final_yolo11m_640_s42_e60p15m60_summary.json), karena kedua file itu melekat langsung ke run final. File [outputs/phase3/final_metrics.csv](final_metrics.csv) tetap disimpan sebagai ringkasan cepat, tetapi ketika pembaca membutuhkan angka rujukan utama, gunakan file run-specific di atas.

| Metrik resmi test set | Nilai | Sumber |
|---|---:|---|
| Precision | 0.4763 | [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json) |
| Recall | 0.5538 | [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json) |
| mAP50 | 0.4677 | [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json) |
| mAP50-95 | 0.2215 | [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json) |
| All classes `AP50 >= 0.70` | False | [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json) |

Per-class metrics di [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json) menunjukkan pola berikut:

- `B1` adalah kelas paling kuat (`mAP50=0.7821`)
- `B3` berada di tengah (`mAP50=0.4880`)
- `B2` masih lemah (`mAP50=0.3266`)
- `B4` adalah kelas tersulit (`mAP50=0.2742`)

## 5. Threshold operasi, error dominan, dan deploy status

Threshold operasi dibahas di [outputs/phase3/threshold_sweep.csv](threshold_sweep.csv). File itu menunjukkan kandidat threshold terbaik pada artefak saat ini berada di `conf=0.1`. Angka pada [outputs/phase3/threshold_sweep.csv](threshold_sweep.csv) dipakai untuk memilih **operating point**, bukan untuk menggantikan skor resmi evaluasi test pada [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json).

Analisis error ringkas ada di [outputs/phase3/error_analysis.md](error_analysis.md), dan daftar contoh image terburuk ada di [outputs/phase3/error_stratification.csv](error_stratification.csv). Dua sumber itu menunjukkan error dominan masih terkonsentrasi pada:

- `false_positive`
- confusion `B2_B3`
- `B4_missed`
- confusion `B3_B4`

File [outputs/phase3/confusion_matrix.csv](confusion_matrix.csv) tetap disimpan sebagai artefak ekspor, tetapi untuk pembacaan praktis run ini, [outputs/phase3/error_stratification.csv](error_stratification.csv) dan [outputs/phase3/error_analysis.md](error_analysis.md) jauh lebih informatif.

Status deploy untuk sesi ini tetap **deferred**, sesuai [outputs/phase3/deploy_check.md](deploy_check.md). Jadi keluaran utama sesi ini adalah weight final dan laporan evaluasi, bukan artefak TFLite/INT8.

## 6. Kesimpulan akhir

Eksperimen E0 repo ini berakhir dengan keputusan final berikut:

1. dataset valid dan cukup siap dipakai, berdasarkan [outputs/phase0/dataset_audit.json](../phase0/dataset_audit.json)
2. resolusi kerja terbaik yang realistis adalah `640`, berdasarkan [outputs/phase0/phase0_summary.md](../phase0/phase0_summary.md)
3. pipeline yang dipertahankan adalah **one-stage**, berdasarkan [outputs/phase1/one_stage_results.csv](../phase1/one_stage_results.csv) dan [outputs/phase1/two_stage_results.csv](../phase1/two_stage_results.csv)
4. model yang di-lock sampai akhir adalah `yolo11m.pt`, berdasarkan [outputs/phase1/architecture_benchmark.csv](../phase1/architecture_benchmark.csv) dan [outputs/phase1/locked_setup.yaml](../phase1/locked_setup.yaml)
5. recipe final yang dipakai sampai Phase 3 adalah recipe baseline stabil, berdasarkan [outputs/phase2/tuning_results.csv](../phase2/tuning_results.csv) dan [outputs/phase2/final_hparams.yaml](../phase2/final_hparams.yaml)
6. final `best.pt` berhasil diamankan di [runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt](../../runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt)
7. hasil akhir masih masuk bucket **needs work**, sehingga prioritas pengembangan berikutnya sebaiknya fokus ke pemisahan `B2/B3`, peningkatan recall `B4`, dan validasi deploy nyata setelah konversi benar-benar dilakukan

## 7. File yang sebaiknya dibuka setelah dokumen ini

- Buka [outputs/phase3/final_evaluation.md](final_evaluation.md) untuk evaluasi teknis run final.
- Buka [outputs/phase3/threshold_sweep.csv](threshold_sweep.csv) untuk threshold operasi.
- Buka [outputs/phase3/error_stratification.csv](error_stratification.csv) untuk melihat contoh image tersulit.
- Buka [outputs/phase2/final_hparams.yaml](../phase2/final_hparams.yaml) untuk recipe final yang bisa direplikasi.
- Buka [outputs/reports/run_ledger.csv](../reports/run_ledger.csv) untuk audit seluruh run E0.
