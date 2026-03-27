# Brand New YOLO — Laporan Akhir E0

`README.md` ini adalah **lapisan terluar** untuk pembaca repo. Mulailah dari sini, lalu telusuri file-file sumber yang disebut langsung pada setiap bagian.

Jika Anda ingin ringkasan keputusan lintas fase, buka [outputs/phase3/final_report.md](outputs/phase3/final_report.md). Jika Anda ingin evaluasi teknis test-set final, buka [outputs/phase3/final_evaluation.md](outputs/phase3/final_evaluation.md).

## Mulai dari file mana?

- Untuk narasi akhir end-to-end, buka [outputs/phase3/final_report.md](outputs/phase3/final_report.md).
- Untuk metrik test akhir yang lebih rinci, buka [outputs/phase3/final_evaluation.md](outputs/phase3/final_evaluation.md).
- Untuk skor resmi run final, jadikan [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json) dan [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json](outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json) sebagai acuan utama.
- Untuk weight final yang diamankan, buka [runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt](runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt).

## Sumber protokol, runbook, dan konteks

Protokol canonical eksperimen dirujuk di [E0.md](E0.md) dan sumber flowchart aslinya adalah `https://github.com/muhammad-zainal-muttaqin/YOLOBench/blob/main/E0_Protocol_Flowchart.html`. Override operasional repo ini dicatat di [GUIDE.md](GUIDE.md), sedangkan konteks keputusan dan semantik label tambahan dicatat di [CONTEXT.md](CONTEXT.md).

Konfigurasi yang benar-benar dikunci setelah seleksi model dapat dilihat langsung di [outputs/phase1/locked_setup.yaml](outputs/phase1/locked_setup.yaml), dan konfigurasi final tuning yang masuk ke Phase 3 dapat dilihat langsung di [outputs/phase2/final_hparams.yaml](outputs/phase2/final_hparams.yaml).

## Semantic mapping yang dipakai di repo ini

Semantic mapping label yang dipakai konsisten di [E0.md](E0.md), [GUIDE.md](GUIDE.md), [CONTEXT.md](CONTEXT.md), dan [outputs/phase1/locked_setup.yaml](outputs/phase1/locked_setup.yaml):

- `B1`: buah **merah**, **besar**, **bulat**, posisi **paling bawah** pada tandan → **paling matang / ripe**
- `B2`: buah masih **hitam** namun mulai **transisi ke merah**, sudah **besar** dan **bulat**, posisi **di atas B1**
- `B3`: buah **full hitam**, masih **berduri**, masih **lonjong**, posisi **di atas B2**
- `B4`: buah **paling kecil**, **paling dalam di batang/tandan**, sulit terlihat, masih banyak **duri**, warna **hitam sampai hijau**, dan masih bisa berkembang lebih besar → **paling belum matang**

Urutan biologis yang dipakai adalah **`B1 -> B2 -> B3 -> B4` = paling matang ke paling belum matang**.

## Ringkasan keputusan sesuai alur E0

| Fase E0 | Keputusan inti | File sumber yang harus dibuka |
|---|---|---|
| Phase 0 | Dataset lolos validasi dan resolusi kerja dipilih `640` | [outputs/phase0/dataset_audit.json](outputs/phase0/dataset_audit.json), [outputs/phase0/phase0_summary.md](outputs/phase0/phase0_summary.md), [outputs/phase0/learning_curve.csv](outputs/phase0/learning_curve.csv) |
| Phase 1A | Pipeline **one-stage** dipilih, two-stage tidak cukup meyakinkan secara end-to-end | [outputs/phase1/one_stage_results.csv](outputs/phase1/one_stage_results.csv), [outputs/phase1/two_stage_results.csv](outputs/phase1/two_stage_results.csv), [outputs/phase1/phase1_summary.md](outputs/phase1/phase1_summary.md) |
| Phase 1B | Model terbaik yang di-lock adalah `yolo11m.pt` | [outputs/phase1/architecture_benchmark.csv](outputs/phase1/architecture_benchmark.csv), [outputs/phase1/phase1b_top3.csv](outputs/phase1/phase1b_top3.csv), [outputs/phase1/locked_setup.yaml](outputs/phase1/locked_setup.yaml) |
| Phase 2 | Hyperparameter final tetap pada recipe `lr0=0.001`, `batch=16`, `imbalance=none`, `ordinal=standard`, `aug=medium` | [outputs/phase2/phase2_summary.md](outputs/phase2/phase2_summary.md), [outputs/phase2/tuning_results.csv](outputs/phase2/tuning_results.csv), [outputs/phase2/final_hparams.yaml](outputs/phase2/final_hparams.yaml), [outputs/phase2/p2confirm_yolo11m_640_s3_e30p10m30_eval.json](outputs/phase2/p2confirm_yolo11m_640_s3_e30p10m30_eval.json) |
| Phase 3 | Final retrain dijalankan pada test protocol repo ini; weight final diamankan; deploy check ditunda | [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json), [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json](outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json), [outputs/phase3/threshold_sweep.csv](outputs/phase3/threshold_sweep.csv), [outputs/phase3/error_analysis.md](outputs/phase3/error_analysis.md), [outputs/phase3/deploy_check.md](outputs/phase3/deploy_check.md) |

## Hasil akhir singkat

Keputusan model final tetap mengikuti lock di [outputs/phase1/locked_setup.yaml](outputs/phase1/locked_setup.yaml): model yang dipakai adalah `yolo11m.pt` pada resolusi `640`. Recipe final untuk Phase 3 tetap mengikuti [outputs/phase2/final_hparams.yaml](outputs/phase2/final_hparams.yaml): `lr0=0.001`, `batch=16`, `imbalance_strategy=none`, `ordinal_strategy=standard`, dan `aug_profile=medium`.

Untuk skor resmi evaluasi akhir, gunakan angka pada [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json) dan [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json](outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json):

- precision: **0.4763**
- recall: **0.5538**
- mAP50: **0.4677**
- mAP50-95: **0.2215**
- all classes `AP50 >= 0.70`: **False**

Untuk threshold operasi, file [outputs/phase3/threshold_sweep.csv](outputs/phase3/threshold_sweep.csv) menunjukkan kandidat terbaik di artefak ini berada pada `conf=0.1`. Untuk keputusan error yang dominan, buka [outputs/phase3/error_analysis.md](outputs/phase3/error_analysis.md) dan [outputs/phase3/error_stratification.csv](outputs/phase3/error_stratification.csv). Untuk weight final yang dipakai, buka [runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt](runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt).

Status deploy untuk sesi ini tetap **deferred**, sesuai catatan di [outputs/phase3/deploy_check.md](outputs/phase3/deploy_check.md). Artinya weight final sudah diamankan, tetapi ekspor TFLite/INT8 dan validasi device nyata masih menjadi pekerjaan lanjutan.

## Urutan baca yang disarankan

1. Baca [README.md](README.md) ini untuk peta besar.
2. Lanjut ke [outputs/phase3/final_report.md](outputs/phase3/final_report.md) untuk melihat alur keputusan Phase 0 sampai Phase 3.
3. Lanjut ke [outputs/phase3/final_evaluation.md](outputs/phase3/final_evaluation.md) untuk evaluasi teknis run final.
4. Jika ingin audit keputusan per fase, buka berurutan [outputs/phase0/phase0_summary.md](outputs/phase0/phase0_summary.md), [outputs/phase1/architecture_benchmark.csv](outputs/phase1/architecture_benchmark.csv), [outputs/phase2/tuning_results.csv](outputs/phase2/tuning_results.csv), lalu [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json).
5. Jika ingin audit seluruh run, buka [outputs/reports/run_ledger.csv](outputs/reports/run_ledger.csv) dan histori sinkronisasi pada [outputs/reports/git_sync_log.md](outputs/reports/git_sync_log.md).

## Navigasi artefak penting

- Audit dataset: [outputs/phase0/dataset_audit.json](outputs/phase0/dataset_audit.json)
- Ringkasan Phase 0: [outputs/phase0/phase0_summary.md](outputs/phase0/phase0_summary.md)
- Benchmark arsitektur: [outputs/phase1/architecture_benchmark.csv](outputs/phase1/architecture_benchmark.csv)
- Top-3 Phase 1B: [outputs/phase1/phase1b_top3.csv](outputs/phase1/phase1b_top3.csv)
- Lock setup: [outputs/phase1/locked_setup.yaml](outputs/phase1/locked_setup.yaml)
- Ringkasan tuning: [outputs/phase2/phase2_summary.md](outputs/phase2/phase2_summary.md)
- Hyperparameter final: [outputs/phase2/final_hparams.yaml](outputs/phase2/final_hparams.yaml)
- Confirm run Phase 2: [outputs/phase2/p2confirm_yolo11m_640_s3_e30p10m30_eval.json](outputs/phase2/p2confirm_yolo11m_640_s3_e30p10m30_eval.json)
- Evaluasi run final: [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json)
- Metadata run final: [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json](outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json)
- Threshold sweep: [outputs/phase3/threshold_sweep.csv](outputs/phase3/threshold_sweep.csv)
- Analisis error: [outputs/phase3/error_analysis.md](outputs/phase3/error_analysis.md)
- Stratifikasi error: [outputs/phase3/error_stratification.csv](outputs/phase3/error_stratification.csv)
- Final report: [outputs/phase3/final_report.md](outputs/phase3/final_report.md)
- Final evaluation: [outputs/phase3/final_evaluation.md](outputs/phase3/final_evaluation.md)
- Weight final: [runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt](runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt)

## Audit trail

Untuk melihat seluruh ledger run gunakan [outputs/reports/run_ledger.csv](outputs/reports/run_ledger.csv). Untuk melihat jejak commit/push artefak gunakan [outputs/reports/git_sync_log.md](outputs/reports/git_sync_log.md). Untuk status akhir orchestrator gunakan [outputs/reports/master_state.json](outputs/reports/master_state.json).
