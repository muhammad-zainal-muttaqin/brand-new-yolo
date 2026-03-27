# Reproducibility and RunPod Termination Checklist

Dokumen ini dibuat sebagai **checklist terakhir sebelum RunPod diterminasi**. Tujuannya ada dua: 
1. memastikan artefak penting sudah aman di GitHub, dan 
2. mencatat kondisi eksekusi dengan cukup rinci agar eksperimen ini bisa direproduksi.

Jika Anda mulai membaca dari root repo, buka juga [README.md](../../README.md). Untuk laporan lintas fase buka [outputs/phase3/final_report.md](../phase3/final_report.md). Untuk evaluasi teknis akhir buka [outputs/phase3/final_evaluation.md](../phase3/final_evaluation.md).

## Navigasi mini

- Ringkasan repo: [README.md](../../README.md)
- Final report: [outputs/phase3/final_report.md](../phase3/final_report.md)
- Final evaluation: [outputs/phase3/final_evaluation.md](../phase3/final_evaluation.md)
- Lock setup: [outputs/phase1/locked_setup.yaml](../phase1/locked_setup.yaml)
- Final hparams: [outputs/phase2/final_hparams.yaml](../phase2/final_hparams.yaml)
- Eval JSON final: [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](../phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json)
- Weight final: [runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt](../../runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt)
- Audit seluruh run: [outputs/reports/run_ledger.csv](run_ledger.csv)

## 1. Snapshot repositori yang harus dipakai untuk reproduksi

- Remote GitHub: `https://github.com/muhammad-zainal-muttaqin/brand-new-yolo.git`
- Branch: `main`
- Final archive tag: `runpod-archive-e0-final-20260327`
- Audit timestamp UTC: `2026-03-27T13:55:26Z`
- Working directory saat eksperimen berjalan: `/workspace/brand-new-yolo`
- Dataset root saat eksperimen berjalan: `/workspace/Dataset-Sawit-YOLO`

Untuk reproduksi, **clone repo ini lalu checkout tag `runpod-archive-e0-final-20260327`**, bukan hanya branch `main` terbaru, agar isi dokumen, script, dan artefak yang dirujuk tetap konsisten. Tag ini adalah referensi arsip final yang stabil untuk snapshot sebelum RunPod dimatikan.

## 2. Snapshot environment saat eksperimen berjalan

Environment yang benar-benar dipakai pada sesi ini:

- OS/platform: `Linux-6.8.0-52-generic-x86_64-with-glibc2.35`
- Python: `3.11.15`
- PyTorch: `2.4.1+cu124`
- Ultralytics: `8.4.30`
- CUDA available: `True`
- GPU count: `1`
- GPU: `NVIDIA A40`
- GPU memory: `46068 MiB`
- NVIDIA driver: `570.195.03`

Pemeriksaan dependency ringan bisa dijalankan dari [scripts/setup_env.sh](../../scripts/setup_env.sh). Script itu memeriksa ketersediaan modul inti seperti `torch`, `ultralytics`, `pandas`, `yaml`, `PIL`, `matplotlib`, dan `seaborn`.

## 3. Kondisi dataset yang dipakai

Dataset aktif repo ini dirujuk melalui [Dataset-YOLO/data.yaml](../../Dataset-YOLO/data.yaml). Isi intinya menunjuk ke:

- `path: /workspace/Dataset-Sawit-YOLO`
- `train: images/train`
- `val: images/val`
- `test: images/test`
- `nc: 4`
- kelas: `B1`, `B2`, `B3`, `B4`

Audit dataset mentah dapat dibaca di [outputs/phase0/dataset_audit.json](../phase0/dataset_audit.json). Snapshot audit yang penting:

- total images: `3992`
- total labels: `3992`
- total instances: `17987`
- split images: `train 2764`, `val 604`, `test 624`
- split instances: `train 12360`, `val 2786`, `test 2841`
- group counts: `train 663`, `val 144`, `test 146`
- empty-label images: `83`
- invalid label issues after self-healing: `0`

Semantic mapping label yang harus dipakai tetap mengikuti [outputs/phase1/locked_setup.yaml](../phase1/locked_setup.yaml), [E0.md](../../E0.md), [GUIDE.md](../../GUIDE.md), dan [CONTEXT.md](../../CONTEXT.md):

- `B1`: paling matang / ripe
- `B2`: transisi menuju matang
- `B3`: lebih mentah dari `B2`
- `B4`: paling belum matang

Urutannya adalah **`B1 -> B2 -> B3 -> B4` = paling matang ke paling belum matang**.

## 4. Kondisi data untuk final retrain / final evaluation

Final retrain Phase 3 tidak memakai `train` bawaan mentah begitu saja, tetapi memakai file [outputs/phase3/final_data.yaml](../phase3/final_data.yaml). File ini penting untuk reproduksi final run karena berisi:

- `path: /workspace/Dataset-Sawit-YOLO`
- `train: /workspace/brand-new-yolo/outputs/phase3/trainval.txt`
- `val: images/val`
- `test: images/test`

Artinya, final retrain Phase 3 dijalankan dengan daftar train gabungan yang dirujuk melalui [outputs/phase3/trainval.txt](../phase3/trainval.txt), lalu dievaluasi pada split `test`.

## 5. Source of truth per fase

Untuk membaca keputusan dan angka resmi, gunakan file-file berikut sebagai source of truth:

### Phase 0
- dataset audit: [outputs/phase0/dataset_audit.json](../phase0/dataset_audit.json)
- ringkasan keputusan: [outputs/phase0/phase0_summary.md](../phase0/phase0_summary.md)
- learning curve mentah: [outputs/phase0/learning_curve.csv](../phase0/learning_curve.csv)

### Phase 1
- ringkasan pipeline decision: [outputs/phase1/phase1_summary.md](../phase1/phase1_summary.md)
- one-stage baseline: [outputs/phase1/one_stage_results.csv](../phase1/one_stage_results.csv)
- two-stage component results: [outputs/phase1/two_stage_results.csv](../phase1/two_stage_results.csv)
- benchmark arsitektur: [outputs/phase1/architecture_benchmark.csv](../phase1/architecture_benchmark.csv)
- top-3 arsitektur: [outputs/phase1/phase1b_top3.csv](../phase1/phase1b_top3.csv)
- lock setup final: [outputs/phase1/locked_setup.yaml](../phase1/locked_setup.yaml)

### Phase 2
- ringkasan tuning: [outputs/phase2/phase2_summary.md](../phase2/phase2_summary.md)
- hasil agregasi tuning: [outputs/phase2/tuning_results.csv](../phase2/tuning_results.csv)
- hyperparameter final: [outputs/phase2/final_hparams.yaml](../phase2/final_hparams.yaml)
- confirm run: [outputs/phase2/p2confirm_yolo11m_640_s3_e30p10m30_eval.json](../phase2/p2confirm_yolo11m_640_s3_e30p10m30_eval.json)

### Phase 3
- final report: [outputs/phase3/final_report.md](../phase3/final_report.md)
- final evaluation: [outputs/phase3/final_evaluation.md](../phase3/final_evaluation.md)
- final metrics resmi per run: [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](../phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json)
- metadata run final: [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json](../phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json)
- threshold operasi: [outputs/phase3/threshold_sweep.csv](../phase3/threshold_sweep.csv)
- analisis error: [outputs/phase3/error_analysis.md](../phase3/error_analysis.md)
- stratifikasi error: [outputs/phase3/error_stratification.csv](../phase3/error_stratification.csv)
- deploy status: [outputs/phase3/deploy_check.md](../phase3/deploy_check.md)
- final best weight: [runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt](../../runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt)

## 6. Final setup yang berhasil diamankan

Final setup yang dikunci dan harus dianggap resmi adalah:

- model final: `yolo11m.pt`
- pipeline final: `one-stage`
- imgsz: `640`
- lr0: `0.001`
- batch: `16`
- imbalance strategy: `none`
- ordinal strategy: `standard`
- aug profile: `medium`
- optimizer: `AdamW`
- final Phase 2 confirm seed: `3`
- final Phase 3 seed: `42`
- final Phase 3 epochs: `60`
- final Phase 3 patience: `15`
- final Phase 3 min_epochs: `60`

Konfigurasi ini dapat diverifikasi langsung di [outputs/phase1/locked_setup.yaml](../phase1/locked_setup.yaml), [outputs/phase2/final_hparams.yaml](../phase2/final_hparams.yaml), dan [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json](../phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json).

## 7. Perintah yang relevan untuk reproduksi

### Validasi dataset
Gunakan [scripts/validate_dataset.py](../../scripts/validate_dataset.py):

```bash
python scripts/validate_dataset.py --data Dataset-YOLO/data.yaml --outdir outputs/phase0
```

### Menjalankan satu eksperimen YOLO manual
Gunakan [scripts/run_yolo_experiment.py](../../scripts/run_yolo_experiment.py). Antarmuka argumennya bisa dilihat lewat:

```bash
python scripts/run_yolo_experiment.py --help
```

### Menjalankan orkestrator E0
Gunakan [scripts/e0_master_autonomous.py](../../scripts/e0_master_autonomous.py) dari root repo:

```bash
python scripts/e0_master_autonomous.py
```

### Bentuk perintah final retrain yang dipakai pada sesi ini
Final Phase 3 dijalankan dengan recipe berikut:

```bash
python scripts/run_yolo_experiment.py \
  --phase phase3 \
  --task detect \
  --model yolo11m.pt \
  --data /workspace/brand-new-yolo/outputs/phase3/final_data.yaml \
  --imgsz 640 \
  --epochs 60 \
  --batch 16 \
  --seed 42 \
  --project runs/e0 \
  --name p3_final_yolo11m_640_s42_e60p15m60 \
  --split test \
  --device 0 \
  --workers 8 \
  --patience 15 \
  --min-epochs 60 \
  --imbalance-strategy none \
  --ordinal-strategy standard \
  --focal-gamma 1.5 \
  --pretrained \
  --optimizer AdamW \
  --lr0 0.001 \
  --hsv-h 0.015 \
  --hsv-s 0.7 \
  --hsv-v 0.4 \
  --translate 0.1 \
  --scale 0.5 \
  --mosaic 1.0 \
  --close-mosaic 10
```

## 8. Audit keamanan sebelum terminate RunPod

Checklist audit pada saat dokumen ini disiapkan:

- [x] `HEAD` lokal sudah sama dengan `origin/main`
- [x] tidak ada file `untracked` yang tertinggal (`untracked = 0`)
- [x] file penting final sudah ada di `HEAD` Git, termasuk [README.md](../../README.md), [outputs/phase3/final_report.md](../phase3/final_report.md), [outputs/phase3/final_evaluation.md](../phase3/final_evaluation.md), [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](../phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json), dan [runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt](../../runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt)
- [x] final best weight dan last weight sudah tercatat di Git `HEAD` dengan ukuran blob `40510437` byte masing-masing
- [x] tidak ada proses aktif `scripts/run_yolo_experiment.py`, `scripts/e0_master_autonomous.py`, atau `scripts/e0_root_readme_finalizer.py` saat audit termination dilakukan
- [x] source of truth lintas fase sudah tertulis dan saling merujuk langsung

Catatan penting tentang working tree lokal:

- working tree lokal masih menunjukkan banyak file `D` pada folder `runs/` karena sebelumnya dilakukan **safe local cleanup** untuk menghemat disk RunPod
- deletion lokal itu **tidak sedang dipush sebagai deletion ke remote**
- artefak yang penting sudah diverifikasi ada di `HEAD` / `origin/main`
- artinya, ketika RunPod diterminasi, yang hilang hanya salinan lokal yang memang sudah dibersihkan atau cache lingkungan lokal; **remote GitHub tetap menyimpan artefak eksperimen yang sudah di-commit**

## 9. Status aman atau belum?

Jika setelah dokumen ini dibuat:

- `HEAD == origin/main`
- tidak ada proses training/orchestrator yang masih aktif
- tidak ada file `untracked` penting
- dan file penting final di atas sudah ada di GitHub

maka statusnya adalah: **aman untuk terminate RunPod**.

## 10. Yang harus dibuka lagi setelah clone ulang

Jika nanti ingin melanjutkan dari mesin lain setelah RunPod ini dimatikan, urutan baca yang disarankan adalah:

1. [README.md](../../README.md)
2. [outputs/reports/reproducibility_and_termination.md](reproducibility_and_termination.md)
3. [outputs/phase3/final_report.md](../phase3/final_report.md)
4. [outputs/phase3/final_evaluation.md](../phase3/final_evaluation.md)
5. [outputs/phase1/locked_setup.yaml](../phase1/locked_setup.yaml)
6. [outputs/phase2/final_hparams.yaml](../phase2/final_hparams.yaml)
7. [runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt](../../runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt)
