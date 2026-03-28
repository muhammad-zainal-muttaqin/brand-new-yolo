# Reproducibility & RunPod Termination Checklist

Dokumen ini adalah checklist terakhir sebelum environment RunPod dimatikan. Dua tujuan utamanya: (1) memastikan semua artefak penting sudah aman di GitHub, dan (2) mendokumentasikan informasi yang diperlukan agar eksperimen ini bisa direproduksi di environment lain setelah RunPod asli tidak lagi tersedia.

Peta baca repo di [README.md](../../README.md). Keputusan lintas fase di [final_report.md](../phase3/final_report.md). Evaluasi teknis run final di [final_evaluation.md](../phase3/final_evaluation.md).

## 1. Snapshot repositori yang dipakai untuk reproduksi

- Remote GitHub: `https://github.com/muhammad-zainal-muttaqin/brand-new-yolo.git`
- Branch: `main`
- Final archive tag: `runpod-archive-e0-final-20260327`
- Audit timestamp UTC: `2026-03-27T13:55:26Z`
- Working directory saat eksperimen berjalan: `/workspace/brand-new-yolo`
- Dataset root saat eksperimen berjalan: `/workspace/Dataset-Sawit-YOLO`

Untuk mereproduksi snapshot ini secara eksak, **clone repo lalu checkout tag `runpod-archive-e0-final-20260327`**. Branch `main` bisa saja sudah bergerak setelah titik ini — gunakan tag ini kalau tujuannya adalah mereplikasi hasil final persis seperti yang terdokumentasi.

## 2. Snapshot environment

Environment yang dipakai pada sesi final:

- OS/platform: `Linux-6.8.0-52-generic-x86_64-with-glibc2.35`
- Python: `3.11.15`
- PyTorch: `2.4.1+cu124`
- Ultralytics: `8.4.30`
- CUDA available: `True`
- GPU count: `1`
- GPU: `NVIDIA A40`
- GPU memory: `46068 MiB`
- NVIDIA driver: `570.195.03`

Pemeriksaan dependency ringan bisa dijalankan dari [scripts/setup_env.sh](../../scripts/setup_env.sh).

## 3. Snapshot dataset yang dipakai

Dataset aktif repo ini dirujuk lewat [Dataset-YOLO/data.yaml](../../Dataset-YOLO/data.yaml). Isi utamanya menunjuk ke:

- `path: /workspace/Dataset-Sawit-YOLO`
- `train: images/train`
- `val: images/val`
- `test: images/test`
- `nc: 4`
- kelas: `B1`, `B2`, `B3`, `B4`

Audit dataset mentah bisa dicek di [outputs/phase0/dataset_audit.json](../phase0/dataset_audit.json). Snapshot pentingnya:

- total images: `3992`
- total labels: `3992`
- total instances: `17987`
- split images: `train 2764`, `val 604`, `test 624`
- split instances: `train 12360`, `val 2786`, `test 2841`
- group counts: `train 663`, `val 144`, `test 146`
- empty-label images: `83`
- invalid label issues after self-healing: `0`

Semantic mapping label yang dipakai tetap mengikuti:

- [E0.md](../../E0.md)
- [GUIDE.md](../../GUIDE.md)
- [CONTEXT.md](../../CONTEXT.md)
- [outputs/phase1/locked_setup.yaml](../phase1/locked_setup.yaml)

Urutannya adalah **`B1 -> B2 -> B3 -> B4` = paling matang ke paling belum matang**.

## 4. Data untuk final retrain dan final evaluation

Final retrain Phase 3 tidak memakai `train` mentah apa adanya. Repo ini memakai [outputs/phase3/final_data.yaml](../phase3/final_data.yaml), yang menunjuk ke:

- `path: /workspace/Dataset-Sawit-YOLO`
- `train: /workspace/brand-new-yolo/outputs/phase3/trainval.txt`
- `val: images/val`
- `test: images/test`

Artinya, final retrain berjalan dengan daftar `train+val` yang dirujuk melalui [outputs/phase3/trainval.txt](../phase3/trainval.txt), lalu dievaluasi pada split `test`.

## 5. Source of truth per fase

### Phase 0
- [outputs/phase0/dataset_audit.json](../phase0/dataset_audit.json)
- [outputs/phase0/eda_report.md](../phase0/eda_report.md)
- [outputs/phase0/phase0_summary.md](../phase0/phase0_summary.md)
- [outputs/phase0/learning_curve.csv](../phase0/learning_curve.csv)

### Phase 1
- [outputs/phase1/phase1_summary.md](../phase1/phase1_summary.md)
- [outputs/phase1/one_stage_results.csv](../phase1/one_stage_results.csv)
- [outputs/phase1/two_stage_results.csv](../phase1/two_stage_results.csv)
- [outputs/phase1/architecture_benchmark.csv](../phase1/architecture_benchmark.csv)
- [outputs/phase1/phase1b_top3.csv](../phase1/phase1b_top3.csv)
- [outputs/phase1/locked_setup.yaml](../phase1/locked_setup.yaml)

### Phase 2
- [outputs/phase2/phase2_summary.md](../phase2/phase2_summary.md)
- [outputs/phase2/tuning_results.csv](../phase2/tuning_results.csv)
- [outputs/phase2/final_hparams.yaml](../phase2/final_hparams.yaml)
- [outputs/phase2/p2confirm_yolo11m_640_s3_e30p10m30_eval.json](../phase2/p2confirm_yolo11m_640_s3_e30p10m30_eval.json)

### Phase 3
- [outputs/phase3/final_report.md](../phase3/final_report.md)
- [outputs/phase3/final_evaluation.md](../phase3/final_evaluation.md)
- [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](../phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json)
- [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json](../phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json)
- [outputs/phase3/threshold_sweep.csv](../phase3/threshold_sweep.csv)
- [outputs/phase3/error_analysis.md](../phase3/error_analysis.md)
- [outputs/phase3/error_stratification.csv](../phase3/error_stratification.csv)
- [outputs/phase3/deploy_check.md](../phase3/deploy_check.md)
- [runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt](../../runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt)

## 6. Final setup yang harus dianggap resmi

Konfigurasi final yang dikunci dan berhasil dibawa sampai akhir adalah:

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

Verifikasi langsung ada di:

- [outputs/phase1/locked_setup.yaml](../phase1/locked_setup.yaml)
- [outputs/phase2/final_hparams.yaml](../phase2/final_hparams.yaml)
- [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json](../phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json)

## 7. Perintah yang relevan untuk reproduksi

### Validasi dataset

Gunakan [scripts/validate_dataset.py](../../scripts/validate_dataset.py):

```bash
python scripts/validate_dataset.py --data Dataset-YOLO/data.yaml --outdir outputs/phase0
```

### Menjalankan satu eksperimen YOLO manual

Gunakan [scripts/run_yolo_experiment.py](../../scripts/run_yolo_experiment.py):

```bash
python scripts/run_yolo_experiment.py --help
```

### Menjalankan orkestrator E0

Gunakan [scripts/e0_master_autonomous.py](../../scripts/e0_master_autonomous.py):

```bash
python scripts/e0_master_autonomous.py
```

### Bentuk command final retrain yang dipakai

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

Checklist audit saat dokumen ini dibuat:

- [x] `HEAD` lokal sama dengan `origin/main`
- [x] tidak ada file `untracked` penting yang tertinggal
- [x] file penting final sudah ada di Git `HEAD`
- [x] `best.pt` dan `last.pt` final sudah tercatat di Git `HEAD`
- [x] tidak ada proses aktif `scripts/run_yolo_experiment.py`, `scripts/e0_master_autonomous.py`, atau `scripts/e0_root_readme_finalizer.py`
- [x] source of truth lintas fase sudah tertulis dan saling merujuk

Catatan penting tentang working tree lokal:

- working tree lokal memang masih menunjukkan banyak file `D` pada folder `runs/`
- itu terjadi karena sebelumnya dilakukan **safe local cleanup** untuk menghemat disk RunPod
- deletion lokal itu **tidak** sedang dipush sebagai deletion ke remote
- artefak penting sudah diverifikasi aman di `HEAD` / `origin/main`

## 9. Status aman atau belum?

Jika kondisi berikut terpenuhi:

- `HEAD == origin/main`
- tidak ada proses training aktif
- tidak ada file penting yang belum terlacak
- artefak final sudah ada di GitHub

maka statusnya: **aman untuk mematikan RunPod**.

## 10. Urutan baca setelah clone ulang

Jika nanti melanjutkan dari mesin lain, urutan baca yang paling aman adalah:

1. [README.md](../../README.md)
2. [outputs/reports/reproducibility_and_termination.md](reproducibility_and_termination.md)
3. [outputs/phase3/final_report.md](../phase3/final_report.md)
4. [outputs/phase3/final_evaluation.md](../phase3/final_evaluation.md)
5. [outputs/phase1/locked_setup.yaml](../phase1/locked_setup.yaml)
6. [outputs/phase2/final_hparams.yaml](../phase2/final_hparams.yaml)
7. [runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt](../../runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt)
