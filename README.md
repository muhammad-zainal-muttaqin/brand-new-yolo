# Brand New YOLO — Deteksi Kematangan Buah Sawit

Repo ini mendokumentasikan eksperimen E0: membangun baseline detector kematangan buah kelapa sawit menggunakan YOLO. Task-nya adalah mendeteksi dan mengklasifikasikan buah sawit ke dalam 4 tingkat kematangan (B1–B4) pada satu tandan, dalam satu shot — tanpa pipeline multi-stage.

Eksperimen berjalan dalam 4 fase yang sequential: validasi dataset → pemilihan pipeline & arsitektur → hyperparameter tuning → retrain final dan evaluasi. Setiap fase menghasilkan keputusan yang di-lock dan dibawa ke fase berikutnya, sehingga pipeline secara keseluruhan reproducible dan traceable.

Hasil akhir: model `yolo11m.pt` pada resolusi 640 menghasilkan **mAP50 = 0.4677** pada test set. B1 (matang) terdeteksi dengan baik (mAP50 = 0.78), tapi B4 (belum matang) masih sangat sulit (mAP50 = 0.27). Baseline ini solid dan terdokumentasi, tapi belum production-ready.

---

## Definisi kelas

Repo ini membedakan 4 tingkat kematangan buah sawit berdasarkan visual appearance dan posisi di tandan:

- **B1** — buah **merah**, **besar**, **bulat**, posisi **paling bawah** pada tandan → **paling matang**
- **B2** — buah masih **hitam**, mulai **transisi ke merah**, sudah **besar** dan **bulat**, posisi **di atas B1**
- **B3** — buah **full hitam**, masih **berduri**, masih **lonjong**, posisi **di atas B2**
- **B4** — buah **paling kecil**, **paling dalam di tandan**, sulit terlihat, masih banyak **duri**, warna **hitam sampai hijau** → **paling belum matang**

Urutan biologis: **B1 → B2 → B3 → B4 = paling matang ke paling mentah**.

Mapping ini konsisten di seluruh repo: [E0.md](E0.md), [GUIDE.md](GUIDE.md), [CONTEXT.md](CONTEXT.md), [locked_setup.yaml](outputs/phase1/locked_setup.yaml).

---

## Keputusan akhir & angka resmi

Sebelum masuk ke narasi per fase, berikut keputusan dan angka final yang di-lock:

| Komponen | Keputusan |
|---|---|
| Pipeline | `one-stage` |
| Model | `yolo11m.pt` |
| Resolusi | `640` |
| Recipe | `lr0=0.001`, `batch=16`, `imbalance=none`, `ordinal=standard`, `aug=medium` |
| Run final | `p3_final_yolo11m_640_s42_e60p15m60` |
| Weight | [best.pt](runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt) |

**Metrik resmi test set** (sumber: [eval.json](outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json)):

| Metrik | Nilai |
|---|---:|
| Precision | 0.4763 |
| Recall | 0.5538 |
| mAP50 | 0.4677 |
| mAP50-95 | 0.2215 |
| All classes AP50 ≥ 0.70 | **False** |

---

## Perjalanan eksperimen: dari data mentah ke baseline final

### Progres keseluruhan

Empat chart di bawah merangkum evolusi metrik di seluruh run yang dijalankan, dari Phase 0 sampai Phase 3. Setiap titik adalah satu training run; garis step menunjukkan running best.

![mAP50 across all runs](outputs/figures/e0_research_progress_map50.png)

![mAP50-95 across all runs](outputs/figures/e0_research_progress_map50_95.png)

![Precision across all runs](outputs/figures/e0_research_progress_precision.png)

![Recall across all runs](outputs/figures/e0_research_progress_recall.png)

Dari chart di atas terlihat bahwa setelah Phase 1B (benchmark arsitektur), peningkatan metrik mulai plateau. Phase 2 (tuning) dan Phase 3 (retrain final) tidak menghasilkan lompatan — mereka mengonfirmasi bahwa ceiling performa sudah tercapai pada konfigurasi ini.

Sumber data: [run_ledger.csv](outputs/reports/run_ledger.csv). Komponen interaktif: [yolo_e0_research_progress.jsx](yolo_e0_research_progress.jsx).

---

### Phase 0 — Dataset & resolusi kerja

> Detail lengkap: [eda_report.md](outputs/phase0/eda_report.md) · [phase0_summary.md](outputs/phase0/phase0_summary.md)

Phase 0 memvalidasi dataset dan memilih resolusi kerja. Dataset terdiri dari **3,992 image** dengan **17,987 instance** buah sawit, dibagi ke train (2,764), val (604), dan test (624). Audit otomatis tidak menemukan blocker: tidak ada label yang invalid, tidak ada leakage antar split.

#### Distribusi kelas

![Distribusi kelas](outputs/phase0/figures/eda_class_distribution.png)

Distribusi sangat tidak merata — B3 mendominasi dengan 46% dari seluruh instance, sementara B1 hanya 12%. Ini bukan kebetulan: secara biologis, buah di tahap B3 memang paling banyak ditemukan di tandan. Implikasinya, model akan terekspos B3 hampir 4× lebih sering dari B1 selama training, yang bisa menyebabkan bias prediksi ke arah majority class.

#### Ukuran bounding box

![Ukuran bbox per kelas](outputs/phase0/figures/eda_bbox_size_comparison.png)

Ada gradasi ukuran yang konsisten dari B1 ke B4 — mencerminkan tahap biologis. B4 (median area 0.0072 normalized) hampir 2× lebih kecil dari B1 (0.0140). Kombinasi ukuran kecil + posisi tersembunyi membuat B4 menjadi kelas tersulit di sepanjang eksperimen.

Sumber: [class_distribution.csv](outputs/phase0/class_distribution.csv), [bbox_stats.csv](outputs/phase0/bbox_stats.csv), [dataset_audit.json](outputs/phase0/dataset_audit.json)

#### Resolution sweep: 640 vs 1024

![Perbandingan resolusi](outputs/phase0/figures/p0_resolution_comparison.png)

Resolusi 1024 memang sedikit lebih baik (mAP50-95: 0.258 vs 0.253), tapi gain-nya hanya **+2.15%** — sementara compute cost-nya hampir 2.5× lebih berat. Dalam pipeline E0 dengan puluhan run, 640 adalah pilihan yang paling realistis.

**Keputusan: resolusi kerja = 640.**

#### Learning curve

![Learning curve](outputs/phase0/figures/p0_learning_curve.png)

Kurva belajar menunjukkan gain yang konsisten dari 25% ke 75% data (~+0.02 mAP50-95 per step), tapi melambat drastis dari 75% ke 100% (+0.009). Dataset belum benar-benar saturasi, tapi diminishing returns sudah mulai terlihat — menambah data secara random tanpa memperhatikan distribusi kelas kemungkinan hanya memberi gain kecil.

Sumber: [resolution_sweep.csv](outputs/phase0/resolution_sweep.csv), [learning_curve.csv](outputs/phase0/learning_curve.csv)

---

### Phase 1A — One-stage vs two-stage

> Detail lengkap: [phase1_summary.md](outputs/phase1/phase1_summary.md)

Phase 1A membandingkan dua pipeline: one-stage (satu YOLO detector langsung 4-class) vs two-stage (detector single-class + classifier pada crops).

![One-stage vs two-stage](outputs/phase1/figures/p1_one_vs_two_stage.png)

Stage-2 classifier pada **ground-truth crops** (kondisi ideal, tanpa noise dari detector) hanya mencapai top-1 accuracy **63.8%**. Yang lebih penting, confusion matrix-nya menunjukkan masalah fundamental:

| | Prediksi B2 | Prediksi B3 |
|---|---:|---:|
| GT B2 | 211 (correct) | **94 → B3** (31%) |
| GT B3 | **334 → B2** (23%) | 1,112 (correct) |

Bahkan pada kondisi ideal (crop sempurna dari GT box), classifier tidak bisa membedakan B2 dan B3 secara reliable. Menambahkan complexity pipeline two-stage tidak menyelesaikan masalah fundamental ini — one-stage detector yang langsung mengoptimasi 4-class detection setidaknya bisa memanfaatkan konteks spasial (posisi relatif dalam tandan) yang hilang saat objek di-crop.

**Keputusan: pipeline one-stage.**

Sumber: [one_stage_results.csv](outputs/phase1/one_stage_results.csv), [two_stage_results.csv](outputs/phase1/two_stage_results.csv)

---

### Phase 1B — Benchmark arsitektur

> Detail lengkap: [phase1_summary.md](outputs/phase1/phase1_summary.md)

11 arsitektur YOLO di-benchmark dalam kondisi identik (resolusi 640, lr0=0.001, batch=16, medium aug, 2 seed per model):

![Architecture benchmark](outputs/phase1/figures/p1_architecture_benchmark.png)

**Top-3** (sumber: [phase1b_top3.csv](outputs/phase1/phase1b_top3.csv)):

| Rank | Model | Mean mAP50 | Mean mAP50-95 |
|---:|---|---:|---:|
| 1 | `yolo11m.pt` | 0.5298 | 0.2570 |
| 2 | `yolov9c.pt` | 0.5292 | 0.2518 |
| 3 | `yolov8s.pt` | 0.5256 | 0.2521 |

Gap antar model teratas sangat kecil (selisih #1 dan #2 hanya 0.0006) — ini menandakan bahwa **bottleneck bukan di arsitektur model, tapi di task difficulty dan data quality**. Ganti family model tidak menghasilkan lompatan; ceiling performa ditentukan oleh sifat task itu sendiri.

Tidak ada model yang melewati gate canonical mAP50 ≥ 0.70. Repo ini menggunakan override operasional agar pipeline end-to-end tetap berjalan sampai Phase 3 — menghasilkan satu baseline lengkap yang bisa dijadikan referensi.

**Keputusan: model = `yolo11m.pt`.** Lock file: [locked_setup.yaml](outputs/phase1/locked_setup.yaml).

Yang menarik: kalau kita lihat performa per kelas di **semua** arsitektur, pola yang sama muncul di mana-mana — B1 selalu tinggi, B4 selalu rendah, terlepas dari model yang dipakai:

![Per-class heatmap across architectures](outputs/phase1/figures/p1_per_class_heatmap.png)

Heatmap ini mengonfirmasi bahwa difficulty ranking antar kelas (B1 > B3 > B2 > B4) bukan artefak dari satu model tertentu — ini adalah sifat inherent dari task dan dataset.

Sumber: [architecture_benchmark.csv](outputs/phase1/architecture_benchmark.csv), [per_class_metrics.csv](outputs/phase1/per_class_metrics.csv)

---

### Phase 2 — Hyperparameter tuning (dan kenapa hasilnya revert)

> Detail lengkap: [phase2_summary.md](outputs/phase2/phase2_summary.md)

Phase 2 menguji apakah tuning hyperparameter bisa mendorong performa melewati ceiling Phase 1. Tuning dilakukan sequential: loss function → learning rate → batch size → augmentation.

#### Loss function: bukan bottleneck

Temuan paling informatif datang dari step pertama. Tiga strategi loss (`none`, `class_weighted`, `focal`) menghasilkan **metrik yang identik** — bukan "mirip", tapi persis sama sampai 4 desimal:

| Strategi | Mean mAP50 | Mean mAP50-95 | Mean B4 Recall |
|---|---:|---:|---:|
| `none` | 0.5298 | 0.2570 | 0.3673 |
| `class_weighted` | 0.5298 | 0.2570 | 0.3673 |
| `focal15` | 0.5298 | 0.2570 | 0.3673 |

![Imbalance/loss sweep — semua identik](outputs/phase2/figures/p2_imbalance_sweep.png)

Secara visual langsung terlihat: bar-bar untuk ketiga strategi persis sama. Ini kuat mengindikasikan bahwa model sudah mengekstrak sinyal seefisien yang bisa dari data yang ada — mengubah objective function tidak mengubah apa yang dipelajari.

#### LR, batch, dan augmentation sweep

![Learning rate sweep](outputs/phase2/figures/p2_lr_sweep.png)

![Batch & augmentation sweep](outputs/phase2/figures/p2_batch_aug_sweep.png)

LR sweep menunjukkan `lr0=0.0005` sedikit lebih baik (+0.52% mAP50), tapi gain-nya marginal dan variance antar seed masih overlap. `lr0=0.002` malah menjatuhkan B4 recall ke 0.334. Batch dan augmentation sweep juga tidak menghasilkan breakthrough — setiap gain di satu metrik diikuti penurunan di metrik lain.

#### Keputusan revert

![Tuning progression](outputs/phase2/figures/p2_tuning_summary.png)

Kandidat terbaik (0.5350) vs baseline (0.5298) — selisih hanya **0.52%**, terlalu kecil untuk membenarkan perubahan recipe. **Keputusan: revert ke baseline Phase 1B.**

Pesan terbesar: **bottleneck ada di task difficulty dan data quality, bukan di hyperparameter.** Sweep di ruang standar sudah saturated.

Konfigurasi final: [final_hparams.yaml](outputs/phase2/final_hparams.yaml).

Sumber: [lr_sweep.csv](outputs/phase2/lr_sweep.csv), [batch_sweep.csv](outputs/phase2/batch_sweep.csv), [aug_sweep.csv](outputs/phase2/aug_sweep.csv), [tuning_results.csv](outputs/phase2/tuning_results.csv)

---

### Phase 3 — Retrain final & evaluasi test set

> Detail lengkap: [final_evaluation.md](outputs/phase3/final_evaluation.md) · [error_analysis.md](outputs/phase3/error_analysis.md)

Phase 3 melakukan retrain final dengan budget lebih besar (60 epoch, patience 15, seed 42) pada konfigurasi yang sudah di-lock, lalu evaluasi pada test set yang tidak pernah disentuh selama training maupun tuning.

#### Training curves

![Training curves final run](outputs/phase3/figures/p3_training_curves.png)

Training curves menunjukkan model converge dengan baik: training loss menurun monoton, validation loss stabil setelah ~epoch 20, dan mAP50 mencapai puncaknya di pertengahan training sebelum plateau. Model tidak menunjukkan tanda overfitting yang parah — validation loss tidak naik kembali, menandakan bahwa patience 15 dan epoch budget 60 sudah tepat.

#### Performa per kelas

![Metrik per kelas](outputs/phase3/figures/p3_per_class_metrics.png)

| Kelas | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| B1 | 0.7237 | 0.7262 | 0.7821 | 0.4246 |
| B2 | 0.3932 | 0.4182 | 0.3266 | 0.1481 |
| B3 | 0.4603 | 0.6910 | 0.4880 | 0.2138 |
| B4 | 0.3280 | 0.3798 | 0.2742 | 0.0993 |

Polanya konsisten dengan prediksi sejak Phase 0:

- **B1** mendominasi — warna merah terang dan ukuran besar membuat visual signature-nya paling distinct
- **B3** punya recall tinggi (0.69) tapi precision rendah (0.46) — model sering salah melabel objek lain sebagai B3, terutama B2
- **B2** paling ambigu — gradasi warna antara B2 dan B3 sangat halus, membuat discriminative learning sulit
- **B4** konsisten tersulit — ukuran kecil, posisi tersembunyi, dan jumlah instance lebih sedikit

Secara praktis, mAP50 overall 0.47 artinya model lebih bisa diandalkan untuk mengonfirmasi kematangan tinggi (B1) daripada mendeteksi buah muda (B4) secara komprehensif.

#### Threshold operasi

![Threshold sweep](outputs/phase3/figures/p3_threshold_sweep_detail.png)

| conf | precision | recall | mAP50 | mAP50-95 | B4 recall |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.7032 | 0.6995 | 0.7395 | 0.4499 | 0.5415 |
| 0.2 | 0.7032 | 0.6995 | 0.7218 | 0.4475 | 0.5415 |
| 0.3 | 0.7152 | 0.6821 | 0.7086 | 0.4484 | 0.5181 |
| 0.4 | 0.7853 | 0.5860 | 0.6862 | 0.4429 | 0.4152 |
| 0.5 | 0.8307 | 0.4717 | 0.6515 | 0.4311 | 0.2798 |

Di `conf=0.1`, precision dan recall seimbang (~0.70) dan mAP50 melompat ke 0.74. Yang paling menarik: B4 recall naik ke 0.54 (dari 0.38 default) — model sebenarnya "melihat" B4 tapi dengan confidence rendah. Operating point terbaik: **`conf=0.1`**.

Catatan: angka threshold sweep menentukan operating point deployment, bukan mengganti skor resmi di eval JSON.

Sumber: [threshold_sweep.csv](outputs/phase3/threshold_sweep.csv)

#### Evolusi performa lintas fase

![Cross-phase comparison](outputs/phase3/figures/p3_cross_phase_comparison.png)

Chart ini merangkum perjalanan per-class mAP50 dari Phase 1B baseline (val set, 30 epoch) → Phase 2 confirmation (val set, seed 3) → Phase 3 final (test set, 60 epoch). Beberapa hal yang terlihat:

- **B1 konsisten di atas target 0.70** di semua fase — satu-satunya kelas yang lolos
- **B2, B3, B4 tidak pernah mendekati target** di fase manapun
- Drop dari Phase 2 (val) ke Phase 3 (test) pada B2 dan B4 menandakan bahwa val set sedikit lebih "mudah" dari test set untuk kelas-kelas ini
- Gap terbesar ada di B4: dari ~0.38 di Phase 1B ke 0.27 di Phase 3 test — menunjukkan bahwa B4 bahkan lebih sulit di test set

#### Error dominan

![Distribusi error](outputs/phase3/figures/p3_error_distribution.png)

Dari 20 image tersulit:

| Kategori | Jumlah Image |
|---|---:|
| `false_positive` | 20 |
| `B2_B3_confusion` | 13 |
| `B4_missed` | 11 |
| `B3_B4_confusion` | 10 |

![Top-20 image tersulit](outputs/phase3/figures/p3_error_by_image_score.png)

Polanya koheren: error terbesar terjadi pada **kelas berdekatan secara ordinal** (B2↔B3, B3↔B4) dan pada **scene dense** (banyak buah dalam satu frame → banyak false positive). Image dari tandan yang sama cenderung muncul berulang di daftar tersulit — beberapa tandan memang secara inheren lebih sulit.

Sumber: [error_stratification.csv](outputs/phase3/error_stratification.csv), detail di [error_analysis.md](outputs/phase3/error_analysis.md)

#### Status deploy

**Deferred.** Weight `best.pt` aman, tapi konversi ke format deployment (TFLite, ONNX, INT8) ditunda — menjalankan deploy pipeline di atas model yang belum memenuhi target AP50 ≥ 0.70 di semua kelas akan menghasilkan artefak yang perlu di-redo. Detail: [deploy_check.md](outputs/phase3/deploy_check.md).

---

## Kesimpulan

Eksperimen E0 menghasilkan **satu baseline final yang konsisten, terlacak, dan reproducible** untuk task deteksi kematangan buah sawit. Setiap keputusan terdokumentasi, setiap angka bisa ditelusuri ke artefak aslinya.

Dari sisi performa, model ini solid sebagai baseline tapi bukan production-ready. Temuan lintas fase yang paling penting:

1. **Dataset cukup bersih untuk baseline**, tapi distribusi kelas tidak merata (B3 mendominasi 46%)
2. **Resolusi 640 sudah cukup** — gain dari 1024 hanya 2.15%, tidak sebanding compute cost
3. **One-stage lebih realistis** — two-stage gagal menyelesaikan confusion B2/B3 bahkan di GT crops
4. **Bottleneck bukan di arsitektur** — gap antar 11 model YOLO sangat kecil
5. **Bottleneck bukan di hyperparameter** — loss function sweep identik, LR/batch/aug sweep marginal
6. **Bottleneck ada di task difficulty dan data quality** — confusion antar kelas berdekatan (B2↔B3, B3↔B4) dan kesulitan mendeteksi B4 yang kecil

### Arah perbaikan

Karena hyperparameter sweep sudah saturated, peningkatan berikutnya harus datang dari perubahan yang lebih fundamental:

1. **Mengurangi confusion B2/B3** — eksplorasi fine-grained feature learning (attention pada region warna), augmentasi domain-specific yang memanipulasi gradasi warna zona transisi
2. **Meningkatkan recall B4** — threshold sweep menunjukkan model "melihat" B4 di confidence rendah; bisa didekati via resolusi lebih tinggi atau penambahan data B4 berkualitas
3. **Menekan false positive** — NMS tuning dan confidence calibration bisa membantu tanpa retrain
4. Baru setelah ketiga perbaikan menunjukkan progress, lanjutkan ke deploy pipeline

---

## Referensi teknis

### Hierarki acuan

Kalau ada konflik antar dokumen, pakai urutan prioritas ini:

1. **Artefak run final** (source of truth):
   - [p3_final_yolo11m_640_s42_e60p15m60_eval.json](outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json)
   - [p3_final_yolo11m_640_s42_e60p15m60_summary.json](outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json)
2. **Lock files**:
   - [locked_setup.yaml](outputs/phase1/locked_setup.yaml)
   - [final_hparams.yaml](outputs/phase2/final_hparams.yaml)
3. **Ringkasan per fase**:
   - [phase0_summary.md](outputs/phase0/phase0_summary.md) · [eda_report.md](outputs/phase0/eda_report.md)
   - [phase1_summary.md](outputs/phase1/phase1_summary.md)
   - [phase2_summary.md](outputs/phase2/phase2_summary.md)
   - [final_report.md](outputs/phase3/final_report.md) · [final_evaluation.md](outputs/phase3/final_evaluation.md)
4. **Dokumen konteks**: [E0.md](E0.md) · [GUIDE.md](GUIDE.md) · [CONTEXT.md](CONTEXT.md) · [CONTEXT_Less.md](CONTEXT_Less.md)

### Audit trail

- Ledger seluruh run: [run_ledger.csv](outputs/reports/run_ledger.csv)
- Status eksekusi: [latest_status.md](outputs/reports/latest_status.md)
- Git sync log: [git_sync_log.md](outputs/reports/git_sync_log.md)
- Reproduksi & terminasi: [reproducibility_and_termination.md](outputs/reports/reproducibility_and_termination.md)
- State orchestrator: [master_state.json](outputs/reports/master_state.json)
