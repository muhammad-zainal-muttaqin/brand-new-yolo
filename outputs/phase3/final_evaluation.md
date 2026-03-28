# Evaluasi Final — Phase 3 Test Set

Dokumen ini membahas evaluasi teknis run final `p3_final_yolo11m_640_s42_e60p15m60` secara menyeluruh. Di sini kita membedah angka-angka yang dihasilkan model, melihat pola per kelas, mengeksplorasi threshold operasi, dan mengidentifikasi error dominan — semuanya dievaluasi pada split **test** yang tidak pernah disentuh selama training maupun tuning.

Untuk narasi keputusan lintas fase (kenapa model ini dipilih, kenapa hyperparameter ini di-lock), silakan ke [final_report.md](final_report.md). Peta baca keseluruhan repo ada di [README.md](../../README.md).

## Sumber data

Semua angka di dokumen ini bersumber langsung dari artefak run berikut:

- [p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json) — metrik resmi per kelas dan overall
- [p3_final_yolo11m_640_s42_e60p15m60_summary.json](p3_final_yolo11m_640_s42_e60p15m60_summary.json) — metadata training lengkap
- [final_metrics.csv](final_metrics.csv) — ringkasan metrik tabular
- [threshold_sweep.csv](threshold_sweep.csv) — hasil sweep confidence threshold
- [error_analysis.md](error_analysis.md) — analisis kualitatif error
- [error_stratification.csv](error_stratification.csv) — stratifikasi error per image
- [deploy_check.md](deploy_check.md) — status deployment

Kalau ada keraguan tentang angka mana yang "benar", prioritaskan selalu **eval JSON** dan **summary JSON** — keduanya dihasilkan langsung oleh pipeline evaluasi tanpa post-processing manual.

## 1. Identitas run yang dievaluasi

Run final ini menggunakan konfigurasi yang sudah di-lock sejak akhir Phase 2. Tidak ada perubahan hyperparameter atau arsitektur sejak lock file ditetapkan — ini murni retrain dengan epoch budget lebih besar dan seed produksi (`42`).

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

Perlu dicatat bahwa epoch budget 60 dengan patience 15 memberi ruang cukup bagi model untuk converge sepenuhnya — berbeda dengan run-run eksplorasi di Phase 1-2 yang hanya 30 epoch. Training curves di bawah menunjukkan bahwa model memang memanfaatkan budget ini dengan baik:

![Training curves final run](figures/p3_training_curves.png)

Loss menurun monoton di training set, validation loss stabil setelah ~epoch 20, dan mAP50 mencapai puncaknya di pertengahan training sebelum plateau. Model tidak menunjukkan tanda overfitting yang parah.

## 2. Metrik resmi test set

Angka-angka di bawah ini adalah skor resmi yang dihasilkan model pada test set, dievaluasi menggunakan confidence threshold default YOLO (`conf=0.001`, `iou=0.6`):

| Metrik | Nilai |
|---|---:|
| Precision | 0.4763 |
| Recall | 0.5538 |
| mAP50 | 0.4677 |
| mAP50-95 | 0.2215 |
| B4 recall | 0.3798 |
| All classes `AP50 >= 0.70` | False |

Apa yang bisa kita baca dari sini? Pertama, recall lebih tinggi dari precision — artinya model cenderung "agresif" dalam mendeteksi, menghasilkan banyak prediksi tapi dengan tingkat false positive yang cukup tinggi. Kedua, gap antara mAP50 (0.47) dan mAP50-95 (0.22) cukup lebar, yang menandakan bahwa meskipun model bisa menemukan objek di IoU rendah, kualitas lokalisasinya masih perlu perbaikan — bounding box yang dihasilkan belum terlalu presisi.

Yang paling krusial: target keras **semua kelas AP50 ≥ 0.70 belum tercapai**. Hanya B1 yang melewati batas ini. Ini mengonfirmasi bahwa model cukup untuk dijadikan baseline akhir eksperimen E0, tapi belum siap untuk deployment produksi tanpa iterasi tambahan.

## 3. Metrik per kelas

Breakdown per kelas di bawah ini mengungkap di mana sebenarnya model kuat dan di mana masih lemah:

| Kelas | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| `B1` | 0.7237 | 0.7262 | 0.7821 | 0.4246 |
| `B2` | 0.3932 | 0.4182 | 0.3266 | 0.1481 |
| `B3` | 0.4603 | 0.6910 | 0.4880 | 0.2138 |
| `B4` | 0.3280 | 0.3798 | 0.2742 | 0.0993 |

![Metrik per kelas — Final Run](figures/p3_per_class_metrics.png)

Ada pola yang sangat jelas di sini, dan pola ini konsisten dengan apa yang sudah diprediksi sejak Phase 0:

**B1 (buah matang, merah, besar)** mendominasi di semua metrik. Ini masuk akal — B1 punya visual signature yang paling distinct: warna merah terang, ukuran besar, posisi di luar tandan. Model tidak kesulitan membedakannya dari kelas lain.

**B3 (buah hitam, berduri)** punya recall tinggi (0.69) tapi precision rendah (0.46). Artinya model cukup baik menemukan B3, tapi sering salah melabel objek lain sebagai B3 — terutama B2 yang secara visual memang berada di zona transisi antara hitam dan merah.

**B2 (transisi hitam-merah)** adalah kelas yang paling "ambigu" secara visual. Precision dan recall-nya nyaris seimbang di ~0.40, tapi keduanya rendah. Ini bukan masalah ukuran objek (B2 relatif besar), melainkan masalah fitur diskriminatif — gradasi warna antara B2 dan B3 sangat halus, membuat model sulit memisahkan keduanya.

**B4 (buah terkecil, tersembunyi)** konsisten menjadi kelas tersulit. Dengan mAP50 hanya 0.27 dan mAP50-95 di bawah 0.10, model nyaris gagal pada kelas ini. Penyebabnya gabungan: ukuran fisik kecil, posisi tersembunyi di dalam tandan, dan jumlah training instance yang lebih sedikit dibanding B3.

## 4. Threshold operasi

Pada evaluasi di atas, YOLO menggunakan confidence threshold sangat rendah (`conf=0.001`) yang memaksimalkan recall tapi membanjiri output dengan low-confidence predictions. Dalam skenario deployment nyata, kita perlu memilih operating point yang lebih masuk akal.

Untuk itu, kita menjalankan threshold sweep pada rentang `conf=0.1` sampai `conf=0.5`:

| conf | precision | recall | mAP50 | mAP50-95 | b4_recall |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.7032 | 0.6995 | 0.7395 | 0.4499 | 0.5415 |
| 0.2 | 0.7032 | 0.6995 | 0.7218 | 0.4475 | 0.5415 |
| 0.3 | 0.7152 | 0.6821 | 0.7086 | 0.4484 | 0.5181 |
| 0.4 | 0.7853 | 0.5860 | 0.6862 | 0.4429 | 0.4152 |
| 0.5 | 0.8307 | 0.4717 | 0.6515 | 0.4311 | 0.2798 |

![Threshold sweep — trade-off metrik vs confidence](figures/p3_threshold_sweep_detail.png)

Threshold sweep ini menunjukkan trade-off klasik precision-recall, tapi dengan twist yang menarik. Di `conf=0.1`, precision dan recall nyaris seimbang (~0.70), dan mAP50 melompat ke 0.74 — jauh lebih tinggi dari skor resmi 0.47. Ini terjadi karena pada threshold rendah, YOLO memasukkan banyak prediksi borderline yang menurunkan precision tanpa menambah recall secara proporsional.

Yang paling menarik adalah perilaku B4 recall: di `conf=0.1` naik ke 0.54 (dari 0.38 di default), tapi di `conf=0.5` anjlok ke 0.28. Ini menandakan bahwa model sebenarnya "melihat" B4 tapi dengan confidence rendah — informasinya ada di feature map, hanya saja model belum cukup yakin.

Kandidat operating point terbaik adalah **`conf=0.1`**, yang memberikan keseimbangan terbaik antara precision, recall, dan cakupan B4. Catatan penting: angka threshold sweep ini dipakai untuk menentukan **operating point deployment**, bukan untuk menggantikan skor resmi run final di eval JSON.

## 5. Error dominan

Analisis error detail ada di [error_analysis.md](error_analysis.md). Di sini kita rangkum pola utama dari [error_stratification.csv](error_stratification.csv), yang menganalisis 20 image tersulit berdasarkan error score:

- `false_positive`: **20** image — hampir semua image sulit punya masalah ini
- `B2_B3_confusion`: **13** image — confusion terbanyak terjadi di boundary kelas ini
- `B4_missed`: **11** image — B4 yang terlewat, konsisten dengan recall rendah
- `B3_B4_confusion`: **10** image — confusion antara kelas yang berdekatan secara ordinal

![Distribusi kategori error](figures/p3_error_distribution.png)

Pola ini menceritakan satu narasi yang koheren: model kesulitan paling besar pada **boundary antar kelas yang berdekatan secara ordinal** (B2↔B3, B3↔B4) dan pada **scene yang dense** (banyak buah dalam satu frame → banyak false positive). Ini bukan surprising — kedua masalah ini adalah konsekuensi langsung dari cara dataset ini di-annotate dan dari sifat biologis buah sawit itu sendiri.

![Top-20 image tersulit by error score](figures/p3_error_by_image_score.png)

Dari chart di atas, terlihat bahwa image-image tersulit cenderung datang dari tandan yang sama (lihat prefix `DAMIMAS_A21B_0075`, `DAMIMAS_A21B_0838`) — ini mengindikasikan bahwa beberapa tandan tertentu memang secara inheren lebih sulit karena komposisi buah yang padat dan pencahayaan yang kurang ideal.

Beberapa contoh image tersulit:

- `/workspace/Dataset-Sawit-YOLO/images/test/DAMIMAS_A21B_0838_5.jpg` — error score 21, 6 missed GT + 9 false positive, tanpa true positive sama sekali
- `/workspace/Dataset-Sawit-YOLO/images/test/DAMIMAS_A21B_0075_2.jpg` — error score 19, gabungan B2/B3 confusion + B3/B4 confusion
- `/workspace/Dataset-Sawit-YOLO/images/test/DAMIMAS_A21B_0292_2.jpg` — error score 19, 13 false positive meskipun punya 4 true positive

## 6. Hubungan hasil final dengan Phase 2

Run final ini **bukan model baru atau eksperimen tambahan** di luar apa yang sudah di-lock. Ini adalah kelanjutan langsung dari keputusan Phase 1-2: arsitektur `yolo11m.pt` dipilih di Phase 1B, hyperparameter di-lock di Phase 2 (setelah tuning menunjukkan gain marginal < 1%), dan Phase 3 hanya melakukan retrain dengan budget lebih besar (60 epoch vs 30) plus seed produksi.

Jejak keputusan lengkapnya:

- [outputs/phase1/locked_setup.yaml](../phase1/locked_setup.yaml) — lock arsitektur dan baseline config
- [outputs/phase2/final_hparams.yaml](../phase2/final_hparams.yaml) — lock hyperparameter setelah tuning
- [outputs/phase2/p2confirm_yolo11m_640_s3_e30p10m30_eval.json](../phase2/p2confirm_yolo11m_640_s3_e30p10m30_eval.json) — confirmation run Phase 2 dengan seed ke-3

Perbandingan cepat: confirmation run Phase 2 menghasilkan mAP50 ~0.53 pada val set, sementara run final menghasilkan 0.47 pada test set. Drop ini expected — test set sengaja dipisahkan sejak awal dan tidak pernah dipakai untuk model selection, jadi angka yang lebih rendah justru menandakan evaluasi yang jujur.

![Evolusi per-class mAP50 lintas fase](figures/p3_cross_phase_comparison.png)

Chart di atas memperlihatkan evolusi per-class mAP50 dari Phase 1B → Phase 2 → Phase 3. Hanya B1 yang konsisten di atas target 0.70. Kelas lainnya tidak pernah mendekati target, bahkan dengan budget training lebih besar.

## 7. Status deploy

Status deploy saat ini tetap **deferred**, sesuai [deploy_check.md](deploy_check.md).

Weight `best.pt` sudah aman tersimpan, tapi konversi ke format deployment (TFLite, ONNX, INT8 quantization) dan validasi pasca-konversi belum dilakukan. Keputusan ini disengaja — menjalankan deploy pipeline di atas model yang belum memenuhi target AP50 ≥ 0.70 di semua kelas akan menghasilkan artefak yang perlu di-redo nantinya.

Keluaran utama yang siap dipakai dari sesi ini:

- [outputs/phase3/final_report.md](final_report.md) — narasi keputusan lengkap
- [outputs/phase3/final_evaluation.md](final_evaluation.md) — dokumen ini
- [runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt](../../runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt) — weight model
- [outputs/phase2/final_hparams.yaml](../phase2/final_hparams.yaml) — konfigurasi yang bisa langsung di-reproduce

## 8. Kesimpulan evaluasi

Evaluasi final ini mengonfirmasi bahwa eksperimen E0 berhasil menghasilkan **satu setup akhir yang konsisten, terlacak, dan reproducible**. Dari sisi metodologi, pipeline berjalan sesuai protokol — setiap keputusan terdokumentasi, setiap angka bisa ditelusuri ke artefak aslinya.

Dari sisi performa, model ini adalah **baseline yang solid tapi bukan model yang production-ready**. mAP50 overall di 0.47 dan ketimpangan tajam antar kelas (B1 di 0.78 vs B4 di 0.27) menunjukkan bahwa masih ada ruang perbaikan yang substansial.

Jalur perbaikan yang paling promising, diurutkan berdasarkan expected impact:

1. **Mengurangi confusion B2/B3** — ini penyumbang error terbesar kedua. Pendekatan yang bisa dieksplorasi: fine-grained feature learning (attention mechanism pada region warna), atau augmentasi yang secara spesifik memanipulasi gradasi warna di zona transisi hitam-merah.

2. **Meningkatkan recall B4** — threshold sweep menunjukkan model sebenarnya mendeteksi B4 di confidence rendah. Ini bisa didekati dari dua arah: meningkatkan resolusi input (trade-off compute vs gain), atau menambah data B4 yang berkualitas tinggi.

3. **Menekan false positive** — hampir semua image sulit punya masalah ini. NMS tuning dan post-processing confidence calibration bisa membantu tanpa perlu retrain.

4. **Baru setelah ketiga perbaikan di atas menunjukkan progress**, lanjutkan ke deploy pipeline dengan validasi end-to-end.
