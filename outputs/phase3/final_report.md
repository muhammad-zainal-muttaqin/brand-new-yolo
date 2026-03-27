# Final Report - E0 End-to-End

Dokumen ini merangkum keputusan akhir eksperimen E0 dari **Phase 0** sampai **Phase 3**. Laporan ini sengaja dibaca sebagai peta keputusan: pembaca baru bisa cepat menangkap konteks, sementara pembaca teknis tetap bisa menelusuri artefak sumber tanpa kehilangan detail. Jika Anda baru membuka repo ini, mulai dari [README.md](../../README.md). Jika Anda ingin langsung melihat metrik teknis run final, buka [outputs/phase3/final_evaluation.md](final_evaluation.md).

## Navigasi singkat

Bagian ini membantu pembaca melompat ke artefak yang paling relevan sesuai kebutuhan, baik untuk membaca ringkasan, memeriksa protokol, maupun menelusuri bukti teknis per fase.

- Ringkasan repo: [README.md](../../README.md)
- Protokol canonical: [E0.md](../../E0.md)
- Runbook repo ini: [GUIDE.md](../../GUIDE.md)
- Ringkasan Phase 0: [outputs/phase0/phase0_summary.md](../phase0/phase0_summary.md)
- Ringkasan Phase 1: [outputs/phase1/phase1_summary.md](../phase1/phase1_summary.md)
- Ringkasan Phase 2: [outputs/phase2/phase2_summary.md](../phase2/phase2_summary.md)
- Evaluasi final teknis: [outputs/phase3/final_evaluation.md](final_evaluation.md)
- Reproduksi dan terminasi: [outputs/reports/reproducibility_and_termination.md](../reports/reproducibility_and_termination.md)

**So what:** pembaca tidak perlu menebak urutan baca. Dokumen ini adalah ringkasan keputusan, sedangkan link di atas adalah jalur cepat menuju bukti detailnya.

## 1. Dokumen acuan yang dipakai

Sebelum membaca hasil, penting untuk menetapkan source of truth. Bagian ini menunjukkan file mana yang benar-benar dipakai untuk mengambil keputusan akhir, sehingga pembacaan tidak tercampur dengan catatan eksperimen lain yang sifatnya sekunder.

Keputusan di dokumen ini merujuk langsung ke file berikut:

- [outputs/phase1/locked_setup.yaml](../phase1/locked_setup.yaml)
- [outputs/phase2/final_hparams.yaml](../phase2/final_hparams.yaml)
- [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json)
- [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json](p3_final_yolo11m_640_s42_e60p15m60_summary.json)
- [runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt](../../runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt)

**So what:** jika ada ringkasan lain yang terasa berbeda, file-file di atas yang harus diprioritaskan sebagai referensi resmi.

## 2. Keputusan akhir yang dikunci

Bagian ini merangkum keputusan operasional yang benar-benar di-lock. Ini bukan sekadar preferensi sementara, tetapi konfigurasi yang dipakai sampai run final dan menjadi dasar reproduksi.

Eksperimen ini berakhir dengan keputusan berikut:

- pipeline final: **`one-stage`**
- model final: **`yolo11m.pt`**
- resolusi kerja final: **`640`**
- recipe final: **`lr0=0.001`, `batch=16`, `imbalance=none`, `ordinal=standard`, `aug=medium`**
- run final: **`p3_final_yolo11m_640_s42_e60p15m60`**
- final best weight: [runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt](../../runs/detect/runs/e0/p3_final_yolo11m_640_s42_e60p15m60/weights/best.pt)

**So what:** titik akhirnya jelas. Repo ini tidak berhenti di ruang eksplorasi, tetapi di satu pipeline, satu model, dan satu recipe yang bisa ditelusuri ulang.

## 3. Narasi keputusan per fase

Angka akhir tidak muncul tiba-tiba. Setiap fase menjawab pertanyaan yang berbeda: apakah data cukup sehat, pipeline mana yang paling realistis, model mana yang paling stabil, dan apakah tuning benar-benar memberi alasan untuk mengubah baseline.

### Phase 0 - dataset valid, resolusi `640`

Phase 0 menjawab dua hal paling mendasar: apakah dataset aktif layak dipakai untuk baseline, dan resolusi kerja mana yang paling masuk akal jika performa dan biaya dihitung bersama.

Phase 0 menunjukkan bahwa dataset aktif lolos audit dasar dan tidak memiliki blocker teknis yang terdeteksi. Buktinya ada di [outputs/phase0/dataset_audit.json](../phase0/dataset_audit.json) dan [outputs/phase0/eda_report.md](../phase0/eda_report.md).

Resolution sweep pada [outputs/phase0/resolution_sweep.csv](../phase0/resolution_sweep.csv) menunjukkan `1024` memang sedikit lebih baik daripada `640`, tetapi selisihnya kecil. Karena itu, Phase 0 mengunci **`640`** sebagai resolusi kerja yang paling realistis. Ringkasannya ada di [outputs/phase0/phase0_summary.md](../phase0/phase0_summary.md).

![Trade-off resolusi kerja Phase 0](figures/phase0_resolution_tradeoff.png)

Visual pendukung: [outputs/phase3/figures/phase0_resolution_tradeoff.png](figures/phase0_resolution_tradeoff.png)

**Interpretasi singkat:** kenaikan dari `640` ke `1024` ada, tetapi tidak cukup besar untuk otomatis membenarkan biaya eksperimen yang lebih berat.

**So what:** keputusan `640` terlihat konservatif, tetapi justru itu yang membuat baseline tetap efisien dan konsisten untuk fase-fase berikutnya.

### Phase 1A - pipeline `one-stage`

Phase 1A membandingkan dua pendekatan yang secara praktis akan menentukan arah repo: apakah sistem sebaiknya tetap sederhana dalam satu detector, atau dipecah menjadi alur dua tahap.

Perbandingan pipeline dibaca dari:

- [outputs/phase1/one_stage_results.csv](../phase1/one_stage_results.csv)
- [outputs/phase1/two_stage_results.csv](../phase1/two_stage_results.csv)

Hasilnya: pipeline **`one-stage`** dipilih. Alasan utamanya bukan karena two-stage tidak bisa belajar, tetapi karena komponen classifier two-stage masih menunjukkan confusion `B2/B3` yang besar, bahkan pada GT crops. Ringkasannya ada di [outputs/phase1/phase1_summary.md](../phase1/phase1_summary.md).

**Interpretasi singkat:** masalah utamanya bukan sekadar akurasi detector, tetapi ketidakstabilan klasifikasi saat kelas yang berdekatan harus dipisahkan.

**So what:** memilih `one-stage` membuat fokus repo tetap tajam. Upaya berikutnya lebih masuk akal diarahkan ke detector utama, bukan menambah kompleksitas di classifier two-stage yang belum cukup meyakinkan.

### Phase 1B - `yolo11m.pt` menang tipis dan di-lock

Phase 1B adalah fase seleksi model. Tujuannya bukan mencari model yang tampak bagus pada satu run, tetapi model yang tetap paling aman ketika dibandingkan secara langsung.

Benchmark arsitektur lengkap ada di [outputs/phase1/architecture_benchmark.csv](../phase1/architecture_benchmark.csv). Top-3 resminya ada di [outputs/phase1/phase1b_top3.csv](../phase1/phase1b_top3.csv).

Tiga model teratas:

| Rank | Model | mean mAP50 | mean mAP50-95 |
|---:|---|---:|---:|
| 1 | `yolo11m.pt` | 0.5298 | 0.2570 |
| 2 | `yolov9c.pt` | 0.5292 | 0.2518 |
| 3 | `yolov8s.pt` | 0.5256 | 0.2521 |

Gate canonical `mAP50 >= 0.70` memang tidak lolos. Namun repo ini memakai override operasional agar baseline end-to-end tetap selesai. Lock resminya tetap ada di [outputs/phase1/locked_setup.yaml](../phase1/locked_setup.yaml).

![Benchmark arsitektur Phase 1B](figures/phase1_architecture_benchmark.png)

Visual pendukung: [outputs/phase3/figures/phase1_architecture_benchmark.png](figures/phase1_architecture_benchmark.png)

**Interpretasi singkat:** `yolo11m.pt` bukan menang telak, tetapi menang tipis dengan cukup alasan untuk di-lock sebagai opsi paling stabil.

**So what:** karena selisih antar model atas relatif rapat, bottleneck utama repo ini tampaknya bukan lagi soal memilih family model yang "ajaib", melainkan batas kualitas task dan data.

### Phase 2 - tuning selesai, tetapi repo kembali ke baseline stabil

Phase 2 menguji satu pertanyaan yang sangat praktis: apakah tuning cukup kuat untuk menggantikan baseline Phase 1, atau justru hanya memberi variasi kecil tanpa keuntungan yang tegas.

Phase 2 merangkum hasil tuning di [outputs/phase2/tuning_results.csv](../phase2/tuning_results.csv). File itu mencatat:

- `final_source = phase1_baseline_reverted`
- `reverted_to_phase1_baseline = True`

Artinya, tuning dilakukan, tetapi hasil akhirnya tetap kembali ke recipe baseline yang paling stabil. Konfigurasi final yang dibawa ke Phase 3 ada di [outputs/phase2/final_hparams.yaml](../phase2/final_hparams.yaml). Confirm run-nya ada di [outputs/phase2/p2confirm_yolo11m_640_s3_e30p10m30_eval.json](../phase2/p2confirm_yolo11m_640_s3_e30p10m30_eval.json).

**Interpretasi singkat:** tuning tetap berguna karena menutup pertanyaan eksperimen, tetapi hasilnya belum cukup kuat untuk menggeser baseline yang sudah stabil.

**So what:** ini sinyal penting bahwa ruang optimasi ringan sudah banyak dijelajahi. Gain besar kemungkinan tidak akan datang dari tweak recipe kecil.

### Phase 3 - final retrain selesai, deploy ditunda

Phase 3 adalah fase penguncian akhir. Di sini baseline yang sudah dipilih tidak lagi diuji sebagai hipotesis, tetapi dijalankan ulang secara final untuk menghasilkan bobot, evaluasi, dan status deploy yang resmi.

Run final Phase 3 didokumentasikan di:

- [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json](p3_final_yolo11m_640_s42_e60p15m60_summary.json)
- [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json)

Phase 3 menyelesaikan final retrain, evaluasi test set, threshold sweep, dan error review. Weight final berhasil diamankan. Status deploy tetap **deferred**, sesuai [outputs/phase3/deploy_check.md](deploy_check.md).

**Interpretasi singkat:** dari sisi eksperimen, fase ini selesai dengan rapi. Dari sisi operasional, repo masih memilih menahan deploy karena kualitas antar kelas belum merata.

**So what:** baseline final sudah ada dan bisa direproduksi, tetapi status deploy yang ditunda menegaskan bahwa "selesai eksperimen" tidak sama dengan "siap dipakai".

## 4. Hasil akhir yang harus dianggap resmi

Bagian ini memuat angka yang harus dipakai ketika seseorang mengutip performa final repo. Jika ada angka dari artefak intermediate atau dari threshold tertentu, anggap bagian ini sebagai acuan utama.

Untuk angka resmi final, prioritaskan file run-specific berikut:

- [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json)
- [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_summary.json](p3_final_yolo11m_640_s42_e60p15m60_summary.json)

| Metrik resmi test set | Nilai |
|---|---:|
| Precision | 0.4763 |
| Recall | 0.5538 |
| mAP50 | 0.4677 |
| mAP50-95 | 0.2215 |
| All classes `AP50 >= 0.70` | False |

Per-class highlights:

- `B1` paling kuat (`mAP50 = 0.7821`)
- `B3` berada di tengah (`mAP50 = 0.4880`)
- `B2` masih lemah (`mAP50 = 0.3266`)
- `B4` paling sulit (`mAP50 = 0.2742`)

![Performa final per kelas](figures/phase3_per_class_ap.png)

Visual pendukung: [outputs/phase3/figures/phase3_per_class_ap.png](figures/phase3_per_class_ap.png)

**Interpretasi singkat:** performa final tidak runtuh total, tetapi jelas tidak seimbang. `B1` sudah kuat, `B3` masih menengah, sedangkan `B2` dan terutama `B4` masih menjadi sumber kelemahan utama.

**So what:** model ini sudah cukup untuk menjadi baseline final yang jujur, tetapi belum cukup aman jika targetnya adalah performa yang rata di semua kelas.

## 5. Threshold operasi, error dominan, dan deploy status

Bagian ini menerjemahkan metrik menjadi implikasi operasional. Di sinilah pembaca bisa melihat threshold mana yang masuk akal, error apa yang paling sering muncul, dan kenapa deploy masih ditunda.

Threshold operasi dibaca dari [outputs/phase3/threshold_sweep.csv](threshold_sweep.csv). Kandidat terbaik pada artefak ini berada di **`conf=0.1`**.

![Threshold sweep Phase 3](figures/phase3_threshold_sweep.png)

Visual pendukung: [outputs/phase3/figures/phase3_threshold_sweep.png](figures/phase3_threshold_sweep.png)

Analisis error praktis ada di:

- [outputs/phase3/error_analysis.md](error_analysis.md)
- [outputs/phase3/error_stratification.csv](error_stratification.csv)

Pola error dominannya tetap sama seperti yang terlihat di fase-fase sebelumnya:

- `false_positive`
- confusion `B2/B3`
- `B4_missed`
- confusion `B3/B4`

Status deploy saat ini tetap **deferred**, jadi keluaran utama sesi ini adalah:

- weight final `.pt`
- laporan akhir
- evaluasi final
- recipe final yang bisa direproduksi

**Interpretasi singkat:** threshold rendah di **`conf=0.1`** menunjukkan bahwa menjaga recall masih penting, tetapi pola error membuktikan bahwa masalah inti belum selesai hanya dengan menggeser threshold.

**So what:** keputusan deploy **deferred** terlihat masuk akal. Jika dipaksakan ke tahap operasional sekarang, risiko noise prediksi dan miss pada kelas sulit masih terlalu nyata.

## 6. Kesimpulan akhir

Eksperimen E0 repo ini menghasilkan **satu baseline final yang konsisten dan terdokumentasi**, tetapi belum menghasilkan model yang kuat merata di semua kelas.

Kesimpulan paling aman adalah:

1. dataset aktif cukup layak untuk baseline
2. `640` adalah resolusi kerja yang paling realistis
3. `one-stage` adalah pipeline yang paling masuk akal untuk baseline repo ini
4. `yolo11m.pt` adalah model terbaik yang lolos sampai akhir
5. tuning Phase 2 tidak memberi alasan kuat untuk meninggalkan baseline stabil
6. bottleneck utama tetap berada pada `B2/B3`, `B4`, dan error deteksi berlebih

**So what:** langkah berikutnya sebaiknya tidak lagi berupa sweep kecil yang generik. Prioritas yang lebih masuk akal adalah pekerjaan yang langsung menyentuh `B2/B3`, `B4`, kualitas label, dan formulasi task.

## 7. File yang sebaiknya dibuka setelah dokumen ini

Bagian ini adalah jalur lanjut baca yang paling efisien jika Anda ingin bergerak dari ringkasan ke bukti teknis yang lebih rinci.

- [outputs/phase3/final_evaluation.md](final_evaluation.md)
- [outputs/phase3/error_analysis.md](error_analysis.md)
- [outputs/phase3/threshold_sweep.csv](threshold_sweep.csv)
- [outputs/phase2/final_hparams.yaml](../phase2/final_hparams.yaml)
- [outputs/reports/reproducibility_and_termination.md](../reports/reproducibility_and_termination.md)

**So what:** urutan baca yang paling praktis biasanya dimulai dari evaluasi final, lalu turun ke error analysis dan threshold sweep, baru setelah itu ke recipe dan catatan reproduksi.
