# GUIDE.md

# Panduan Operasional E0 Baseline Protocol

Dokumen ini adalah **guide eksekusi praktis** untuk menjalankan protokol **E0** secara bertahap di repo ini, lengkap dengan **checklist**, **aturan lanjut/stop**, **artefak yang harus dihasilkan**, dan **batasan otomatisasi**.

---

## 1. Tujuan Dokumen

Tujuan `GUIDE.md`:
- menjadi **runbook utama** untuk menjalankan E0,
- menjadi **checklist progres** yang bisa diperbarui bertahap,
- menjadi **kontrak operasional**: apa yang bisa dijalankan otomatis oleh agent, apa yang wajib berhenti dan dilaporkan,
- mencegah eksperimen acak di luar protokol.

---

## 2. Jawaban Singkat: Bisakah E0 dijalankan tanpa supervisi?

### Jawaban jujur
**Ya, E0 dapat dijalankan sangat otonom.** Agent akan bekerja sendiri dari bootstrap sampai finalisasi, dan hanya boleh berhenti jika benar-benar terblokir secara teknis atau jika perbaikan yang diperlukan akan menjadi terlalu spekulatif.

### Yang bisa dijalankan otomatis
- setup environment,
- validasi struktur dataset,
- EDA dataset,
- resolution sweep,
- learning curve,
- one-stage baseline,
- perbandingan model baseline antar arsitektur,
- hyperparameter tuning terstruktur,
- logging hasil,
- pembuatan tabel, ringkasan, dan laporan phase-by-phase,
- update checklist progres.

### Yang tetap menantang meski dikerjakan otomatis
- audit visual label yang membutuhkan penilaian manusia masih punya keterbatasan akurasi,
- keputusan ilmiah final bisa tetap ambigu jika hasil saling bertentangan,
- perubahan definisi kelas tetap berisiko jika bertentangan dengan label guide asli.

### Kesimpulan operasional
Mode kerja yang dipakai adalah:

> **agent-driven, self-healing first, human escalation only if truly blocked**

Artinya saya menjalankan E0 **secara bertahap dan otomatis**, dan jika muncul masalah kritis yang **masih bisa diperbaiki secara teknis**, saya harus **memperbaikinya sendiri terlebih dahulu**, misalnya:
- membetulkan path dataset,
- memperbaiki split yang tidak valid,
- membersihkan file/label rusak,
- memperbaiki mismatch image-label,
- menyusun ulang artefak eksperimen yang gagal.

Saya hanya boleh berhenti jika masalah itu memang **tidak bisa diselesaikan secara aman dan teknis** di workspace ini.

---

## 3. Status Repo Saat Dokumen Ini Dibuat

### Temuan saat audit awal
- Repo ini **belum memuat kode training lengkap**.
- Folder `Dataset-YOLO/` di repo hanya berisi metadata.
- Namun dataset aktual **tersedia** di path yang dirujuk `data.yaml`, yaitu:
  - `/workspace/Dataset-Sawit-YOLO`
- Struktur dataset aktual terdeteksi lengkap:
  - `images/train`, `images/val`, `images/test`
  - `labels/train`, `labels/val`, `labels/test`
- Hitungan aktual yang terdeteksi di workspace:
  - train `2764`
  - val `604`
  - test `624`
  - total image `3992`
  - total label `3992`
- Dependency penting belum lengkap:
  - `torch` ada
  - `ultralytics` belum ada
  - `pandas` belum ada
- GPU tersedia: **NVIDIA A40**
- `git lfs` belum tersedia

### Implikasi
E0 **bisa mulai dipersiapkan** karena dataset aktual ada, tetapi environment dan script eksekusi masih perlu dilengkapi.

---

## 4. Prinsip Eksekusi

Semua eksekusi E0 harus mengikuti prinsip berikut:

1. **Data-first**
   - Jangan training sebelum data tervalidasi.

2. **Satu fase, satu keputusan**
   - Jangan lompat ke phase berikut jika gate phase sebelumnya belum jelas.

3. **Semua hasil harus tersimpan**
   - Setiap phase harus menghasilkan file artefak yang bisa ditinjau ulang.

4. **Bandingkan apple-to-apple**
   - Split, resolusi, seed, dan rejim evaluasi harus tercatat.

5. **Self-healing pada kondisi kritis yang masih bisa diperbaiki**
   - Bila quality gate atau audit menemukan masalah, agent harus **mencoba memperbaiki sendiri terlebih dahulu**.
   - Stop hanya boleh dilakukan jika masalah itu memang tidak bisa diperbaiki secara aman/teknis, atau jika perbaikannya akan mengubah makna ilmiah task secara spekulatif.

6. **Lock setup inti setelah dipilih**
   - Pipeline, arsitektur/model, dan resolusi final **tidak dipreset dari awal**, tetapi **dipilih berdasarkan hasil run yang sah**.
   - Setelah pipeline, arsitektur/model, dan resolusi final dipilih, setup inti itu **dikunci** sampai Phase 3.
   - Contoh: jika hasil terbaik menunjukkan **`yolo11s` + `640px`**, maka sampai finalisasi Phase 3 agent **tidak boleh** mengganti ke `yolo11m`, `yolo11l`, `1024px`, atau setup inti lain.
   - Jika hasil terbaik justru model lain, maka model itulah yang dikunci.
   - Setelah lock, yang boleh berubah hanya komponen yang memang termasuk ruang tuning Phase 2, seperti learning rate, batch size, loss strategy, dan augmentasi.

7. **Batas ukuran model untuk semua training/run E0**
   - Semua training, benchmark, tuning, retrain final, dan testing dalam E0 **harus memakai model dengan ukuran kurang dari atau sama dengan medium**.
   - Varian **large (`l`)**, **xlarge (`x`)**, atau yang lebih besar **dilarang** dipakai dalam bentuk apa pun pada E0.
   - Secara praktis, kandidat model dibatasi ke varian seperti `n`, `s`, dan `m` atau ekuivalennya dalam family model yang diuji.
   - Dengan policy ini, weight hasil run diharapkan cukup kecil untuk ikut disimpan ke GitHub.

8. **`GUIDE.md` adalah living document**
   - `GUIDE.md` harus diperbarui secara berkala selama eksekusi berlangsung.
   - Jika environment berubah, misalnya dependency yang sebelumnya belum terpasang menjadi sudah terpasang, status di `GUIDE.md` harus ikut diperbarui.
   - Jika alur kerja aktual berubah agar lebih sesuai dengan realitas eksekusi, perubahan itu harus dicerminkan di `GUIDE.md` tanpa melanggar policy inti yang sudah disetujui.
   - Setiap akhir phase atau checkpoint penting, agent harus menyinkronkan kondisi nyata eksekusi ke `GUIDE.md`.

---

## 5. Struktur Folder yang Direkomendasikan

Target struktur kerja minimal:

```text
.
├── GUIDE.md
├── CONTEXT.md
├── E0.md
├── Dataset-YOLO/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── data.yaml
│   └── README.md
├── scripts/
│   ├── setup_env.sh
│   ├── validate_dataset.py
│   ├── phase0_eda.py
│   ├── phase0_resolution.py
│   ├── phase0_learning_curve.py
│   ├── phase1_one_stage.py
│   ├── phase1_arch_sweep.py
│   ├── phase2_tuning.py
│   ├── phase3_finalize.py
│   └── aggregate_results.py
├── outputs/
│   ├── phase0/
│   ├── phase1/
│   ├── phase2/
│   ├── phase3/
│   └── reports/
└── runs/
```

Catatan: folder `scripts/`, `outputs/`, dan `runs/` bisa dibuat bertahap.

---

## 6. Checklist Master E0

Gunakan checklist ini sebagai tracker utama.

## 6.1 Readiness Checklist

- [ ] Repo berhasil diaudit
- [ ] Dataset image tersedia di `Dataset-YOLO/images/...`
- [ ] Dataset label tersedia di `Dataset-YOLO/labels/...`
- [ ] `data.yaml` menunjuk path yang benar
- [ ] Semua split (`train/val/test`) ada
- [ ] Pairing image-label valid
- [ ] Dependency Python terpasang
- [ ] `ultralytics` terpasang
- [ ] `pandas` terpasang
- [ ] GPU terdeteksi
- [ ] Direktori output dibuat
- [ ] Format logging eksperimen disiapkan
- [ ] `GITHUB_TOKEN` tersedia di environment
- [ ] Remote GitHub valid dan bisa di-push

## 6.2 Phase 0 — Validation & Calibration

### Task A — EDA dan Validasi Dataset
- [ ] Hitung jumlah image per split
- [ ] Hitung jumlah label per split
- [ ] Hitung distribusi kelas B1-B4
- [ ] Cek image tanpa label
- [ ] Cek bbox invalid / out-of-range
- [ ] Cek ukuran bbox kecil ekstrem
- [ ] Cek ukuran objek per kelas
- [ ] Cek konsistensi nama file image-label
- [ ] Cek tree-group leakage antar split
- [ ] Simpan report EDA
- [ ] Simpan visual sample / hard cases

### Task B — Resolution Sweep
- [ ] Jalankan baseline pada `imgsz=640`, seed 1
- [ ] Jalankan baseline pada `imgsz=640`, seed 2
- [ ] Jalankan baseline pada `imgsz=1024`, seed 1
- [ ] Jalankan baseline pada `imgsz=1024`, seed 2
- [ ] Bandingkan mean metric 640 vs 1024
- [ ] Tetapkan resolusi kerja

### Task C — Learning Curve
- [ ] Train subset 25%
- [ ] Train subset 50%
- [ ] Train subset 75%
- [ ] Train subset 100%
- [ ] Plot learning curve
- [ ] Tentukan indikasi saturasi / masih butuh data

### Gate Phase 0
- [ ] Jika ditemukan label error berat, lakukan perbaikan otomatis yang aman semampunya
- [ ] Jika split tidak valid, lakukan perbaikan split secara otomatis
- [ ] Split final valid dan bebas leakage?
- [ ] Resolusi final dipilih
- [ ] Kecukupan data dinilai

## 6.3 Phase 1A — Pipeline Decision

- [ ] Jalankan one-stage baseline 4 kelas
- [ ] Siapkan two-stage design (jika data mendukung)
- [ ] Jalankan stage-1 detector 1 kelas
- [ ] Jalankan stage-2 maturity classifier
- [ ] Bandingkan one-stage vs two-stage
- [ ] Analisis confusion B2/B3
- [ ] Tetapkan pipeline final

## 6.4 Phase 1B — Architecture Sweep

**Policy ukuran model untuk Phase 1B:**
- Kandidat model untuk benchmark **harus** berukuran **kurang dari atau sama dengan medium**.
- Varian **large (`l`)**, **xlarge (`x`)**, atau yang lebih besar **tidak boleh** masuk daftar uji.
- Jika sebuah family model memakai naming berbeda, hanya ambil varian yang ekuivalen sampai kelas medium.

- [ ] Tentukan daftar model yang akan diuji (maksimal size medium)
- [ ] Pastikan tidak ada model `large` atau di atasnya di benchmark
- [ ] Tetapkan hyperparameter default yang terkunci
- [ ] Jalankan seed 1 untuk semua model
- [ ] Jalankan seed 2 untuk semua model
- [ ] Rekap mean dan std metrik
- [ ] Identifikasi top-2 / top-3 model
- [ ] Analisis per-class failure
- [ ] Evaluasi khusus B4

### Gate Phase 1
- [ ] mAP cukup layak untuk lanjut?
- [ ] Top model terpilih
- [ ] Model terpilih memenuhi policy size maksimal medium
- [ ] Failure mode terdokumentasi
- [ ] Pipeline final dikunci
- [ ] Arsitektur/model final dikunci
- [ ] Resolusi final dikunci
- [ ] File kontrak setup final dibuat

## 6.5 Phase 2 — Hyperparameter Optimization

**Aturan keras Phase 2:**
- Phase 2 **tidak boleh** mengganti pipeline, arsitektur/model, atau resolusi yang sudah dikunci pada akhir Phase 1.
- Jika setup final yang terkunci adalah `yolo11s-640`, maka semua eksperimen Phase 2 wajib tetap memakai `yolo11s-640`.
- Model yang dipakai di Phase 2 tetap **harus** memenuhi policy ukuran **<= medium**.
- Phase 2 hanya boleh mengubah hyperparameter training dan komponen optimasi yang memang termasuk ruang tuning.

### Step 0a — Imbalance Handling
- [ ] Baseline no weighting
- [ ] Class weighting
- [ ] Focal loss / alternatif imbalance-aware
- [ ] Pilih strategi terbaik

### Step 0b — Ordinal / Class-Relationship Handling
- [ ] Standard loss baseline
- [ ] Ordinal-aware variant (jika tersedia)
- [ ] CORAL / CORN untuk cabang classifier (jika two-stage)
- [ ] Pilih strategi terbaik

### Step 1 — Learning Rate
- [ ] Uji `0.0005`
- [ ] Uji `0.001`
- [ ] Uji `0.002`
- [ ] Pilih LR terbaik

### Step 2 — Batch Size
- [ ] Uji `8`
- [ ] Uji `16`
- [ ] Uji `32`
- [ ] Pilih batch terbaik

### Step 3 — Augmentation
- [ ] Light
- [ ] Medium
- [ ] Heavy
- [ ] Pilih augment terbaik

### Gate Phase 2
- [ ] Improvement bermakna atas baseline?
- [ ] Jika <1% dan tidak stabil, kembali ke baseline terbaik
- [ ] Konfigurasi final phase 2 dipilih

## 6.6 Phase 3 — Final Validation

**Aturan keras Phase 3:**
- Retrain final dan evaluasi final wajib memakai **setup inti yang sama** dengan yang sudah dikunci.
- Jika setup final yang terkunci adalah `yolo11s-640`, maka Phase 3 tetap memakai `yolo11s-640`.
- Model yang dipakai di Phase 3 tetap **harus** memenuhi policy ukuran **<= medium**.
- Phase 3 bukan tempat mengganti model yang sudah kalah di phase sebelumnya.

- [ ] Retrain final pada train+val
- [ ] Nonaktifkan early stopping jika sesuai protokol
- [ ] Simpan weight final
- [ ] Jalankan evaluasi final pada test/holdout yang sah
- [ ] Threshold sweep confidence
- [ ] Optimalkan recall B4 dan confusion B2/B3
- [ ] Export model deploy format
- [ ] Cek ukuran model
- [ ] Cek inference viability
- [ ] Ambil 20 error terburuk
- [ ] Lakukan error analysis
- [ ] Tulis ringkasan final

### Gate Phase 3
- [ ] Jika error analysis menemukan masalah label yang bisa diperbaiki, lakukan perbaikan lalu ulang evaluasi yang diperlukan
- [ ] Final metric tervalidasi
- [ ] Final decision tercatat

## 6.7 Final Deliverables Checklist

- [ ] Report EDA
- [ ] Report resolution sweep
- [ ] Report learning curve
- [ ] Report pipeline decision
- [ ] Report architecture benchmark
- [ ] Report tuning phase 2
- [ ] Final confusion matrix
- [ ] Final summary report
- [ ] Ledger hasil eksperimen
- [ ] Semua weight run sudah ikut tersimpan dan ter-push
- [ ] Checklist ini diperbarui lengkap

---

## 7. Apa yang Akan Diotomatisasi oleh Agent

Jika Anda meminta saya menjalankan E0 secara bertahap, default perilaku saya seharusnya seperti ini:

### Policy penyimpanan hasil setiap run
Setiap kali **satu run eksperimen selesai**, agent harus:
1. menyimpan artefak run ke folder output yang sesuai,
2. mengekstrak metrik utama ke file ringkasan/ledger,
3. memperbarui checklist progres jika ada milestone yang selesai,
4. membuat commit Git,
5. melakukan push ke repository GitHub yang ditentukan.

### Definisi minimal “hasil run” yang wajib disimpan
Untuk setiap run, minimal simpan:
- konfigurasi run,
- nama model,
- resolusi,
- seed,
- split / evaluation regime,
- metrik utama (`mAP50`, `mAP50-95`, per-class bila ada),
- weight hasil run minimal `best` dan `last` jika tersedia,
- path artefak penting,
- timestamp,
- status run (`success`, `failed`, `stopped`).

### Default artefak yang di-push ke GitHub
Secara default, agent mendorong artefak berikut ke GitHub:
- file konfigurasi,
- CSV/TSV hasil,
- ringkasan Markdown,
- plot/gambar laporan yang ukurannya wajar,
- checklist dan ledger,
- **semua weight hasil run**.

### Policy weight wajib push
- Karena eksperimen dibatasi hanya sampai model size **medium**, setiap weight hasil run dianggap cukup aman untuk di-push ke GitHub.
- Maka, untuk setiap run, agent **wajib** ikut menyimpan dan mendorong weight yang relevan, minimal `best` dan `last` jika tersedia.
- Weight final Phase 3 juga **wajib** ikut di-commit dan di-push.
- Jika suatu weight ternyata gagal di-push karena masalah teknis GitHub atau jaringan, agent tetap menyimpan lokal, mencatat kegagalan sinkronisasi, lalu berhenti untuk melapor.

### Ledger hasil run
Disarankan ada ledger terstruktur, misalnya:
- `outputs/reports/run_ledger.csv`
- atau `outputs/reports/run_ledger.tsv`

Setiap run baru harus menambah satu baris ke ledger itu.

### Mode otomatis yang diinginkan
1. **Audit readiness**
2. **Setup environment**
3. **Validasi dataset**
4. **Jalankan Phase 0**
5. **Simpan hasil run + update ledger + commit + push**
6. **Tulis ringkasan + update checklist**
7. **Jika gate lulus, lanjut Phase 1**
8. **Simpan hasil run + update ledger + commit + push**
9. **Tulis ringkasan + update checklist**
10. **Jika gate lulus, lanjut Phase 2**
11. **Simpan hasil run + update ledger + commit + push**
12. **Tulis ringkasan + update checklist**
13. **Jika gate lulus, lanjut Phase 3**
14. **Simpan hasil run + update ledger + commit + push**
15. **Tulis laporan final**

### Aturan update progres
Setelah tiap phase, checklist pada file ini harus diperbarui dari `[ ]` menjadi `[x]` hanya jika artefak benar-benar ada.

---

## 8. Batasan Otomatisasi: Kapan Agent Harus Stop

Agent **wajib mencoba memperbaiki sendiri lebih dulu**. Stop dan lapor hanya jika terjadi salah satu kondisi berikut **setelah upaya perbaikan yang wajar gagal**:

### Git persistence / push policy
- Push dilakukan ke repository:
  - `https://github.com/muhammad-zainal-muttaqin/brand-new-yolo`
- Autentikasi memakai environment variable:
  - `GITHUB_TOKEN`
- Token **tidak boleh** ditulis ke file, log, atau commit.
- Agent hanya boleh memakai token pada saat operasi push.
- Push setiap run harus mencakup artefak laporan **dan weight run**.
- Jika push gagal karena auth/network/permission, agent harus:
  1. tetap menyimpan hasil run secara lokal,
  2. mencatat bahwa sinkronisasi GitHub gagal,
  3. berhenti untuk melapor, atau menunggu retry policy jika nanti ditetapkan.

### A. Data tidak siap dan tidak berhasil diperbaiki
- folder image kosong,
- folder label kosong,
- banyak file label hilang yang tidak bisa direkonstruksi,
- `data.yaml` salah path dan tidak bisa dibetulkan secara aman,
- split train/val/test tidak ada dan tidak bisa dibangun ulang secara sah,
- tree leakage berat yang tidak bisa diperbaiki dengan resplit yang valid,
- label format rusak dan tidak bisa dipulihkan.

### B. Quality gate gagal dan tidak berhasil diperbaiki
- indikasi label error berat yang tidak bisa dibersihkan secara aman,
- terlalu banyak bbox invalid yang tidak bisa diperbaiki,
- hasil eksperimen tidak valid karena data mismatch yang tidak bisa diselesaikan.

### C. Konflik ilmiah / kebijakan eksperimen
- split pada dokumen berbeda dengan split aktual,
- definisi kelas tidak konsisten dan memengaruhi evaluasi,
- prosedur two-stage memerlukan keputusan desain yang belum dipaku.

### D. Keterbatasan teknis
- dependency gagal terpasang,
- disk tidak cukup,
- runtime melebihi batas wajar,
- training crash berulang.

---

## 9. Artefak yang Harus Dihasilkan per Fase

## 9.1 Phase 0
Minimal hasil:

```text
outputs/phase0/
├── dataset_audit.json
├── eda_report.md
├── class_distribution.csv
├── bbox_stats.csv
├── leakage_report.json
├── resolution_sweep.csv
├── learning_curve.csv
└── phase0_summary.md
```

## 9.2 Phase 1

```text
outputs/phase1/
├── one_stage_results.csv
├── two_stage_results.csv
├── architecture_benchmark.csv
├── per_class_metrics.csv
├── locked_setup.yaml
└── phase1_summary.md
```

## 9.3 Phase 2

```text
outputs/phase2/
├── tuning_results.csv
├── lr_sweep.csv
├── batch_sweep.csv
├── aug_sweep.csv
└── phase2_summary.md
```

## 9.4 Phase 3

```text
outputs/phase3/
├── final_metrics.csv
├── confusion_matrix.csv
├── threshold_sweep.csv
├── error_analysis.md
├── deploy_check.md
└── final_report.md
```

## 9.5 Artefak ledger dan sinkronisasi Git

```text
outputs/reports/
├── run_ledger.csv
├── latest_status.md
└── git_sync_log.md
```

Keterangan:
- `run_ledger.csv` menyimpan ringkasan semua run
- `latest_status.md` menyimpan status eksekusi terakhir
- `git_sync_log.md` mencatat commit/push yang berhasil atau gagal

---

## 10. Kebijakan Keputusan Otomatis per Gate

### Phase 0
- Jika split valid, data valid, dan tidak ada red flag label berat → **lanjut**
- Jika ditemukan label error atau split bermasalah, agent harus **memperbaikinya terlebih dahulu** lalu mengulang validasi yang relevan
- Jika 1024 hanya naik kecil dan cost jauh lebih tinggi → pilih **640**

### Phase 1A
- Jika two-stage unggul jelas dan stabil → pilih **two-stage**
- Jika tidak unggul berarti → pilih **one-stage**

### Phase 1B
- Pilih top model berdasarkan:
  1. `mAP50-95`
  2. stabilitas antar seed
  3. perilaku pada B2/B3/B4
- Model final **harus berasal dari hasil benchmark/run**, bukan asumsi awal.
- Setelah top model dipilih, agent harus membuat file **`outputs/phase1/locked_setup.yaml`**.
- Isi file lock minimal:
  - pipeline final,
  - nama model final,
  - resolusi final,
  - split/evaluation regime,
  - seed policy,
  - data path.
- Setelah file lock dibuat, setup inti **tidak boleh diubah** sampai Phase 3 kecuali ada instruksi eksplisit dari Anda.

### Phase 2
- Jika kenaikan <1% dan tidak konsisten → **rollback ke baseline terbaik**
- Jangan lanjut tuning liar di luar grid yang sudah ditentukan tanpa alasan baru
- Phase 2 **harus** membaca dan mematuhi `locked_setup.yaml`
- Dilarang mengganti model size, backbone family, atau `imgsz` setelah lock

### Phase 3
- Final retrain **harus** memakai setup yang sama dengan `locked_setup.yaml`
- Jika final model valid dan artefak lengkap → tulis final report
- Jika error analysis menunjukkan masalah label dominan yang masih bisa diperbaiki, agent harus **memperbaiki lalu mengulang evaluasi yang diperlukan**

---

## 11. Kebutuhan Minimum Sebelum Eksekusi Nyata

Checklist ini wajib terpenuhi dulu sebelum saya mulai menjalankan E0 sungguhan.

- [ ] Dataset aktual tersedia di workspace
- [ ] Path dataset final sudah dipastikan
- [ ] Package `ultralytics` tersedia
- [ ] Package `pandas` tersedia
- [ ] Lokasi simpan hasil eksperimen disetujui
- [ ] Rejim evaluasi yang akan dipakai dipastikan
- [ ] `GITHUB_TOKEN` tersedia dan valid untuk push

### Rejim evaluasi harus dipilih jelas
Sebelum eksekusi, kita harus menegaskan salah satu:
- split repo saat ini,
- split kanonik lokal,
- atau split benchmark lain.

Karena ada konflik angka split antar dokumen, agent **tidak boleh mengarang sendiri** split final tanpa evidence dari data aktual.

---

## 12. Rencana Eksekusi Bertahap yang Direkomendasikan

### Tahap 1 — Bootstrap
Tujuan:
- siapkan dependency,
- validasi dataset,
- kunci source-of-truth split,
- siapkan struktur output.

### Tahap 2 — Phase 0 penuh
Tujuan:
- audit kualitas data,
- pilih resolusi,
- ukur learning curve.

### Tahap 3 — Phase 1 baseline
Tujuan:
- tentukan pipeline,
- tentukan model kandidat utama.

### Tahap 4 — Phase 2 tuning terbatas
Tujuan:
- cari konfigurasi yang benar-benar layak dibanding baseline.

### Tahap 5 — Phase 3 finalisasi
Tujuan:
- retrain final,
- validasi final,
- laporan final.

---

## 13. Policy: Apa yang Saya Lakukan Setelah Anda Bilang “Lanjut”

Jika Anda memberi instruksi untuk menjalankan E0, urutan kerja saya adalah:

1. audit repo dan dataset,
2. setup environment,
3. buat script pendukung bila belum ada,
4. jalankan Phase 0,
5. simpan hasil run, update ledger, commit, dan push ke GitHub,
6. update `GUIDE.md`,
7. jalankan Phase 1,
8. jika Phase 1 sudah memilih setup final, buat `outputs/phase1/locked_setup.yaml`,
9. simpan hasil run, update ledger, commit, dan push ke GitHub,
10. lanjut otomatis ke Phase 2 dan Phase 3 **hanya dengan setup yang sudah dikunci**,
11. pada akhir setiap run, ulangi langkah simpan hasil + commit + push,
12. berhenti dan lapor jika masuk kondisi stop.

---

## 14. Jawaban Praktis untuk Pertanyaan Anda

### “Apakah semua E0 bisa dilakukan tanpa supervisi saya?”
**Ya, target operasionalnya adalah sangat otonom.**

Versi yang paling akurat:

> Saya menjalankan E0 secara **otonom dengan prinsip self-healing**: agent mengeksekusi semua langkah teknis, memperbaiki masalah data/split/path/artefak jika masih memungkinkan, lalu hanya berhenti jika benar-benar mentok secara teknis atau jika perbaikannya akan menjadi spekulatif dan berisiko merusak validitas ilmiah.

Contoh hal yang tetap bisa menjadi batas keras:
- dataset benar-benar korup atau hilang,
- push GitHub terus gagal,
- dependency inti tidak bisa dipasang,
- conflict definisi kelas tidak bisa diselesaikan secara evidence-based dari data yang ada.

---

## 15. Status Eksekusi Saat Ini

### Belum mulai
- [ ] Bootstrap environment
- [ ] Dataset validation
- [ ] Phase 0
- [ ] Phase 1A
- [ ] Phase 1B
- [ ] Phase 2
- [ ] Phase 3
- [ ] Final report

---

## 16. Next Action Recommended

Urutan aksi paling masuk akal dari kondisi repo saat ini:

1. pastikan dataset aktual tersedia,
2. pasang dependency,
3. validasi `data.yaml`,
4. mulai Phase 0.

---

## 17. Catatan Penting untuk Repo Ini

Dari audit awal, ada conflict metadata yang harus dibereskan sebelum eksperimen sah:
- `CONTEXT.md` menyebut split tertentu,
- `Dataset-YOLO/README.md` menyebut split lain,
- `Dataset-YOLO/data.yaml` menunjuk path yang tampaknya tidak ada di repo ini saat ini.

Maka:

> **Langkah pertama sebelum E0 adalah menegaskan source-of-truth dataset aktual.**

Tanpa itu, semua benchmark berisiko tidak valid.
