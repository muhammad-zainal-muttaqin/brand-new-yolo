# CONTEXT

Dokumen ini adalah **problem definition + decision context** untuk riset deteksi kematangan TBS sawit 4 kelas (`B1`-`B4`).

Tujuannya bukan mencatat semua eksperimen secara kronologis, tetapi menjawab 4 hal sebelum membuka branch baru:

1. **Masalah inti apa yang benar-benar terbukti?**
2. **Angka pembanding yang benar untuk rejim evaluasi ini apa?**
3. **Cabang apa yang sudah cukup difalsifikasi dan tidak layak dibuka lagi?**
4. **Hipotesis apa yang masih hidup dan pantas diuji?**

---

## 1. Source of Truth dan Urutan Prioritas

Gunakan sumber dengan urutan ini jika ada konflik:

1. **Ledger eksperimen aktif**
   - `C:/Users/Zainal/Desktop/autoresearch/results.tsv`
   - `C:/Users/Zainal/Desktop/bbc-autoresearch-v1/experiments/results.tsv`
2. **Laporan formal baseline / E0**
   - `C:/Users/Zainal/Desktop/bbc-autoresearch-v1/LAPORAN_EKSPERIMEN.md`
   - `C:/Users/Zainal/Desktop/bbc-autoresearch-v1/e0_results/results.csv`
3. **Analisis dataset dan audit label**
   - `D:/Work/Assisten Dosen/YOLOBench/analysis_dataset_640/report.md`
   - `D:/Work/Assisten Dosen/YOLOBench/analysis_dataset_640/metrics_summary.json`
   - `D:/Work/Assisten Dosen/YOLOBench/analysis_dataset_640/tables/model_class_summary.csv`
   - `D:/Work/Assisten Dosen/YOLOBench/analysis_dataset_640/tables/source_class_summary.csv`
   - `D:/Work/Assisten Dosen/YOLOBench/analysis_dataset_640/tables/probe_summary.csv`
   - `D:/Work/Assisten Dosen/YOLOBench/analysis/bbox_outliers/summary.md`
   - `D:/Work/Assisten Dosen/YOLOBench/analysis/bbox_outliers/shortlist_review.md`
4. **Ringkasan historis / mirror**
   - `D:/Work/Assisten Dosen/YOLOBench/rangkum2.md`
   - `G:/My Drive/Asisten_Dosen/YOLOBench/rangkum5.md`
5. **Steering notes repo**
   - `C:/Users/Zainal/Desktop/autoresearch/context.md`
   - `C:/Users/Zainal/Desktop/bbc-autoresearch-v1/context/research_overview.md`
   - `C:/Users/Zainal/Desktop/bbc-autoresearch-v1/context/domain_knowledge.md`

### Caveat besar

- **Nama tekstual B1-B4 memang pernah tidak konsisten antar dokumen historis.**
- Namun untuk repo/workspace ini, source of truth operasional yang dipakai adalah:
  - `B1 = buah paling matang / ripe`
  - `B2 = kelas transisi setelah B1, masih hitam namun mulai ke merah`
  - `B3 = lebih mentah dari B2, full hitam, masih berduri, masih lonjong`
  - `B4 = buah paling mengkal / belum matang`
- Jadi urutan biologis `B1 → B4` bergerak dari **paling matang** ke **paling belum matang**.
- Jika ada dokumen historis yang bertentangan, ikuti mapping ini untuk eksekusi aktif repo ini.

### Panduan visual/posisional label aktif

Untuk interpretasi eksperimen aktif repo ini, pakai panduan berikut:
- `B1`: buah **merah**, **besar**, **bulat**, dan posisinya **paling bawah** pada tandan; ini indikator kematangan tertinggi.
- `B2`: buah masih **hitam** namun mulai **transisi ke merah**; buah sudah **besar** dan **bulat**; posisinya **di atas B1**.
- `B3`: buah **full hitam**, masih ada **duri** pada buah, bentuknya masih **lonjong**, dan posisinya **di atas B2**.
- `B4`: buah **paling kecil**, **paling dalam di batang/tandan**, sering **susah terlihat**, masih banyak **duri**, warnanya **hitam bahkan hijau**, dan buah pada tandannya masih bisa berkembang lebih besar; ini kelas **paling belum matang**.

---

## 2. Problem Statement yang Berlaku Saat Ini

Problem saat ini **bukan** “model gagal belajar”.

Problem yang lebih tepat:

> Sistem 4-kelas RGB-only tampak mentok karena kombinasi **ambiguity B2/B3**, **kesulitan small-object pada B4**, **domain imbalance DAMIMAS/LONSUM**, dan **kualitas/ketelitian bbox pada kasus kecil/ambigu**, sehingga recipe tuning biasa tidak lagi memberi lompatan berarti.

Implikasi operasional:
- bottleneck utama **bukan** sekadar optimizer atau epoch,
- bottleneck utama kemungkinan ada pada **struktur sinyal pembelajaran** dan **formulasi task**.

---

## 3. Fakta Dataset yang Paling Penting

### 3.1 Integritas dataset lokal `dataset_640`
Sumber utama:
- `D:/Work/Assisten Dosen/YOLOBench/analysis_dataset_640/metrics_summary.json`
- `D:/Work/Assisten Dosen/YOLOBench/analysis_dataset_640/report.md`

Angka yang paling aman dipakai untuk analisis lokal:
- Total image: **3992**
- Total instance: **17949**
- Split lokal:
  - train **2772**
  - val **608**
  - test **612**
- Total tree: **953**
- Tree leakage: **0**
- Empty-label images: **83**
- Resolusi unik: **640 × 853**

### 3.2 Konflik hitungan split antar sumber
Ada dua keluarga angka yang sama-sama muncul di repo:
- analisis lokal: `2772 / 608 / 612`
- branch cleaning lama / v2: `2780 / 620 / 592`

Aturan pakai:
- untuk **analisis dataset_640 lokal**, pakai `2772/608/612`
- untuk **benchmark v2 / dataset_cleaned lama**, boleh muncul `2780/620/592`
- **jangan campur** tanpa menyebut sumbernya

### 3.3 Domain imbalance sangat besar
Sumber utama:
- `D:/Work/Assisten Dosen/YOLOBench/analysis_dataset_640/report.md`
- `D:/Work/Assisten Dosen/YOLOBench/analysis_dataset_640/tables/source_class_summary.csv`

Fakta:
- DAMIMAS menyumbang sekitar **90.1% image** dan **94.3% instance**
- LONSUM minoritas
- LONSUM **B1 hanya 17 instance**
- Rata-rata objek per image:
  - DAMIMAS: **4.71**
  - LONSUM: **2.59**

Makna:
- model “combined” pada praktiknya belajar prior DAMIMAS
- gap lintas source bukan hanya appearance shift, tapi juga **scene-structure shift**

### 3.4 B4 memang objek paling kecil
Sumber utama:
- `D:/Work/Assisten Dosen/YOLOBench/analysis/bbox_outliers/summary.md`

Median geometry per kelas:

| Class | Count | Median rel_area | Median width px | Median height px |
|---|---:|---:|---:|---:|
| B1 | 2177 | 0.0140 | 125 | 137 |
| B2 | 4075 | 0.0107 | 109 | 121 |
| B3 | 8296 | 0.0096 | 105 | 114 |
| B4 | 3442 | 0.0072 | 94 | 96 |

Makna:
- **B4 paling kecil secara geometri**
- semua pasangan kelas overlap ukuran pada rentang P10-P90
- ukuran saja tidak cukup untuk memisahkan B2/B3/B4 secara bersih

### 3.5 Dataset sendiri sudah menunjukkan separability problem
Sumber utama:
- `D:/Work/Assisten Dosen/YOLOBench/analysis_dataset_640/tables/probe_summary.csv`

Probe linear di embedding menghasilkan overall accuracy hanya **0.528**:

| Class | Precision | Recall |
|---|---:|---:|
| B1 | 0.770 | 0.713 |
| B2 | 0.394 | 0.463 |
| B3 | 0.420 | 0.363 |
| B4 | 0.554 | 0.575 |

Makna:
- **B1 paling separable**
- **B2/B3 paling campur**
- ini mendukung dugaan bahwa bottleneck B2/B3 bukan sekadar kekurangan tuning detector

---

## 4. Bukti Performa: Apa yang Benar-Benar Terlihat di Eksperimen

## 4.1 Benchmark harus dipisah per rejim
Jangan mencampur angka-angka berikut sebagai satu leaderboard tunggal.

| Rejim | Best run | mAP50 | mAP50-95 | Makna |
|---|---|---:|---:|---|
| Legacy unfair | `exp7` YOLOv9c `damimas-full` | 0.650 | 0.328 | Historis, terinflasi leakage |
| V2 fair benchmark | `exp17` YOLOv9c `damimas_only`, seed 42 | 0.505 | 0.230 | Benchmark fair lama yang paling jujur |
| `dataset_640` baseline | YOLOv9c AdamW | 0.509 | 0.240 | Baseline kuat branch YOLOBench |
| Active standard val | `AR29` YOLO11l 640 b16 | 0.555 | 0.264 | Best non-`train+test` saat ini |
| Active upper-bound | `AR34` YOLO11l 80 ep 640 b16 `train+test` | 0.554 | 0.269 | Bukan fair benchmark |
| E0 final | `p3_final_yolo11s_s42` | 0.558 | 0.265 | E0 masih berujung `insufficient` |

### 4.2 Apa yang bisa disimpulkan dari angka-angka ini
- sistem 4-kelas modern stabil di sekitar **0.24–0.27 mAP50-95** tergantung rejim
- ada perbaikan dari baseline lama ke branch aktif, tapi **tidak ada lompatan besar**
- hasil `train+test` hanya menaikkan upper-bound sedikit, bukan mengubah nature problem

### 4.3 Per-class difficulty sangat konsisten
Sumber utama:
- `D:/Work/Assisten Dosen/YOLOBench/analysis_dataset_640/tables/model_class_summary.csv`

Rata-rata performa model per kelas:

| Class | mean mAP50 | mean mAP50-95 |
|---|---:|---:|
| B1 | 0.700 | 0.330 |
| B3 | 0.410 | 0.170 |
| B2 | 0.285 | 0.126 |
| B4 | 0.229 | 0.085 |

Makna:
- **B1 termudah**
- **B4 tersulit secara deteksi**
- **B2 sangat buruk secara klasifikasi/lokalisasi** meski bukan yang paling kecil

### 4.4 E0 menegaskan bottleneck yang sama
Sumber utama:
- `C:/Users/Zainal/Desktop/bbc-autoresearch-v1/LAPORAN_EKSPERIMEN.md`

Temuan E0 yang penting:
- 1024 hanya memberi gain kecil atas 640
- YOLO11s dan YOLOv8s sangat dekat
- keputusan akhir: **INSUFFICIENT**
- bottleneck eksplisit di laporan:
  - **B2/B3 confusion** tinggi
  - **B4 sering missed ke background**

Angka confusion yang paling penting dari E0:
- B2 correct sekitar **32–34%**
- B2 → B3 sekitar **34–35%**
- B4 → background sekitar **41–43%**

Makna:
- resolusi/arsitektur saja **tidak menyelesaikan** masalah utama

---

## 5. Apa yang Sudah Terbukti / High-Confidence Conclusions

### 5.1 Sistem 4-kelas memang belajar
Bukti:
- one-stage modern mencapai `0.24–0.27 mAP50-95`
- single-class stage-1 detector mencapai **0.390 mAP50-95**

Makna:
- masalahnya bukan “detector tidak bisa mendeteksi tandan sama sekali”
- masalahnya muncul saat task menjadi **4-kelas dengan boundary ambigu**

### 5.2 B2/B3 adalah bottleneck semantik / label boundary
Bukti:
- probe embedding B2/B3 lemah
- confusion E0 B2→B3 sangat tinggi
- classifier crops, DINOv2, CORN, SupCon, hierarchical branch semuanya gagal memberi terobosan B2/B3

Makna:
- bottleneck ini kemungkinan **lebih dekat ke ambiguity definisi kelas / kualitas sinyal visual**, bukan murni kekurangan backbone

### 5.3 B4 adalah bottleneck small-object + density
Bukti:
- B4 paling kecil secara geometri
- mean mAP50-95 B4 paling rendah
- E0 menunjukkan B4 sering missed ke background

Makna:
- B4 butuh treatment small-object yang lebih spesifik daripada recipe umum

### 5.4 Domain imbalance menahan generalisasi
Bukti:
- DAMIMAS dominan hampir seluruh dataset
- `lonsum_only` runtuh di benchmark fair
- LONSUM B1 nyaris tidak ada

Makna:
- evaluasi lintas source harus dibaca hati-hati
- domain-aware analysis lebih penting daripada sweep recipe kecil

### 5.5 Recipe tuning biasa sudah diminishing returns
Bukti:
- banyak tweak kecil gagal mengalahkan baseline dengan margin bermakna
- long-run training tidak menembus ceiling baru
- arsitektur alternatif tidak mengubah pola gagal

Makna:
- jika lanjut, branch baru harus **struktural**, bukan knob-turning biasa

---

## 6. Branch yang Sudah Cukup Ditutup

Jangan buka lagi branch berikut **tanpa alasan struktural baru**.

### 6.1 Knob-turning / recipe tweaks
Sudah cukup difalsifikasi:
- `imgsz 800`
- `patience 30`
- `erasing 0.2`
- `flipud 0.5`
- `scale 0.7`
- `scale 0.7 + degrees 5.0`
- `BOX=10 CLS=1.5 DFL=2.0`
- `lr0=0.0005 lrf=0.1`
- `LR0=0.002`
- `SGD` mengganti AdamW
- `copy_paste 0.3`
- `label_smoothing=0.1`
- `long-run 2h / 300 epoch`
- `model soup`

### 6.2 Data-centric versi lama yang sudah gagal
- oversampling sederhana B1/B4 berbasis duplikasi/flip
- tiled dataset training naif
- label correction otomatis berbasis model confidence
- HSV/color-only branch
- SAHI yang pernah diuji pada setup lama dan dilaporkan memperburuk hasil

### 6.3 Architecture swap yang sudah tidak layak diulang apa adanya
- YOLOv9e
- RT-DETR-L
- RF-DETR Base DINOv2
- YOLO11m 1024
- YOLO11x train+test
- YOLO10n / YOLO10s di E0

### 6.4 Two-stage 4-class branch
Sudah cukup jelas kalah dari one-stage best:
- detector + EfficientNet
- detector + DINOv2 CE
- detector + DINOv2 CORN
- hierarchical coarse + B23 specialist
- wide-context crop classifier

Makna penting:
- detector stage-1 bisa bagus,
- tapi **pipeline 4-kelas tetap kalah** karena bottleneck klasifikasi boundary B2/B3/B4

---

## 7. Hal yang Masih Layak Dianggap Hidup

Ini bukan daftar “harus dikerjakan”, tapi daftar hipotesis yang **belum benar-benar mati**.

### 7.1 Label ceiling / human ambiguity pada B2-B3
Ini hipotesis paling penting yang belum benar-benar ditutup.

Bukti yang mendukung:
- B2/B3 overlap visual dan ukuran
- probe embedding lemah
- crop classifier kuat pun gagal
- one-stage dan two-stage sama-sama mentok

Konsekuensi:
- sebelum branch arsitektur baru, lebih bernilai melakukan:
  - audit label boundary B2/B3,
  - re-review human agreement,
  - atau pertimbangan reformulasi task

### 7.2 B4 masih mungkin naik lewat treatment small-object yang benar-benar spesifik
Hipotesis ini masih hidup, tapi **khusus untuk B4**, bukan solusi total 4-kelas.

### 7.3 Data quality masih punya ROI lebih tinggi daripada tuning biasa
Bukti:
- audit bbox menemukan outlier nyata
- shortlist review berisi **3 DROP + 9 high-priority review + 21 review tambahan**
- cleaning lama memang menghasilkan perubahan label nyata

Makna:
- perbaikan label/bbox terarah masih lebih rasional daripada sweep baru yang luas

### 7.4 Domain-aware evaluation masih penting
Karena source imbalance besar, eksperimen baru sebaiknya punya breakdown minimal:
- per-domain
- per-class
- kalau bisa per-size bucket

---

## 8. Aturan Perbandingan yang Aman

- Jika eksperimen memakai **legacy split**, bandingkan hanya ke legacy split.
- Jika eksperimen memakai **V2 tree-level benchmark**, bandingkan ke `exp10–exp21`, terutama `exp17`.
- Jika eksperimen memakai **active standard val**, bandingkan ke `AR29`.
- Jika eksperimen memakai **train+test**, bandingkan ke `AR31–AR39`, terutama `AR34`.
- Jika eksperimen memakai **binary task** atau **single-class detector**, jangan klaim itu sebagai solusi final 4-kelas.

---

## 9. Decision Context untuk Eksperimen Berikutnya

### 9.1 Jika targetnya “naik sedikit”
Masih mungkin ada ruang kecil pada:
- evaluasi lebih granular,
- sedikit perbaikan data,
- atau treatment small-object khusus B4.

### 9.2 Jika targetnya “lompatan nyata”
Jangan asumsikan lompatan akan datang dari recipe tuning biasa.

Branch baru hanya masuk akal jika menyasar salah satu dari ini:
- **membuktikan atau mematahkan label ceiling B2/B3**,
- **membenahi data quality pada slice tersulit**,
- **menguji treatment small-object yang benar-benar baru untuk B4**,
- atau **mereformulasi task** bila B2/B3 memang tidak reliably separable.

### 9.3 Prinsip kerja yang berlaku sekarang
- prefer **satu hipotesis falsifiable** per branch
- jangan buka branch besar baru tanpa metrik slice yang jelas
- jangan reopen branch tertutup hanya dengan kombinasi knob sedikit berbeda

---

## 10. Ringkasan Satu Kalimat

> Riset ini tampak mentok bukan karena model YOLO gagal belajar, tetapi karena sistem 4-kelas RGB-only sudah menabrak kombinasi **ambiguity B2/B3**, **small-object burden pada B4**, **domain imbalance DAMIMAS/LONSUM**, dan **noise/ketelitian bbox pada kasus sulit**, sehingga tuning recipe biasa hanya memberi gain kecil dan tidak menyentuh akar masalah.

---

## 11. Working Defaults

- main ceiling: **B2/B3 discrimination**
- secondary ceiling: **B4 small-object recall / localization**
- main decision metric: **`mAP50-95`**
- safest active baseline for standard val: **`AR29` = 0.264147**
- safest active upper-bound: **`AR34` = 0.269424**
- current stance: **problem is structural first, optimization second**
