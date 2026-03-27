# CONTEXT

File ini menjelaskan **masalah inti** dan **konteks keputusan** riset deteksi kematangan TBS sawit 4 kelas (`B1`–`B4`). Ini bukan kronologi lengkap, tetapi panduan untuk menjawab: apa yang sudah terbukti, apa yang belum, dan eksperimen mana yang masih layak dibuka.

Kalau hanya butuh versi ringkas, buka [CONTEXT_Less.md](CONTEXT_Less.md). Untuk protokol canonical, buka [E0.md](E0.md). Untuk runbook repo ini, buka [GUIDE.md](GUIDE.md).

## 1. Urutan source of truth

Kalau ada konflik antar dokumen, ikuti urutan berikut.

### 1.1 Ledger eksperimen aktif

- `C:/Users/Zainal/Desktop/autoresearch/results.tsv`
- `C:/Users/Zainal/Desktop/bbc-autoresearch-v1/experiments/results.tsv`

### 1.2 Laporan formal baseline dan E0

- `C:/Users/Zainal/Desktop/bbc-autoresearch-v1/LAPORAN_EKSPERIMEN.md`
- `C:/Users/Zainal/Desktop/bbc-autoresearch-v1/e0_results/results.csv`

### 1.3 Analisis dataset dan audit label

- `D:/Work/Assisten Dosen/YOLOBench/analysis_dataset_640/report.md`
- `D:/Work/Assisten Dosen/YOLOBench/analysis_dataset_640/metrics_summary.json`
- `D:/Work/Assisten Dosen/YOLOBench/analysis_dataset_640/tables/model_class_summary.csv`
- `D:/Work/Assisten Dosen/YOLOBench/analysis_dataset_640/tables/source_class_summary.csv`
- `D:/Work/Assisten Dosen/YOLOBench/analysis_dataset_640/tables/probe_summary.csv`
- `D:/Work/Assisten Dosen/YOLOBench/analysis/bbox_outliers/summary.md`
- `D:/Work/Assisten Dosen/YOLOBench/analysis/bbox_outliers/shortlist_review.md`

### 1.4 Ringkasan historis dan mirror

- `D:/Work/Assisten Dosen/YOLOBench/rangkum2.md`
- `G:/My Drive/Asisten_Dosen/YOLOBench/rangkum5.md`

### 1.5 Steering notes repo

- `C:/Users/Zainal/Desktop/autoresearch/context.md`
- `C:/Users/Zainal/Desktop/bbc-autoresearch-v1/context/research_overview.md`
- `C:/Users/Zainal/Desktop/bbc-autoresearch-v1/context/domain_knowledge.md`

### 1.6 Catatan penting

Nama `B1`–`B4` pernah tidak konsisten di dokumen lama. Untuk workspace ini, mapping yang berlaku:

- `B1 = buah paling matang / ripe`
- `B2 = kelas transisi setelah B1`
- `B3 = lebih mentah dari B2`
- `B4 = buah paling mengkal / belum matang`

Jadi urutan biologis yang dipakai di repo ini adalah **`B1 -> B2 -> B3 -> B4` = paling matang ke paling belum matang**.

### 1.7 Panduan visual/posisional label aktif

Pakai interpretasi ini secara konsisten:

- `B1`: buah **merah**, **besar**, **bulat**, posisi **paling bawah** pada tandan
- `B2`: buah masih **hitam**, mulai **transisi ke merah**, sudah **besar** dan **bulat**, posisi **di atas B1**
- `B3`: buah **full hitam**, masih **berduri**, masih **lonjong**, posisi **di atas B2**
- `B4`: buah **paling kecil**, **paling dalam di tandan**, sulit terlihat, masih banyak **duri**, warna **hitam sampai hijau**, dan masih bisa berkembang lebih besar

## 2. Problem statement yang berlaku sekarang

Masalah proyek ini **bukan** “model gagal belajar”.

Yang lebih tepat:

> Sistem 4-kelas RGB-only mentok karena kombinasi **ambiguitas `B2/B3`**, **small-object burden pada `B4`**, **domain imbalance DAMIMAS/LONSUM**, dan **kualitas bbox pada kasus kecil atau ambigu** — sehingga tuning recipe biasa tidak lagi memberi lompatan berarti.

Implikasinya:

- bottleneck utama **bukan** optimizer atau jumlah epoch
- bottleneck utama lebih mungkin ada pada **struktur sinyal belajar** dan **formulasi task**

## 3. Fakta dataset yang paling penting

### 3.1 Integritas dataset lokal `dataset_640`

Sumber utama:

- `D:/Work/Assisten Dosen/YOLOBench/analysis_dataset_640/metrics_summary.json`
- `D:/Work/Assisten Dosen/YOLOBench/analysis_dataset_640/report.md`

Angka yang paling aman dipakai untuk analisis lokal:

- total image: **3992**
- total instance: **17949**
- split lokal:
  - train **2772**
  - val **608**
  - test **612**
- total tree: **953**
- tree leakage: **0**
- empty-label images: **83**
- resolusi unik: **640 × 853**

### 3.2 Konflik hitungan split antar sumber

Ada dua keluarga angka yang sering muncul:

- analisis lokal: `2772 / 608 / 612`
- branch cleaning lama / v2: `2780 / 620 / 592`

Aturan pakainya:

- untuk **analisis lokal `dataset_640`**, pakai `2772/608/612`
- untuk **benchmark v2 atau dataset_cleaned lama**, angka `2780/620/592` masih bisa muncul
- **jangan campur** dua keluarga angka itu tanpa menyebut sumbernya

### 3.3 Domain imbalance sangat besar

Sumber utama:

- `D:/Work/Assisten Dosen/YOLOBench/analysis_dataset_640/report.md`
- `D:/Work/Assisten Dosen/YOLOBench/analysis_dataset_640/tables/source_class_summary.csv`

Fakta penting:

- DAMIMAS menyumbang sekitar **90.1% image** dan **94.3% instance**
- LONSUM adalah domain minoritas
- LONSUM `B1` hanya **17 instance**
- rata-rata objek per image:
  - DAMIMAS: **4.71**
  - LONSUM: **2.59**

Artinya:

- model “combined” pada praktiknya belajar prior DAMIMAS
- gap lintas domain bukan hanya appearance shift, tetapi juga **scene-structure shift**

### 3.4 `B4` memang objek paling kecil

Sumber utama:

- `D:/Work/Assisten Dosen/YOLOBench/analysis/bbox_outliers/summary.md`

Median geometry per kelas:

| Class | Count | Median rel_area | Median width px | Median height px |
|---|---:|---:|---:|---:|
| B1 | 2177 | 0.0140 | 125 | 137 |
| B2 | 4075 | 0.0107 | 109 | 121 |
| B3 | 8296 | 0.0096 | 105 | 114 |
| B4 | 3442 | 0.0072 | 94 | 96 |

Artinya:

- `B4` paling kecil secara geometri
- semua pasangan kelas tetap overlap pada rentang P10–P90
- ukuran saja tidak cukup untuk memisahkan `B2/B3/B4`

### 3.5 Dataset sendiri sudah menunjukkan separability problem

Sumber utama:

- `D:/Work/Assisten Dosen/YOLOBench/analysis_dataset_640/tables/probe_summary.csv`

Linear probe pada embedding hanya memberi overall accuracy **0.528**:

| Class | Precision | Recall |
|---|---:|---:|
| B1 | 0.770 | 0.713 |
| B2 | 0.394 | 0.463 |
| B3 | 0.420 | 0.363 |
| B4 | 0.554 | 0.575 |

Artinya:

- `B1` paling separable
- `B2/B3` paling campur
- bottleneck `B2/B3` kemungkinan bukan sekadar kurang tuning detector

## 4. Bukti performa: apa yang benar-benar terlihat

### 4.1 Benchmark perlu dipisah per rejim

Jangan baca angka-angka berikut sebagai satu leaderboard tunggal.

| Rejim | Best run | mAP50 | mAP50-95 | Makna |
|---|---|---:|---:|---|
| Legacy unfair | `exp7` YOLOv9c `damimas-full` | 0.650 | 0.328 | Historis. Terinflasi leakage. |
| V2 fair benchmark | `exp17` YOLOv9c `damimas_only`, seed 42 | 0.505 | 0.230 | Benchmark fair lama yang paling jujur. |
| `dataset_640` baseline | YOLOv9c AdamW | 0.509 | 0.240 | Baseline kuat branch YOLOBench. |
| Active standard val | `AR29` YOLO11l 640 b16 | 0.555 | 0.264 | Best non-`train+test` saat ini. |
| Active upper-bound | `AR34` YOLO11l 80 ep 640 b16 `train+test` | 0.554 | 0.269 | Bukan fair benchmark. |
| E0 final | `p3_final_yolo11s_s42` | 0.558 | 0.265 | E0 tetap berakhir `insufficient`. |

### 4.2 Kesimpulan yang bisa ditarik

- sistem 4-kelas modern stabil di kisaran **0.24–0.27 mAP50-95**, tergantung rejim
- ada perbaikan dari baseline lama ke branch aktif, tetapi **tidak ada lompatan besar**
- hasil `train+test` hanya menaikkan upper-bound sedikit; ia tidak mengubah sifat masalah

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

Artinya:

- `B1` paling mudah
- `B4` paling sulit secara deteksi
- `B2` sangat lemah walaupun bukan yang paling kecil

### 4.4 E0 menegaskan bottleneck yang sama

Sumber utama:

- `C:/Users/Zainal/Desktop/bbc-autoresearch-v1/LAPORAN_EKSPERIMEN.md`

Temuan E0 yang paling penting:

- `1024` hanya memberi gain kecil atas `640`
- YOLO11s dan YOLOv8s sangat dekat
- keputusan akhir: **INSUFFICIENT**
- bottleneck eksplisit:
  - confusion `B2/B3` tinggi
  - `B4` sering missed ke background

Angka confusion yang paling penting:

- `B2` correct sekitar **32–34%**
- `B2 -> B3` sekitar **34–35%**
- `B4 -> background` sekitar **41–43%**

Artinya: resolusi dan arsitektur saja **tidak menyelesaikan** akar masalah.

## 5. Kesimpulan yang sudah cukup kuat

### 5.1 Sistem 4-kelas memang belajar

Bukti:

- one-stage modern mencapai sekitar `0.24–0.27 mAP50-95`
- single-class stage-1 detector mencapai **0.390 mAP50-95**

Makna: masalahnya bukan detector gagal mendeteksi tandan sama sekali. Masalahnya muncul saat task menjadi **4 kelas dengan boundary ambigu**.

### 5.2 `B2/B3` adalah bottleneck semantik atau label boundary

Bukti:

- probe embedding `B2/B3` lemah
- confusion E0 `B2 -> B3` tinggi
- classifier crops, DINOv2, CORN, SupCon, dan hierarchical branch gagal memberi terobosan jelas

Makna: bottleneck ini kemungkinan lebih dekat ke **ambiguity definisi kelas** atau **kualitas sinyal visual**, bukan kekurangan backbone semata.

### 5.3 `B4` adalah bottleneck small-object dan density

Bukti:

- `B4` paling kecil secara geometri
- mean mAP50-95 `B4` paling rendah
- E0 menunjukkan `B4` sering missed ke background

Makna: `B4` perlu treatment small-object yang lebih spesifik daripada recipe umum.

### 5.4 Domain imbalance menahan generalisasi

Bukti:

- DAMIMAS mendominasi hampir seluruh dataset
- `lonsum_only` runtuh di benchmark fair
- `B1` LONSUM nyaris tidak ada

Makna: evaluasi lintas domain harus dibaca hati-hati. Domain-aware analysis lebih penting daripada sweep recipe kecil.

### 5.5 Recipe tuning biasa sudah diminishing returns

Bukti:

- banyak tweak kecil gagal mengalahkan baseline dengan margin bermakna
- long-run training tidak menembus ceiling baru
- arsitektur alternatif tidak mengubah pola gagal

Makna: branch baru harus **struktural**, bukan knob-turning biasa.

## 6. Branch yang sudah cukup ditutup

Jangan buka ulang branch berikut tanpa alasan struktural yang baru.

### 6.1 Knob-turning / recipe tweaks

- `imgsz 800`
- `patience 30`
- `erasing 0.2`
- `flipud 0.5`
- `scale 0.7`
- `scale 0.7 + degrees 5.0`
- `BOX=10 CLS=1.5 DFL=2.0`
- `lr0=0.0005 lrf=0.1`
- `lr0=0.002`
- SGD mengganti AdamW
- `copy_paste 0.3`
- `label_smoothing=0.1`
- brute-force long-run `2h / 300 epoch`
- model soup

### 6.2 Data-centric versi lama yang sudah gagal

- oversampling sederhana `B1/B4`
- tiled dataset training naif
- label correction otomatis berbasis confidence model
- HSV/color-only branch
- SAHI pada setup lama yang justru memperburuk hasil

### 6.3 Architecture swap yang tidak layak diulang apa adanya

- YOLOv9e
- RT-DETR-L
- RF-DETR Base DINOv2
- YOLO11m 1024
- YOLO11x train+test
- YOLO10n / YOLO10s pada jalur lama

### 6.4 Two-stage 4-class branch

- detector + EfficientNet
- detector + DINOv2 CE
- detector + DINOv2 CORN
- hierarchical coarse + B23 specialist
- wide-context crop classifier

Intinya: stage-1 detector bisa bagus, tapi **pipeline 4-kelas tetap kalah** karena boundary `B2/B3/B4` masih bermasalah.

## 7. Hipotesis yang masih terbuka

### 7.1 Label ceiling atau human ambiguity pada `B2/B3`

Hipotesis paling penting yang belum benar-benar tertutup.

Langkah yang lebih masuk akal ketimbang ganti model:

- audit label boundary `B2/B3`
- re-review agreement manusia
- slice review pada kasus confusion tertinggi
- pertimbangkan reformulasi task jika perlu

### 7.2 `B4` masih mungkin naik lewat treatment small-object yang benar-benar spesifik

Masih terbuka, tapi **khusus untuk `B4`** — bukan solusi untuk semua 4 kelas.

### 7.3 Data quality masih memberi ROI lebih tinggi daripada tuning biasa

Bukti:

- audit bbox menemukan outlier nyata
- shortlist review berisi **3 DROP + 9 high-priority review + 21 review tambahan**
- cleaning lama memang menghasilkan koreksi nyata

### 7.4 Domain-aware evaluation tetap penting

Eksperimen baru minimal sebaiknya punya breakdown:

- per-domain
- per-class
- kalau bisa per-size bucket

## 8. Aturan perbandingan

- Jika eksperimen memakai **legacy split**, bandingkan hanya ke legacy split.
- Jika eksperimen memakai **V2 tree-level benchmark**, bandingkan ke `exp10–exp21`, terutama `exp17`.
- Jika eksperimen memakai **active standard val**, bandingkan ke `AR29`.
- Jika eksperimen memakai **train+test**, bandingkan ke `AR31–AR39`, terutama `AR34`.
- Jika eksperimen memakai **binary task** atau **single-class detector**, jangan klaim itu sebagai solusi final 4 kelas.

## 9. Decision context untuk eksperimen berikutnya

### 9.1 Jika targetnya hanya “naik sedikit”

Ruang kecil masih mungkin ada pada:

- evaluasi yang lebih granular
- sedikit perbaikan data
- treatment small-object khusus `B4`

### 9.2 Jika targetnya “lompatan nyata”

Jangan harap lompatan datang dari recipe tuning biasa.

Branch baru masuk akal jika menyasar salah satu dari ini:

- membuktikan atau mematahkan **label ceiling `B2/B3`**
- membenahi **data quality pada slice tersulit**
- menguji **treatment small-object yang benar-benar baru untuk `B4`**
- mereformulasi task bila `B2/B3` memang tidak reliably separable

### 9.3 Prinsip kerja yang sebaiknya dipakai sekarang

- satu hipotesis falsifiable per branch
- slice metric harus jelas sejak awal
- jangan buka ulang branch tertutup hanya dengan kombinasi knob yang hampir sama

## 10. Ringkasan satu kalimat

> Riset ini tampak mentok bukan karena model YOLO gagal belajar, tetapi karena sistem 4-kelas RGB-only sudah menabrak kombinasi **ambiguity `B2/B3`**, **small-object burden pada `B4`**, **domain imbalance DAMIMAS/LONSUM**, dan **noise/ketelitian bbox pada kasus sulit**.

## 11. Acuan kerja

- main ceiling: **`B2/B3` discrimination**
- secondary ceiling: **`B4` small-object recall / localization**
- main decision metric: **`mAP50-95`**
- safest active baseline for standard val: **`AR29 = 0.264147`**
- safest active upper-bound: **`AR34 = 0.269424`**
- sikap saat ini: **masalah utama struktural, optimisasi nomor dua**
