# CONTEXT_Less

`CONTEXT_Less.md` adalah versi ringkas [CONTEXT.md](CONTEXT.md). Pakai file ini sebagai pegangan cepat sebelum membuka branch baru atau menyusun rencana eksperimen.

Kalau butuh bukti lengkap, angka rinci, atau daftar sumber historis, kembali ke [CONTEXT.md](CONTEXT.md).

## 1. Inti masalah saat ini

Ceiling proyek ini datang dari gabungan empat hal:

- **ambiguity `B2/B3`**: dua kelas ini sulit dipisahkan secara visual
- **small-object burden pada `B4`**: `B4` paling kecil dan paling sering missed
- **domain imbalance dan domain shift**: DAMIMAS dominan, LONSUM minoritas
- **bbox dan label noise**: terutama pada kasus kecil, padat, dan ambigu

Artinya sederhana: masalah utama proyek ini **bukan lagi tuning recipe biasa**. Akarnya tampak **struktural** — gabungan data, definisi task, dan kualitas sinyal belajar.

## 2. Angka acuan cepat

Pakai benchmark sesuai rejimnya. Jangan campur jadi satu leaderboard.

| Rejim | Best run | mAP50 | mAP50-95 |
|---|---|---:|---:|
| V2 fair benchmark | `exp17` YOLOv9c `damimas_only` | 0.505 | 0.230 |
| `dataset_640` baseline | YOLOv9c AdamW | 0.509 | 0.240 |
| Active standard val | `AR29` YOLO11l 640 b16 | 0.555 | 0.264 |
| Active upper-bound | `AR34` YOLO11l 80 ep 640 b16 `train+test` | 0.554 | 0.269 |
| E0 final | `p3_final_yolo11s_s42` | 0.558 | 0.265 |

Aturan baca:

- **jangan** bandingkan `train+test` dengan fair benchmark
- **jangan** bandingkan binary task atau single-class detector dengan task final 4 kelas
- jika ada konflik dokumentasi, cek dulu sumber yang diprioritaskan di [CONTEXT.md](CONTEXT.md)

## 3. Fakta dataset penting

### Dataset lokal `dataset_640`

- total image: **3992**
- total instance: **17949**
- split lokal utama: **2772 / 608 / 612**
- tree leakage: **0**
- resolusi: **640 × 853**
- empty-label images: **83**

### Domain

- DAMIMAS ≈ **90.1% image**, **94.3% instance**
- LONSUM minoritas
- LONSUM `B1` hanya **17 instance**
- DAMIMAS: **4.71 objek/image**
- LONSUM: **2.59 objek/image**

### Ukuran objek

Median bbox:

- `B1`: rel_area **0.0140**
- `B2`: **0.0107**
- `B3`: **0.0096**
- `B4`: **0.0072**

Artinya:

- `B4` memang paling kecil
- `B2/B3/B4` punya overlap ukuran yang besar
- ukuran saja tidak cukup untuk memisahkan kelas sulit

### Probe separability

Linear probe di embedding hanya memberi overall accuracy **0.528**:

- `B1` paling mudah dipisahkan
- `B2/B3` paling campur

Ini memperkuat dugaan bahwa bottleneck utama bukan sekadar kurang tuning.

## 4. Apa yang sudah terbukti

### Fakta yang sudah kuat

- model **bisa belajar**
- sistem one-stage modern cenderung mentok di sekitar **0.24–0.27 mAP50-95** tergantung rejim
- single-class detector bisa jauh lebih tinggi daripada task 4 kelas
- `B2/B3` adalah bottleneck utama klasifikasi
- `B4` adalah bottleneck utama recall dan lokalisasi objek kecil
- domain imbalance menahan generalisasi
- tweak recipe kecil sudah masuk fase **diminishing returns**

### Ditegaskan oleh E0

- `1024` hanya membantu sedikit dibanding `640`
- arsitektur small modern saling berdekatan hasilnya
- keputusan akhir E0: **INSUFFICIENT**
- confusion penting:
  - `B2 -> B3` sekitar **34–35%**
  - `B4 -> background` sekitar **41–43%**

## 5. Branch yang tidak perlu diulang

### Recipe tweaks

- imgsz 800
- patience 30
- erasing 0.2
- flipud 0.5
- scale 0.7
- BOX/CLS/DFL heavy reweight
- LR tweaks minor
- SGD swap
- copy_paste 0.3
- label smoothing standalone
- brute-force long run
- model soup

### Data-centric lama

- oversampling sederhana `B1/B4`
- tiled training naif
- auto label correction berbasis confidence model
- HSV/color-only branch
- SAHI pada setup lama yang sudah dilaporkan memburuk

### Architecture swaps lama

- YOLOv9e
- RT-DETR-L
- RF-DETR Base DINOv2
- YOLO11m 1024
- YOLO11x train+test
- jalur lama YOLO10n/s

### Two-stage 4-class yang sudah cukup tertutup

- detector + EfficientNet
- detector + DINOv2 CE
- detector + CORN
- hierarchical coarse + B23 specialist
- wide-context crop classifier

## 6. Hipotesis yang masih terbuka

### H1. `B2/B3` punya label ceiling atau human ambiguity

Hipotesis paling penting yang belum tertutup.

Langkah yang masih masuk akal:

- audit boundary `B2/B3`
- cek agreement antar reviewer
- review slice confusion tertinggi

### H2. `B4` masih bisa naik lewat treatment small-object yang lebih spesifik

Fokusnya bukan semua kelas, tetapi khusus:

- `B4` recall
- AP small
- bbox quality pada objek kecil

### H3. Data quality masih lebih menjanjikan daripada knob tuning

Bukti historis kuat:

- shortlist audit: **3 DROP + 9 high-priority review + 21 review tambahan**
- cleaning lama memang menghasilkan koreksi nyata

### H4. Domain-aware evaluation tetap penting

Eksperimen baru minimal harus melaporkan:

- per-source
- per-class
- kalau bisa per-size bucket

## 7. Logging minimal untuk branch baru

Catat setidaknya:

### Model
- model variant
- pretrained/freeze state
- imgsz

### Augment, loss, optimizer
- mosaic, mixup, copy-paste, HSV, geometry
- box/cls/dfl/focal/weighting
- optimizer, lr0, lrf, wd, warmup, cosine

### Training
- batch size
- epoch
- seed
- AMP
- accumulation
- EMA

### Data
- dataset version
- split type (`tree` vs `image`)
- jumlah image dan instance
- source composition
- catatan kualitas anotasi

### Metrics
- mAP50-95
- AP per class
- precision/recall
- PR curve
- confusion matrix
- kalau ada: TIDE atau breakdown error serupa

### Artifacts
- `args.yaml`
- `results.csv`
- `best.pt`
- `last.pt`
- commit hash

## 8. Diagnostik yang lebih berguna dari sweep biasa

Sebelum buka branch baru, prioritaskan:

- cek duplicate dan leakage per-tree
- small no-aug overfit sanity test
- audit label pada slice `B2/B3` dan `B4`
- cek imbalance handling yang benar-benar terukur
- cek kebutuhan `P2` atau small-object head
- sweep inference setting: conf, IoU, NMS
- LR-range dan warmup sanity

## 9. Tiga arah yang masih masuk akal

### E1 — Label noise / relabel audit
**Tujuan:** membuktikan apakah ceiling `B2/B3` datang dari label boundary yang kabur.

### E2 — Domain-aware sampling / evaluation
**Tujuan:** melihat apakah worst-domain bisa naik tanpa merusak domain utama.

### E3 — Small-object specific branch
**Tujuan:** membantu `B4` dengan treatment yang memang dirancang untuk objek kecil.

## 10. Stop rule

Hentikan branch baru kalau:

- gain hanya kecil, misalnya `< 0.01`, tanpa insight baru
- slice metric utama tidak berubah (`B2/B3`, `B4`, per-source)
- branch itu hanya mengulang branch lama dengan knob yang sedikit berbeda

Lanjutkan hanya kalau branch itu:

- menguji hipotesis struktural baru
- atau memberi bukti baru tentang ceiling task

## 11. Ringkasan satu kalimat

> Masalah proyek ini lebih mungkin berakar pada **struktur dataset, ambiguitas task, dan small-object burden** — bukan karena kurang mencoba optimizer atau recipe baru.
