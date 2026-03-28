# Error Analysis

Dokumen ini menganalisis pola error pada run final `p3_final_yolo11m_640_s42_e60p15m60`, berdasarkan evaluasi terhadap 20 image tersulit di test set. Tujuannya bukan sekadar menyebutkan angka, tapi memahami *kenapa* model gagal di kasus-kasus tertentu dan apa implikasinya bagi iterasi selanjutnya.

Untuk skor test-set resmi dan metrik per kelas, lihat [final_evaluation.md](final_evaluation.md). Data mentah stratifikasi error ada di [error_stratification.csv](error_stratification.csv).

## Sumber data

- [p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json) — metrik per kelas
- [threshold_sweep.csv](threshold_sweep.csv) — hasil sweep confidence
- [error_stratification.csv](error_stratification.csv) — stratifikasi 20 image tersulit
- [confusion_matrix.csv](confusion_matrix.csv) — matriks konfusi (catatan: matriks ini berisi nol karena pipeline evaluasi tidak mempopulasikan nilainya; analisis confusion di dokumen ini bertumpu pada stratifikasi per-image)

## 1. Ringkasan cepat

| Parameter | Nilai |
|---|---|
| Optimized confidence threshold | `0.1` |
| Confusion `B2/B3` (eval JSON) | `None` (tidak diisi pipeline) |
| Confusion `B3/B4` (eval JSON) | `None` (tidak diisi pipeline) |
| All classes `AP50 >= 0.70` | **False** |

Karena eval JSON tidak menyediakan confusion numerik antar kelas secara agregat, seluruh analisis error di dokumen ini bertumpu pada [error_stratification.csv](error_stratification.csv) — yang meng-categorize setiap image berdasarkan pola error yang diamati.

## 2. Pola error yang paling sering muncul

Dari 20 image dengan error score tertinggi, berikut distribusi kategori error:

- `false_positive`: **20** image — muncul di hampir setiap kasus sulit
- `B2_B3_confusion`: **13** image
- `B4_missed`: **11** image
- `B3_B4_confusion`: **10** image

![Distribusi kategori error pada 20 image tersulit](figures/p3_error_distribution.png)

Perhatikan bahwa satu image bisa punya beberapa kategori error sekaligus — itulah sebabnya total kategori melebihi 20. Misalnya, satu image bisa bersamaan mengalami `B2_B3_confusion`, `B4_missed`, dan `false_positive`.

Ada narasi yang koheren dari distribusi ini: **error terbesar terjadi pada kelas-kelas yang berdekatan secara ordinal**. B2↔B3 dan B3↔B4 sama-sama berada di zona transisi visual yang halus. Sementara false positive mendominasi karena scene sawit secara inheren padat — dalam satu frame bisa ada puluhan buah, dan model cenderung over-predict di area dense.

## 3. Image tersulit

![Top-20 image tersulit by error score](figures/p3_error_by_image_score.png)

Beberapa image tersulit dan apa yang membuat mereka problematik:

| Image | Error Score | TP | Confusion | Missed | FP | Kategori |
|---|---:|---:|---:|---:|---:|---|
| `DAMIMAS_A21B_0838_5.jpg` | 21 | 0 | 0 | 6 | 9 | B4 missed, false positive |
| `DAMIMAS_A21B_0075_2.jpg` | 19 | 1 | 6 | 0 | 7 | B2/B3 + B3/B4 confusion |
| `DAMIMAS_A21B_0075_4.jpg` | 19 | 3 | 5 | 1 | 7 | B2/B3 + B3/B4 confusion |
| `DAMIMAS_A21B_0292_2.jpg` | 19 | 4 | 1 | 2 | 13 | B2/B3 + B4 missed + FP |
| `DAMIMAS_A21B_0107_2.jpg` | 17 | 1 | 4 | 2 | 5 | Semua kategori error |

Pola yang menarik: image-image dari tandan yang sama (prefix `0075`, `0838`, `0107`) cenderung muncul berulang di daftar tersulit. Ini mengindikasikan bahwa **kesulitan bukan hanya per-objek tapi per-scene** — beberapa tandan memang secara inheren lebih sulit karena komposisi pencahayaan, sudut pandang, atau densitas buah yang ekstrem.

Image `DAMIMAS_A21B_0838_5.jpg` layak disorot: error score-nya tertinggi (21) dengan nol true positive dan 6 ground truth yang seluruhnya missed. Ini kemungkinan scene di mana semua objek adalah B4 kecil yang tersembunyi, dan model justru memproduksi 9 false positive — artinya model "melihat sesuatu" di area itu tapi salah total dalam mengidentifikasinya.

## 4. Anatomi tiap kategori error

### `false_positive` — muncul di 20/20 image

Ini error paling universal. Setiap image sulit punya masalah ini, yang menandakan bahwa ini bukan kasus edge tapi perilaku sistemik. Penyebab utama:

- **Densitas objek tinggi** — tandan sawit bisa berisi puluhan buah dalam satu frame. Model yang agresif (confidence rendah) akan memproduksi banyak deteksi di area dense, sebagian besar tidak match dengan ground truth.
- **Background clutter** — daun, tangkai, dan bagian tandan lain yang secara tekstural mirip buah bisa memicu false detection.
- **Annotasi konservatif** — jika annotator hanya melabel buah yang jelas terlihat, sementara model mendeteksi buah yang partially occluded, ini tercatat sebagai false positive meskipun secara teknis model "benar".

### `B2_B3_confusion` — 13 image

Ini adalah confusion paling dominan, dan ini sudah diprediksi sejak Phase 0. B2 dan B3 berada di zona transisi warna: B2 mulai berubah dari hitam ke merah, sementara B3 masih full hitam. Dalam kondisi pencahayaan tertentu, perbedaan keduanya bisa sangat subtle — terutama saat cahaya matahari membuat B3 tampak agak kemerahan, atau saat shadow membuat B2 tampak lebih gelap dari seharusnya.

Ini bukan masalah yang bisa diselesaikan hanya dengan tuning hyperparameter. Pendekatan yang lebih promising: augmentasi warna yang agresif pada zona transisi, atau feature extraction yang lebih fokus pada tekstur permukaan (duri vs halus) ketimbang warna.

### `B4_missed` — 11 image

B4 adalah kelas terkecil secara fisik dan paling tersembunyi dalam tandan. Median bounding box area B4 hanya 0.0072 (normalized) — hampir separuh dari B1. Di resolusi 640, banyak B4 yang hanya menempati beberapa piksel, membuat deteksi sangat sulit.

Menariknya, threshold sweep menunjukkan B4 recall naik dari 0.38 (default) ke 0.54 di `conf=0.1`. Artinya model sebenarnya menghasilkan prediksi B4 yang benar, tapi dengan confidence rendah. Ini membuka dua jalur perbaikan: (1) meningkatkan resolusi input agar B4 punya lebih banyak pixel, atau (2) menambah training data B4 supaya model lebih "yakin" saat mendeteksinya.

### `B3_B4_confusion` — 10 image

Selain B2/B3, boundary B3/B4 juga bermasalah. Keduanya sama-sama gelap/hitam, dan perbedaan utamanya ada di ukuran dan posisi dalam tandan — fitur spasial yang tidak selalu konsisten antar scene. Ini memperkuat argumen bahwa task 4-kelas ini memang punya boundary visual yang sangat rapat, terutama di tiga kelas tengah-bawah (B2, B3, B4).

## 5. Hubungan dengan threshold sweep

Data [threshold_sweep.csv](threshold_sweep.csv) menunjukkan bahwa `conf=0.1` adalah operating point terbaik — di titik ini, precision dan recall seimbang di ~0.70 dan B4 recall mencapai 0.54. Menaikkan threshold ke 0.3+ mulai mengorbankan recall B4 secara signifikan.

Namun perlu ditekankan: threshold sweep memilih **operating point untuk deployment**, bukan menggantikan skor resmi yang dihitung pada `conf=0.001`. Skor di eval JSON tetap menjadi angka referensi untuk perbandingan antar eksperimen.

## 6. Kesimpulan dan arah perbaikan

Analisis error ini mengonsolidasikan temuan yang sudah terlihat sejak fase-fase awal ke dalam gambaran yang lebih jelas:

1. **False positive adalah masalah sistemik**, bukan edge case. Ini perlu didekati dari sisi post-processing (NMS tuning, confidence calibration) dan mungkin juga dari sisi arsitektur (deformable attention untuk menangani scene dense).

2. **Confusion B2/B3 adalah bottleneck diskriminatif utama**. Kedua kelas ini secara biologis berada di zona transisi yang sama, dan model belum punya fitur yang cukup untuk memisahkannya. Pendekatan berbasis warna saja tidak cukup — perlu eksplorasi fitur tekstural dan spatial.

3. **B4 recall bisa ditingkatkan** tanpa perubahan arsitektur drastis. Threshold sweep menunjukkan sinyal ada — tinggal diperkuat melalui data augmentation, resolusi lebih tinggi, atau focal attention pada small objects.

4. **Beberapa scene secara inheren sangat sulit** (tandan tertentu muncul berulang di top-20 hardest). Ini bisa menjadi basis untuk hard-example mining di iterasi selanjutnya.

Jika repo ini dilanjutkan ke eksperimen berikutnya, ketiga pola error di atas harus menjadi prioritas pertama — dan keberhasilannya bisa diukur langsung terhadap baseline yang sudah di-establish di sini.
