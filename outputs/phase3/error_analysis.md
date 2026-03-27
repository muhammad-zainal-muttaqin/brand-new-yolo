# Error Analysis

Dokumen ini merangkum pola error utama pada run final `p3_final_yolo11m_640_s42_e60p15m60`. Untuk skor test-set resminya, buka [outputs/phase3/final_evaluation.md](final_evaluation.md). Untuk daftar gambar terberat secara mentah, buka [outputs/phase3/error_stratification.csv](error_stratification.csv).

## Sumber utama

Dokumen ini ditulis dari artefak berikut:

- [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json)
- [outputs/phase3/threshold_sweep.csv](threshold_sweep.csv)
- [outputs/phase3/error_stratification.csv](error_stratification.csv)
- [outputs/phase3/confusion_matrix.csv](confusion_matrix.csv)

## 1. Ringkasan cepat

- optimized confidence threshold: **`0.1`**
- confusion `B2/B3` di eval JSON: **`None`**
- confusion `B3/B4` di eval JSON: **`None`**
- all classes `AP50 >= 0.70`: **False**

Catatan penting: file eval JSON final tidak memuat confusion numerik `B2/B3` dan `B3/B4` sebagai angka agregat. Karena itu, pembacaan error praktis di dokumen ini bertumpu pada [outputs/phase3/error_stratification.csv](error_stratification.csv).

## 2. Pola error yang paling sering muncul

Berdasarkan [outputs/phase3/error_stratification.csv](error_stratification.csv), kategori error dominan adalah:

- `false_positive`: **20** image
- `B2_B3_confusion`: **13** image
- `B4_missed`: **11** image
- `B3_B4_confusion`: **10** image

Urutannya memberi pesan yang cukup jelas:

1. model masih terlalu sering membuat deteksi berlebih
2. boundary `B2/B3` tetap bermasalah
3. `B4` masih sering hilang
4. kedekatan visual `B3/B4` juga belum benar-benar teratasi

## 3. Contoh gambar yang paling berat

Baris teratas pada [outputs/phase3/error_stratification.csv](error_stratification.csv) menunjukkan beberapa contoh kasus yang paling sulit:

- `/workspace/Dataset-Sawit-YOLO/images/test/DAMIMAS_A21B_0838_5.jpg`
- `/workspace/Dataset-Sawit-YOLO/images/test/DAMIMAS_A21B_0075_2.jpg`
- `/workspace/Dataset-Sawit-YOLO/images/test/DAMIMAS_A21B_0292_2.jpg`
- `/workspace/Dataset-Sawit-YOLO/images/test/DAMIMAS_A21B_0107_2.jpg`
- `/workspace/Dataset-Sawit-YOLO/images/test/DAMIMAS_A21B_0382_3.jpg`

File ini layak dibuka lebih dulu untuk audit visual manual.

## 4. Cara membaca error ini

### `false_positive`

Ini adalah error paling dominan. Artinya model masih menghasilkan deteksi yang tidak didukung ground truth. Implikasinya bisa datang dari:

- threshold operasi yang terlalu longgar
- kepadatan objek yang membuat model terlalu agresif
- bbox atau label yang belum cukup bersih pada area sulit

### `B2_B3_confusion`

Ini menguatkan temuan dari fase-fase sebelumnya: `B2` dan `B3` masih sulit dipisahkan, bahkan setelah Phase 2 tuning selesai.

### `B4_missed`

Ini sejalan dengan fakta bahwa `B4` adalah kelas paling kecil dan paling sulit. Error ini lebih dekat ke masalah small-object recall daripada sekadar thresholding biasa.

### `B3_B4_confusion`

Selain `B2/B3`, kelas tetangga lain juga masih saling tertukar. Ini menunjukkan bahwa task 4-kelas memang punya boundary visual yang rapat.

## 5. Hubungan dengan threshold sweep

Data pada [outputs/phase3/threshold_sweep.csv](threshold_sweep.csv) menunjukkan kandidat threshold terbaik pada artefak ini ada di `conf=0.1`.

Namun angka threshold sweep dipakai untuk memilih **operating point**. Ia tidak menggantikan skor resmi run final pada [outputs/phase3/p3_final_yolo11m_640_s42_e60p15m60_eval.json](p3_final_yolo11m_640_s42_e60p15m60_eval.json).

## 6. Kesimpulan praktis

Analisis error final ini mengulang tiga pesan yang sama dari seluruh repo:

1. `false_positive` masih tinggi
2. confusion `B2/B3` belum selesai
3. `B4` recall masih lemah

Jika repo ini dibuka lagi untuk eksperimen lanjutan, tiga slice itu harus jadi prioritas pertama.
