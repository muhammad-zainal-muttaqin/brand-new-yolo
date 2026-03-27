---
license: other
license_name: proprietary
license_link: LICENSE
task_categories:
  - image-classification
  - object-detection
language:
  - id
tags:
  - agriculture
  - oil-palm
  - computer-vision
  - indonesia
  - yolo
size_categories:
  - 1K<n<10K
---

# Dataset Deteksi Tandan Buah Segar (TBS) Sawit — Format YOLO

Dataset ini berisi citra lapangan tandan buah segar kelapa sawit dari dua domain di Indonesia: **Damimas** (Blok A21B) dan **Lonsum** (Blok A21A). Format anotasinya mengikuti format YOLO untuk deteksi objek dengan 4 kelas kematangan: `B1`, `B2`, `B3`, dan `B4`.

Split yang dipublikasikan di sini mengikuti **split kanonik lokal** yang dipakai pada workflow `autoresearch` dan eksperimen repo ini.

## Ringkasan dataset

| Split | Jumlah gambar | Persentase |
|---|---:|---:|
| Train | 2,764 | 69.2% |
| Validation | 604 | 15.1% |
| Test | 624 | 15.6% |
| **Total** | **3,992** | **100%** |

Ringkasan cepat:

- total gambar: **3,992** JPG
- total label: **3,992** TXT
- total pohon/sekuens: **953**
- domain/varietas: **2** (`Damimas`, `Lonsum`)
- jumlah kelas: **4** (`B1`, `B2`, `B3`, `B4`)

## Arti kelas yang dipakai di repo ini

Repo ini memakai mapping berikut secara konsisten:

- `B1`: buah paling matang
- `B2`: transisi setelah `B1`
- `B3`: lebih mentah dari `B2`
- `B4`: paling belum matang

Panduan visual singkat:

- `B1`: merah, besar, bulat, paling bawah pada tandan
- `B2`: hitam mulai merah, besar, bulat, di atas `B1`
- `B3`: full hitam, berduri, lonjong, di atas `B2`
- `B4`: paling kecil, paling dalam di tandan, sulit terlihat, hitam sampai hijau

Untuk pembaca repo ini, mapping yang sama juga muncul di:

- [../README.md](../README.md)
- [../E0.md](../E0.md)
- [../GUIDE.md](../GUIDE.md)
- [../CONTEXT.md](../CONTEXT.md)

## Struktur folder

```text
Dataset-YOLO/
|-- images/
|   |-- train/
|   |-- val/
|   `-- test/
|-- labels/
|   |-- train/
|   |-- val/
|   `-- test/
|-- data.yaml
|-- LICENSE
`-- README.md
```

Setiap gambar memiliki file label pasangan dengan stem yang sama.

## Isi `data.yaml`

```yaml
path: .
train: images/train
val: images/val
test: images/test

nc: 4
names:
  0: B1
  1: B2
  2: B3
  3: B4
```

## Format label YOLO

Setiap file `.txt` berisi satu baris per bounding box:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Contoh:

```text
2 0.456789 0.345678 0.123456 0.234567
1 0.678901 0.789012 0.098765 0.087654
```

Keterangan kelas:

- `0 = B1`
- `1 = B2`
- `2 = B3`
- `3 = B4`

## Konvensi penamaan file

Format nama file:

```text
{VARIETAS}_{BLOK}_{NOMOR_POHON}_{SISI_FOTO}.jpg
```

Contoh:

```text
DAMIMAS_A21B_0001_1.jpg
DAMIMAS_A21B_0001_1.txt
```

Satu pohon difoto dari beberapa sisi. Semua sisi dari pohon yang sama ditempatkan pada split yang sama agar kebocoran antar split bisa dihindari.

## Cara pakai

### Ultralytics YOLO

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="data.yaml", epochs=100, imgsz=640)
model.val(data="data.yaml")
```

### Hugging Face `datasets`

```python
from datasets import load_dataset

dataset = load_dataset("ULM-DS-Lab/Dataset-Sawit-YOLO")
train_ds = dataset["train"]
val_ds = dataset["validation"]
test_ds = dataset["test"]
```

## Catatan penting

- split train/val/test pada repo ini **fixed**
- split ini diselaraskan dengan dataset lokal kanonik yang dipakai pada workflow autoresearch YOLO
- jika Anda ingin eksperimen dengan split lain, buat salinan dataset terpisah agar benchmark repo tetap bersih

## Dokumen repo yang relevan

- Peta baca repo: [../README.md](../README.md)
- Protokol E0: [../E0.md](../E0.md)
- Runbook operasional: [../GUIDE.md](../GUIDE.md)
- Audit dataset repo ini: [../outputs/phase0/eda_report.md](../outputs/phase0/eda_report.md)
- Ringkasan keputusan Phase 0: [../outputs/phase0/phase0_summary.md](../outputs/phase0/phase0_summary.md)

## Lisensi

Proprietary. Lihat file [LICENSE](LICENSE).
