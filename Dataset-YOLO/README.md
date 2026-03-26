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

# Dataset Deteksi Tandan Buah Segar (TBS) Kelapa Sawit - YOLO Format

Dataset citra lapangan pohon kelapa sawit dari dua varietas di Indonesia: **Damimas** (Blok A21B) dan **Lonsum** (Blok A21A). Dataset ini menggunakan format YOLO untuk deteksi objek dengan 4 kelas tingkat kematangan buah: `B1`, `B2`, `B3`, dan `B4`.

Split yang dipublikasikan di sini mengikuti **split kanonik lokal** yang dipakai pada workflow `autoresearch` untuk YOLO.

## Ringkasan Dataset

| Split | Jumlah Gambar | Persentase |
|---|---:|---:|
| Train | 2,764 | 69.2% |
| Validation | 604 | 15.1% |
| Test | 624 | 15.6% |
| **Total** | **3,992** | **100%** |

- Total gambar: 3,992 JPG
- Total label: 3,992 TXT (format YOLO)
- Total pohon/sekuens: 953
- Varietas: 2 (Damimas, Lonsum)
- Kelas: 4 (`B1`, `B2`, `B3`, `B4`)

## Struktur Folder

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

Semua gambar memiliki file label pasangan dengan nama stem yang sama.

## Konfigurasi data.yaml

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

## Format Label YOLO

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

## Konvensi Penamaan File

Format nama file:

```text
{VARIETAS}_{BLOK}_{NOMOR_POHON}_{SISI_FOTO}.jpg
```

Contoh:

```text
DAMIMAS_A21B_0001_1.jpg
DAMIMAS_A21B_0001_1.txt
```

Satu pohon difoto dari beberapa sisi, dan seluruh sisi dari pohon yang sama berada pada split yang sama.

## Penggunaan

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

## Catatan

- Split train/val/test pada repo ini bersifat fixed.
- Split ini diselaraskan dengan dataset lokal kanonik yang dipakai pada workflow autoresearch YOLO.
- Jika Anda ingin eksperimen dengan split lain, lakukan resplit di salinan dataset terpisah.

## Lisensi

Proprietary. Lihat file [LICENSE](LICENSE).
