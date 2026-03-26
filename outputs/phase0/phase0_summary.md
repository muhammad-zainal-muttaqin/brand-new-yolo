# Phase 0 Summary

## Dataset validation

- Status: **ok**
- Total images: **3992**
- Total labels: **3992**
- Total instances: **17987**
- Split images: train `2764`, val `604`, test `624`
- Empty-label images: **83**
- Invalid label issues after self-healing: **0**

## Resolution sweep progress (valid runs: >=30 epoch)

| run_name | imgsz | epochs | seed | mAP50 | mAP50-95 | precision | recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| p0_yolo11n_640_s1_e30 | 640 | 30 | 1 | 0.5237 | 0.2538 | 0.4906 | 0.5864 |
| p0_yolo11n_1024_s1_e30 | 1024 | 30 | 1 | 0.5363 | 0.2571 | 0.4888 | 0.6016 |

## Provisional reading

- Seed 1 menunjukkan `1024` lebih tinggi daripada `640` sebesar **0.0126 mAP50** dan **0.0034 mAP50-95**.
- Kenaikan relatif pada `mAP50-95` saat ini sekitar **1.33%**.
- Keputusan resolusi **belum dikunci** karena seed 2 valid (>=30 epoch) belum selesai.
- Run 3-epoch sebelumnya diperlakukan sebagai smoke test, bukan evidence utama.
