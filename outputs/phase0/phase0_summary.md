# Phase 0 Summary

## Dataset validation

- Status: **ok**
- Total images: **3992**
- Total labels: **3992**
- Total instances: **17987**
- Split images: train `2764`, val `604`, test `624`
- Empty-label images: **83**
- Invalid label issues after self-healing: **0**

## Resolution sweep (valid runs: >=30 epoch aktual, patience 10)

| imgsz | seed | mAP50 | mAP50-95 | precision | recall |
|---:|---:|---:|---:|---:|---:|
| 640 | 1 | 0.5237 | 0.2538 | 0.4906 | 0.5864 |
| 1024 | 1 | 0.5363 | 0.2571 | 0.4888 | 0.6016 |
| 640 | 2 | 0.5245 | 0.2514 | 0.4923 | 0.5838 |
| 1024 | 2 | 0.5276 | 0.2589 | 0.4952 | 0.6004 |

### Mean per resolution

- `640`: mAP50 **0.5241**, mAP50-95 **0.2526**
- `1024`: mAP50 **0.5320**, mAP50-95 **0.2580**
- Relative gain `1024` vs `640` on mAP50-95: **2.15%**
- Sesuai aturan E0, gain ini masuk band **2–5%**, sehingga keputusan ditentukan dengan mempertimbangkan biaya compute/inference.
- Karena kenaikan relatif kecil tetapi biaya `1024` hampir 2x lebih berat, **resolusi kerja Phase 0 dipilih = `640`**.

## Learning curve @ 640

| fraction | mAP50 | mAP50-95 | precision | recall |
|---:|---:|---:|---:|---:|
| 0.25 | 0.4444 | 0.1984 | 0.4187 | 0.5758 |
| 0.50 | 0.4637 | 0.2202 | 0.4410 | 0.5791 |
| 0.75 | 0.5033 | 0.2444 | 0.4683 | 0.5906 |
| 1.00 | 0.5237 | 0.2538 | 0.4906 | 0.5864 |

### Reading learning curve

- `25% -> 50%`: +0.0217 mAP50-95
- `50% -> 75%`: +0.0243 mAP50-95
- `75% -> 100%`: +0.0093 mAP50-95
- Interpretasi: performa masih naik saat data bertambah, tetapi kenaikannya mulai mengecil di rentang akhir.
- Kesimpulan Phase 0: data **belum benar-benar saturasi**, namun **diminishing returns** sudah mulai terlihat mendekati 100%.
