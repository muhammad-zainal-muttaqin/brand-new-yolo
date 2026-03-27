# Final Report

- Canonical protocol source: `https://github.com/muhammad-zainal-muttaqin/YOLOBench/blob/main/E0_Protocol_Flowchart.html`
- Final model: `yolo11m.pt`
- Resolution: `640`
- Optimized confidence threshold: `0.1`
- Precision: `0.4763`
- Recall: `0.5538`
- mAP50: `0.4654`
- mAP50-95: `0.2367`
- Confusion B2/B3: `None`
- All classes >= 70% AP50: `False`
- Decision bucket: **NEEDS WORK**
- Deploy check in this run: `deferred`

Per semantic mapping repo ini:
- `B1 = buah merah, besar, bulat, posisi paling bawah tandan; paling matang / ripe`
- `B2 = buah masih hitam namun mulai transisi ke merah, sudah besar dan bulat, posisi di atas B1`
- `B3 = buah full hitam, masih berduri, masih lonjong, posisi di atas B2`
- `B4 = buah paling kecil, paling dalam di batang/tandan, sulit terlihat, masih banyak duri, hitam sampai hijau, dan masih bisa berkembang lebih besar; paling belum matang`
