# Phase 2 Summary

## Operational override aktif

- `Observed plateau/identical curves on Phase 2 Step 0a loss variants; repo override keeps baseline loss setup, reuses Phase 1B baseline for lr0=0.001, and continues with a reduced sweep over LR, batch, and augmentation.`
- `yolo11m.pt`: Step 0a plateau/identical, jadi baseline loss setup dikunci (`imbalance=none`, `ordinal=standard`) dan sisa branch Step 0b dilewati; sweep lanjut hanya untuk LR/batch/augmentation.
- `yolo11m.pt`: kandidat Step 1 `lr0=0.001` direuse dari baseline Phase 1B, jadi run `p2s1_lr001_*` dilewati.
- `yolo11m.pt`: kandidat Step 2 `batch=32` dilewati untuk menghemat 2 run dan menjaga sweep tetap fokus pada `8` vs `16`.
- `yolo11m.pt`: kandidat Step 3 `heavy` dilewati untuk menghemat 2 run; sweep augmentation dibatasi pada `light` vs `medium`.

- `yolo11m.pt` -> mAP50 `0.5329`, mAP50-95 `0.2578`, config: imbalance=`none`, ordinal=`standard`, lr0=`0.001`, batch=`16`, aug=`medium`, reverted=`True`
