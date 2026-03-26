# Phase 0 Dataset Audit

- Data YAML: `Dataset-YOLO/data.yaml`
- Dataset root: `/workspace/Dataset-Sawit-YOLO`
- Status: **ok**
- Total images: **3992**
- Total labels: **3992**
- Total instances: **17987**
- Split images: train `2764`, val `604`, test `624`
- Empty-label images: **83**
- Missing label images: **0**
- Orphan labels: **0**
- Invalid label issues: **0**
- Group overlap counts: `{'train__val': 0, 'train__test': 0, 'val__test': 0}`

## Class distribution

```
 class_id class_name  count    share
        0         B1   2177 0.121032
        1         B2   4073 0.226441
        2         B3   8295 0.461166
        3         B4   3442 0.191360
```

## BBox stats

```
 class_id class_name  count  median_width_px  median_height_px  median_area_norm  p10_width_px  p90_width_px  p10_height_px  p90_height_px  p10_area_norm  p90_area_norm
        0         B1   2177        124.99968         136.79232          0.014006      80.18880     199.12800       90.62528      215.10400       0.006153       0.033894
        1         B2   4073        109.09056         120.66304          0.010655      69.94080     178.17216       76.68864      193.46432       0.004504       0.026868
        2         B3   8295        105.31296         114.36800          0.009645      59.25888     176.63040       66.29632      189.10720       0.003417       0.026090
        3         B4   3442         93.60960          96.12032          0.007221      53.35776     147.63552       57.15840      151.10656       0.002634       0.017125
```

## Attention

- No blocking issue detected in automatic audit.
