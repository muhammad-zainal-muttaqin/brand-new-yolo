#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

import pandas as pd
import yaml
from PIL import Image

IMG_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def group_key(stem: str) -> str:
    parts = stem.split('_')
    if parts and parts[-1].isdigit():
        return '_'.join(parts[:-1])
    return stem


@dataclass
class LabelIssue:
    split: str
    image: str
    issue: str
    detail: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--outdir', default='outputs/phase0')
    return p.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open('r', encoding='utf-8-sig') as f:
        return yaml.safe_load(f)


def list_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_SUFFIXES])


def quantiles(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    s = sorted(values)
    def q(pos: float) -> float:
        idx = max(0, min(len(s) - 1, int(round((len(s) - 1) * pos))))
        return float(s[idx])
    return q(0.1), float(median(s)), q(0.9)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    cfg = load_yaml(data_path)
    root = Path(cfg['path'])
    names = cfg.get('names', {})
    if isinstance(names, list):
        class_map = {i: n for i, n in enumerate(names)}
    else:
        class_map = {int(k): v for k, v in names.items()}
    nc = int(cfg['nc'])

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    class_counts = Counter()
    split_image_counts = {}
    split_label_counts = {}
    split_instance_counts = Counter()
    empty_label_images = 0
    missing_label_images = []
    orphan_labels = []
    invalid_issues: list[LabelIssue] = []
    bbox_rows = []
    image_rows = []
    group_sets = defaultdict(set)
    file_sets = defaultdict(set)
    exact_overlap = {}
    group_overlap = {}

    for split in ['train', 'val', 'test']:
        img_dir = root / cfg[split]
        label_dir = root / 'labels' / split
        images = list_images(img_dir)
        labels = sorted(label_dir.glob('*.txt')) if label_dir.exists() else []
        split_image_counts[split] = len(images)
        split_label_counts[split] = len(labels)
        label_map = {p.stem: p for p in labels}
        image_stems = {p.stem for p in images}

        for p in images:
            file_sets[split].add(p.name)
            group_sets[split].add(group_key(p.stem))

        for stem, lp in label_map.items():
            if stem not in image_stems:
                orphan_labels.append({'split': split, 'label': str(lp)})

        for img_path in images:
            stem = img_path.stem
            label_path = label_map.get(stem)
            width = height = None
            try:
                with Image.open(img_path) as im:
                    width, height = im.size
            except Exception as e:
                invalid_issues.append(LabelIssue(split, img_path.name, 'image_open_error', str(e)))
                continue

            n_boxes = 0
            has_invalid = False
            if label_path is None:
                missing_label_images.append({'split': split, 'image': str(img_path)})
                has_invalid = True
            else:
                raw = label_path.read_text(encoding='utf-8').strip()
                if not raw:
                    empty_label_images += 1
                else:
                    for ln, line in enumerate(raw.splitlines(), start=1):
                        parts = line.strip().split()
                        if len(parts) != 5:
                            invalid_issues.append(LabelIssue(split, img_path.name, 'label_format', f'line {ln}: {line}'))
                            has_invalid = True
                            continue
                        try:
                            cls = int(float(parts[0]))
                            x, y, w, h = map(float, parts[1:])
                        except Exception:
                            invalid_issues.append(LabelIssue(split, img_path.name, 'label_parse', f'line {ln}: {line}'))
                            has_invalid = True
                            continue
                        if cls < 0 or cls >= nc:
                            invalid_issues.append(LabelIssue(split, img_path.name, 'class_out_of_range', f'line {ln}: {cls}'))
                            has_invalid = True
                            continue
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                            invalid_issues.append(LabelIssue(split, img_path.name, 'bbox_out_of_range', f'line {ln}: {line}'))
                            has_invalid = True
                            continue
                        x1 = x - w / 2
                        y1 = y - h / 2
                        x2 = x + w / 2
                        y2 = y + h / 2
                        if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
                            invalid_issues.append(LabelIssue(split, img_path.name, 'bbox_crosses_border', f'line {ln}: {line}'))
                            has_invalid = True
                            continue
                        n_boxes += 1
                        class_counts[cls] += 1
                        split_instance_counts[split] += 1
                        bbox_rows.append({
                            'split': split,
                            'image': img_path.name,
                            'class_id': cls,
                            'class_name': class_map.get(cls, str(cls)),
                            'x_center': x,
                            'y_center': y,
                            'width_norm': w,
                            'height_norm': h,
                            'width_px': w * width,
                            'height_px': h * height,
                            'area_norm': w * h,
                            'area_px': w * h * width * height,
                            'group_key': group_key(stem),
                        })

            image_rows.append({
                'split': split,
                'image': img_path.name,
                'width': width,
                'height': height,
                'boxes': n_boxes,
                'is_empty_label': int(label_path is not None and label_path.read_text(encoding='utf-8').strip() == ''),
                'has_invalid': int(has_invalid),
                'group_key': group_key(stem),
            })

    splits = ['train', 'val', 'test']
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            exact_overlap[f'{a}__{b}'] = sorted(file_sets[a] & file_sets[b])
            group_overlap[f'{a}__{b}'] = sorted(group_sets[a] & group_sets[b])

    bbox_df = pd.DataFrame(bbox_rows)
    img_df = pd.DataFrame(image_rows)
    issues_df = pd.DataFrame([vars(x) for x in invalid_issues]) if invalid_issues else pd.DataFrame(columns=['split', 'image', 'issue', 'detail'])

    class_dist_rows = []
    total_instances = sum(class_counts.values())
    for cls in range(nc):
        cnt = class_counts[cls]
        class_dist_rows.append({
            'class_id': cls,
            'class_name': class_map.get(cls, str(cls)),
            'count': cnt,
            'share': (cnt / total_instances) if total_instances else 0.0,
        })
    class_df = pd.DataFrame(class_dist_rows)

    bbox_stats_rows = []
    for cls in range(nc):
        subset = bbox_df[bbox_df['class_id'] == cls] if not bbox_df.empty else pd.DataFrame()
        width_p10, width_med, width_p90 = quantiles(subset['width_px'].tolist()) if not subset.empty else (None, None, None)
        height_p10, height_med, height_p90 = quantiles(subset['height_px'].tolist()) if not subset.empty else (None, None, None)
        area_p10, area_med, area_p90 = quantiles(subset['area_norm'].tolist()) if not subset.empty else (None, None, None)
        bbox_stats_rows.append({
            'class_id': cls,
            'class_name': class_map.get(cls, str(cls)),
            'count': int(len(subset)),
            'median_width_px': width_med,
            'median_height_px': height_med,
            'median_area_norm': area_med,
            'p10_width_px': width_p10,
            'p90_width_px': width_p90,
            'p10_height_px': height_p10,
            'p90_height_px': height_p90,
            'p10_area_norm': area_p10,
            'p90_area_norm': area_p90,
        })
    bbox_stats_df = pd.DataFrame(bbox_stats_rows)

    group_counts = {split: len(group_sets[split]) for split in splits}
    leakage_summary = {
        'exact_filename_overlap_counts': {k: len(v) for k, v in exact_overlap.items()},
        'group_overlap_counts': {k: len(v) for k, v in group_overlap.items()},
        'exact_filename_overlap_examples': {k: v[:20] for k, v in exact_overlap.items()},
        'group_overlap_examples': {k: v[:20] for k, v in group_overlap.items()},
    }

    status = 'ok'
    blockers = []
    if missing_label_images:
        blockers.append(f'missing_label_images={len(missing_label_images)}')
    if orphan_labels:
        blockers.append(f'orphan_labels={len(orphan_labels)}')
    if not issues_df.empty:
        blockers.append(f'invalid_label_issues={len(issues_df)}')
    if sum(leakage_summary['group_overlap_counts'].values()) > 0:
        blockers.append('group_leakage_detected')
    if blockers:
        status = 'needs_attention'

    audit = {
        'data_yaml': str(data_path),
        'dataset_root': str(root),
        'status': status,
        'split_image_counts': split_image_counts,
        'split_label_counts': split_label_counts,
        'split_instance_counts': dict(split_instance_counts),
        'group_counts': group_counts,
        'total_images': int(sum(split_image_counts.values())),
        'total_labels': int(sum(split_label_counts.values())),
        'total_instances': int(total_instances),
        'empty_label_images': int(empty_label_images),
        'missing_label_images': len(missing_label_images),
        'orphan_labels': len(orphan_labels),
        'invalid_label_issues': int(len(issues_df)),
        'blockers': blockers,
    }

    class_df.to_csv(outdir / 'class_distribution.csv', index=False)
    bbox_stats_df.to_csv(outdir / 'bbox_stats.csv', index=False)
    with (outdir / 'leakage_report.json').open('w', encoding='utf-8') as f:
        json.dump(leakage_summary, f, indent=2)
    with (outdir / 'dataset_audit.json').open('w', encoding='utf-8') as f:
        json.dump(audit, f, indent=2)

    summary_lines = [
        '# Phase 0 Dataset Audit',
        '',
        f'- Data YAML: `{data_path}`',
        f'- Dataset root: `{root}`',
        f'- Status: **{status}**',
        f'- Total images: **{audit["total_images"]}**',
        f'- Total labels: **{audit["total_labels"]}**',
        f'- Total instances: **{audit["total_instances"]}**',
        f'- Split images: train `{split_image_counts.get("train", 0)}`, val `{split_image_counts.get("val", 0)}`, test `{split_image_counts.get("test", 0)}`',
        f'- Empty-label images: **{empty_label_images}**',
        f'- Missing label images: **{len(missing_label_images)}**',
        f'- Orphan labels: **{len(orphan_labels)}**',
        f'- Invalid label issues: **{len(issues_df)}**',
        f'- Group overlap counts: `{leakage_summary["group_overlap_counts"]}`',
        '',
        '## Class distribution',
        '',
        '```',
        class_df.to_string(index=False),
        '```',
        '',
        '## BBox stats',
        '',
        '```',
        bbox_stats_df.to_string(index=False),
        '```',
    ]
    if blockers:
        summary_lines += ['', '## Attention', ''] + [f'- {x}' for x in blockers]
    else:
        summary_lines += ['', '## Attention', '', '- No blocking issue detected in automatic audit.']

    text = '\n'.join(summary_lines) + '\n'
    (outdir / 'eda_report.md').write_text(text, encoding='utf-8')
    (outdir / 'phase0_summary.md').write_text(text, encoding='utf-8')

    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
