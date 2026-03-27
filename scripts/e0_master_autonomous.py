#!/usr/bin/env python3
from __future__ import annotations

import base64
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
from ultralytics import YOLO

ROOT = Path('/workspace/brand-new-yolo')
GUIDE = ROOT / 'GUIDE.md'
LEDGER = ROOT / 'outputs/reports/run_ledger.csv'
LOCK_PATH = ROOT / 'outputs/phase1/locked_setup.yaml'
MASTER_LOG = ROOT / 'outputs/reports/master_autopilot.log'
MASTER_STATE = ROOT / 'outputs/reports/master_state.json'
SYNC_LOG = ROOT / 'outputs/reports/git_sync_log.md'
CANONICAL_SOURCE = 'https://github.com/muhammad-zainal-muttaqin/YOLOBench/blob/main/E0_Protocol_Flowchart.html'
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
PHASE1B_MODELS = [
    'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov9c.pt', 'yolov10n.pt', 'yolov10s.pt',
    'yolov10m.pt', 'yolo26n.pt', 'yolo26s.pt', 'yolo26m.pt', 'yolo11m.pt',
]
PHASE1B_SEEDS = [1, 2]
PHASE2_SEEDS = [1, 2]
PHASE2_CONFIRM_SEED = 3
PHASE3_FINAL_SEED = 42
PHASE1B_OVERRIDE_IGNORE_MAP70_STOP = True
PHASE3_DEPLOY_CHECK_DEFERRED = True
GUIDE_STATUS_START = '<!-- AUTOSTATUS:START -->'
GUIDE_STATUS_END = '<!-- AUTOSTATUS:END -->'
AUG_PROFILES = {
    'light': {
        'hsv_h': 0.01,
        'hsv_s': 0.4,
        'hsv_v': 0.25,
        'translate': 0.05,
        'scale': 0.25,
        'mosaic': 0.5,
        'close_mosaic': 5,
    },
    'medium': {
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'translate': 0.1,
        'scale': 0.5,
        'mosaic': 1.0,
        'close_mosaic': 10,
    },
    'heavy': {
        'hsv_h': 0.02,
        'hsv_s': 0.9,
        'hsv_v': 0.5,
        'translate': 0.15,
        'scale': 0.7,
        'mosaic': 1.0,
        'mixup': 0.1,
        'copy_paste': 0.1,
        'close_mosaic': 10,
    },
}


@dataclass
class RunSpec:
    phase: str
    name: str
    model: str
    imgsz: int
    epochs: int
    batch: int
    seed: int
    split: str = 'val'
    task: str = 'detect'
    patience: int = 10
    min_epochs: int = 30
    pretrained: bool = True
    data: str = 'Dataset-YOLO/data.yaml'
    workers: int = 8
    device: str = '0'
    project: str = 'runs/e0'
    optimizer: str | None = 'AdamW'
    lr0: float | None = None
    imbalance_strategy: str = 'none'
    ordinal_strategy: str = 'standard'
    focal_gamma: float = 1.5
    aug_profile: str = 'medium'


def utc_now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def log(msg: str) -> None:
    line = f'[{utc_now()}] {msg}'
    print(line, flush=True)
    MASTER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with MASTER_LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def write_state(data: dict[str, Any]) -> None:
    MASTER_STATE.parent.mkdir(parents=True, exist_ok=True)
    MASTER_STATE.write_text(json.dumps(data, indent=2), encoding='utf-8')


def sh(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    log('RUN ' + ' '.join(cmd))
    return subprocess.run(cmd, cwd=ROOT, check=check, text=True)


def sh_capture(cmd: str) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, shell=True, text=True).strip()


def pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def active_external_training_processes() -> list[str]:
    out = sh_capture("pgrep -af 'scripts/run_yolo_experiment.py' || true")
    lines = [line for line in out.splitlines() if line.strip()]
    filtered = []
    for line in lines:
        if str(os.getpid()) in line:
            continue
        if "pgrep -af 'scripts/run_yolo_experiment.py'" in line:
            continue
        filtered.append(line)
    return filtered


def wait_for_external_activity() -> None:
    wait_pids = [int(x) for x in os.getenv('E0_WAIT_FOR_PIDS', '').split(',') if x.strip()]
    while any(pid_exists(pid) for pid in wait_pids):
        log(f'waiting for external pids: {wait_pids}')
        time.sleep(30)
    while active_external_training_processes():
        log('waiting for external training processes to finish')
        time.sleep(30)


def cleanup_downloaded_root_weights() -> None:
    candidates = [
        'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov9c.pt', 'yolov10n.pt', 'yolov10s.pt',
        'yolov10m.pt', 'yolo26n.pt', 'yolo26s.pt', 'yolo26m.pt', 'yolo11m.pt', 'yolo11n.pt',
        'yolo11s.pt', 'yolo11n-cls.pt', 'yolo26n.pt',
    ]
    for name in candidates:
        p = ROOT / name
        if p.exists():
            p.unlink()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def ledger_lookup() -> dict[str, dict[str, str]]:
    return {row['run_name']: row for row in read_csv(LEDGER)}


def run_done(run_name: str) -> bool:
    row = ledger_lookup().get(run_name)
    return bool(row and row.get('status') == 'completed')


def summary_path(phase: str, run_name: str) -> Path:
    return ROOT / f'outputs/{phase}/{run_name}_summary.json'


def eval_path(phase: str, run_name: str) -> Path:
    return ROOT / f'outputs/{phase}/{run_name}_eval.json'


def read_json(path: Path) -> dict[str, Any]:
    path = restore_tracked_file(path)
    return json.loads(path.read_text(encoding='utf-8'))


def checkpoint(message: str) -> bool:
    cleanup_downloaded_root_weights()
    sh(['git', 'add', '--ignore-removal', 'GUIDE.md', 'E0.md', 'CONTEXT.md', 'outputs', 'runs', 'scripts'])
    diff = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT)
    if diff.returncode == 0:
        log(f'no changes to commit for checkpoint: {message}')
        return True
    sh(['git', 'commit', '-m', message])
    commit_hash = sh_capture('git rev-parse --short HEAD')
    with SYNC_LOG.open('a', encoding='utf-8') as f:
        f.write(f'- {utc_now()} | commit {commit_hash} | {message}\n')
    sh(['git', 'add', str(SYNC_LOG.relative_to(ROOT))])
    sh(['git', 'commit', '--amend', '--no-edit'])
    token = os.getenv('GITHUB_TOKEN', '')
    auth = base64.b64encode(f'x-access-token:{token}'.encode()).decode()
    for attempt in range(1, 4):
        r = subprocess.run(
            ['git', '-c', f'http.https://github.com/.extraheader=AUTHORIZATION: basic {auth}', 'push', 'origin', 'main'],
            cwd=ROOT,
            text=True,
        )
        if r.returncode == 0:
            log(f'push success on attempt {attempt}: {message}')
            return True
        log(f'push failed attempt {attempt}: {message}')
        time.sleep(10 * attempt)
    with SYNC_LOG.open('a', encoding='utf-8') as f:
        f.write(f'- {utc_now()} | PENDING SYNC | {message}\n')
    log(f'pending sync recorded: {message}')
    return False


def load_data_cfg(data_yaml: str) -> tuple[dict[str, Any], Path]:
    yaml_path = Path(data_yaml).resolve()
    cfg = yaml.safe_load(yaml_path.read_text(encoding='utf-8-sig'))
    return cfg, yaml_path


def dataset_root(cfg: dict[str, Any], yaml_path: Path) -> Path:
    root = Path(cfg.get('path', yaml_path.parent))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    return root


def resolve_entry(root: Path, entry: str) -> Path:
    path = Path(entry)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def iter_split_images(data_yaml: str, split: str) -> list[Path]:
    cfg, yaml_path = load_data_cfg(data_yaml)
    root = dataset_root(cfg, yaml_path)
    entry = cfg[split]
    target = resolve_entry(root, entry)
    if target.is_file():
        paths = []
        for line in target.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line:
                paths.append(Path(line))
        return paths
    if target.is_dir():
        return sorted([p for p in target.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES])
    raise FileNotFoundError(f'Cannot resolve split {split}: {target}')


def image_to_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if 'images' in parts:
        idx = parts.index('images')
        parts[idx] = 'labels'
        return Path(*parts).with_suffix('.txt')
    return Path(str(image_path).replace('/images/', '/labels/')).with_suffix('.txt')


def restore_tracked_file(path: str | Path) -> Path:
    p = Path(path)
    if p.exists():
        return p
    rel = p.resolve().relative_to(ROOT.resolve())
    sh(['git', 'checkout', 'HEAD', '--', str(rel)])
    return p


def load_gt_boxes(image_path: Path) -> tuple[list[dict[str, Any]], tuple[int, int]]:
    with Image.open(image_path) as img:
        width, height = img.size
    boxes: list[dict[str, Any]] = []
    label_path = image_to_label_path(image_path)
    if label_path.exists():
        for line in label_path.read_text(encoding='utf-8').splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
            x1 = (x - w / 2) * width
            y1 = (y - h / 2) * height
            x2 = (x + w / 2) * width
            y2 = (y + h / 2) * height
            boxes.append({
                'cls': cls,
                'xyxy': (x1, y1, x2, y2),
                'width': max(x2 - x1, 0.0),
                'height': max(y2 - y1, 0.0),
            })
    return boxes, (width, height)


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(inter_x2 - inter_x1, 0.0)
    ih = max(inter_y2 - inter_y1, 0.0)
    inter = iw * ih
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match(gt_boxes: list[dict[str, Any]], pred_boxes: list[dict[str, Any]], iou_thresh: float = 0.5) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]], list[int], list[int]]:
    same_cls_candidates: list[tuple[float, int, int]] = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            if gt['cls'] != pred['cls']:
                continue
            iou = box_iou(gt['xyxy'], pred['xyxy'])
            if iou >= iou_thresh:
                same_cls_candidates.append((iou, gi, pi))
    same_cls_candidates.sort(reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    tp_matches: list[tuple[int, int, float]] = []
    for iou, gi, pi in same_cls_candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        tp_matches.append((gi, pi, iou))

    confusion_candidates: list[tuple[float, int, int]] = []
    for gi, gt in enumerate(gt_boxes):
        if gi in matched_gt:
            continue
        for pi, pred in enumerate(pred_boxes):
            if pi in matched_pred:
                continue
            iou = box_iou(gt['xyxy'], pred['xyxy'])
            if iou >= iou_thresh:
                confusion_candidates.append((iou, gi, pi))
    confusion_candidates.sort(reverse=True)
    confusion_matches: list[tuple[int, int, float]] = []
    for iou, gi, pi in confusion_candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        confusion_matches.append((gi, pi, iou))

    missed_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    fp_pred = [i for i in range(len(pred_boxes)) if i not in matched_pred]
    return tp_matches, confusion_matches, missed_gt, fp_pred


def predict_boxes(model: YOLO, image_path: Path, imgsz: int, device: str, conf: float) -> list[dict[str, Any]]:
    result = model.predict(source=str(image_path), imgsz=imgsz, device=device, conf=conf, verbose=False)[0]
    preds: list[dict[str, Any]] = []
    if result.boxes is None:
        return preds
    xyxy = result.boxes.xyxy.cpu().tolist()
    cls = result.boxes.cls.cpu().tolist()
    confs = result.boxes.conf.cpu().tolist()
    for box, c, score in zip(xyxy, cls, confs):
        preds.append({'cls': int(c), 'conf': float(score), 'xyxy': tuple(float(v) for v in box)})
    return preds


def build_error_stratification(
    best_weight: str,
    data_yaml: str,
    split: str,
    imgsz: int,
    device: str,
    conf: float,
    limit: int,
    out_csv: Path,
    extra_fields: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    best_weight = str(restore_tracked_file(best_weight))
    model = YOLO(best_weight)
    names = {int(k): v for k, v in model.names.items()} if isinstance(model.names, dict) else dict(enumerate(model.names))
    rows: list[dict[str, Any]] = []
    for image_path in iter_split_images(data_yaml, split):
        gt_boxes, _ = load_gt_boxes(image_path)
        pred_boxes = predict_boxes(model, image_path, imgsz, device, conf)
        tp, confusions, missed_gt, fp_pred = greedy_match(gt_boxes, pred_boxes, iou_thresh=0.5)
        if not confusions and not missed_gt and not fp_pred:
            continue
        categories: set[str] = set()
        for gi in missed_gt:
            gt = gt_boxes[gi]
            if min(gt['width'], gt['height']) < 16:
                categories.add('small_object_missed')
            if names.get(gt['cls']) == 'B4':
                categories.add('B4_missed')
        for gi, pi, _ in confusions:
            true_name = names.get(gt_boxes[gi]['cls'], str(gt_boxes[gi]['cls']))
            pred_name = names.get(pred_boxes[pi]['cls'], str(pred_boxes[pi]['cls']))
            if {true_name, pred_name} == {'B2', 'B3'}:
                categories.add('B2_B3_confusion')
            if {true_name, pred_name} == {'B3', 'B4'}:
                categories.add('B3_B4_confusion')
        if fp_pred:
            categories.add('false_positive')
        if not categories:
            categories.add('manual_review')
        row = {
            'image_path': str(image_path),
            'tp': len(tp),
            'confusions': len(confusions),
            'missed_gt': len(missed_gt),
            'false_positive': len(fp_pred),
            'error_score': 2 * len(confusions) + 2 * len(missed_gt) + len(fp_pred),
            'categories': ';'.join(sorted(categories)),
        }
        if extra_fields:
            row.update(extra_fields)
        rows.append(row)
    rows.sort(key=lambda x: (x['error_score'], x['confusions'], x['missed_gt'], x['false_positive']), reverse=True)
    top_rows = rows[:limit]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(top_rows[0].keys()) if top_rows else list((extra_fields or {}).keys()) + ['image_path', 'tp', 'confusions', 'missed_gt', 'false_positive', 'error_score', 'categories']
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(top_rows)
    return top_rows


def confusion_rate(cm, idx_a: int | None, idx_b: int | None) -> float | None:
    if idx_a is None or idx_b is None:
        return None
    total = float(sum(cm[idx_a][:-1]) + sum(cm[idx_b][:-1]))
    if total <= 0:
        return None
    return float((cm[idx_a][idx_b] + cm[idx_b][idx_a]) / total)


def materialize_eval_snapshot(phase: str, run_name: str, best_weight: str, data: str, split: str, imgsz: int, batch: int, device: str) -> dict[str, Any]:
    out = eval_path(phase, run_name)
    if out.exists():
        return read_json(out)
    best_weight = str(restore_tracked_file(best_weight))
    model = YOLO(best_weight)
    metrics = model.val(data=data, split=split, imgsz=imgsz, batch=batch, device=device, workers=8, plots=False)
    names = {int(k): v for k, v in metrics.names.items()}
    per_class = []
    for idx in sorted(names):
        p, r, map50, map50_95 = metrics.box.class_result(idx)
        per_class.append({
            'class_idx': idx,
            'class_name': names[idx],
            'precision': float(p),
            'recall': float(r),
            'map50': float(map50),
            'map50_95': float(map50_95),
        })
    name_to_idx = {v: k for k, v in names.items()}
    cm = metrics.confusion_matrix.matrix
    snapshot = {
        'run_name': run_name,
        'split': split,
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
        'map50': float(metrics.box.map50),
        'map50_95': float(metrics.box.map),
        'per_class': per_class,
        'confusion_b2_b3': confusion_rate(cm, name_to_idx.get('B2'), name_to_idx.get('B3')),
        'confusion_b3_b4': confusion_rate(cm, name_to_idx.get('B3'), name_to_idx.get('B4')),
        'b4_recall': next((row['recall'] for row in per_class if row['class_name'] == 'B4'), None),
        'all_classes_ge_70_ap50': all(row['map50'] >= 0.70 for row in per_class),
    }
    out.write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
    return snapshot


def build_command(spec: RunSpec) -> list[str]:
    cmd = [
        sys.executable, 'scripts/run_yolo_experiment.py',
        '--phase', spec.phase,
        '--task', spec.task,
        '--model', spec.model,
        '--data', spec.data,
        '--imgsz', str(spec.imgsz),
        '--epochs', str(spec.epochs),
        '--batch', str(spec.batch),
        '--seed', str(spec.seed),
        '--project', spec.project,
        '--name', spec.name,
        '--split', spec.split,
        '--device', spec.device,
        '--workers', str(spec.workers),
        '--patience', str(spec.patience),
        '--min-epochs', str(spec.min_epochs),
        '--imbalance-strategy', spec.imbalance_strategy,
        '--ordinal-strategy', spec.ordinal_strategy,
        '--focal-gamma', str(spec.focal_gamma),
    ]
    if spec.pretrained:
        cmd.append('--pretrained')
    if spec.optimizer:
        cmd += ['--optimizer', spec.optimizer]
    if spec.lr0 is not None:
        cmd += ['--lr0', str(spec.lr0)]
    for key, value in AUG_PROFILES[spec.aug_profile].items():
        cmd += ['--' + key.replace('_', '-'), str(value)]
    return cmd


def run_experiment(spec: RunSpec) -> dict[str, Any]:
    if run_done(spec.name) and summary_path(spec.phase, spec.name).exists():
        log(f'skip completed run: {spec.name}')
    else:
        wait_for_external_activity()
        if run_done(spec.name) and summary_path(spec.phase, spec.name).exists():
            log(f'skip completed run after wait: {spec.name}')
        else:
            sh(build_command(spec))
            checkpoint(f'{spec.phase}: add {spec.name}')
    summary = read_json(summary_path(spec.phase, spec.name))
    materialize_eval_snapshot(spec.phase, spec.name, summary['best_weight'], spec.data, spec.split, spec.imgsz, spec.batch, spec.device)
    return summary


def _aggregate_summary_eval_pairs(run_pairs: list[tuple[str, str]]) -> dict[str, Any]:
    summaries = [read_json(summary_path(phase, name)) for phase, name in run_pairs]
    evals = [read_json(eval_path(phase, name)) for phase, name in run_pairs]
    run_names = [name for _, name in run_pairs]
    return {
        'run_names': run_names,
        'runs': len(run_names),
        'mean_map50': statistics.mean(float(x['map50']) for x in summaries),
        'std_map50': statistics.pstdev(float(x['map50']) for x in summaries) if len(run_names) > 1 else 0.0,
        'mean_map50_95': statistics.mean(float(x['map50_95']) for x in summaries),
        'std_map50_95': statistics.pstdev(float(x['map50_95']) for x in summaries) if len(run_names) > 1 else 0.0,
        'mean_precision': statistics.mean(float(x['precision']) for x in summaries),
        'mean_recall': statistics.mean(float(x['recall']) for x in summaries),
        'mean_confusion_b2_b3': statistics.mean(x['confusion_b2_b3'] for x in evals if x['confusion_b2_b3'] is not None) if any(x['confusion_b2_b3'] is not None for x in evals) else None,
        'mean_confusion_b3_b4': statistics.mean(x['confusion_b3_b4'] for x in evals if x['confusion_b3_b4'] is not None) if any(x['confusion_b3_b4'] is not None for x in evals) else None,
        'mean_b4_recall': statistics.mean(x['b4_recall'] for x in evals if x['b4_recall'] is not None) if any(x['b4_recall'] is not None for x in evals) else None,
    }


def aggregate_runs(phase: str, run_names: list[str]) -> dict[str, Any]:
    return _aggregate_summary_eval_pairs([(phase, name) for name in run_names])


def aggregate_mixed_runs(run_pairs: list[tuple[str, str]]) -> dict[str, Any]:
    return _aggregate_summary_eval_pairs(run_pairs)


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            row['mean_map50'],
            row['mean_map50_95'],
            -(row['mean_confusion_b2_b3'] if row['mean_confusion_b2_b3'] is not None else 1.0),
            row['mean_b4_recall'] if row['mean_b4_recall'] is not None else 0.0,
        ),
        reverse=True,
    )


def model_stem(model_name: str) -> str:
    return Path(model_name).stem


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        return
    fields = fieldnames or list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fields})


def read_lock() -> dict[str, Any]:
    if LOCK_PATH.exists():
        return yaml.safe_load(LOCK_PATH.read_text(encoding='utf-8')) or {}
    return {}


def write_lock(data: dict[str, Any]) -> None:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


def require_lock_keys(lock: dict[str, Any], keys: list[str], context: str) -> None:
    missing = [k for k in keys if k not in lock or lock[k] in (None, '', [])]
    if missing:
        raise RuntimeError(f'{context}: missing lock keys {missing} in {LOCK_PATH}')


def validate_phase2_lock(lock: dict[str, Any]) -> None:
    require_lock_keys(lock, ['phase0_locked', 'phase1a_locked', 'phase1b_locked'], 'phase2 lock validation')
    if lock['phase1a_locked'].get('pipeline') != 'one-stage':
        raise RuntimeError('phase2 lock validation: pipeline lock mismatch, expected one-stage')
    finalists = lock['phase1b_locked'].get('architecture_finalists') or []
    if not finalists:
        raise RuntimeError('phase2 lock validation: architecture_finalists empty')


def validate_phase3_lock(lock: dict[str, Any]) -> None:
    validate_phase2_lock(lock)
    require_lock_keys(lock, ['final_model', 'final_config'], 'phase3 lock validation')
    finalists = lock['phase1b_locked'].get('architecture_finalists') or []
    if lock['final_model'] not in finalists:
        raise RuntimeError('phase3 lock validation: final_model is not part of locked phase1 finalists')
    cfg = lock['final_config']
    for key in ['model', 'lr0', 'batch', 'imbalance_strategy', 'ordinal_strategy', 'aug_profile', 'imgsz']:
        if key not in cfg:
            raise RuntimeError(f'phase3 lock validation: final_config missing key {key}')
    if cfg['model'] != lock['final_model']:
        raise RuntimeError('phase3 lock validation: final_config.model mismatch with final_model lock')
    if int(cfg['imgsz']) != 640:
        raise RuntimeError('phase3 lock validation: imgsz changed after lock')


def update_guide_status(lines: list[str]) -> None:
    text = GUIDE.read_text(encoding='utf-8')
    if GUIDE_STATUS_START not in text or GUIDE_STATUS_END not in text:
        return
    before, rest = text.split(GUIDE_STATUS_START, 1)
    _, after = rest.split(GUIDE_STATUS_END, 1)
    replacement = GUIDE_STATUS_START + '\n' + '\n'.join(lines) + '\n' + GUIDE_STATUS_END
    GUIDE.write_text(before + replacement + after, encoding='utf-8')


def append_phase_summary(path: Path, heading: str, lines: list[str]) -> None:
    existing = path.read_text(encoding='utf-8') if path.exists() else '# Summary\n\n'
    block = '\n' + heading + '\n\n' + '\n'.join(lines) + '\n'
    path.write_text(existing + block, encoding='utf-8')


def phase1b_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for model in PHASE1B_MODELS:
        stem = model_stem(model)
        for seed in PHASE1B_SEEDS:
            specs.append(RunSpec(
                phase='phase1',
                name=f'p1bfc_{stem}_640_s{seed}_e30p10m30',
                model=model,
                imgsz=640,
                epochs=30,
                batch=16,
                seed=seed,
                split='val',
                patience=10,
                min_epochs=30,
                lr0=0.001,
                optimizer='AdamW',
                aug_profile='medium',
            ))
    return specs


def phase1b() -> dict[str, Any]:
    log('phase1b start')
    rows = []
    for spec in phase1b_specs():
        run_experiment(spec)
    per_class_rows = []
    for spec in phase1b_specs():
        ev = read_json(eval_path('phase1', spec.name))
        for row in ev['per_class']:
            per_class_rows.append({
                'run_name': spec.name,
                'model': spec.model,
                'seed': spec.seed,
                **row,
            })
    write_csv(ROOT / 'outputs/phase1/per_class_metrics.csv', per_class_rows)

    for model in PHASE1B_MODELS:
        stem = model_stem(model)
        run_names = [f'p1bfc_{stem}_640_s{seed}_e30p10m30' for seed in PHASE1B_SEEDS]
        agg = aggregate_runs('phase1', run_names)
        agg.update({'model': model, 'imgsz': 640, 'seeds': '1,2', 'lr0': 0.001, 'batch': 16, 'aug_profile': 'medium', 'patience': 10})
        rows.append(agg)
    ranked = rank_rows(rows)
    arch_rows = []
    for row in ranked:
        arch_rows.append({
            'model': row['model'],
            'imgsz': row['imgsz'],
            'seeds': row['seeds'],
            'lr0': row['lr0'],
            'batch': row['batch'],
            'aug_profile': row['aug_profile'],
            'patience': row['patience'],
            'mean_map50': row['mean_map50'],
            'std_map50': row['std_map50'],
            'mean_map50_95': row['mean_map50_95'],
            'std_map50_95': row['std_map50_95'],
            'mean_confusion_b2_b3': row['mean_confusion_b2_b3'],
            'mean_confusion_b3_b4': row['mean_confusion_b3_b4'],
            'mean_b4_recall': row['mean_b4_recall'],
        })
    write_csv(ROOT / 'outputs/phase1/architecture_benchmark.csv', arch_rows)
    top3_models = [row['model'] for row in ranked[:3]]
    finalists = [ranked[0]['model']]
    write_csv(ROOT / 'outputs/phase1/phase1b_top3.csv', [{
        'rank': i + 1,
        'model': row['model'],
        'mean_map50': row['mean_map50'],
        'mean_map50_95': row['mean_map50_95'],
        'mean_confusion_b2_b3': row['mean_confusion_b2_b3'],
        'mean_b4_recall': row['mean_b4_recall'],
    } for i, row in enumerate(ranked[:3])])

    error_rows: list[dict[str, Any]] = []
    for model in top3_models:
        stem = model_stem(model)
        run_names = [f'p1bfc_{stem}_640_s{seed}_e30p10m30' for seed in PHASE1B_SEEDS]
        best_run = max(run_names, key=lambda n: float(read_json(summary_path('phase1', n))['map50']))
        best_weight = read_json(summary_path('phase1', best_run))['best_weight']
        temp_csv = ROOT / f'outputs/phase1/{best_run}_error_temp.csv'
        rows_top = build_error_stratification(
            best_weight=best_weight,
            data_yaml='Dataset-YOLO/data.yaml',
            split='val',
            imgsz=640,
            device='0',
            conf=0.25,
            limit=20,
            out_csv=temp_csv,
            extra_fields={'architecture': model, 'best_run': best_run},
        )
        error_rows.extend(rows_top)
    write_csv(ROOT / 'outputs/phase1/phase1b_error_stratification.csv', error_rows)

    best_row = ranked[0]
    gate_pass = best_row['mean_map50'] >= 0.70
    lock = read_lock()
    lock.update({
        'protocol_source': CANONICAL_SOURCE,
        'dataset_label_semantics': {
            'maturity_order': ['B1', 'B2', 'B3', 'B4'],
            'direction': 'most_mature_to_least_mature',
            'B1': 'buah merah, besar, bulat, posisi paling bawah tandan; paling matang / ripe',
            'B2': 'buah masih hitam namun mulai transisi ke merah, sudah besar dan bulat, posisi di atas B1',
            'B3': 'buah full hitam, masih berduri, masih lonjong, posisi di atas B2',
            'B4': 'buah paling kecil, paling dalam di batang/tandan, sulit terlihat, masih banyak duri, hitam sampai hijau, masih bisa berkembang lebih besar; paling mengkal / belum matang',
        },
        'phase0_locked': {'imgsz': 640, 'split': 'tree-grouped class-stratified'},
        'phase1a_locked': {'pipeline': 'one-stage'},
        'phase1b_locked': {
            'lock_stage': 'phase1b_single_best_locked',
            'architecture_finalists': finalists,
            'selected_model': finalists[0],
            'selection_policy': 'single_best_only_for_phase2',
            'baseline': {
                'lr0': 0.001,
                'batch': 16,
                'patience': 10,
                'epochs': 30,
                'min_epochs': 30,
                'aug_profile': 'medium',
            },
            'phase1b_gate_map50_pass': gate_pass,
            'phase1b_gate_override_continue': PHASE1B_OVERRIDE_IGNORE_MAP70_STOP,
        },
        'final_model': lock.get('final_model'),
        'final_config': lock.get('final_config'),
    })
    write_lock(lock)

    phase1_summary_lines = [
        f'- Phase 2 locked single best model: `{finalists[0]}`',
        f'- Reference top-3 ranking saved to `outputs/phase1/phase1b_top3.csv`: `{", ".join(top3_models)}`',
        f'- Best Phase 1B mean mAP50: `{best_row["mean_map50"]:.4f}`',
        f'- Best Phase 1B mean mAP50-95: `{best_row["mean_map50_95"]:.4f}`',
        f'- Gate canonical `mAP50 >= 0.70`: `{gate_pass}`',
        f'- Local override continue despite gate: `{PHASE1B_OVERRIDE_IGNORE_MAP70_STOP}`',
    ]
    append_phase_summary(ROOT / 'outputs/phase1/phase1_summary.md', '## Phase 1B — Canonical Flowchart-Synced Sweep', phase1_summary_lines)
    update_guide_status([
        '- Canonical source synced: `E0.md` mengikuti flowchart YOLOBench.',
        '- Phase 1B canonical flowchart-synced selesai untuk roster 11 model × 2 seeds.',
        f'- Model tunggal untuk Phase 2 dikunci di `outputs/phase1/locked_setup.yaml`: `{finalists[0]}`.',
        f'- Ranking referensi top-3 tetap disimpan di `outputs/phase1/phase1b_top3.csv`: `{", ".join(top3_models)}`.',
        f'- Gate canonical `mAP50 >= 70%` tercatat sebagai `{gate_pass}`, tetapi override lokal repo tetap lanjut = `{PHASE1B_OVERRIDE_IGNORE_MAP70_STOP}`.',
    ])
    checkpoint('phase1b canonical sync complete')
    return lock


def summarize_group_rows(rows: list[dict[str, Any]], extra: dict[str, Any]) -> dict[str, Any]:
    ranked = rank_rows(rows)
    best = ranked[0]
    out = dict(extra)
    out.update(best)
    return out


def phase2_run_name(prefix: str, token: str, model: str, seed: int) -> str:
    return f'{prefix}_{token}_{model_stem(model)}_640_s{seed}_e30p10m30'


def phase2_step_runs(prefix: str, token: str, model: str, seeds: list[int], **kwargs) -> list[str]:
    run_names = []
    for seed in seeds:
        spec = RunSpec(
            phase='phase2',
            name=phase2_run_name(prefix, token, model, seed),
            model=model,
            imgsz=640,
            epochs=30,
            batch=kwargs['batch'],
            seed=seed,
            split='val',
            patience=10,
            min_epochs=30,
            lr0=kwargs['lr0'],
            optimizer='AdamW',
            imbalance_strategy=kwargs['imbalance_strategy'],
            ordinal_strategy=kwargs['ordinal_strategy'],
            aug_profile=kwargs['aug_profile'],
        )
        run_experiment(spec)
        run_names.append(spec.name)
    return run_names


def aggregate_phase2_option(model: str, phase_prefix: str, token: str, seeds: list[int], **kwargs) -> dict[str, Any]:
    run_names = phase2_step_runs(phase_prefix, token, model, seeds, **kwargs)
    agg = aggregate_runs('phase2', run_names)
    agg.update({'model': model, 'option': token})
    return agg


def phase2() -> dict[str, Any]:
    log('phase2 start')
    lock = read_lock()
    validate_phase2_lock(lock)
    finalists: list[str] = lock['phase1b_locked']['architecture_finalists']
    baseline = lock['phase1b_locked']['baseline']
    imbalance_rows: list[dict[str, Any]] = []
    ordinal_rows: list[dict[str, Any]] = []
    lr_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    aug_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []

    for model in finalists:
        base_phase1 = aggregate_runs('phase1', [f'p1bfc_{model_stem(model)}_640_s{seed}_e30p10m30' for seed in PHASE1B_SEEDS])
        current = {
            'lr0': float(baseline['lr0']),
            'batch': int(baseline['batch']),
            'imbalance_strategy': 'none',
            'ordinal_strategy': 'standard',
            'aug_profile': baseline['aug_profile'],
        }

        step0a_candidates = []
        for token, imbalance in [('none', 'none'), ('class_weighted', 'class_weighted'), ('focal15', 'focal')]:
            agg = aggregate_phase2_option(model, 'p2s0a', token, PHASE2_SEEDS, **current | {'imbalance_strategy': imbalance})
            agg['imbalance_strategy'] = imbalance
            step0a_candidates.append(agg)
            imbalance_rows.append(agg)
        best0a = rank_rows(step0a_candidates)[0]
        current['imbalance_strategy'] = best0a['imbalance_strategy']

        step0b_tokens = [('standard', 'standard'), ('ordinal', 'ordinal_weighted')]
        step0b_candidates = []
        for token, ordinal in step0b_tokens:
            agg = aggregate_phase2_option(model, 'p2s0b', token, PHASE2_SEEDS, **current | {'ordinal_strategy': ordinal})
            agg['ordinal_strategy'] = ordinal
            step0b_candidates.append(agg)
            ordinal_rows.append(agg)
        best0b = rank_rows(step0b_candidates)[0]
        current['ordinal_strategy'] = best0b['ordinal_strategy']

        step1_candidates = []
        for token, lr in [('lr0005', 0.0005), ('lr001', 0.001), ('lr002', 0.002)]:
            agg = aggregate_phase2_option(model, 'p2s1', token, PHASE2_SEEDS, **current | {'lr0': lr})
            agg['lr0'] = lr
            step1_candidates.append(agg)
            lr_rows.append(agg)
        best1 = rank_rows(step1_candidates)[0]
        current['lr0'] = float(best1['lr0'])

        step2_candidates = []
        for token, batch in [('bs8', 8), ('bs16', 16), ('bs32', 32)]:
            agg = aggregate_phase2_option(model, 'p2s2', token, PHASE2_SEEDS, **current | {'batch': batch})
            agg['batch'] = batch
            step2_candidates.append(agg)
            batch_rows.append(agg)
        best2 = rank_rows(step2_candidates)[0]
        current['batch'] = int(best2['batch'])

        step3_candidates = []
        for token in ['light', 'medium', 'heavy']:
            agg = aggregate_phase2_option(model, 'p2s3', token, PHASE2_SEEDS, **current | {'aug_profile': token})
            agg['aug_profile'] = token
            step3_candidates.append(agg)
            aug_rows.append(agg)
        best3 = rank_rows(step3_candidates)[0]
        current['aug_profile'] = best3['aug_profile']

        tuned_mean_map50 = float(best3['mean_map50'])
        baseline_mean_map50 = float(base_phase1['mean_map50'])
        reverted = (tuned_mean_map50 - baseline_mean_map50) < 0.01
        if reverted:
            current = {
                'lr0': float(baseline['lr0']),
                'batch': int(baseline['batch']),
                'imbalance_strategy': 'none',
                'ordinal_strategy': 'standard',
                'aug_profile': baseline['aug_profile'],
            }

        confirm_name = f'p2confirm_{model_stem(model)}_640_s{PHASE2_CONFIRM_SEED}_e30p10m30'
        confirm_spec = RunSpec(
            phase='phase2',
            name=confirm_name,
            model=model,
            imgsz=640,
            epochs=30,
            batch=current['batch'],
            seed=PHASE2_CONFIRM_SEED,
            split='val',
            patience=10,
            min_epochs=30,
            lr0=current['lr0'],
            optimizer='AdamW',
            imbalance_strategy=current['imbalance_strategy'],
            ordinal_strategy=current['ordinal_strategy'],
            aug_profile=current['aug_profile'],
        )
        run_experiment(confirm_spec)

        if reverted:
            final_pairs = [(
                'phase1', f'p1bfc_{model_stem(model)}_640_s{seed}_e30p10m30'
            ) for seed in PHASE1B_SEEDS] + [('phase2', confirm_name)]
            final_agg = aggregate_mixed_runs(final_pairs)
            final_row = {
                'model': model,
                'final_source': 'phase1_baseline_reverted',
                'reverted_to_phase1_baseline': True,
                'baseline_mean_map50': baseline_mean_map50,
                'tuned_mean_map50': tuned_mean_map50,
                'mean_map50': final_agg['mean_map50'],
                'mean_map50_95': final_agg['mean_map50_95'],
                'lr0': current['lr0'],
                'batch': current['batch'],
                'imbalance_strategy': current['imbalance_strategy'],
                'ordinal_strategy': current['ordinal_strategy'],
                'aug_profile': current['aug_profile'],
                'confirmation_run': confirm_name,
            }
        else:
            final_prefix = phase2_run_name('p2s3', current['aug_profile'], model, 1).rsplit('_s1_', 1)[0]
            final_run_names = [phase2_run_name('p2s3', current['aug_profile'], model, seed) for seed in PHASE2_SEEDS] + [confirm_name]
            final_agg = aggregate_runs('phase2', final_run_names)
            final_row = {
                'model': model,
                'final_source': final_prefix,
                'reverted_to_phase1_baseline': False,
                'baseline_mean_map50': baseline_mean_map50,
                'tuned_mean_map50': tuned_mean_map50,
                'mean_map50': final_agg['mean_map50'],
                'mean_map50_95': final_agg['mean_map50_95'],
                'lr0': current['lr0'],
                'batch': current['batch'],
                'imbalance_strategy': current['imbalance_strategy'],
                'ordinal_strategy': current['ordinal_strategy'],
                'aug_profile': current['aug_profile'],
                'confirmation_run': confirm_name,
            }
        final_rows.append(final_row)

    write_csv(ROOT / 'outputs/phase2/imbalance_sweep.csv', imbalance_rows)
    write_csv(ROOT / 'outputs/phase2/ordinal_sweep.csv', ordinal_rows)
    write_csv(ROOT / 'outputs/phase2/lr_sweep.csv', lr_rows)
    write_csv(ROOT / 'outputs/phase2/batch_sweep.csv', batch_rows)
    write_csv(ROOT / 'outputs/phase2/aug_sweep.csv', aug_rows)
    ranked_final = sorted(final_rows, key=lambda r: (r['mean_map50'], r['mean_map50_95']), reverse=True)
    write_csv(ROOT / 'outputs/phase2/tuning_results.csv', ranked_final)

    best_final = ranked_final[0]
    final_hparams = {
        'model': best_final['model'],
        'lr0': float(best_final['lr0']),
        'batch': int(best_final['batch']),
        'imbalance_strategy': best_final['imbalance_strategy'],
        'ordinal_strategy': best_final['ordinal_strategy'],
        'aug_profile': best_final['aug_profile'],
        'patience': 10,
        'epochs': 30,
        'min_epochs': 30,
        'imgsz': 640,
    }
    (ROOT / 'outputs/phase2').mkdir(parents=True, exist_ok=True)
    (ROOT / 'outputs/phase2/final_hparams.yaml').write_text(yaml.safe_dump(final_hparams, sort_keys=False), encoding='utf-8')
    phase2_lines = ['# Phase 2 Summary', '']
    for row in ranked_final:
        phase2_lines.append(
            "- "
            f"`{row['model']}` -> "
            f"mAP50 `{row['mean_map50']:.4f}`, "
            f"mAP50-95 `{row['mean_map50_95']:.4f}`, "
            f"config: imbalance=`{row['imbalance_strategy']}`, "
            f"ordinal=`{row['ordinal_strategy']}`, "
            f"lr0=`{row['lr0']}`, batch=`{row['batch']}`, "
            f"aug=`{row['aug_profile']}`, reverted=`{row['reverted_to_phase1_baseline']}`"
        )
    (ROOT / 'outputs/phase2/phase2_summary.md').write_text('\n'.join(phase2_lines) + '\n', encoding='utf-8')

    lock = read_lock()
    validate_phase2_lock(lock)
    if best_final['model'] not in lock['phase1b_locked']['architecture_finalists']:
        raise RuntimeError('phase2 final selection is outside locked finalists')
    lock['phase1b_locked']['lock_stage'] = 'phase2_final_model_locked'
    lock['final_model'] = best_final['model']
    lock['final_config'] = final_hparams
    write_lock(lock)
    update_guide_status([
        '- Canonical source synced: `E0.md` mengikuti flowchart YOLOBench.',
        '- Phase 1B canonical selesai dan finalis Phase 2 sudah terkunci.',
        f"- Phase 2 selesai. Model final untuk Phase 3: `{best_final['model']}`.",
        f"- Final config Phase 2 ditulis ke `outputs/phase1/locked_setup.yaml` dan `outputs/phase2/final_hparams.yaml`.",
    ])
    checkpoint('phase2 canonical sync complete')
    return lock


def create_phase3_data_yaml() -> Path:
    cfg, yaml_path = load_data_cfg('Dataset-YOLO/data.yaml')
    root = dataset_root(cfg, yaml_path)
    outdir = ROOT / 'outputs/phase3'
    outdir.mkdir(parents=True, exist_ok=True)
    trainval_txt = outdir / 'trainval.txt'
    with trainval_txt.open('w', encoding='utf-8') as f:
        for split in ['train', 'val']:
            for img in iter_split_images('Dataset-YOLO/data.yaml', split):
                f.write(str(img) + '\n')
    final_cfg = {
        'path': str(root),
        'train': str(trainval_txt),
        'val': 'images/val',
        'test': 'images/test',
        'nc': int(cfg['nc']),
        'names': cfg['names'],
    }
    final_yaml = outdir / 'final_data.yaml'
    final_yaml.write_text(yaml.safe_dump(final_cfg, sort_keys=False), encoding='utf-8')
    return final_yaml


def threshold_sweep(best_weight: str, data_yaml: str, imgsz: int, batch: int, device: str) -> tuple[list[dict[str, Any]], float]:
    model = YOLO(best_weight)
    rows: list[dict[str, Any]] = []
    best_tuple = None
    best_conf = 0.25
    for conf in [0.1, 0.2, 0.3, 0.4, 0.5]:
        metrics = model.val(data=data_yaml, split='val', imgsz=imgsz, batch=batch, device=device, workers=8, plots=False, conf=conf)
        names = {int(k): v for k, v in metrics.names.items()}
        name_to_idx = {v: k for k, v in names.items()}
        cm = metrics.confusion_matrix.matrix
        b2b3 = confusion_rate(cm, name_to_idx.get('B2'), name_to_idx.get('B3'))
        b4_recall = float(metrics.box.class_result(name_to_idx['B4'])[1]) if 'B4' in name_to_idx else None
        row = {
            'conf': conf,
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'map50': float(metrics.box.map50),
            'map50_95': float(metrics.box.map),
            'confusion_b2_b3': b2b3,
            'b4_recall': b4_recall,
        }
        rows.append(row)
        score_tuple = (row['map50'], -(row['confusion_b2_b3'] if row['confusion_b2_b3'] is not None else 1.0), row['b4_recall'] if row['b4_recall'] is not None else 0.0)
        if best_tuple is None or score_tuple > best_tuple:
            best_tuple = score_tuple
            best_conf = conf
    return rows, best_conf


def final_decision_bucket(map50: float, conf_b2_b3: float | None, all_classes_ge70: bool) -> str:
    if map50 >= 0.90 and (conf_b2_b3 is not None and conf_b2_b3 < 0.20) and all_classes_ge70:
        return 'EXCELLENT'
    if 0.85 <= map50 < 0.90 and (conf_b2_b3 is not None and 0.20 <= conf_b2_b3 < 0.30) and all_classes_ge70:
        return 'GOOD'
    if 0.80 <= map50 < 0.85 and (conf_b2_b3 is not None and conf_b2_b3 >= 0.30) and all_classes_ge70:
        return 'ACCEPTABLE'
    if 0.75 <= map50 < 0.80 or not all_classes_ge70:
        return 'NEEDS WORK'
    return 'INSUFFICIENT'


def write_root_readme_and_checkpoint() -> None:
    sh([sys.executable, 'scripts/write_root_readme.py'])
    checkpoint('write final root README report')


def phase3() -> None:
    log('phase3 start')
    lock = read_lock()
    validate_phase3_lock(lock)
    final_model = lock['final_model']
    final_cfg = lock['final_config']
    data_yaml = create_phase3_data_yaml()
    final_name = f'p3_final_{model_stem(final_model)}_640_s{PHASE3_FINAL_SEED}_e120p999m120'
    spec = RunSpec(
        phase='phase3',
        name=final_name,
        model=final_model,
        imgsz=640,
        epochs=120,
        batch=int(final_cfg['batch']),
        seed=PHASE3_FINAL_SEED,
        split='test',
        patience=999,
        min_epochs=120,
        data=str(data_yaml),
        lr0=float(final_cfg['lr0']),
        optimizer='AdamW',
        imbalance_strategy=final_cfg['imbalance_strategy'],
        ordinal_strategy=final_cfg['ordinal_strategy'],
        aug_profile=final_cfg['aug_profile'],
    )
    summary = run_experiment(spec)
    materialize_eval_snapshot('phase3', final_name, summary['best_weight'], str(data_yaml), 'test', 640, int(final_cfg['batch']), '0')
    sweep_rows, best_conf = threshold_sweep(summary['best_weight'], str(data_yaml), 640, int(final_cfg['batch']), '0')
    write_csv(ROOT / 'outputs/phase3/threshold_sweep.csv', sweep_rows)

    model = YOLO(summary['best_weight'])
    metrics = model.val(data=str(data_yaml), split='test', imgsz=640, batch=int(final_cfg['batch']), device='0', workers=8, plots=False, conf=best_conf)
    names = {int(k): v for k, v in metrics.names.items()}
    name_to_idx = {v: k for k, v in names.items()}
    cm = metrics.confusion_matrix.matrix
    conf_b2_b3 = confusion_rate(cm, name_to_idx.get('B2'), name_to_idx.get('B3'))
    per_class = []
    for idx in sorted(names):
        p, r, map50, map50_95 = metrics.box.class_result(idx)
        per_class.append({'class_name': names[idx], 'precision': float(p), 'recall': float(r), 'map50': float(map50), 'map50_95': float(map50_95)})
    all_classes_ge70 = all(row['map50'] >= 0.70 for row in per_class)
    decision = final_decision_bucket(float(metrics.box.map50), conf_b2_b3, all_classes_ge70)

    write_csv(ROOT / 'outputs/phase3/final_metrics.csv', [
        {'metric': 'optimized_conf', 'value': best_conf},
        {'metric': 'precision', 'value': float(metrics.box.mp)},
        {'metric': 'recall', 'value': float(metrics.box.mr)},
        {'metric': 'map50', 'value': float(metrics.box.map50)},
        {'metric': 'map50_95', 'value': float(metrics.box.map)},
        {'metric': 'confusion_b2_b3', 'value': conf_b2_b3},
        {'metric': 'all_classes_ge70_ap50', 'value': all_classes_ge70},
    ], fieldnames=['metric', 'value'])

    cm_rows = []
    ordered_names = [names[i] for i in sorted(names)] + ['background']
    for i, true_name in enumerate(ordered_names):
        row = {'true_class': true_name}
        for j, pred_name in enumerate(ordered_names):
            row[pred_name] = float(cm[i][j])
        cm_rows.append(row)
    write_csv(ROOT / 'outputs/phase3/confusion_matrix.csv', cm_rows)

    deploy_lines = ['# Deploy Check', '']
    if PHASE3_DEPLOY_CHECK_DEFERRED:
        deploy_lines.append('- Status: **deferred by repo override**.')
        deploy_lines.append('- TFLite export: `skipped for now`')
        deploy_lines.append('- TFLite INT8 export: `skipped for now`')
        deploy_lines.append('- Rationale: amankan final `best.pt` dan validasi metrik eksperimen lebih dulu; konversi deploy boleh dilakukan belakangan sebagai langkah engineering terpisah.')
        deploy_lines.append('- Important: jika nanti dikonversi ke TFLite / INT8 / format lain, akurasi, ukuran, latency, dan kompatibilitas hardware **wajib divalidasi ulang** pada artefak hasil konversi itu.')
    else:
        try:
            export_fp = model.export(format='tflite', imgsz=640)
            export_fp = Path(export_fp)
            deploy_lines.append(f'- TFLite export: `{export_fp}`')
            deploy_lines.append(f'- TFLite size MB: `{export_fp.stat().st_size / (1024*1024):.2f}`')
        except Exception as e:
            deploy_lines.append(f'- TFLite export failed: `{type(e).__name__}: {e}`')
        try:
            export_int8 = model.export(format='tflite', imgsz=640, int8=True, data='Dataset-YOLO/data.yaml')
            export_int8 = Path(export_int8)
            deploy_lines.append(f'- TFLite INT8 export: `{export_int8}`')
            deploy_lines.append(f'- TFLite INT8 size MB: `{export_int8.stat().st_size / (1024*1024):.2f}`')
        except Exception as e:
            deploy_lines.append(f'- TFLite INT8 export failed: `{type(e).__name__}: {e}`')
    deploy_lines.append(f'- Best weight size MB: `{Path(summary["best_weight"]).stat().st_size / (1024*1024):.2f}`')
    deploy_lines.append('- Inference viability nyata di tablet tetap perlu pengujian hardware terpisah bila device tersedia.')
    (ROOT / 'outputs/phase3/deploy_check.md').write_text('\n'.join(deploy_lines) + '\n', encoding='utf-8')

    error_rows = build_error_stratification(
        best_weight=summary['best_weight'],
        data_yaml=str(data_yaml),
        split='test',
        imgsz=640,
        device='0',
        conf=best_conf,
        limit=20,
        out_csv=ROOT / 'outputs/phase3/error_stratification.csv',
        extra_fields={'run_name': final_name},
    )
    error_lines = ['# Error Analysis', '']
    error_lines.append(f'- Optimized confidence threshold: `{best_conf}`')
    error_lines.append(f'- Confusion B2/B3: `{conf_b2_b3}`')
    error_lines.append(f'- All classes >= 0.70 AP50: `{all_classes_ge70}`')
    error_lines.append('- Worst-20 image stratification tersedia pada `outputs/phase3/error_stratification.csv`.')
    if error_rows:
        category_counts: dict[str, int] = {}
        for row in error_rows:
            for cat in str(row['categories']).split(';'):
                if cat:
                    category_counts[cat] = category_counts.get(cat, 0) + 1
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            error_lines.append(f'- {cat}: `{count}` image')
    (ROOT / 'outputs/phase3/error_analysis.md').write_text('\n'.join(error_lines) + '\n', encoding='utf-8')

    report_lines = [
        '# Final Report',
        '',
        f'- Canonical protocol source: `{CANONICAL_SOURCE}`',
        f'- Final model: `{final_model}`',
        f'- Resolution: `640`',
        f'- Optimized confidence threshold: `{best_conf}`',
        f'- Precision: `{float(metrics.box.mp):.4f}`',
        f'- Recall: `{float(metrics.box.mr):.4f}`',
        f'- mAP50: `{float(metrics.box.map50):.4f}`',
        f'- mAP50-95: `{float(metrics.box.map):.4f}`',
        f'- Confusion B2/B3: `{conf_b2_b3}`',
        f'- All classes >= 70% AP50: `{all_classes_ge70}`',
        f'- Decision bucket: **{decision}**',
        f'- Deploy check in this run: `{ "deferred" if PHASE3_DEPLOY_CHECK_DEFERRED else "executed" }`',
        '',
        'Per semantic mapping repo ini:',
        '- `B1 = buah merah, besar, bulat, posisi paling bawah tandan; paling matang / ripe`',
        '- `B2 = buah masih hitam namun mulai transisi ke merah, sudah besar dan bulat, posisi di atas B1`',
        '- `B3 = buah full hitam, masih berduri, masih lonjong, posisi di atas B2`',
        '- `B4 = buah paling kecil, paling dalam di batang/tandan, sulit terlihat, masih banyak duri, hitam sampai hijau, dan masih bisa berkembang lebih besar; paling belum matang`',
    ]
    (ROOT / 'outputs/phase3/final_report.md').write_text('\n'.join(report_lines) + '\n', encoding='utf-8')

    update_guide_status([
        '- Canonical source synced: `E0.md` mengikuti flowchart YOLOBench.',
        f'- Phase 3 selesai menggunakan model final `{final_model}`.',
        f'- Final report tersedia di `outputs/phase3/final_report.md` dengan bucket `{decision}`.',
        '- Mapping label repo tetap dikunci: `B1 -> B2 -> B3 -> B4` dari paling matang ke paling belum matang, lengkap dengan ciri visual/posisionalnya.',
    ])
    checkpoint('phase3 canonical sync complete')


def main() -> None:
    log('master orchestrator started')
    write_state({'status': 'running', 'started_utc': utc_now()})
    cleanup_downloaded_root_weights()
    phase1b()
    phase2()
    phase3()
    write_root_readme_and_checkpoint()
    write_state({'status': 'completed', 'completed_utc': utc_now()})
    log('master orchestrator completed')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        write_state({'status': 'failed', 'failed_utc': utc_now(), 'error': f'{type(e).__name__}: {e}'})
        log(f'FATAL {type(e).__name__}: {e}')
        raise
