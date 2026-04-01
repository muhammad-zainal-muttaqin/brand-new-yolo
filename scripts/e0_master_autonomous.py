#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
PHASE3_FINAL_EPOCHS = 60
PHASE3_ONE_STAGE_CANDIDATES = ['yolo11m.pt', 'yolov8s.pt']
PHASE3_DATASET_HF_URL = 'https://huggingface.co/datasets/ULM-DS-Lab/Dataset-Sawit-YOLO'
PHASE3_DATASET_ROOT = Path('/workspace/Dataset-Sawit-YOLO')
PHASE3_STAGE1_MODEL = 'yolo11n.pt'
PHASE3_STAGE1_EPOCHS = 30
PHASE3_STAGE1_PATIENCE = 10
PHASE3_STAGE1_MIN_EPOCHS = 30
PHASE3_STAGE2_MODEL = 'yolo11n-cls.pt'
PHASE3_STAGE2_IMGSZ = 224
PHASE3_STAGE2_EPOCHS = 30
PHASE3_STAGE2_BATCH = 128
PHASE3_STAGE2_PATIENCE = 10
PHASE3_STAGE2_MIN_EPOCHS = 30
PHASE3_TWO_STAGE_DETECTOR_CONF = 0.25
PHASE1B_OVERRIDE_IGNORE_MAP70_STOP = True
PHASE3_DEPLOY_CHECK_DEFERRED = True
PHASE2_OPTION_C_SKIP_REMAINING_LOSS_BRANCHES = True
PHASE2_OPTION_C_SKIP_LR001_RETRAIN = True
PHASE2_OPTION_C_SKIP_BS32 = True
PHASE2_OPTION_C_SKIP_AUG_HEAVY = True
PHASE2_OPTION_C_REASON = (
    'Observed plateau/identical curves on Phase 2 Step 0a loss variants; '
    'repo override keeps baseline loss setup, reuses Phase 1B baseline for lr0=0.001, '
    'and continues with a reduced sweep over LR, batch, and augmentation.'
)
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
    single_cls: bool = False
    eval_checkpoint: str = 'best'
    fixed_epochs: bool = False


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
    wait_for_external_activity()
    cleanup_downloaded_root_weights()
    sh(['git', 'add', '--ignore-removal', 'README.md', 'GUIDE.md', 'E0.md', 'CONTEXT.md', 'outputs', 'runs', 'scripts'])
    diff = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT)
    if diff.returncode == 0:
        log(f'no changes to commit for checkpoint: {message}')
        return True
    sh(['git', 'commit', '-m', message])
    commit_hash = sh_capture('git rev-parse --short HEAD')
    with SYNC_LOG.open('a', encoding='utf-8') as f:
        f.write(f'- {utc_now()} | commit {commit_hash} | {message}\n')
    sh(['git', 'add', str(SYNC_LOG.relative_to(ROOT))])
    diff = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT)
    if diff.returncode != 0:
        sh(['git', 'commit', '-m', f'log sync: {message}'])
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


def materialize_eval_snapshot(phase: str, run_name: str, weight_path: str, data: str, split: str, imgsz: int, batch: int, device: str) -> dict[str, Any]:
    out = eval_path(phase, run_name)
    if out.exists():
        return read_json(out)
    weight_path = str(restore_tracked_file(weight_path))
    model = YOLO(weight_path)
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
        '--eval-checkpoint', spec.eval_checkpoint,
        '--imbalance-strategy', spec.imbalance_strategy,
        '--ordinal-strategy', spec.ordinal_strategy,
        '--focal-gamma', str(spec.focal_gamma),
    ]
    if spec.pretrained:
        cmd.append('--pretrained')
    if spec.single_cls:
        cmd.append('--single-cls')
    if spec.fixed_epochs:
        cmd.append('--fixed-epochs')
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
    checkpoint_key = 'best_weight' if spec.eval_checkpoint == 'best' else 'last_weight'
    eval_weight = summary.get(checkpoint_key) or summary.get('best_weight') or summary.get('last_weight')
    if eval_weight and spec.task == 'detect':
        materialize_eval_snapshot(spec.phase, spec.name, eval_weight, spec.data, spec.split, spec.imgsz, spec.batch, spec.device)
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


def phase2_plateau_like(rows: list[dict[str, Any]], tol: float = 1e-12) -> bool:
    if len(rows) < 2:
        return False
    first = rows[0]
    base_map50 = float(first['mean_map50'])
    base_map50_95 = float(first['mean_map50_95'])
    for row in rows[1:]:
        if abs(float(row['mean_map50']) - base_map50) > tol:
            return False
        if abs(float(row['mean_map50_95']) - base_map50_95) > tol:
            return False
    return True


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


def build_phase3_lock_contract(lock: dict[str, Any]) -> dict[str, Any]:
    base_cfg = lock.get('phase2_locked', {}).get('final_config') or lock.get('final_config') or {}
    if not base_cfg:
        base_cfg = {
            'lr0': 0.001,
            'batch': 16,
            'imbalance_strategy': 'none',
            'ordinal_strategy': 'standard',
            'aug_profile': 'medium',
            'imgsz': 640,
        }
    phase3_locked = lock.get('phase3_locked', {})
    phase3_locked.setdefault('contract_version', 2)
    phase3_locked.setdefault('selection_policy', 'multi_candidate_final_benchmark')
    phase3_locked.setdefault('candidates', PHASE3_ONE_STAGE_CANDIDATES.copy())
    phase3_locked.setdefault('one_stage_config', {
        'lr0': float(base_cfg.get('lr0', 0.001)),
        'batch': int(base_cfg.get('batch', 16)),
        'imbalance_strategy': base_cfg.get('imbalance_strategy', 'none'),
        'ordinal_strategy': base_cfg.get('ordinal_strategy', 'standard'),
        'aug_profile': base_cfg.get('aug_profile', 'medium'),
        'imgsz': int(base_cfg.get('imgsz', 640)),
        'epochs': PHASE3_FINAL_EPOCHS,
        'seed': PHASE3_FINAL_SEED,
        'primary_checkpoint': 'last',
        'secondary_checkpoint': 'best',
        'fixed_epochs': True,
        'eval_splits': ['val', 'test'],
        'train_split': 'train',
    })
    phase3_locked.setdefault('two_stage_config', {
        'enabled': True,
        'seed': PHASE3_FINAL_SEED,
        'stage1': {
            'model': PHASE3_STAGE1_MODEL,
            'single_cls': True,
            'imgsz': 640,
            'epochs': PHASE3_STAGE1_EPOCHS,
            'batch': 16,
            'patience': PHASE3_STAGE1_PATIENCE,
            'min_epochs': PHASE3_STAGE1_MIN_EPOCHS,
            'lr0': 0.001,
            'aug_profile': 'medium',
        },
        'stage2': {
            'model': PHASE3_STAGE2_MODEL,
            'imgsz': PHASE3_STAGE2_IMGSZ,
            'epochs': PHASE3_STAGE2_EPOCHS,
            'batch': PHASE3_STAGE2_BATCH,
            'patience': PHASE3_STAGE2_PATIENCE,
            'min_epochs': PHASE3_STAGE2_MIN_EPOCHS,
            'eval_splits': ['val', 'test'],
        },
        'end_to_end': {
            'detector_conf': PHASE3_TWO_STAGE_DETECTOR_CONF,
            'eval_splits': ['val', 'test'],
        },
    })
    lock['phase3_locked'] = phase3_locked
    lock.pop('final_model', None)
    lock.pop('final_config', None)
    return lock


def ensure_phase3_lock_contract(persist: bool = False) -> dict[str, Any]:
    lock = read_lock()
    validate_phase2_lock(lock)
    lock = build_phase3_lock_contract(lock)
    if persist:
        write_lock(lock)
    return lock


def validate_phase3_lock(lock: dict[str, Any]) -> None:
    validate_phase2_lock(lock)
    require_lock_keys(lock, ['phase3_locked'], 'phase3 lock validation')
    phase3_locked = lock['phase3_locked']
    candidates = phase3_locked.get('candidates') or []
    if candidates != PHASE3_ONE_STAGE_CANDIDATES:
        raise RuntimeError('phase3 lock validation: unexpected candidate list')
    if any(model not in PHASE1B_MODELS for model in candidates):
        raise RuntimeError('phase3 lock validation: candidate outside canonical phase1 roster')
    one_stage = phase3_locked.get('one_stage_config') or {}
    for key in ['lr0', 'batch', 'imbalance_strategy', 'ordinal_strategy', 'aug_profile', 'imgsz', 'epochs', 'seed']:
        if key not in one_stage:
            raise RuntimeError(f'phase3 lock validation: one_stage_config missing key {key}')
    if int(one_stage['imgsz']) != 640:
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
        'phase2_locked': lock.get('phase2_locked'),
        'phase3_locked': lock.get('phase3_locked'),
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
    phase2_override_notes: list[str] = []

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
        step0a_plateau_override = PHASE2_OPTION_C_SKIP_REMAINING_LOSS_BRANCHES and phase2_plateau_like(step0a_candidates)
        current['imbalance_strategy'] = 'none' if step0a_plateau_override else best0a['imbalance_strategy']

        step0b_tokens = [('standard', 'standard'), ('ordinal', 'ordinal_weighted')]
        step0b_candidates = []
        if step0a_plateau_override:
            current['ordinal_strategy'] = 'standard'
            ordinal_rows.append({
                'model': model,
                'option': 'skipped_after_step0a_plateau',
                'ordinal_strategy': 'standard',
                'status': 'skipped',
                'note': PHASE2_OPTION_C_REASON,
            })
            phase2_override_notes.append(
                f'- `{model}`: Step 0a plateau/identical, jadi baseline loss setup dikunci (`imbalance=none`, `ordinal=standard`) dan sisa branch Step 0b dilewati; sweep lanjut hanya untuk LR/batch/augmentation.'
            )
        else:
            for token, ordinal in step0b_tokens:
                agg = aggregate_phase2_option(model, 'p2s0b', token, PHASE2_SEEDS, **current | {'ordinal_strategy': ordinal})
                agg['ordinal_strategy'] = ordinal
                step0b_candidates.append(agg)
                ordinal_rows.append(agg)
            best0b = rank_rows(step0b_candidates)[0]
            current['ordinal_strategy'] = best0b['ordinal_strategy']

        step1_candidates = []
        step1_tokens = [('lr0005', 0.0005), ('lr001', 0.001), ('lr002', 0.002)]
        if step0a_plateau_override and PHASE2_OPTION_C_SKIP_LR001_RETRAIN:
            baseline_lr_ref = dict(base_phase1)
            baseline_lr_ref.update({
                'model': model,
                'option': 'lr001_phase1_reference',
                'lr0': float(baseline['lr0']),
                'source': 'phase1b_baseline_reference',
                'status': 'reused_reference',
            })
            step1_candidates.append(baseline_lr_ref)
            lr_rows.append(baseline_lr_ref)
            step1_tokens = [('lr0005', 0.0005), ('lr002', 0.002)]
            phase2_override_notes.append(
                f'- `{model}`: kandidat Step 1 `lr0=0.001` direuse dari baseline Phase 1B, jadi run `p2s1_lr001_*` dilewati.'
            )
        for token, lr in step1_tokens:
            agg = aggregate_phase2_option(model, 'p2s1', token, PHASE2_SEEDS, **current | {'lr0': lr})
            agg['lr0'] = lr
            step1_candidates.append(agg)
            lr_rows.append(agg)
        best1 = rank_rows(step1_candidates)[0]
        current['lr0'] = float(best1['lr0'])

        step2_candidates = []
        step2_tokens = [('bs8', 8), ('bs16', 16), ('bs32', 32)]
        if PHASE2_OPTION_C_SKIP_BS32:
            step2_tokens = [('bs8', 8), ('bs16', 16)]
            phase2_override_notes.append(
                f'- `{model}`: kandidat Step 2 `batch=32` dilewati untuk menghemat 2 run dan menjaga sweep tetap fokus pada `8` vs `16`.'
            )
        for token, batch in step2_tokens:
            agg = aggregate_phase2_option(model, 'p2s2', token, PHASE2_SEEDS, **current | {'batch': batch})
            agg['batch'] = batch
            step2_candidates.append(agg)
            batch_rows.append(agg)
        best2 = rank_rows(step2_candidates)[0]
        current['batch'] = int(best2['batch'])

        step3_candidates = []
        step3_tokens = ['light', 'medium', 'heavy']
        if PHASE2_OPTION_C_SKIP_AUG_HEAVY:
            step3_tokens = ['light', 'medium']
            phase2_override_notes.append(
                f'- `{model}`: kandidat Step 3 `heavy` dilewati untuk menghemat 2 run; sweep augmentation dibatasi pada `light` vs `medium`.'
            )
        for token in step3_tokens:
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
    if phase2_override_notes:
        phase2_lines.extend([
            '## Operational override aktif',
            '',
            f'- `{PHASE2_OPTION_C_REASON}`',
            *phase2_override_notes,
            '',
        ])
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
    lock['phase2_locked'] = {
        'selected_model': best_final['model'],
        'final_config': final_hparams,
    }
    lock = build_phase3_lock_contract(lock)
    write_lock(lock)
    guide_lines = [
        '- Canonical source synced: `E0.md` mengikuti flowchart YOLOBench.',
        '- Phase 1B canonical selesai dan finalis Phase 2 sudah terkunci.',
        f"- Phase 2 selesai. Model terpilih tunggal tetap `{best_final['model']}` untuk kontrak Phase 2.",
        "- Kontrak Phase 3 sekarang memakai multi-candidate benchmark: `yolo11m.pt` dan `yolov8s.pt`.",
        "- Final config Phase 2 dan kontrak kandidat Phase 3 ditulis ke `outputs/phase1/locked_setup.yaml` dan `outputs/phase2/final_hparams.yaml`.",
    ]
    if phase2_override_notes:
        guide_lines.append('- Phase 2 memakai override plateau-aware: sisa branch loss/ordinal dilewati, lalu sweep dilanjutkan hanya untuk LR, batch, dan augmentation dari baseline loss setup.')
    update_guide_status(guide_lines)
    checkpoint('phase2 canonical sync complete')
    return lock


def ensure_phase3_dataset() -> Path:
    required = [
        PHASE3_DATASET_ROOT / 'images' / 'train',
        PHASE3_DATASET_ROOT / 'images' / 'val',
        PHASE3_DATASET_ROOT / 'images' / 'test',
        PHASE3_DATASET_ROOT / 'labels' / 'train',
        PHASE3_DATASET_ROOT / 'labels' / 'val',
        PHASE3_DATASET_ROOT / 'labels' / 'test',
    ]
    if all(path.exists() for path in required):
        return PHASE3_DATASET_ROOT
    if PHASE3_DATASET_ROOT.exists() and (PHASE3_DATASET_ROOT / '.git').exists():
        sh(['git', '-C', str(PHASE3_DATASET_ROOT), 'lfs', 'pull'], check=False)
        if all(path.exists() for path in required):
            return PHASE3_DATASET_ROOT
    if PHASE3_DATASET_ROOT.exists():
        raise RuntimeError(f'dataset root exists but is incomplete: {PHASE3_DATASET_ROOT}')
    log(f'restoring dataset from {PHASE3_DATASET_HF_URL}')
    sh(['git', 'clone', PHASE3_DATASET_HF_URL, str(PHASE3_DATASET_ROOT)])
    sh(['git', '-C', str(PHASE3_DATASET_ROOT), 'lfs', 'pull'], check=False)
    if not all(path.exists() for path in required):
        raise RuntimeError(f'dataset restore incomplete at {PHASE3_DATASET_ROOT}')
    return PHASE3_DATASET_ROOT


def phase3_path(*parts: str) -> Path:
    return ROOT.joinpath('outputs', 'phase3', *parts)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def names_from_data_yaml(data_yaml: str) -> list[str]:
    cfg, _ = load_data_cfg(data_yaml)
    names = cfg['names']
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def create_phase3_data_yaml() -> Path:
    ensure_phase3_dataset()
    cfg, yaml_path = load_data_cfg('Dataset-YOLO/data.yaml')
    root = dataset_root(cfg, yaml_path)
    outdir = ROOT / 'outputs/phase3'
    outdir.mkdir(parents=True, exist_ok=True)
    final_cfg = {
        'path': str(root),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': int(cfg['nc']),
        'names': cfg['names'],
    }
    final_yaml = outdir / 'final_data.yaml'
    final_yaml.write_text(yaml.safe_dump(final_cfg, sort_keys=False), encoding='utf-8')
    return final_yaml


def threshold_sweep(weight_path: str, data_yaml: str, imgsz: int, batch: int, device: str, split: str = 'val') -> tuple[list[dict[str, Any]], float]:
    model = YOLO(weight_path)
    rows: list[dict[str, Any]] = []
    best_tuple = None
    best_conf = 0.25
    for conf in [0.1, 0.2, 0.3, 0.4, 0.5]:
        metrics = model.val(data=data_yaml, split=split, imgsz=imgsz, batch=batch, device=device, workers=8, plots=False, conf=conf)
        names = {int(k): v for k, v in metrics.names.items()}
        name_to_idx = {v: k for k, v in names.items()}
        cm = metrics.confusion_matrix.matrix
        b2b3 = confusion_rate(cm, name_to_idx.get('B2'), name_to_idx.get('B3'))
        b4_recall = float(metrics.box.class_result(name_to_idx['B4'])[1]) if 'B4' in name_to_idx else None
        row = {
            'split': split,
            'conf': conf,
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'map50': float(metrics.box.map50),
            'map50_95': float(metrics.box.map),
            'confusion_b2_b3': b2b3,
            'b4_recall': b4_recall,
        }
        rows.append(row)
        score_tuple = (
            row['map50'],
            -(row['confusion_b2_b3'] if row['confusion_b2_b3'] is not None else 1.0),
            row['b4_recall'] if row['b4_recall'] is not None else 0.0,
        )
        if best_tuple is None or score_tuple > best_tuple:
            best_tuple = score_tuple
            best_conf = conf
    return rows, best_conf


def empty_square_matrix(size: int) -> list[list[int]]:
    return [[0 for _ in range(size)] for _ in range(size)]


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def normalize_rows(matrix: list[list[int]], missed_by_class: list[int]) -> list[dict[str, float]]:
    rows = []
    for i, counts in enumerate(matrix):
        total = sum(counts) + missed_by_class[i]
        row = {f'pred_{j}': safe_div(counts[j], total) for j in range(len(counts))}
        row['missed_gt'] = safe_div(missed_by_class[i], total)
        rows.append(row)
    return rows


def largest_confusion_pairs(matrix: list[list[int]], class_names: list[str], limit: int = 6) -> list[dict[str, Any]]:
    pairs = []
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            if i == j:
                continue
            count = int(matrix[i][j])
            if count <= 0:
                continue
            pairs.append({'true_class': true_name, 'pred_class': pred_name, 'count': count})
    pairs.sort(key=lambda item: item['count'], reverse=True)
    return pairs[:limit]


def summarize_multiclass_counts(
    matrix: list[list[int]],
    missed_by_class: list[int],
    fp_by_class: list[int],
    class_names: list[str],
) -> dict[str, Any]:
    per_class = []
    support_total = 0
    predicted_total = 0
    tp_total = 0
    for idx, class_name in enumerate(class_names):
        tp = int(matrix[idx][idx])
        support = int(sum(matrix[idx]) + missed_by_class[idx])
        predicted = int(sum(matrix[row][idx] for row in range(len(class_names))) + fp_by_class[idx])
        fp = predicted - tp
        fn = support - tp
        precision = safe_div(tp, predicted)
        recall = safe_div(tp, support)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_class.append({
            'class_idx': idx,
            'class_name': class_name,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'predicted': predicted,
            'true_positive': tp,
            'false_positive': fp,
            'false_negative': fn,
            'missed_gt': int(missed_by_class[idx]),
        })
        support_total += support
        predicted_total += predicted
        tp_total += tp

    macro_precision = statistics.mean(row['precision'] for row in per_class) if per_class else 0.0
    macro_recall = statistics.mean(row['recall'] for row in per_class) if per_class else 0.0
    macro_f1 = statistics.mean(row['f1'] for row in per_class) if per_class else 0.0
    weighted_precision = safe_div(sum(row['precision'] * row['support'] for row in per_class), support_total)
    weighted_recall = safe_div(sum(row['recall'] * row['support'] for row in per_class), support_total)
    weighted_f1 = safe_div(sum(row['f1'] * row['support'] for row in per_class), support_total)

    return {
        'per_class': per_class,
        'macro_avg': {'precision': macro_precision, 'recall': macro_recall, 'f1': macro_f1},
        'weighted_avg': {'precision': weighted_precision, 'recall': weighted_recall, 'f1': weighted_f1},
        'accuracy': safe_div(tp_total, support_total),
        'support_total': support_total,
        'predicted_total': predicted_total,
        'true_positive_total': tp_total,
        'missed_gt_total': int(sum(missed_by_class)),
        'false_positive_total': int(sum(fp_by_class)),
        'largest_confusions': largest_confusion_pairs(matrix, class_names),
        'row_normalized': normalize_rows(matrix, missed_by_class),
    }


def categorize_detection_errors(
    gt_boxes: list[dict[str, Any]],
    pred_boxes: list[dict[str, Any]],
    names: dict[int, str],
    confusions: list[tuple[int, int, float]],
    missed_gt: list[int],
    fp_pred: list[int],
) -> set[str]:
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
    return categories


def evaluate_detection_like_split(
    data_yaml: str,
    split: str,
    class_names: list[str],
    predictor,
    extra_fields: dict[str, Any] | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    names = {idx: name for idx, name in enumerate(class_names)}
    matrix = empty_square_matrix(len(class_names))
    missed_by_class = [0 for _ in class_names]
    fp_by_class = [0 for _ in class_names]
    image_rows: list[dict[str, Any]] = []

    for image_path in iter_split_images(data_yaml, split):
        gt_boxes, _ = load_gt_boxes(image_path)
        pred_boxes = predictor(image_path)
        tp, confusions, missed_gt, fp_pred = greedy_match(gt_boxes, pred_boxes, iou_thresh=0.5)
        for gi, pi, _ in tp:
            matrix[gt_boxes[gi]['cls']][pred_boxes[pi]['cls']] += 1
        for gi, pi, _ in confusions:
            matrix[gt_boxes[gi]['cls']][pred_boxes[pi]['cls']] += 1
        for gi in missed_gt:
            missed_by_class[gt_boxes[gi]['cls']] += 1
        for pi in fp_pred:
            pred_cls = pred_boxes[pi]['cls']
            if 0 <= pred_cls < len(class_names):
                fp_by_class[pred_cls] += 1

        if not confusions and not missed_gt and not fp_pred:
            continue
        categories = categorize_detection_errors(gt_boxes, pred_boxes, names, confusions, missed_gt, fp_pred)
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
        image_rows.append(row)

    image_rows.sort(key=lambda x: (x['error_score'], x['confusions'], x['missed_gt'], x['false_positive']), reverse=True)
    stats = summarize_multiclass_counts(matrix, missed_by_class, fp_by_class, class_names)
    stats.update({
        'counts': matrix,
        'missed_by_class': {class_names[i]: int(v) for i, v in enumerate(missed_by_class)},
        'false_positive_by_class': {class_names[i]: int(v) for i, v in enumerate(fp_by_class)},
        'top_errors': image_rows[:limit],
    })
    return stats


def merge_detection_metrics(per_class_eval: list[dict[str, Any]], confusion_stats: dict[str, Any]) -> list[dict[str, Any]]:
    eval_by_class = {row['class_name']: row for row in per_class_eval}
    merged = []
    for row in confusion_stats['per_class']:
        extra = eval_by_class.get(row['class_name'], {})
        merged.append({
            **row,
            'map50': extra.get('map50'),
            'map50_95': extra.get('map50_95'),
            'ultralytics_precision': extra.get('precision'),
            'ultralytics_recall': extra.get('recall'),
        })
    return merged


def evaluate_one_stage_checkpoint(
    run_name: str,
    candidate: str,
    checkpoint: str,
    weight_path: str,
    data_yaml: str,
    split: str,
    imgsz: int,
    batch: int,
    device: str,
    conf: float,
) -> dict[str, Any]:
    model = YOLO(weight_path)
    metrics = model.val(data=data_yaml, split=split, imgsz=imgsz, batch=batch, device=device, workers=8, plots=False, conf=conf)
    names = {int(k): v for k, v in metrics.names.items()}
    class_names = [names[idx] for idx in sorted(names)]
    per_class_eval = []
    for idx in sorted(names):
        p, r, map50, map50_95 = metrics.box.class_result(idx)
        per_class_eval.append({
            'class_idx': idx,
            'class_name': names[idx],
            'precision': float(p),
            'recall': float(r),
            'map50': float(map50),
            'map50_95': float(map50_95),
        })
    custom = evaluate_detection_like_split(
        data_yaml=data_yaml,
        split=split,
        class_names=class_names,
        predictor=lambda image_path: predict_boxes(model, image_path, imgsz, device, conf),
        extra_fields={'branch': 'one_stage', 'candidate': candidate, 'checkpoint': checkpoint, 'split': split},
    )
    snapshot = {
        'metric_schema': 'detection',
        'run_name': run_name,
        'branch': 'one_stage',
        'candidate': candidate,
        'checkpoint': checkpoint,
        'weight_path': weight_path,
        'split': split,
        'optimized_conf': conf,
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
        'f1': safe_div(2 * float(metrics.box.mp) * float(metrics.box.mr), float(metrics.box.mp) + float(metrics.box.mr)),
        'map50': float(metrics.box.map50),
        'map50_95': float(metrics.box.map),
        'accuracy': custom['accuracy'],
        'support_total': custom['support_total'],
        'predicted_total': custom['predicted_total'],
        'missed_gt_total': custom['missed_gt_total'],
        'false_positive_total': custom['false_positive_total'],
        'macro_avg': custom['macro_avg'],
        'weighted_avg': custom['weighted_avg'],
        'per_class': merge_detection_metrics(per_class_eval, custom),
        'confusion_matrix': {
            'classes': class_names,
            'counts': custom['counts'],
            'row_normalized': custom['row_normalized'],
            'missed_by_class': custom['missed_by_class'],
            'false_positive_by_class': custom['false_positive_by_class'],
            'largest_confusions': custom['largest_confusions'],
        },
        'top_errors': custom['top_errors'],
    }
    return snapshot


def crop_xyxy(image_path: Path, xyxy: tuple[float, float, float, float]) -> Image.Image:
    with Image.open(image_path) as img:
        x1, y1, x2, y2 = xyxy
        x1 = max(int(x1), 0)
        y1 = max(int(y1), 0)
        x2 = min(int(x2), img.width)
        y2 = min(int(y2), img.height)
        if x2 <= x1 or y2 <= y1:
            return img.crop((0, 0, img.width, img.height)).copy()
        return img.crop((x1, y1, x2, y2)).copy()


def rebuild_gt_crop_dataset(data_yaml: str, out_root: Path) -> Path:
    class_names = names_from_data_yaml(data_yaml)
    manifest = out_root / 'manifest.json'
    if manifest.exists():
        return out_root
    for split in ['train', 'val', 'test']:
        for class_name in class_names:
            (out_root / split / class_name).mkdir(parents=True, exist_ok=True)
        for image_path in iter_split_images(data_yaml, split):
            gt_boxes, _ = load_gt_boxes(image_path)
            stem = image_path.stem
            for idx, gt in enumerate(gt_boxes):
                class_name = class_names[gt['cls']]
                crop = crop_xyxy(image_path, gt['xyxy'])
                crop_path = out_root / split / class_name / f'{stem}__gt{idx}.jpg'
                crop.save(crop_path, format='JPEG', quality=95)
    write_json(manifest, {
        'source_data_yaml': data_yaml,
        'dataset_root': str(out_root),
        'class_names': class_names,
        'splits': ['train', 'val', 'test'],
    })
    return out_root


def predict_class_idx(model: YOLO, image_path: Path, imgsz: int, device: str) -> tuple[int, list[int]]:
    result = model.predict(source=str(image_path), imgsz=imgsz, device=device, verbose=False)[0]
    top1 = int(result.probs.top1)
    topk = [int(x) for x in getattr(result.probs, 'top5', [top1])]
    return top1, topk


def summarize_classification_confusion(matrix: list[list[int]], class_names: list[str]) -> dict[str, Any]:
    per_class = []
    total = sum(sum(row) for row in matrix)
    tp_total = 0
    for idx, class_name in enumerate(class_names):
        tp = int(matrix[idx][idx])
        support = int(sum(matrix[idx]))
        predicted = int(sum(matrix[row][idx] for row in range(len(class_names))))
        precision = safe_div(tp, predicted)
        recall = safe_div(tp, support)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_class.append({
            'class_idx': idx,
            'class_name': class_name,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'predicted': predicted,
            'true_positive': tp,
            'false_positive': predicted - tp,
            'false_negative': support - tp,
        })
        tp_total += tp
    macro_avg = {
        'precision': statistics.mean(row['precision'] for row in per_class) if per_class else 0.0,
        'recall': statistics.mean(row['recall'] for row in per_class) if per_class else 0.0,
        'f1': statistics.mean(row['f1'] for row in per_class) if per_class else 0.0,
    }
    weighted_avg = {
        'precision': safe_div(sum(row['precision'] * row['support'] for row in per_class), total),
        'recall': safe_div(sum(row['recall'] * row['support'] for row in per_class), total),
        'f1': safe_div(sum(row['f1'] * row['support'] for row in per_class), total),
    }
    return {
        'per_class': per_class,
        'macro_avg': macro_avg,
        'weighted_avg': weighted_avg,
        'accuracy': safe_div(tp_total, total),
        'support_total': total,
        'row_normalized': normalize_rows(matrix, [0 for _ in class_names]),
        'largest_confusions': largest_confusion_pairs(matrix, class_names),
    }


def evaluate_gt_crop_classifier(
    run_name: str,
    checkpoint: str,
    weight_path: str,
    dataset_root: Path,
    split: str,
    imgsz: int,
    device: str,
) -> dict[str, Any]:
    model = YOLO(weight_path)
    class_names = names_from_data_yaml('Dataset-YOLO/data.yaml')
    matrix = empty_square_matrix(len(class_names))
    topk_correct = 0
    total = 0
    for gt_idx, class_name in enumerate(class_names):
        class_dir = dataset_root / split / class_name
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.glob('*')):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            pred_idx, topk = predict_class_idx(model, image_path, imgsz, device)
            matrix[gt_idx][pred_idx] += 1
            topk_correct += int(gt_idx in topk)
            total += 1
    summary = summarize_classification_confusion(matrix, class_names)
    return {
        'metric_schema': 'classification_gtcrop',
        'run_name': run_name,
        'branch': 'two_stage_gtcrop',
        'candidate': model_stem(PHASE3_STAGE2_MODEL),
        'checkpoint': checkpoint,
        'weight_path': weight_path,
        'split': split,
        'top1_acc': summary['accuracy'],
        'top5_acc': safe_div(topk_correct, total),
        'precision': summary['weighted_avg']['precision'],
        'recall': summary['weighted_avg']['recall'],
        'f1': summary['weighted_avg']['f1'],
        'accuracy': summary['accuracy'],
        'support_total': summary['support_total'],
        'macro_avg': summary['macro_avg'],
        'weighted_avg': summary['weighted_avg'],
        'per_class': summary['per_class'],
        'confusion_matrix': {
            'classes': class_names,
            'counts': matrix,
            'row_normalized': summary['row_normalized'],
            'largest_confusions': summary['largest_confusions'],
        },
    }


def classify_crop_from_box(model: YOLO, image_path: Path, xyxy: tuple[float, float, float, float], imgsz: int, device: str) -> int:
    crop = crop_xyxy(image_path, xyxy)
    temp_path = image_path.parent / f'.__tmp_{os.getpid()}_{time.time_ns()}.jpg'
    crop.save(temp_path, format='JPEG', quality=95)
    try:
        pred_idx, _ = predict_class_idx(model, temp_path, imgsz, device)
        return pred_idx
    finally:
        if temp_path.exists():
            temp_path.unlink()


def evaluate_two_stage_end_to_end(
    stage1_run_name: str,
    stage2_run_name: str,
    checkpoint: str,
    stage1_weight: str,
    stage2_weight: str,
    data_yaml: str,
    split: str,
    detector_imgsz: int,
    classifier_imgsz: int,
    device: str,
    conf: float,
) -> dict[str, Any]:
    stage1_model = YOLO(stage1_weight)
    stage2_model = YOLO(stage2_weight)
    class_names = names_from_data_yaml(data_yaml)
    custom = evaluate_detection_like_split(
        data_yaml=data_yaml,
        split=split,
        class_names=class_names,
        predictor=lambda image_path: [
            {
                **pred,
                'cls': classify_crop_from_box(stage2_model, image_path, pred['xyxy'], classifier_imgsz, device),
            }
            for pred in predict_boxes(stage1_model, image_path, detector_imgsz, device, conf)
        ],
        extra_fields={'branch': 'two_stage_end_to_end', 'candidate': 'detector+classifier', 'checkpoint': checkpoint, 'split': split},
    )
    return {
        'metric_schema': 'detection_like_classification',
        'run_name': f'{stage1_run_name}+{stage2_run_name}',
        'branch': 'two_stage_end_to_end',
        'candidate': 'yolo11n_singlecls+yolo11ncls_gtcrop',
        'checkpoint': checkpoint,
        'stage1_weight': stage1_weight,
        'stage2_weight': stage2_weight,
        'split': split,
        'optimized_conf': conf,
        'precision': safe_div(custom['true_positive_total'], custom['predicted_total']),
        'recall': safe_div(custom['true_positive_total'], custom['support_total']),
        'f1': safe_div(
            2 * safe_div(custom['true_positive_total'], custom['predicted_total']) * safe_div(custom['true_positive_total'], custom['support_total']),
            safe_div(custom['true_positive_total'], custom['predicted_total']) + safe_div(custom['true_positive_total'], custom['support_total']),
        ),
        'accuracy': custom['accuracy'],
        'support_total': custom['support_total'],
        'predicted_total': custom['predicted_total'],
        'missed_gt_total': custom['missed_gt_total'],
        'false_positive_total': custom['false_positive_total'],
        'macro_avg': custom['macro_avg'],
        'weighted_avg': custom['weighted_avg'],
        'per_class': custom['per_class'],
        'confusion_matrix': {
            'classes': class_names,
            'counts': custom['counts'],
            'row_normalized': custom['row_normalized'],
            'missed_by_class': custom['missed_by_class'],
            'false_positive_by_class': custom['false_positive_by_class'],
            'largest_confusions': custom['largest_confusions'],
        },
        'top_errors': custom['top_errors'],
    }


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


def snapshot_metric_row(snapshot: dict[str, Any]) -> dict[str, Any]:
    macro = snapshot.get('macro_avg') or {}
    weighted = snapshot.get('weighted_avg') or {}
    return {
        'branch': snapshot.get('branch'),
        'candidate': snapshot.get('candidate'),
        'checkpoint': snapshot.get('checkpoint'),
        'split': snapshot.get('split'),
        'run_name': snapshot.get('run_name'),
        'metric_schema': snapshot.get('metric_schema'),
        'weight_path': snapshot.get('weight_path', ''),
        'optimized_conf': snapshot.get('optimized_conf', ''),
        'precision': snapshot.get('precision', ''),
        'recall': snapshot.get('recall', ''),
        'f1': snapshot.get('f1', ''),
        'map50': snapshot.get('map50', ''),
        'map50_95': snapshot.get('map50_95', ''),
        'top1_acc': snapshot.get('top1_acc', ''),
        'top5_acc': snapshot.get('top5_acc', ''),
        'accuracy': snapshot.get('accuracy', ''),
        'support_total': snapshot.get('support_total', ''),
        'predicted_total': snapshot.get('predicted_total', ''),
        'missed_gt_total': snapshot.get('missed_gt_total', ''),
        'false_positive_total': snapshot.get('false_positive_total', ''),
        'macro_precision': macro.get('precision', ''),
        'macro_recall': macro.get('recall', ''),
        'macro_f1': macro.get('f1', ''),
        'weighted_precision': weighted.get('precision', ''),
        'weighted_recall': weighted.get('recall', ''),
        'weighted_f1': weighted.get('f1', ''),
    }


def snapshot_per_class_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in snapshot.get('per_class') or []:
        rows.append({
            'branch': snapshot.get('branch'),
            'candidate': snapshot.get('candidate'),
            'checkpoint': snapshot.get('checkpoint'),
            'split': snapshot.get('split'),
            **row,
        })
    return rows


def snapshot_confusion_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    cm = snapshot.get('confusion_matrix') or {}
    classes = cm.get('classes') or []
    counts = cm.get('counts') or []
    row_norm = cm.get('row_normalized') or []
    missed = cm.get('missed_by_class') or {}
    rows = []
    for idx, true_class in enumerate(classes):
        row = {
            'branch': snapshot.get('branch'),
            'candidate': snapshot.get('candidate'),
            'checkpoint': snapshot.get('checkpoint'),
            'split': snapshot.get('split'),
            'true_class': true_class,
            'missed_gt': missed.get(true_class, 0),
            'support': int(sum(counts[idx]) + missed.get(true_class, 0)),
            'norm_missed_gt': (row_norm[idx].get('missed_gt', 0.0) if idx < len(row_norm) else 0.0),
        }
        for jdx, pred_class in enumerate(classes):
            row[pred_class] = int(counts[idx][jdx])
            row[f'norm_{pred_class}'] = row_norm[idx].get(f'pred_{jdx}', 0.0) if idx < len(row_norm) else 0.0
        rows.append(row)
    return rows


def write_phase3_reports(metric_rows: list[dict[str, Any]], error_rows: list[dict[str, Any]]) -> None:
    def find_rows(**filters) -> list[dict[str, Any]]:
        out = []
        for row in metric_rows:
            if all(str(row.get(k)) == str(v) for k, v in filters.items()):
                out.append(row)
        return out

    one_stage_last_test = sorted(
        find_rows(branch='one_stage', checkpoint='last', split='test'),
        key=lambda row: float(row.get('map50') or 0.0),
        reverse=True,
    )
    one_stage_last_val = {
        row['candidate']: row
        for row in find_rows(branch='one_stage', checkpoint='last', split='val')
    }
    two_stage_gtcrop_test = find_rows(branch='two_stage_gtcrop', checkpoint='last', split='test')
    two_stage_e2e_test = find_rows(branch='two_stage_end_to_end', checkpoint='last', split='test')

    report_lines = [
        '# Final Report - Phase 3 Multi-Candidate Benchmark',
        '',
        f'- Canonical protocol source: `{CANONICAL_SOURCE}`',
        '- Phase 3 ini menimpa definisi lama dan sekarang mengikuti split adil: training hanya `train`, evaluasi pada `val` dan `test`.',
        '- Kandidat utama one-stage: `yolo11m.pt` dan `yolov8s.pt`.',
        '- Checkpoint utama untuk pelaporan final: `last.pt`; `best.pt` tetap ikut dievaluasi.',
        '- Cabang two-stage di-run ulang sebagai pembanding pendukung: Stage-1 single-class detector, Stage-2 GT-crop classifier, dan evaluasi end-to-end.',
        '',
        '## One-Stage Test Set — `last.pt`',
        '',
    ]
    for row in one_stage_last_test:
        report_lines.append(
            f"- `{row['candidate']}` | mAP50 `{float(row.get('map50') or 0.0):.4f}` | "
            f"mAP50-95 `{float(row.get('map50_95') or 0.0):.4f}` | precision `{float(row.get('precision') or 0.0):.4f}` | "
            f"recall `{float(row.get('recall') or 0.0):.4f}` | conf `{row.get('optimized_conf', '')}`"
        )
    report_lines.extend(['', '## Gap Val vs Test — `last.pt`', ''])
    for row in one_stage_last_test:
        val_row = one_stage_last_val.get(row['candidate'])
        if not val_row:
            continue
        report_lines.append(
            f"- `{row['candidate']}` | mAP50 val `{float(val_row.get('map50') or 0.0):.4f}` -> test `{float(row.get('map50') or 0.0):.4f}` | "
            f"precision val `{float(val_row.get('precision') or 0.0):.4f}` -> test `{float(row.get('precision') or 0.0):.4f}` | "
            f"recall val `{float(val_row.get('recall') or 0.0):.4f}` -> test `{float(row.get('recall') or 0.0):.4f}`"
        )
    if two_stage_gtcrop_test:
        report_lines.extend(['', '## Two-Stage GT-Crop (`last.pt`, test)', ''])
        for row in two_stage_gtcrop_test:
            report_lines.append(
                f"- top1 `{float(row.get('top1_acc') or 0.0):.4f}` | weighted F1 `{float(row.get('weighted_f1') or 0.0):.4f}` | macro F1 `{float(row.get('macro_f1') or 0.0):.4f}`"
            )
    if two_stage_e2e_test:
        report_lines.extend(['', '## Two-Stage End-to-End (`last.pt`, test)', ''])
        for row in two_stage_e2e_test:
            report_lines.append(
                f"- precision `{float(row.get('precision') or 0.0):.4f}` | recall `{float(row.get('recall') or 0.0):.4f}` | "
                f"F1 `{float(row.get('f1') or 0.0):.4f}` | accuracy `{float(row.get('accuracy') or 0.0):.4f}`"
            )
    phase3_root = ROOT / 'outputs' / 'phase3'
    (phase3_root / 'final_report.md').write_text('\n'.join(report_lines) + '\n', encoding='utf-8')

    eval_lines = [
        '# Final Evaluation — Phase 3',
        '',
        'Dokumen ini memuat ringkasan evaluasi teknis hasil otomasi ulang Phase 3 sesuai `GUIDE.md`: train-only benchmark, dual-candidate one-stage, dan cabang two-stage yang dibangun ulang.',
        '',
        '## Source of truth',
        '',
        '- `outputs/phase3/final_metrics.csv`',
        '- `outputs/phase3/per_class_metrics.csv`',
        '- `outputs/phase3/confusion_matrix.csv`',
        '- `outputs/phase3/threshold_sweep.csv`',
        '- `outputs/phase3/error_stratification.csv`',
        '',
        '## Kandidat One-Stage',
        '',
    ]
    for row in one_stage_last_test:
        eval_lines.append(
            f"- `{row['candidate']}` (`last`, test): mAP50 `{float(row.get('map50') or 0.0):.4f}`, mAP50-95 `{float(row.get('map50_95') or 0.0):.4f}`, "
            f"precision `{float(row.get('precision') or 0.0):.4f}`, recall `{float(row.get('recall') or 0.0):.4f}`, weighted F1 `{float(row.get('weighted_f1') or 0.0):.4f}`"
        )
    if two_stage_gtcrop_test or two_stage_e2e_test:
        eval_lines.extend(['', '## Cabang Two-Stage', ''])
        for row in two_stage_gtcrop_test:
            eval_lines.append(
                f"- GT-crop classifier (`last`, test): top1 `{float(row.get('top1_acc') or 0.0):.4f}`, weighted F1 `{float(row.get('weighted_f1') or 0.0):.4f}`"
            )
        for row in two_stage_e2e_test:
            eval_lines.append(
                f"- End-to-end (`last`, test): precision `{float(row.get('precision') or 0.0):.4f}`, recall `{float(row.get('recall') or 0.0):.4f}`, F1 `{float(row.get('f1') or 0.0):.4f}`"
            )
    (phase3_root / 'final_evaluation.md').write_text('\n'.join(eval_lines) + '\n', encoding='utf-8')

    error_lines = ['# Error Analysis', '']
    if error_rows:
        grouped: dict[tuple[str, str, str, str], dict[str, int]] = {}
        for row in error_rows:
            key = (str(row.get('branch')), str(row.get('candidate')), str(row.get('checkpoint')), str(row.get('split')))
            grouped.setdefault(key, {})
            for cat in str(row.get('categories', '')).split(';'):
                if not cat:
                    continue
                grouped[key][cat] = grouped[key].get(cat, 0) + 1
        for key, counts in sorted(grouped.items()):
            branch, candidate, checkpoint, split = key
            error_lines.append(f'- `{branch}` / `{candidate}` / `{checkpoint}` / `{split}`')
            for cat, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
                error_lines.append(f'  - {cat}: `{count}` image')
    else:
        error_lines.append('- Belum ada error stratification.')
    (phase3_root / 'error_analysis.md').write_text('\n'.join(error_lines) + '\n', encoding='utf-8')


def phase3() -> None:
    log('phase3 start')
    ensure_phase3_dataset()
    lock = ensure_phase3_lock_contract(persist=True)
    validate_phase3_lock(lock)
    data_yaml = create_phase3_data_yaml()
    phase3_locked = lock['phase3_locked']
    one_stage_cfg = phase3_locked['one_stage_config']
    metric_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    tracked_weights: list[str] = []

    phase3_path('detail').mkdir(parents=True, exist_ok=True)
    phase3_path('figures').mkdir(parents=True, exist_ok=True)

    for model_name in phase3_locked['candidates']:
        candidate = model_stem(model_name)
        run_name = f'p3os_{candidate}_640_s{one_stage_cfg["seed"]}_e{one_stage_cfg["epochs"]}fix'
        spec = RunSpec(
            phase='phase3',
            name=run_name,
            model=model_name,
            imgsz=int(one_stage_cfg['imgsz']),
            epochs=int(one_stage_cfg['epochs']),
            batch=int(one_stage_cfg['batch']),
            seed=int(one_stage_cfg['seed']),
            split='val',
            patience=0,
            min_epochs=0,
            data=str(data_yaml),
            lr0=float(one_stage_cfg['lr0']),
            optimizer='AdamW',
            imbalance_strategy=one_stage_cfg['imbalance_strategy'],
            ordinal_strategy=one_stage_cfg['ordinal_strategy'],
            aug_profile=one_stage_cfg['aug_profile'],
            eval_checkpoint='last',
            fixed_epochs=bool(one_stage_cfg['fixed_epochs']),
        )
        summary = run_experiment(spec)
        tracked_weights.extend([summary.get('best_weight', ''), summary.get('last_weight', '')])
        detail_dir = phase3_path('detail', 'one_stage', candidate)
        detail_dir.mkdir(parents=True, exist_ok=True)

        for checkpoint_name in ['last', 'best']:
            weight_path = summary.get(f'{checkpoint_name}_weight')
            if not weight_path:
                continue
            sweep_rows, best_conf = threshold_sweep(weight_path, str(data_yaml), int(one_stage_cfg['imgsz']), int(one_stage_cfg['batch']), '0', split='val')
            for row in sweep_rows:
                row.update({'branch': 'one_stage', 'candidate': candidate, 'checkpoint': checkpoint_name, 'run_name': run_name})
            threshold_rows.extend(sweep_rows)
            write_csv(detail_dir / f'{checkpoint_name}_threshold_sweep.csv', sweep_rows)
            for split in one_stage_cfg['eval_splits']:
                snapshot = evaluate_one_stage_checkpoint(
                    run_name=run_name,
                    candidate=candidate,
                    checkpoint=checkpoint_name,
                    weight_path=weight_path,
                    data_yaml=str(data_yaml),
                    split=split,
                    imgsz=int(one_stage_cfg['imgsz']),
                    batch=int(one_stage_cfg['batch']),
                    device='0',
                    conf=best_conf,
                )
                write_json(detail_dir / f'{checkpoint_name}_{split}_eval.json', snapshot)
                metric_rows.append(snapshot_metric_row(snapshot))
                per_class_rows.extend(snapshot_per_class_rows(snapshot))
                confusion_rows.extend(snapshot_confusion_rows(snapshot))
                error_rows.extend(snapshot.get('top_errors') or [])

    two_stage_cfg = phase3_locked['two_stage_config']
    if two_stage_cfg.get('enabled'):
        stage1_cfg = two_stage_cfg['stage1']
        stage2_cfg = two_stage_cfg['stage2']
        gtcrop_root = rebuild_gt_crop_dataset(str(data_yaml), Path('/workspace/phase3_cls_gtcrops'))

        stage1_name = f'p3ts_stage1_singlecls_{model_stem(stage1_cfg["model"])}_640_s{two_stage_cfg["seed"]}_e{stage1_cfg["epochs"]}p{stage1_cfg["patience"]}m{stage1_cfg["min_epochs"]}'
        stage1_spec = RunSpec(
            phase='phase3',
            name=stage1_name,
            model=stage1_cfg['model'],
            imgsz=int(stage1_cfg['imgsz']),
            epochs=int(stage1_cfg['epochs']),
            batch=int(stage1_cfg['batch']),
            seed=int(two_stage_cfg['seed']),
            split='val',
            patience=int(stage1_cfg['patience']),
            min_epochs=int(stage1_cfg['min_epochs']),
            data=str(data_yaml),
            lr0=float(stage1_cfg['lr0']),
            optimizer='AdamW',
            aug_profile=stage1_cfg['aug_profile'],
            single_cls=bool(stage1_cfg['single_cls']),
            eval_checkpoint='last',
        )
        stage1_summary = run_experiment(stage1_spec)
        tracked_weights.extend([stage1_summary.get('best_weight', ''), stage1_summary.get('last_weight', '')])

        stage1_detail_dir = phase3_path('detail', 'two_stage', 'stage1')
        stage1_detail_dir.mkdir(parents=True, exist_ok=True)
        for checkpoint_name in ['last', 'best']:
            weight_path = stage1_summary.get(f'{checkpoint_name}_weight')
            if not weight_path:
                continue
            model = YOLO(weight_path)
            for split in ['val', 'test']:
                metrics = model.val(
                    data=str(data_yaml),
                    split=split,
                    imgsz=int(stage1_cfg['imgsz']),
                    batch=int(stage1_cfg['batch']),
                    device='0',
                    workers=8,
                    plots=False,
                    single_cls=True,
                )
                snapshot = {
                    'metric_schema': 'single_class_detection',
                    'run_name': stage1_name,
                    'branch': 'two_stage_stage1',
                    'candidate': model_stem(stage1_cfg['model']),
                    'checkpoint': checkpoint_name,
                    'weight_path': weight_path,
                    'split': split,
                    'precision': float(metrics.box.mp),
                    'recall': float(metrics.box.mr),
                    'f1': safe_div(2 * float(metrics.box.mp) * float(metrics.box.mr), float(metrics.box.mp) + float(metrics.box.mr)),
                    'map50': float(metrics.box.map50),
                    'map50_95': float(metrics.box.map),
                }
                write_json(stage1_detail_dir / f'{checkpoint_name}_{split}_eval.json', snapshot)
                metric_rows.append(snapshot_metric_row(snapshot))

        stage2_name = f'p3ts_stage2_cls_{model_stem(stage2_cfg["model"])}_{stage2_cfg["imgsz"]}_s{two_stage_cfg["seed"]}_e{stage2_cfg["epochs"]}p{stage2_cfg["patience"]}m{stage2_cfg["min_epochs"]}'
        stage2_spec = RunSpec(
            phase='phase3',
            name=stage2_name,
            model=stage2_cfg['model'],
            imgsz=int(stage2_cfg['imgsz']),
            epochs=int(stage2_cfg['epochs']),
            batch=int(stage2_cfg['batch']),
            seed=int(two_stage_cfg['seed']),
            split='val',
            task='classify',
            patience=int(stage2_cfg['patience']),
            min_epochs=int(stage2_cfg['min_epochs']),
            data=str(gtcrop_root),
            optimizer='AdamW',
            eval_checkpoint='last',
        )
        stage2_summary = run_experiment(stage2_spec)
        tracked_weights.extend([stage2_summary.get('best_weight', ''), stage2_summary.get('last_weight', '')])

        stage2_detail_dir = phase3_path('detail', 'two_stage', 'gtcrop')
        stage2_detail_dir.mkdir(parents=True, exist_ok=True)
        for checkpoint_name in ['last', 'best']:
            weight_path = stage2_summary.get(f'{checkpoint_name}_weight')
            if not weight_path:
                continue
            for split in stage2_cfg['eval_splits']:
                snapshot = evaluate_gt_crop_classifier(
                    run_name=stage2_name,
                    checkpoint=checkpoint_name,
                    weight_path=weight_path,
                    dataset_root=gtcrop_root,
                    split=split,
                    imgsz=int(stage2_cfg['imgsz']),
                    device='0',
                )
                write_json(stage2_detail_dir / f'{checkpoint_name}_{split}_eval.json', snapshot)
                metric_rows.append(snapshot_metric_row(snapshot))
                per_class_rows.extend(snapshot_per_class_rows(snapshot))
                confusion_rows.extend(snapshot_confusion_rows(snapshot))

        e2e_detail_dir = phase3_path('detail', 'two_stage', 'end_to_end')
        e2e_detail_dir.mkdir(parents=True, exist_ok=True)
        for checkpoint_name in ['last', 'best']:
            stage1_weight = stage1_summary.get(f'{checkpoint_name}_weight')
            stage2_weight = stage2_summary.get(f'{checkpoint_name}_weight')
            if not stage1_weight or not stage2_weight:
                continue
            for split in two_stage_cfg['end_to_end']['eval_splits']:
                snapshot = evaluate_two_stage_end_to_end(
                    stage1_run_name=stage1_name,
                    stage2_run_name=stage2_name,
                    checkpoint=checkpoint_name,
                    stage1_weight=stage1_weight,
                    stage2_weight=stage2_weight,
                    data_yaml=str(data_yaml),
                    split=split,
                    detector_imgsz=int(stage1_cfg['imgsz']),
                    classifier_imgsz=int(stage2_cfg['imgsz']),
                    device='0',
                    conf=float(two_stage_cfg['end_to_end']['detector_conf']),
                )
                write_json(e2e_detail_dir / f'{checkpoint_name}_{split}_eval.json', snapshot)
                metric_rows.append(snapshot_metric_row(snapshot))
                per_class_rows.extend(snapshot_per_class_rows(snapshot))
                confusion_rows.extend(snapshot_confusion_rows(snapshot))
                error_rows.extend(snapshot.get('top_errors') or [])

    metric_rows.sort(key=lambda row: (row['branch'], row['candidate'], row['checkpoint'], row['split']))
    per_class_rows.sort(key=lambda row: (row['branch'], row['candidate'], row['checkpoint'], row['split'], row['class_name']))
    confusion_rows.sort(key=lambda row: (row['branch'], row['candidate'], row['checkpoint'], row['split'], row['true_class']))
    threshold_rows.sort(key=lambda row: (row['candidate'], row['checkpoint'], float(row['conf'])))
    error_rows.sort(key=lambda row: (row.get('branch', ''), row.get('candidate', ''), row.get('checkpoint', ''), row.get('split', ''), -int(row.get('error_score', 0))))

    write_csv(phase3_path('final_metrics.csv'), metric_rows)
    write_csv(phase3_path('per_class_metrics.csv'), per_class_rows)
    write_csv(phase3_path('confusion_matrix.csv'), confusion_rows)
    write_csv(phase3_path('threshold_sweep.csv'), threshold_rows)
    write_csv(phase3_path('error_stratification.csv'), error_rows)

    deploy_lines = ['# Deploy Check', '']
    if PHASE3_DEPLOY_CHECK_DEFERRED:
        deploy_lines.append('- Status: **deferred by repo override**.')
        deploy_lines.append('- TFLite export: `skipped for now`')
        deploy_lines.append('- TFLite INT8 export: `skipped for now`')
        deploy_lines.append('- Rationale: Phase 3 ini memprioritaskan benchmark adil, dokumentasi, dan sinkronisasi artefak sebelum deployment engineering.')
        deploy_lines.append('- Important: konversi deployment wajib divalidasi ulang terhadap artefak hasil konversi.')
    for weight_path in sorted({path for path in tracked_weights if path}):
        weight_file = Path(weight_path)
        if weight_file.exists():
            deploy_lines.append(f'- Weight: `{weight_file}` | size MB `{weight_file.stat().st_size / (1024 * 1024):.2f}`')
    (phase3_path('deploy_check.md')).write_text('\n'.join(deploy_lines) + '\n', encoding='utf-8')

    write_phase3_reports(metric_rows, error_rows)
    sh([sys.executable, 'scripts/generate_doc_figures.py', '--phase', '3'])
    sh([sys.executable, 'scripts/generate_e0_research_progress_charts.py'])

    update_guide_status([
        '- Canonical source synced: `E0.md` mengikuti flowchart YOLOBench.',
        '- Phase 3 sekarang ditimpa sebagai benchmark otomatis multi-candidate dengan split adil (`train` -> `val` + `test`).',
        '- Kandidat utama one-stage yang dijalankan: `yolo11m.pt` dan `yolov8s.pt`.',
        '- Cabang two-stage dibangun ulang: single-class detector, GT-crop classifier, dan evaluasi end-to-end.',
        '- Artefak Phase 3 baru tersedia di `outputs/phase3/` dan detail per-branch ada di `outputs/phase3/detail/`.',
    ])
    checkpoint('phase3 canonical sync complete')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--from-phase', choices=['phase1b', 'phase2', 'phase3'], default='phase1b')
    p.add_argument('--skip-root-readme', action='store_true')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log('master orchestrator started')
    write_state({'status': 'running', 'started_utc': utc_now(), 'from_phase': args.from_phase})
    cleanup_downloaded_root_weights()
    if args.from_phase == 'phase1b':
        phase1b()
        phase2()
        phase3()
    elif args.from_phase == 'phase2':
        phase2()
        phase3()
    else:
        phase3()
    if not args.skip_root_readme:
        write_root_readme_and_checkpoint()
    write_state({'status': 'completed', 'completed_utc': utc_now(), 'from_phase': args.from_phase})
    log('master orchestrator completed')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        write_state({'status': 'aborted', 'failed_utc': utc_now(), 'error': 'KeyboardInterrupt'})
        log('FATAL KeyboardInterrupt: orchestration interrupted')
        raise
    except Exception as e:
        write_state({'status': 'failed', 'failed_utc': utc_now(), 'error': f'{type(e).__name__}: {e}'})
        log(f'FATAL {type(e).__name__}: {e}')
        raise
