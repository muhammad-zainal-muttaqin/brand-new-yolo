#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss


LEDGER_COLUMNS = [
    'timestamp_utc','phase','run_name','model','imgsz','epochs','batch','seed','split',
    'data','save_dir','best_weight','last_weight','precision','recall','map50','map50_95',
    'top1_acc','top5_acc','task','single_cls','eval_checkpoint','fixed_epochs','train_no_val',
    'train_entry','eval_conf','status'
]
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


class ProtocolDetectionLoss(v8DetectionLoss):
    def __init__(
        self,
        model,
        imbalance_strategy: str = 'none',
        ordinal_strategy: str = 'standard',
        class_weights: list[float] | None = None,
        focal_gamma: float = 1.5,
        focal_alpha: float = 0.25,
    ):
        super().__init__(model)
        self.imbalance_strategy = imbalance_strategy
        self.ordinal_strategy = ordinal_strategy
        self.class_weights = torch.tensor(class_weights or [1.0] * self.nc, device=self.device, dtype=torch.float)
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def _build_ordinal_weights(self, target_scores: torch.Tensor, fg_mask: torch.Tensor, gt_labels: torch.Tensor, target_gt_idx: torch.Tensor, dtype) -> torch.Tensor:
        weights = torch.ones_like(target_scores, dtype=dtype)
        if self.ordinal_strategy != 'ordinal_weighted':
            return weights
        if not fg_mask.any() or gt_labels.shape[1] == 0 or self.nc <= 1:
            return weights
        safe_idx = target_gt_idx.clamp(min=0, max=max(gt_labels.shape[1] - 1, 0))
        assigned_cls = gt_labels.squeeze(-1).gather(1, safe_idx).long()
        class_ids = torch.arange(self.nc, device=self.device).view(1, 1, -1)
        dist = (class_ids - assigned_cls.unsqueeze(-1)).abs().float()
        if self.nc == 2:
            neg_weight = torch.where(dist > 0, torch.full_like(dist, 0.5), torch.ones_like(dist))
        else:
            denom = max(self.nc - 2, 1)
            neg_weight = 0.5 + 0.5 * ((dist - 1).clamp(min=0) / denom)
        ordinal = torch.where(target_scores > 0, torch.ones_like(target_scores), neg_weight)
        return torch.where(fg_mask.unsqueeze(-1), ordinal.to(dtype), weights)

    def _classification_loss(self, pred_scores: torch.Tensor, target_scores: torch.Tensor, fg_mask: torch.Tensor, gt_labels: torch.Tensor, target_gt_idx: torch.Tensor) -> torch.Tensor:
        dtype = pred_scores.dtype
        target_scores = target_scores.to(dtype)
        loss = F.binary_cross_entropy_with_logits(pred_scores, target_scores, reduction='none')

        if self.imbalance_strategy == 'focal':
            pred_prob = pred_scores.sigmoid()
            p_t = target_scores * pred_prob + (1 - target_scores) * (1 - pred_prob)
            modulating_factor = (1.0 - p_t) ** self.focal_gamma
            alpha = self.focal_alpha
            alpha_factor = target_scores * alpha + (1 - target_scores) * (1 - alpha)
            loss *= modulating_factor * alpha_factor
        elif self.imbalance_strategy == 'class_weighted':
            loss *= self.class_weights.to(dtype).view(1, 1, -1)

        if self.ordinal_strategy == 'ordinal_weighted':
            loss *= self._build_ordinal_weights(target_scores, fg_mask, gt_labels, target_gt_idx, dtype)

        return loss

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple:
        loss = torch.zeros(3, device=self.device)
        pred_distri, pred_scores = (
            preds['boxes'].permute(0, 2, 1).contiguous(),
            preds['scores'].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds['feats'], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds['feats'][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        cls_loss = self._classification_loss(pred_scores, target_scores, fg_mask, gt_labels, target_gt_idx)
        loss[1] = cls_loss.sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )


# Imported late in source order to keep the custom loss readable.
from ultralytics.utils.loss import make_anchors  # noqa: E402


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_ledger(row: dict, ledger_path: Path) -> None:
    ensure_parent(ledger_path)
    rows: list[dict] = []
    if ledger_path.exists():
        with ledger_path.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    rows = [r for r in rows if not (r.get('phase') == str(row.get('phase')) and r.get('run_name') == str(row.get('run_name')))]
    rows.append({k: row.get(k, '') for k in LEDGER_COLUMNS})
    with ledger_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_latest_status(row: dict, save_dir: Path, status_path: Path) -> None:
    ensure_parent(status_path)
    lines = [
        '# Latest Execution Status',
        '',
        f"- Timestamp UTC: `{row['timestamp_utc']}`",
        f"- Phase: `{row['phase']}`",
        f"- Run name: `{row['run_name']}`",
        f"- Model: `{row['model']}`",
        f"- Image size: `{row['imgsz']}`",
        f"- Epochs: `{row['epochs']}`",
        f"- Batch: `{row['batch']}`",
        f"- Seed: `{row['seed']}`",
        f"- Eval split: `{row['split']}`",
        f"- Status: **{row['status']}**",
        f"- Save dir: `{save_dir}`",
        f"- Best weight: `{row['best_weight']}`",
        f"- Last weight: `{row['last_weight']}`",
        f"- Eval checkpoint: `{row.get('eval_checkpoint', 'best')}`",
        f"- Fixed epochs: `{row.get('fixed_epochs', False)}`",
        f"- Train no val: `{row.get('train_no_val', False)}`",
        f"- Train entry: `{row.get('train_entry', '')}`",
        f"- Eval conf: `{row.get('eval_conf', '')}`",
        f"- Precision: `{row['precision']}`",
        f"- Recall: `{row['recall']}`",
        f"- mAP50: `{row['map50']}`",
        f"- mAP50-95: `{row['map50_95']}`",
    ]
    if row.get('top1_acc') is not None:
        lines.append(f"- Top1 acc: `{row['top1_acc']}`")
    if row.get('top5_acc') is not None:
        lines.append(f"- Top5 acc: `{row['top5_acc']}`")
    if row.get('metric_schema_note'):
        lines.append(f"- Metric note: `{row['metric_schema_note']}`")
    status_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))


def resolve_dataset_root(cfg: dict, yaml_path: Path) -> Path:
    root = Path(cfg.get('path', yaml_path.parent))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    return root


def resolve_entry_path(base: Path, entry: str) -> Path:
    p = Path(entry)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def iter_split_images(data_yaml: str, split: str) -> list[Path]:
    yaml_path = Path(data_yaml).resolve()
    cfg = load_yaml(str(yaml_path))
    root = resolve_dataset_root(cfg, yaml_path)
    entry = cfg[split]
    target = resolve_entry_path(root, entry)
    if target.is_file():
        images = []
        for line in target.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line:
                images.append(Path(line))
        return images
    if target.is_dir():
        return sorted([p for p in target.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES])
    raise FileNotFoundError(f'Could not resolve split {split}: {target}')


def image_to_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if 'images' in parts:
        idx = parts.index('images')
        parts[idx] = 'labels'
        return Path(*parts).with_suffix('.txt')
    return Path(str(image_path).replace('/images/', '/labels/')).with_suffix('.txt')


def compute_class_weights(data_yaml: str, nc: int) -> list[float]:
    counts = [0] * nc
    for img_path in iter_split_images(data_yaml, 'train'):
        label_path = image_to_label_path(img_path)
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding='utf-8').splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(float(parts[0]))
            if 0 <= cls < nc:
                counts[cls] += 1
    safe_counts = [c if c > 0 else 1 for c in counts]
    inv = [sum(safe_counts) / c for c in safe_counts]
    mean_inv = sum(inv) / len(inv)
    return [x / mean_inv for x in inv]


def install_custom_detection_loss(
    model: YOLO,
    task: str,
    imbalance_strategy: str,
    ordinal_strategy: str,
    data_yaml: str,
    focal_gamma: float,
) -> None:
    if task != 'detect':
        return
    if imbalance_strategy == 'none' and ordinal_strategy == 'standard':
        return
    class_weights = None
    if imbalance_strategy == 'class_weighted':
        cfg = load_yaml(data_yaml)
        nc = int(cfg['nc'])
        class_weights = compute_class_weights(data_yaml, nc)
    detection_model = model.model

    def init_criterion():
        return ProtocolDetectionLoss(
            detection_model,
            imbalance_strategy=imbalance_strategy,
            ordinal_strategy=ordinal_strategy,
            class_weights=class_weights,
            focal_gamma=focal_gamma,
        )

    detection_model.init_criterion = init_criterion
    detection_model.criterion = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--phase', required=True)
    p.add_argument('--task', choices=['detect', 'classify'], default='detect')
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--imgsz', type=int, required=True)
    p.add_argument('--epochs', type=int, required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--project', default='runs/e0')
    p.add_argument('--name', required=True)
    p.add_argument('--split', default='val')
    p.add_argument('--device', default='0')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--min-epochs', type=int, default=0)
    p.add_argument('--fraction', type=float, default=1.0)
    p.add_argument('--single-cls', action='store_true')
    p.add_argument('--pretrained', action='store_true')
    p.add_argument('--plots', action='store_true')
    p.add_argument('--optimizer', default=None)
    p.add_argument('--lr0', type=float, default=None)
    p.add_argument('--lrf', type=float, default=None)
    p.add_argument('--hsv-h', type=float, default=None)
    p.add_argument('--hsv-s', type=float, default=None)
    p.add_argument('--hsv-v', type=float, default=None)
    p.add_argument('--degrees', type=float, default=None)
    p.add_argument('--translate', type=float, default=None)
    p.add_argument('--scale', type=float, default=None)
    p.add_argument('--mosaic', type=float, default=None)
    p.add_argument('--mixup', type=float, default=None)
    p.add_argument('--copy-paste', type=float, default=None)
    p.add_argument('--close-mosaic', type=int, default=None)
    p.add_argument('--conf', type=float, default=None)
    p.add_argument('--eval-checkpoint', choices=['best', 'last'], default='best')
    p.add_argument('--fixed-epochs', action='store_true')
    p.add_argument('--train-no-val', action='store_true')
    p.add_argument('--imbalance-strategy', choices=['none', 'class_weighted', 'focal'], default='none')
    p.add_argument('--ordinal-strategy', choices=['standard', 'ordinal_weighted', 'coral'], default='standard')
    p.add_argument('--focal-gamma', type=float, default=1.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.ordinal_strategy == 'coral' and args.task != 'classify':
        raise ValueError('CORAL hanya relevan untuk pipeline classify / two-stage classifier.')

    model = YOLO(args.model)
    install_custom_detection_loss(
        model=model,
        task=args.task,
        imbalance_strategy=args.imbalance_strategy,
        ordinal_strategy=args.ordinal_strategy,
        data_yaml=args.data,
        focal_gamma=args.focal_gamma,
    )

    if args.min_epochs:
        def keep_training_until_min_epochs(trainer):
            current_epoch = trainer.epoch + 1
            if current_epoch < args.min_epochs:
                trainer.stop = False
                if getattr(trainer, 'stopper', None) is not None:
                    trainer.stopper.possible_stop = False
        model.add_callback('on_fit_epoch_end', keep_training_until_min_epochs)
    train_kwargs = dict(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        seed=args.seed,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=args.pretrained,
        patience=args.patience,
        fraction=args.fraction,
        single_cls=args.single_cls,
        plots=args.plots,
    )
    if args.fixed_epochs:
        # Disable early stopping by making patience unreachable within the scheduled epoch budget.
        train_kwargs['patience'] = max(args.patience, args.epochs + 1)
    if args.train_no_val:
        train_kwargs['val'] = False
    optional_train_args = {
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'hsv_h': args.hsv_h,
        'hsv_s': args.hsv_s,
        'hsv_v': args.hsv_v,
        'degrees': args.degrees,
        'translate': args.translate,
        'scale': args.scale,
        'mosaic': args.mosaic,
        'mixup': args.mixup,
        'copy_paste': args.copy_paste,
        'close_mosaic': args.close_mosaic,
    }
    for k, v in optional_train_args.items():
        if v is not None:
            train_kwargs[k] = v

    train_results = model.train(**train_kwargs)
    save_dir = Path(train_results.save_dir)
    best_weight = save_dir / 'weights' / 'best.pt'
    last_weight = save_dir / 'weights' / 'last.pt'
    if args.train_no_val and best_weight.exists():
        # In no-validation runs, best.pt is not a meaningful selection artifact.
        best_weight.unlink()

    eval_weight = best_weight if args.eval_checkpoint == 'best' else last_weight
    if not eval_weight.exists():
        fallback_weight = best_weight if best_weight.exists() else last_weight
        eval_weight = fallback_weight if fallback_weight.exists() else Path(args.model)
    best_model = YOLO(str(eval_weight))
    val_kwargs = dict(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        single_cls=args.single_cls if args.task == 'detect' else False,
        plots=args.plots,
    )
    if args.conf is not None:
        val_kwargs['conf'] = args.conf
    metrics = best_model.val(**val_kwargs)

    train_entry = ''
    data_path = Path(args.data)
    if data_path.exists() and data_path.suffix.lower() in {'.yaml', '.yml'}:
        train_entry = str(load_yaml(args.data).get('train', ''))

    top1_acc = None
    top5_acc = None
    metric_schema_note = None
    if args.task == 'classify':
        top1_acc = float(getattr(metrics, 'top1', 0.0))
        top5_acc = float(getattr(metrics, 'top5', 0.0))
        precision = top1_acc
        recall = top5_acc
        map50 = top1_acc
        map50_95 = top5_acc
        metric_schema_note = 'classification task: precision/top1 and recall/top5 are aliases for compatibility; mAP fields are not detection mAP'
    else:
        precision = float(getattr(metrics.box, 'mp', 0.0))
        recall = float(getattr(metrics.box, 'mr', 0.0))
        map50 = float(getattr(metrics.box, 'map50', 0.0))
        map50_95 = float(getattr(metrics.box, 'map', 0.0))

    row = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'phase': args.phase,
        'run_name': args.name,
        'model': args.model,
        'imgsz': args.imgsz,
        'epochs': args.epochs,
        'batch': args.batch,
        'seed': args.seed,
        'split': args.split,
        'data': args.data,
        'save_dir': str(save_dir),
        'best_weight': str(best_weight) if best_weight.exists() else '',
        'last_weight': str(last_weight) if last_weight.exists() else '',
        'precision': precision,
        'recall': recall,
        'map50': map50,
        'map50_95': map50_95,
        'top1_acc': top1_acc,
        'top5_acc': top5_acc,
        'metric_schema_note': metric_schema_note,
        'eval_checkpoint': args.eval_checkpoint,
        'fixed_epochs': args.fixed_epochs,
        'train_no_val': args.train_no_val,
        'train_entry': train_entry,
        'eval_conf': args.conf,
        'status': 'completed',
        'fraction': args.fraction,
        'single_cls': args.single_cls,
        'min_epochs': args.min_epochs,
        'patience': args.patience,
        'task': args.task,
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'imbalance_strategy': args.imbalance_strategy,
        'ordinal_strategy': args.ordinal_strategy,
        'focal_gamma': args.focal_gamma,
        'augmentations': {
            'hsv_h': args.hsv_h,
            'hsv_s': args.hsv_s,
            'hsv_v': args.hsv_v,
            'degrees': args.degrees,
            'translate': args.translate,
            'scale': args.scale,
            'mosaic': args.mosaic,
            'mixup': args.mixup,
            'copy_paste': args.copy_paste,
            'close_mosaic': args.close_mosaic,
        },
    }

    summary_path = Path(f'outputs/{args.phase}/{args.name}_summary.json')
    ensure_parent(summary_path)
    summary_path.write_text(json.dumps(row, indent=2), encoding='utf-8')

    append_ledger(row, Path('outputs/reports/run_ledger.csv'))
    write_latest_status(row, save_dir, Path('outputs/reports/latest_status.md'))
    print(json.dumps(row, indent=2))


if __name__ == '__main__':
    main()
