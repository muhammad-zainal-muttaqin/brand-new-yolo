#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO


LEDGER_COLUMNS = [
    'timestamp_utc','phase','run_name','model','imgsz','epochs','batch','seed','split',
    'data','save_dir','best_weight','last_weight','precision','recall','map50','map50_95','status'
]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_ledger(row: dict, ledger_path: Path) -> None:
    ensure_parent(ledger_path)
    exists = ledger_path.exists()
    with ledger_path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, '') for k in LEDGER_COLUMNS})


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
        f"- Precision: `{row['precision']}`",
        f"- Recall: `{row['recall']}`",
        f"- mAP50: `{row['map50']}`",
        f"- mAP50-95: `{row['map50_95']}`",
    ]
    status_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--phase', required=True)
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
    p.add_argument('--pretrained', action='store_true')
    p.add_argument('--plots', action='store_true')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    train_results = model.train(
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
        plots=args.plots,
    )
    save_dir = Path(train_results.save_dir)
    best_weight = save_dir / 'weights' / 'best.pt'
    last_weight = save_dir / 'weights' / 'last.pt'

    best_model = YOLO(str(best_weight if best_weight.exists() else args.model))
    metrics = best_model.val(data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch, device=args.device, workers=args.workers, plots=args.plots)

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
        'precision': float(getattr(metrics.box, 'mp', 0.0)),
        'recall': float(getattr(metrics.box, 'mr', 0.0)),
        'map50': float(getattr(metrics.box, 'map50', 0.0)),
        'map50_95': float(getattr(metrics.box, 'map', 0.0)),
        'status': 'completed',
    }

    summary_path = Path(f'outputs/{args.phase}/{args.name}_summary.json')
    ensure_parent(summary_path)
    summary_path.write_text(json.dumps(row, indent=2), encoding='utf-8')

    append_ledger(row, Path('outputs/reports/run_ledger.csv'))
    write_latest_status(row, save_dir, Path('outputs/reports/latest_status.md'))
    print(json.dumps(row, indent=2))


if __name__ == '__main__':
    main()
