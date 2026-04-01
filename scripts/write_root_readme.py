#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path('/workspace/brand-new-yolo')


def read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8').strip() if path.exists() else ''


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def bullet_list(items: list[str]) -> str:
    return '\n'.join(f'- {item}' for item in items)


def extract_phase_status() -> list[str]:
    state_path = ROOT / 'outputs/reports/master_state.json'
    if not state_path.exists():
        return ['master_state.json belum tersedia']
    state = read_json(state_path)
    items = [f"Status orchestrator: `{state.get('status', 'unknown')}`"]
    if 'started_utc' in state:
        items.append(f"Started UTC: `{state['started_utc']}`")
    if 'completed_utc' in state:
        items.append(f"Completed UTC: `{state['completed_utc']}`")
    if 'error' in state:
        items.append(f"Error: `{state['error']}`")
    return items


def summarize_phase0() -> str:
    return read_text(ROOT / 'outputs/phase0/phase0_summary.md')


def summarize_phase1() -> str:
    top3_rows = read_csv_rows(ROOT / 'outputs/phase1/phase1b_top3.csv')
    lock = read_yaml(ROOT / 'outputs/phase1/locked_setup.yaml') if (ROOT / 'outputs/phase1/locked_setup.yaml').exists() else {}
    lines = [read_text(ROOT / 'outputs/phase1/phase1_summary.md')]
    if top3_rows:
        lines.append('## Phase 1B Top-3 Canonical Architectures\n')
        for row in top3_rows:
            lines.append(
                f"- Rank {row['rank']}: `{row['model']}` | mean mAP50 `{float(row['mean_map50']):.4f}` | "
                f"mean mAP50-95 `{float(row['mean_map50_95']):.4f}` | "
                f"mean B4 recall `{row['mean_b4_recall']}`"
            )
    if lock:
        lines.append('\n## Locked Setup\n')
        semantics = lock.get('dataset_label_semantics', {})
        if semantics:
            lines.append(f"- Label order: `{semantics.get('maturity_order', [])}`")
            lines.append(f"- Direction: `{semantics.get('direction', '')}`")
            for key in ['B1', 'B2', 'B3', 'B4']:
                if key in semantics:
                    lines.append(f"- `{key}`: {semantics[key]}")
        if lock.get('phase1b_locked', {}).get('architecture_finalists'):
            locked_models = ', '.join(lock['phase1b_locked']['architecture_finalists'])
            policy = lock.get('phase1b_locked', {}).get('selection_policy')
            if policy == 'single_best_only_for_phase2':
                lines.append(f"- Phase 2 locked model: `{locked_models}`")
            else:
                lines.append(f"- Phase 2 finalists: `{locked_models}`")
        phase2_locked = lock.get('phase2_locked', {})
        if phase2_locked.get('selected_model'):
            lines.append(f"- Phase 2 selected model: `{phase2_locked['selected_model']}`")
        phase3_locked = lock.get('phase3_locked', {})
        if phase3_locked.get('candidates'):
            lines.append(f"- Phase 3 candidates: `{', '.join(phase3_locked['candidates'])}`")
    return '\n'.join([x for x in lines if x])


def summarize_phase2() -> str:
    lines = [read_text(ROOT / 'outputs/phase2/phase2_summary.md')]
    hparams_path = ROOT / 'outputs/phase2/final_hparams.yaml'
    if hparams_path.exists():
        hp = read_yaml(hparams_path)
        lines.append('\n## Final Phase 2 Configuration\n')
        for k, v in hp.items():
            lines.append(f'- {k}: `{v}`')
    return '\n'.join([x for x in lines if x])


def summarize_phase3() -> str:
    lines = [read_text(ROOT / 'outputs/phase3/final_report.md')]
    metrics = read_csv_rows(ROOT / 'outputs/phase3/final_metrics.csv')
    if metrics:
        lines.append('\n## Final Metrics Table\n')
        if 'metric' in metrics[0] and 'value' in metrics[0]:
            for row in metrics:
                lines.append(f"- {row['metric']}: `{row['value']}`")
        else:
            for row in metrics:
                lines.append(
                    f"- [{row.get('branch')}] `{row.get('candidate')}` / `{row.get('checkpoint')}` / `{row.get('split')}` | "
                    f"precision `{row.get('precision')}` | recall `{row.get('recall')}` | "
                    f"mAP50 `{row.get('map50')}` | weighted F1 `{row.get('weighted_f1')}`"
                )
    final_eval = read_text(ROOT / 'outputs/phase3/final_evaluation.md')
    if final_eval:
        lines.append('\n' + final_eval)
    deploy = read_text(ROOT / 'outputs/phase3/deploy_check.md')
    if deploy:
        lines.append('\n' + deploy)
    error = read_text(ROOT / 'outputs/phase3/error_analysis.md')
    if error:
        lines.append('\n' + error)
    return '\n'.join([x for x in lines if x])


def summarize_artifacts() -> str:
    items = [
        '`outputs/phase0/phase0_summary.md`',
        '`outputs/phase1/phase1_summary.md`',
        '`outputs/phase1/architecture_benchmark.csv`',
        '`outputs/phase1/per_class_metrics.csv`',
        '`outputs/phase1/locked_setup.yaml`',
        '`outputs/phase2/phase2_summary.md`',
        '`outputs/phase2/tuning_results.csv`',
        '`outputs/phase2/final_hparams.yaml`',
        '`outputs/phase3/final_report.md`',
        '`outputs/phase3/final_evaluation.md`',
        '`outputs/phase3/final_metrics.csv`',
        '`outputs/phase3/per_class_metrics.csv`',
        '`outputs/phase3/confusion_matrix.csv`',
        '`outputs/phase3/threshold_sweep.csv`',
        '`outputs/phase3/error_stratification.csv`',
        '`outputs/phase3/detail/`',
        '`outputs/phase3/figures/`',
        '`outputs/reports/run_ledger.csv`',
        '`outputs/reports/git_sync_log.md`',
    ]
    return bullet_list(items)


def summarize_weights() -> str:
    ledger_rows = read_csv_rows(ROOT / 'outputs/reports/run_ledger.csv')
    if not ledger_rows:
        return '- Ledger belum tersedia.'
    last_rows = ledger_rows[-10:]
    lines = []
    for row in last_rows:
        lines.append(
            f"- `{row['run_name']}` | model `{row['model']}` | best `{row['best_weight']}` | last `{row['last_weight']}`"
        )
    return '\n'.join(lines)


def main() -> None:
    readme = f"""# Brand New YOLO — E0 End-to-End Report

Repositori ini memuat eksekusi **E0 Baseline Experimental Protocol** untuk task deteksi tingkat kematangan tandan buah sawit 4 kelas pada dataset aktif repo ini.

## Canonical Protocol Source
- `E0.md`
- `https://github.com/muhammad-zainal-muttaqin/YOLOBench/blob/main/E0_Protocol_Flowchart.html`

## Root Semantic Mapping Used in This Repo
- `B1`: buah **merah**, **besar**, **bulat**, posisi **paling bawah** pada tandan → **paling matang / ripe**
- `B2`: buah masih **hitam** namun mulai **transisi ke merah**, sudah **besar** dan **bulat**, posisi **di atas B1**
- `B3`: buah **full hitam**, masih **berduri**, masih **lonjong**, posisi **di atas B2**
- `B4`: buah **paling kecil**, **paling dalam di batang/tandan**, sulit terlihat, masih banyak **duri**, warna **hitam sampai hijau**, masih bisa berkembang lebih besar → **paling belum matang**

Urutan biologis yang dipakai konsisten di repo ini adalah: **`B1 -> B2 -> B3 -> B4` = paling matang ke paling belum matang**.

## Orchestrator Status
{bullet_list(extract_phase_status())}

## Phase 0 — Validation & Calibration

{summarize_phase0()}

## Phase 1 — Pipeline Decision + Architecture Sweep

{summarize_phase1()}

## Phase 2 — Hyperparameter Optimization

{summarize_phase2()}

## Phase 3 — Final Validation

{summarize_phase3()}

## Key Artifacts
{summarize_artifacts()}

## Recent Weight Outputs
{summarize_weights()}

## Notes
- `GUIDE.md` adalah runbook operasional.
- `CONTEXT.md` memuat decision context dan caveat riset.
- `outputs/reports/run_ledger.csv` adalah ledger utama semua run.
- Seluruh workflow diatur untuk menyimpan hasil, commit, lalu push.
"""
    (ROOT / 'README.md').write_text(readme.strip() + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
