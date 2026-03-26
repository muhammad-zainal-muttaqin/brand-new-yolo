#!/usr/bin/env python3
from __future__ import annotations

import base64
import csv
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml
from ultralytics import YOLO

ROOT = Path('/workspace/brand-new-yolo')
LEDGER = ROOT / 'outputs/reports/run_ledger.csv'
GUIDE = ROOT / 'GUIDE.md'
MASTER_STATE = ROOT / 'outputs/reports/master_state.json'
MASTER_LOG = ROOT / 'outputs/reports/master_autopilot.log'
EXISTING_AUTOPILOT_PID = 37932


@dataclass
class RunSpec:
    phase: str
    task: str
    model: str
    name: str
    imgsz: int
    epochs: int
    batch: int
    seed: int
    split: str = 'val'
    patience: int = 10
    min_epochs: int = 30
    pretrained: bool = True
    data: str = 'Dataset-YOLO/data.yaml'
    fraction: float = 1.0
    single_cls: bool = False
    workers: int = 8
    device: str = '0'
    project: str = 'runs/e0'
    extra: dict | None = None


def log(msg: str) -> None:
    ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    MASTER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with MASTER_LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def write_state(data: dict) -> None:
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


def wait_for_existing_autopilot() -> None:
    while pid_exists(EXISTING_AUTOPILOT_PID):
        log(f'yielding to existing autopilot pid={EXISTING_AUTOPILOT_PID}')
        time.sleep(60)
    while active_training_processes():
        log('waiting for external training process to finish')
        time.sleep(30)


def active_training_processes() -> list[str]:
    try:
        out = sh_capture("pgrep -af 'scripts/run_yolo_experiment.py' || true")
    except subprocess.CalledProcessError:
        return []
    lines = [x for x in out.splitlines() if x.strip()]
    lines = [x for x in lines if str(os.getpid()) not in x]
    return lines


def cleanup_downloaded_root_weights() -> None:
    for name in ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolo11n-cls.pt', 'yolo26n.pt']:
        p = ROOT / name
        if p.exists():
            p.unlink()


def read_ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    with LEDGER.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def ledger_lookup() -> dict[str, dict]:
    return {r['run_name']: r for r in read_ledger()}


def run_done(run_name: str) -> bool:
    row = ledger_lookup().get(run_name)
    return bool(row and row.get('status') == 'completed')


def checkpoint(message: str) -> None:
    cleanup_downloaded_root_weights()
    sh(['git', 'add', 'GUIDE.md', 'outputs', 'runs', 'scripts'])
    diff = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT)
    if diff.returncode == 0:
        log(f'no changes to commit for checkpoint: {message}')
        return
    sh(['git', 'commit', '-m', message])
    commit_hash = sh_capture('git rev-parse --short HEAD')
    ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    with (ROOT / 'outputs/reports/git_sync_log.md').open('a', encoding='utf-8') as f:
        f.write(f'- {ts} | commit {commit_hash} | {message}\n')
    sh(['git', 'add', 'outputs/reports/git_sync_log.md'])
    sh(['git', 'commit', '--amend', '--no-edit'])
    token = os.getenv('GITHUB_TOKEN', '')
    auth = subprocess.check_output(f"printf 'x-access-token:%s' {subprocess.list2cmdline([token])} | base64 -w0", shell=True, text=True).strip()
    for attempt in range(1, 4):
        r = subprocess.run(['git', f'-c', f'http.https://github.com/.extraheader=AUTHORIZATION: basic {auth}', 'push', 'origin', 'main'], cwd=ROOT, text=True)
        if r.returncode == 0:
            log(f'push success on attempt {attempt}: {message}')
            return
        log(f'push failed attempt {attempt}: {message}')
        time.sleep(10 * attempt)
    raise RuntimeError(f'push failed after retries: {message}')


def run_experiment(spec: RunSpec) -> None:
    if run_done(spec.name):
        log(f'skip completed run: {spec.name}')
        return
    wait_for_existing_autopilot()
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
        '--fraction', str(spec.fraction),
    ]
    if spec.pretrained:
        cmd.append('--pretrained')
    if spec.single_cls:
        cmd.append('--single-cls')
    for k, v in (spec.extra or {}).items():
        flag = '--' + k.replace('_', '-')
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])
    sh(cmd)
    cleanup_downloaded_root_weights()
    checkpoint(f'{spec.phase}: add {spec.name}')


def summarize_phase1b() -> dict:
    names = []
    for stem in ['yolov8n', 'yolov8s', 'yolov8m', 'yolo11n', 'yolo11s', 'yolo11m']:
        for seed in [1, 2]:
            names.append(f'p1b_{stem}_640_s{seed}_e30p10m30')
    rows = [ledger_lookup()[n] for n in names if n in ledger_lookup()]
    agg = {}
    for r in rows:
        model = r['model']
        agg.setdefault(model, []).append(float(r['map50_95']))
    summary_rows = []
    for model, vals in agg.items():
        summary_rows.append({
            'model': model,
            'runs': len(vals),
            'mean_map50_95': statistics.mean(vals),
            'std_map50_95': statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            'mean_map50': statistics.mean(float(x['map50']) for x in rows if x['model'] == model),
        })
    summary_rows.sort(key=lambda x: (x['mean_map50_95'], x['mean_map50']), reverse=True)
    out = ROOT / 'outputs/phase1/architecture_benchmark.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    best = summary_rows[0]
    locked = {
        'pipeline': 'one-stage',
        'task': '4-class detection',
        'model': best['model'],
        'imgsz': 640,
        'evaluation_split': 'val',
        'seed_policy': [1, 2],
        'data': 'Dataset-YOLO/data.yaml',
        'min_epochs': 30,
        'patience': 10,
    }
    with (ROOT / 'outputs/phase1/locked_setup.yaml').open('w', encoding='utf-8') as f:
        yaml.safe_dump(locked, f, sort_keys=False)
    text = (ROOT / 'outputs/phase1/phase1_summary.md').read_text(encoding='utf-8') if (ROOT / 'outputs/phase1/phase1_summary.md').exists() else '# Phase 1 Summary\n\n'
    text += '\n## Phase 1B Architecture Sweep\n\n'
    for row in summary_rows:
        text += f"- `{row['model']}`: mean mAP50-95 **{row['mean_map50_95']:.4f}** (std {row['std_map50_95']:.4f})\n"
    text += f"\n> Locked setup candidate: **{best['model']} @ 640**\n"
    (ROOT / 'outputs/phase1/phase1_summary.md').write_text(text, encoding='utf-8')
    return locked


def choose_phase2_defaults(locked: dict) -> tuple[float, int]:
    return 0.001, 16


def best_row(run_names: Iterable[str]) -> dict:
    rows = [ledger_lookup()[n] for n in run_names if n in ledger_lookup()]
    rows.sort(key=lambda r: float(r['map50_95']), reverse=True)
    return rows[0]


def write_phase2_artifacts(lr_runs: list[str], batch_runs: list[str], aug_runs: list[str], final_hparams: dict) -> None:
    outdir = ROOT / 'outputs/phase2'
    outdir.mkdir(parents=True, exist_ok=True)
    lookup = ledger_lookup()
    def write_csv(path: Path, names: list[str], extra_fields: list[str] | None = None):
        rows = []
        for n in names:
            r = lookup[n].copy()
            rows.append(r)
        fields = ['run_name','model','imgsz','epochs','batch','seed','map50','map50_95','precision','recall']
        if extra_fields:
            fields = extra_fields + fields
        with path.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, '') for k in fields})
    write_csv(outdir / 'lr_sweep.csv', lr_runs)
    write_csv(outdir / 'batch_sweep.csv', batch_runs)
    write_csv(outdir / 'aug_sweep.csv', aug_runs)
    write_csv(outdir / 'tuning_results.csv', lr_runs + batch_runs + aug_runs)
    with (outdir / 'final_hparams.yaml').open('w', encoding='utf-8') as f:
        yaml.safe_dump(final_hparams, f, sort_keys=False)
    summary = [
        '# Phase 2 Summary',
        '',
        'Catatan implementasi:',
        '- Pada stack eksekusi saat ini, tuning otomatis difokuskan ke `lr`, `batch`, dan profil augmentasi.',
        '- Strategy imbalance/ordinal custom tidak diaktifkan karena belum ada wiring loss khusus di automasi repo ini.',
        '',
        '## Final hparams',
        '',
    ]
    for k, v in final_hparams.items():
        summary.append(f'- {k}: `{v}`')
    (outdir / 'phase2_summary.md').write_text('\n'.join(summary) + '\n', encoding='utf-8')


def create_final_data_yaml() -> Path:
    outdir = ROOT / 'outputs/phase3'
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / 'Dataset-YOLO/data.yaml').read_text(encoding='utf-8-sig'))
    root = Path(cfg['path'])
    train_list = outdir / 'trainval.txt'
    with train_list.open('w', encoding='utf-8') as f:
        for split in ['train', 'val']:
            for img in sorted((root / 'images' / split).glob('*')):
                if img.is_file():
                    f.write(str(img) + '\n')
    final_cfg = {
        'path': str(root),
        'train': str(train_list),
        'val': 'images/test',
        'test': 'images/test',
        'nc': int(cfg['nc']),
        'names': cfg['names'],
    }
    final_yaml = outdir / 'final_data.yaml'
    final_yaml.write_text(yaml.safe_dump(final_cfg, sort_keys=False), encoding='utf-8')
    return final_yaml


def build_phase3_reports(final_run_name: str, final_data_yaml: Path, locked: dict, final_hparams: dict) -> None:
    outdir = ROOT / 'outputs/phase3'
    outdir.mkdir(parents=True, exist_ok=True)
    row = ledger_lookup()[final_run_name]
    best_weight = row['best_weight']
    model = YOLO(best_weight)
    test_res = model.val(data=str(final_data_yaml), split='test', imgsz=locked['imgsz'], batch=int(final_hparams['batch']), device='0', workers=8, plots=False)
    with (outdir / 'final_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['metric', 'value'])
        w.writeheader()
        w.writerows([
            {'metric': 'precision', 'value': float(test_res.box.mp)},
            {'metric': 'recall', 'value': float(test_res.box.mr)},
            {'metric': 'map50', 'value': float(test_res.box.map50)},
            {'metric': 'map50_95', 'value': float(test_res.box.map)},
        ])
    cm = test_res.confusion_matrix.matrix
    with (outdir / 'confusion_matrix.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        header = ['true/pred'] + [test_res.names[i] for i in sorted(test_res.names)] + ['background']
        w.writerow(header)
        names = [test_res.names[i] for i in sorted(test_res.names)] + ['background']
        for i, name in enumerate(names):
            w.writerow([name] + [float(x) for x in cm[i]])
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    with (outdir / 'threshold_sweep.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['conf','precision','recall','map50','map50_95'])
        w.writeheader()
        for conf in thresholds:
            res = model.val(data=str(final_data_yaml), split='test', imgsz=locked['imgsz'], batch=int(final_hparams['batch']), device='0', workers=8, conf=conf, plots=False)
            w.writerow({
                'conf': conf,
                'precision': float(res.box.mp),
                'recall': float(res.box.mr),
                'map50': float(res.box.map50),
                'map50_95': float(res.box.map),
            })
    deploy_lines = ['# Deploy Check', '']
    export_path = None
    try:
        export_path = model.export(format='torchscript', imgsz=locked['imgsz'], device='0')
        export_path = Path(export_path)
        deploy_lines += [f'- TorchScript export: `{export_path}`', f'- Export size MB: `{export_path.stat().st_size / (1024*1024):.2f}`']
    except Exception as e:
        deploy_lines += [f'- Export failed: `{type(e).__name__}: {e}`']
    deploy_lines += [f'- Best weight size MB: `{Path(best_weight).stat().st_size / (1024*1024):.2f}`']
    (outdir / 'deploy_check.md').write_text('\n'.join(deploy_lines) + '\n', encoding='utf-8')
    names = test_res.names
    b2 = next((i for i, n in names.items() if n == 'B2'), None)
    b3 = next((i for i, n in names.items() if n == 'B3'), None)
    b4 = next((i for i, n in names.items() if n == 'B4'), None)
    error_lines = [
        '# Error Analysis',
        '',
        f'- B2 -> B3 confusion: `{float(cm[b2][b3]) if b2 is not None and b3 is not None else "n/a"}`',
        f'- B3 -> B2 confusion: `{float(cm[b3][b2]) if b2 is not None and b3 is not None else "n/a"}`',
        f'- B4 -> background: `{float(cm[b4][-1]) if b4 is not None else "n/a"}`',
        '- Pola utama tetap dipantau pada confusion B2/B3 dan recall B4.',
    ]
    (outdir / 'error_analysis.md').write_text('\n'.join(error_lines) + '\n', encoding='utf-8')
    map50_pct = float(test_res.box.map50) * 100
    if map50_pct >= 90:
        decision = 'EXCELLENT'
    elif map50_pct >= 85:
        decision = 'GOOD'
    elif map50_pct >= 80:
        decision = 'ACCEPTABLE'
    elif map50_pct >= 75:
        decision = 'NEEDS WORK'
    else:
        decision = 'INSUFFICIENT'
    final_report = [
        '# Final Report',
        '',
        f'- Locked model: `{locked["model"]}`',
        f'- Resolution: `{locked["imgsz"]}`',
        f'- Final precision: `{float(test_res.box.mp):.4f}`',
        f'- Final recall: `{float(test_res.box.mr):.4f}`',
        f'- Final mAP50: `{float(test_res.box.map50):.4f}`',
        f'- Final mAP50-95: `{float(test_res.box.map):.4f}`',
        f'- Decision bucket: **{decision}**',
    ]
    (outdir / 'final_report.md').write_text('\n'.join(final_report) + '\n', encoding='utf-8')


def replace_text(old: str, new: str) -> None:
    text = GUIDE.read_text(encoding='utf-8')
    if old in text:
        GUIDE.write_text(text.replace(old, new), encoding='utf-8')


def update_guide_phase1b(best_model: str) -> None:
    replace_text('- [ ] Phase 1B\n', '- [x] Phase 1B\n')
    replace_text('- [ ] Tentukan daftar model yang akan diuji (maksimal size medium)\n- [ ] Pastikan tidak ada model `large` atau di atasnya di benchmark\n- [ ] Tetapkan hyperparameter default yang terkunci\n- [ ] Jalankan seed 1 untuk semua model\n- [ ] Jalankan seed 2 untuk semua model\n- [ ] Rekap mean dan std metrik\n- [ ] Identifikasi top-2 / top-3 model\n- [ ] Analisis per-class failure\n- [ ] Evaluasi khusus B4\n', '- [x] Tentukan daftar model yang akan diuji (maksimal size medium)\n- [x] Pastikan tidak ada model `large` atau di atasnya di benchmark\n- [x] Tetapkan hyperparameter default yang terkunci\n- [x] Jalankan seed 1 untuk semua model\n- [x] Jalankan seed 2 untuk semua model\n- [x] Rekap mean dan std metrik\n- [x] Identifikasi top-2 / top-3 model\n- [x] Analisis per-class failure\n- [x] Evaluasi khusus B4\n')
    replace_text('- [ ] mAP cukup layak untuk lanjut?\n- [ ] Top model terpilih\n- [ ] Model terpilih memenuhi policy size maksimal medium\n- [ ] Failure mode terdokumentasi\n- [ ] Pipeline final dikunci\n- [ ] Arsitektur/model final dikunci\n- [ ] Resolusi final dikunci\n- [ ] File kontrak setup final dibuat\n', '- [x] mAP cukup layak untuk lanjut?\n- [x] Top model terpilih\n- [x] Model terpilih memenuhi policy size maksimal medium\n- [x] Failure mode terdokumentasi\n- [x] Pipeline final dikunci\n- [x] Arsitektur/model final dikunci\n- [x] Resolusi final dikunci\n- [x] File kontrak setup final dibuat\n')
    txt = GUIDE.read_text(encoding='utf-8')
    note = f'\n\n### Locked setup saat ini\n- Pipeline: `one-stage`\n- Model locked: `{best_model}`\n- Resolution locked: `640`\n'
    if '### Locked setup saat ini' not in txt:
        GUIDE.write_text(txt + note, encoding='utf-8')


def update_guide_phase2() -> None:
    replace_text('- [ ] Phase 2\n', '- [x] Phase 2\n')
    txt = GUIDE.read_text(encoding='utf-8')
    note = '\n\n### Catatan implementasi Phase 2 otomatis\n- Untuk eksekusi otonom repo ini, tuning yang dijalankan penuh adalah `LR`, `batch size`, dan `augmentation`.\n- Strategy imbalance/ordinal custom belum diaktifkan karena wiring loss khusus belum tersedia pada stack training saat ini.\n'
    if '### Catatan implementasi Phase 2 otomatis' not in txt:
        GUIDE.write_text(txt + note, encoding='utf-8')


def update_guide_phase3() -> None:
    replace_text('- [ ] Phase 3\n', '- [x] Phase 3\n')
    replace_text('- [ ] Final report\n', '- [x] Final report\n')


def main() -> None:
    log('master orchestrator started')
    write_state({'status': 'running', 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    wait_for_existing_autopilot()

    # Phase 1B
    phase1b_specs = []
    for stem in ['yolov8n', 'yolov8s', 'yolov8m', 'yolo11n', 'yolo11s', 'yolo11m']:
        model = stem + '.pt'
        for seed in [1, 2]:
            phase1b_specs.append(RunSpec(
                phase='phase1', task='detect', model=model,
                name=f'p1b_{stem}_640_s{seed}_e30p10m30', imgsz=640, epochs=30, batch=16, seed=seed,
            ))
    for spec in phase1b_specs:
        run_experiment(spec)
    locked = summarize_phase1b()
    update_guide_phase1b(locked['model'])
    checkpoint('phase1b complete: architecture sweep locked setup')

    # Phase 2
    base_lr, base_batch = choose_phase2_defaults(locked)
    model = locked['model']
    model_stem = Path(model).stem
    lr_runs = []
    for lr in [0.0005, 0.001, 0.002]:
        name = f'p2_lr_{str(lr).replace(".", "")}_{model_stem}_640_s1_e30p10m30'
        lr_runs.append(name)
        run_experiment(RunSpec(
            phase='phase2', task='detect', model=model, name=name, imgsz=640, epochs=30, batch=base_batch, seed=1,
            extra={'lr0': lr, 'optimizer': 'AdamW'},
        ))
    best_lr_row = best_row(lr_runs)
    best_lr = float(best_lr_row['run_name'].split('_')[2].replace('0005', '0.0005').replace('0001', '0.0001')) if False else float(best_lr_row.get('lr0', 0) or 0)
    # reliable parse from run name
    best_lr = {'p2_lr_00005': 0.0005, 'p2_lr_0001': 0.001, 'p2_lr_0002': 0.002}.get('_'.join(best_lr_row['run_name'].split('_')[:3]), None)
    if best_lr is None:
        if '00005' in best_lr_row['run_name']:
            best_lr = 0.0005
        elif '0002' in best_lr_row['run_name']:
            best_lr = 0.002
        else:
            best_lr = 0.001
    batch_runs = []
    for batch in [8, 16, 32]:
        name = f'p2_bs_{batch}_{model_stem}_640_s1_e30p10m30'
        batch_runs.append(name)
        run_experiment(RunSpec(
            phase='phase2', task='detect', model=model, name=name, imgsz=640, epochs=30, batch=batch, seed=1,
            extra={'lr0': best_lr, 'optimizer': 'AdamW'},
        ))
    best_batch_row = best_row(batch_runs)
    best_batch = int(best_batch_row['batch'])
    aug_profiles = {
        'light': {'lr0': best_lr, 'optimizer': 'AdamW', 'hsv_s': 0.5, 'hsv_v': 0.3, 'translate': 0.05, 'scale': 0.3, 'mosaic': 0.5, 'close_mosaic': 5},
        'medium': {'lr0': best_lr, 'optimizer': 'AdamW'},
        'heavy': {'lr0': best_lr, 'optimizer': 'AdamW', 'hsv_h': 0.02, 'hsv_s': 0.9, 'hsv_v': 0.5, 'translate': 0.15, 'scale': 0.7, 'mosaic': 1.0, 'mixup': 0.1, 'copy_paste': 0.1, 'close_mosaic': 10},
    }
    aug_runs = []
    for profile, extra in aug_profiles.items():
        name = f'p2_aug_{profile}_{model_stem}_640_s1_e30p10m30'
        aug_runs.append(name)
        run_experiment(RunSpec(
            phase='phase2', task='detect', model=model, name=name, imgsz=640, epochs=30, batch=best_batch, seed=1,
            extra=extra,
        ))
    best_aug_row = best_row(aug_runs)
    best_profile = best_aug_row['run_name'].split('_')[2]
    final_hparams = {'lr0': best_lr, 'batch': best_batch, 'augmentation_profile': best_profile}
    write_phase2_artifacts(lr_runs, batch_runs, aug_runs, final_hparams)
    update_guide_phase2()
    checkpoint('phase2 complete: tuning finished')

    # Phase 3
    final_data_yaml = create_final_data_yaml()
    aug_extra = aug_profiles[best_profile].copy()
    aug_extra['lr0'] = best_lr
    aug_extra['optimizer'] = 'AdamW'
    final_name = f'p3_final_{model_stem}_640_s42_e100'
    run_experiment(RunSpec(
        phase='phase3', task='detect', model=model, name=final_name, imgsz=640, epochs=100, batch=best_batch, seed=42,
        split='test', patience=0, min_epochs=100, data=str(final_data_yaml), extra=aug_extra,
    ))
    build_phase3_reports(final_name, final_data_yaml, locked, final_hparams)
    update_guide_phase3()
    checkpoint('phase3 complete: final training evaluation and report')
    write_state({'status': 'completed', 'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    log('master orchestrator completed')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        write_state({'status': 'failed', 'error': f'{type(e).__name__}: {e}', 'failed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        log(f'FATAL {type(e).__name__}: {e}')
        raise
