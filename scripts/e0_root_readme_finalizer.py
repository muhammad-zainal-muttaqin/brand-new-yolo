#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/brand-new-yolo')
STATE = ROOT / 'outputs/reports/master_state.json'
LOG = ROOT / 'outputs/reports/root_readme_finalizer.log'
SYNC_LOG = ROOT / 'outputs/reports/git_sync_log.md'
README = ROOT / 'README.md'


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def sh(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    log('RUN ' + ' '.join(cmd))
    return subprocess.run(cmd, cwd=ROOT, check=check, text=True)


def read_state() -> dict:
    if not STATE.exists():
        return {}
    return json.loads(STATE.read_text(encoding='utf-8'))


def wait_until_ready() -> None:
    while True:
        state = read_state()
        if state.get('status') == 'completed' and (ROOT / 'outputs/phase3/final_report.md').exists() and (ROOT / 'outputs/phase2/phase2_summary.md').exists():
            return
        if state.get('status') == 'failed':
            raise RuntimeError(f"master failed: {state.get('error')}")
        log('waiting for phase2/phase3 completion before writing root README')
        time.sleep(60)


def commit_and_push() -> None:
    sh([sys.executable, 'scripts/write_root_readme.py'])
    sh(['git', 'add', 'README.md', 'outputs/reports/root_readme_finalizer.log'])
    diff = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT)
    if diff.returncode == 0:
        log('README already up to date; nothing to commit')
        return
    sh(['git', 'commit', '-m', 'write final root README report'])
    commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, text=True).strip()
    ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    with SYNC_LOG.open('a', encoding='utf-8') as f:
        f.write(f'- {ts} | commit {commit_hash} | write final root README report\n')
    sh(['git', 'add', str(SYNC_LOG.relative_to(ROOT))])
    sh(['git', 'commit', '--amend', '--no-edit'])
    auth = base64.b64encode(f"x-access-token:{os.environ['GITHUB_TOKEN']}".encode()).decode()
    for attempt in range(1, 4):
        r = subprocess.run(['git', '-c', f'http.https://github.com/.extraheader=AUTHORIZATION: basic {auth}', 'push', 'origin', 'main'], cwd=ROOT, text=True)
        if r.returncode == 0:
            log(f'push success on attempt {attempt}')
            return
        log(f'push failed attempt {attempt}')
        time.sleep(10 * attempt)
    raise RuntimeError('push failed for final root README')


def main() -> None:
    wait_until_ready()
    commit_and_push()
    log('root README finalizer completed')


if __name__ == '__main__':
    main()
