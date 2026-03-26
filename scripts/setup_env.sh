#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

mkdir -p outputs/reports outputs/phase0 outputs/phase1 outputs/phase2 outputs/phase3 runs/e0

python - <<'PY'
import importlib
mods=['torch','ultralytics','pandas','yaml','PIL','matplotlib','seaborn']
for m in mods:
    try:
        importlib.import_module(m)
        print(f'{m}: OK')
    except Exception as e:
        print(f'{m}: MISSING ({type(e).__name__})')
PY
