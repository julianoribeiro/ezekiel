#!/usr/bin/env bash
# Cria o ambiente fixado e roda a verificação.
#   bash scripts/setup_env.sh            # .venv com Python 3.12
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3.12}
"$PY" -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e . --no-deps
.venv/bin/python -m pytest tests -q
echo "Ambiente pronto: source .venv/bin/activate"
