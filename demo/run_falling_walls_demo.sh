#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m pip install -r demo/requirements-falling-walls.txt
exec "$PYTHON_BIN" -m streamlit run demo/falling_walls_demo.py --server.headless true
