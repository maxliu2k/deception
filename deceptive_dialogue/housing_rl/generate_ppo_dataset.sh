#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Optional: if conda is available, try to activate the expected env.
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate "${CONDA_ENV_NAME:-openrlhf}" >/dev/null 2>&1 || true
fi

PYTHON_BIN="python3"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: neither python3 nor python is available in PATH." >&2
  exit 1
fi

mkdir -p data/in/ppo_data

"$PYTHON_BIN" conglomerate_json.py
"$PYTHON_BIN" jaxseq_jsonl_gen.py
mv -f train.jsonl data/in/ppo_data/train.jsonl
mv -f test.jsonl data/in/ppo_data/test.jsonl
mv -f metadata.json data/in/ppo_data/metadata.json
rm -f conglomerated_data.json
