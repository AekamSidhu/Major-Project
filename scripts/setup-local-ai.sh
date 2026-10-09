#!/usr/bin/env bash
# Creates the OpenMeet local AI sidecar virtual environment (local-ai/.venv)
# and installs faster-whisper + fastembed. Everything stays on this device.
#
# Usage: scripts/setup-local-ai.sh [--dev] [--download-models]
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
dir="$root/local-ai"
venv="$dir/.venv"
dev=0
download=0
for arg in "$@"; do
  case "$arg" in
    --dev) dev=1 ;;
    --download-models) download=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

python=""
for candidate in python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)'; then
      python="$candidate"
      break
    fi
  fi
done
if [ -z "$python" ]; then
  echo "Python 3.10 - 3.12 is required (faster-whisper / onnxruntime wheels)." >&2
  exit 1
fi

if [ ! -x "$venv/bin/python" ]; then
  echo "Creating virtual environment in $venv"
  "$python" -m venv "$venv"
fi
"$venv/bin/python" -m pip install --upgrade pip
if [ "$dev" = 1 ]; then
  "$venv/bin/python" -m pip install -r "$dir/requirements-dev.txt"
else
  "$venv/bin/python" -m pip install -r "$dir/requirements.txt"
fi
if [ "$download" = 1 ]; then
  "$venv/bin/python" "$dir/server.py" --download-models
fi
echo
echo "Local AI is installed. OpenMeet starts it automatically when a meeting needs it,"
echo "or run it yourself: $venv/bin/python $dir/server.py"
