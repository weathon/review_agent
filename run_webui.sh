#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python gradio_app.py
