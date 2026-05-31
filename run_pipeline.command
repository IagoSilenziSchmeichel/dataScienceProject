#!/bin/zsh

cd "$(dirname "$0")"
.venv/bin/python tech_stock_prediction/run_pipeline.py

echo
echo "Pipeline finished. You can close this window."
read -k 1 "?Press any key to close..."
