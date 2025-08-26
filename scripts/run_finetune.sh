#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-finetune_config.yaml}
OUTDIR=${2:-outputs/zamai-phi3-pashto}
HUB_MODEL_ID=${3:-tasal9/ZamAI-Phi-3-Mini-Pashto}

echo "[INFO] Starting fine-tune with config=$CONFIG"
python train_lora.py \
  --config "$CONFIG" \
  --output_dir "$OUTDIR" \
  --push_to_hub \
  --hub_model_id "$HUB_MODEL_ID"
