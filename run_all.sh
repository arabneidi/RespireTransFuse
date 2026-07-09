#!/usr/bin/env bash
set -euo pipefail

python -u scripts/train/train_ehr.py --config configs/experiments/ehr_only.yaml
python -u scripts/train/train_image.py --config configs/experiments/image_only.yaml
python -u scripts/train/train_early_fusion.py --config configs/experiments/early_fusion.yaml
python -u scripts/train/train_medfuse.py --config configs/experiments/medfuse.yaml
python -u scripts/train/train_respire_transfuse.py --config configs/experiments/respire_transfuse.yaml
