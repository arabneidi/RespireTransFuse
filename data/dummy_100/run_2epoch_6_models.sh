#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

python -u scripts/train/train_ehr.py \
  --paths data/dummy_100/configs/paths_dummy_100.yaml \
  --config configs/experiments/ehr_only_natural_sampling.yaml \
  --save_dir outputs/dummy_100/ehr_only_2e \
  --epochs 2

python -u scripts/train/train_image.py \
  --paths data/dummy_100/configs/paths_dummy_100.yaml \
  --config configs/experiments/image_only.yaml \
  --save_dir outputs/dummy_100/image_only_2e \
  --epochs 2

python -u scripts/train/train_respire_transfuse.py \
  --paths data/dummy_100/configs/paths_dummy_100.yaml \
  --config configs/experiments/respire_transfuse.yaml \
  --save_dir outputs/dummy_100/respire_transfuse_2e \
  --epochs 2

python -u scripts/train/train_early_fusion.py \
  --paths data/dummy_100/configs/paths_dummy_100.yaml \
  --config configs/experiments/early_fusion.yaml \
  --save_dir outputs/dummy_100/early_fusion_2e \
  --epochs 2

python -u scripts/train/run_medfuse_dummy_yaml.py \
  --config data/dummy_100/configs/medfuse_dummy_2e.yaml \
  --run uni_ehr

python -u scripts/train/run_medfuse_dummy_yaml.py \
  --config data/dummy_100/configs/medfuse_dummy_2e.yaml \
  --run lstm
