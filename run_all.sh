#!/usr/bin/env bash
# Train the five primary project model families with their standard configurations.
# The commands run sequentially from the repository root and stop at the first
# failure, leaving each trainer responsible for its own checkpoints, metrics,
# predictions, and plots under the configured output directory.

set -euo pipefail

python -u scripts/train/train_ehr.py --config configs/experiments/ehr_only.yaml
python -u scripts/train/train_image.py --config configs/experiments/image_only.yaml
python -u scripts/train/train_early_fusion.py --config configs/experiments/early_fusion.yaml
python -u scripts/train/train_medfuse.py --config configs/experiments/medfuse.yaml
python -u scripts/train/train_respire_transfuse.py --config configs/experiments/respire_transfuse.yaml
