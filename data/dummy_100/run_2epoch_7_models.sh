#!/usr/bin/env bash
# Run two-epoch smoke tests for all seven evaluated model configurations.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

python -u data/dummy_100/check_dummy_requirements.py

echo
echo "======================================================================"
echo "1/7 EHR-Only Transformer"
echo "======================================================================"

python -u scripts/train/train_ehr.py \
  --paths data/dummy_100/configs/paths_dummy_100.yaml \
  --config configs/experiments/ehr_only.yaml \
  --save_dir outputs/dummy_100/ehr_only_2e \
  --epochs 2

echo
echo "======================================================================"
echo "2/7 Image-Only CNN"
echo "======================================================================"

python -u scripts/train/train_image.py \
  --paths data/dummy_100/configs/paths_dummy_100.yaml \
  --config configs/experiments/image_only.yaml \
  --save_dir outputs/dummy_100/image_only_2e \
  --epochs 2

echo
echo "======================================================================"
echo "3/7 Early Fusion"
echo "======================================================================"

python -u scripts/train/train_early_fusion.py \
  --paths data/dummy_100/configs/paths_dummy_100.yaml \
  --config configs/experiments/early_fusion.yaml \
  --save_dir outputs/dummy_100/early_fusion_2e \
  --epochs 2

echo
echo "======================================================================"
echo "4/7 RespireTransFuse"
echo "======================================================================"

python -u scripts/train/train_respire_transfuse.py \
  --paths data/dummy_100/configs/paths_dummy_100.yaml \
  --config configs/experiments/respire_transfuse.yaml \
  --cohort_csv data/dummy_100/cohort_dummy_100.csv \
  --ehr_npz data/dummy_100/ehr_dummy_100.npz \
  --save_dir outputs/dummy_100/respire_transfuse_2e \
  --warmup_epochs 0 \
  --epochs 2

echo
echo "======================================================================"
echo "5/7 MedFuse Uni-CXR"
echo "======================================================================"

python -u data/dummy_100/run_medfuse_dummy_yaml.py \
  --config data/dummy_100/configs/medfuse_dummy_2e.yaml \
  --run uni_cxr

MEDFUSE_CXR_CHECKPOINT="outputs/dummy_100/medfuse_cxr_2e/best_checkpoint.pth.tar"

if [[ ! -f "$MEDFUSE_CXR_CHECKPOINT" ]]; then
  echo "Missing MedFuse CXR checkpoint: $MEDFUSE_CXR_CHECKPOINT"
  exit 1
fi

echo
echo "======================================================================"
echo "6/7 MedFuse Uni-EHR"
echo "======================================================================"

python -u data/dummy_100/run_medfuse_dummy_yaml.py \
  --config data/dummy_100/configs/medfuse_dummy_2e.yaml \
  --run uni_ehr

MEDFUSE_EHR_CHECKPOINT="outputs/dummy_100/medfuse_ehr_2e/best_checkpoint.pth.tar"

if [[ ! -f "$MEDFUSE_EHR_CHECKPOINT" ]]; then
  echo "Missing MedFuse EHR checkpoint: $MEDFUSE_EHR_CHECKPOINT"
  exit 1
fi

echo
echo "======================================================================"
echo "7/7 MedFuse Multimodal LSTM"
echo "======================================================================"

python -u data/dummy_100/run_medfuse_dummy_yaml.py \
  --config data/dummy_100/configs/medfuse_dummy_2e.yaml \
  --run multimodal_lstm

echo
echo "======================================================================"
echo "Seven-model dummy run completed"
echo "======================================================================"
