# RespireTransFuse

RespireTransFuse is a multimodal deep learning project for respiratory deterioration prediction using chest X-ray images and 24-hour pre-index EHR time-series features.

## Models

This repository supports six experiments:

1. EHR-only
2. Image-only
3. Early fusion
4. MedFuse EHR-only baseline
5. MedFuse multimodal baseline
5. RespireTransFuse

## Data

Raw MIMIC-CXR and MIMIC-IV files are not included in this repository.

Expected local structure:

data/raw/mimic_cxr/metadata/
data/raw/mimic_cxr/images/
data/raw/mimiciv/icu/
data/raw/mimiciv/hosp/

Generated cohorts, tensors, checkpoints, logs, and figures are ignored by Git.

## Preprocessing

python -u scripts/preprocess/build_cohort.py --config configs/preprocess.yaml
python -u scripts/preprocess/build_ehr_tensor.py --config configs/preprocess.yaml
python -u scripts/preprocess/filter_ehr_features.py --config configs/preprocess.yaml
python -u scripts/preprocess/select_ehr_features.py --config configs/preprocess.yaml
python -u scripts/preprocess/build_train_ready_ehr.py --config configs/preprocess.yaml
python -u scripts/preprocess/build_image_index.py --config configs/preprocess.yaml
python -u scripts/preprocess/build_multimodal_manifest.py --config configs/preprocess.yaml

## Training

Run one model:

python -u scripts/train/train_ehr.py --config configs/experiments/ehr_only.yaml

Run all six experiments:

bash run_all.sh

## Outputs

Training outputs are saved under:

outputs/runs/
outputs/checkpoints/

