# RespireTransFuse

RespireTransFuse is a multimodal deep learning project for respiratory deterioration prediction. It combines chest X-ray imaging with 24-hour pre-index EHR time-series features and compares unimodal, early-fusion, MedFuse-style, and cross-attention fusion models.

The repository is organized so that the full MIMIC-IV/MIMIC-CXR preprocessing pipeline and a small local smoke test can both be run from a fresh checkout.

## Repository Contents

- `src/respire_transfuse/` - reusable model, dataset, training, metric, and utility code.
- `scripts/preprocess/` - cohort construction, EHR tensor building, feature filtering, feature selection, and final train-ready tensor creation.
- `scripts/train/` - training entry points for EHR-only, image-only, early fusion, RespireTransFuse, and MedFuse baselines.
- `configs/` - preprocessing and experiment YAML files.
- `data/dummy_100/` - a small committed smoke-test pack with 100 sample rows, EHR tensors, image files, and 2-epoch run configs.
- `external/medfuse_original/` - adapted MedFuse baseline code used by the MedFuse training entry point.

## External Resources

Large clinical data and model assets are not stored directly in GitHub.

Required data folder:

[Download required MIMIC data folders from Google Drive](https://drive.google.com/drive/folders/1l-LYWFxiVTThrFozhGGk8xNpNA8jbJlt?usp=sharing)

CXR image model folder:

[Download CXR image model resources from Google Drive](https://drive.google.com/drive/folders/1PeLDVnkaq7b-tPSXCB-zGD6pHQpzsqEv?usp=sharing)

After downloading the required data, place or mount the raw folders so the project can see this structure:

```text
data/raw/mimic_cxr/metadata/
data/raw/mimic_cxr/images/
data/raw/mimiciv/icu/
data/raw/mimiciv/hosp/
```

The dummy 100 test already includes its small sample images and tensors, so it can be used before downloading the full raw data.

## Setup

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA environment if the default `pip install` does not provide CUDA support.

## Full Preprocessing Pipeline

The full preprocessing stage is controlled by:

```text
start_preprocessing.py
```

It runs the main data preparation steps in sequence:

1. Check that the required MIMIC-CXR and MIMIC-IV input files are present.
2. Build a CXR-indexed cohort from MIMIC metadata and ICU/hospital tables.
3. Build 24-hour EHR tensors aligned to each sample's index time.
4. Remove EHR features with zero observations in the training split.
5. Run clinical EHR feature selection.
6. Build a broader EHR candidate tensor and run stricter broad feature selection.
7. Combine selected evidence into the final EHR feature list.
8. Normalize train-ready EHR tensors using train-only statistics.
9. Verify the expected cohort and EHR output files.

Run it from the repository root:

```bash
python start_preprocessing.py
```

The lower-level bash helper remains available for environments that prefer shell scripts:

```bash
bash scripts/preprocess/run_preprocessing_before_training.sh
```

The preprocessing pipeline writes generated files under:

```text
data/processed/cohorts/
data/processed/ehr/
```

## Training

Run all main model experiments:

```bash
bash run_all.sh
```

`run_all.sh` launches:

1. EHR-only Transformer
2. Image-only CXR model
3. Early fusion model
4. MedFuse baseline
5. RespireTransFuse

The primary training outputs are saved under `outputs/`, including model checkpoints, predictions, histories, summaries, and configuration snapshots.

Individual models can also be run directly. For example:

```bash
python -u scripts/train/train_ehr.py --config configs/experiments/ehr_only_natural_sampling.yaml
python -u scripts/train/train_image.py --config configs/experiments/image_only.yaml
python -u scripts/train/train_respire_transfuse.py --config configs/experiments/respire_transfuse.yaml
```

## Dummy 100 Smoke Test

The next recommended stage after setup is the dummy 100 test. This is a fast check that the repository can load data, instantiate the models, run training loops, and write outputs.

The shell script is:

```text
data/dummy_100/run_2epoch_6_models.sh
```

It runs 2-epoch versions of:

1. EHR-only
2. Image-only
3. RespireTransFuse
4. Early fusion
5. MedFuse EHR-only baseline
6. MedFuse multimodal baseline

Run it directly:

```bash
bash data/dummy_100/run_2epoch_6_models.sh
```

Or use the local Python launcher, which runs the same steps without requiring bash:

```bash
python start_dummy_test.py
```

Dummy outputs are written to:

```text
outputs/dummy_100/
```

The dummy test is only a smoke test. Do not report its metrics as scientific results.

## Model Summary

- **EHR-only**: Transformer encoder over 24 hourly EHR time steps, with observation masks and attention pooling.
- **Image-only**: EfficientNet-based CXR classifier with conservative regularization and optional EMA.
- **Early fusion**: Concatenates projected image and EHR representations.
- **MedFuse baselines**: Runs adapted original MedFuse-style EHR-only and multimodal baselines.
- **RespireTransFuse**: A multimodal bidirectional cross-attention model that connects image tokens and EHR tokens, then predicts a bounded residual around the EHR risk logit.
