# RespireTransFuse

**Bidirectional cross-attention fusion of chest X-rays and longitudinal EHR data for respiratory deterioration prediction.**

RespireTransFuse is a multimodal research model that combines a chest X-ray acquired during an ICU stay with the preceding 24 hours of structured clinical measurements. An EfficientNet image branch and a mask-aware Transformer EHR branch produce spatial and temporal tokens, which interact through reciprocal cross-attention before risk prediction.

<p align="center">
  <img src="docs/assets/respiretransfuse_architecture.jpg" alt="RespireTransFuse model architecture" width="820">
</p>

## Study task

The prediction index is an eligible frontal chest radiograph from MIMIC-CXR linked to a MIMIC-IV ICU stay. EHR observations from the 24 hours ending at the image time form the temporal input.

- **Positive:** first qualifying respiratory-support event occurs within 48 hours after the index image.
- **Negative:** no qualifying event occurs through 72 hours.
- **Excluded:** a qualifying intervention occurred at or before the index, or the first future event occurs between 48 and 72 hours.
- **Split policy:** all samples from the same patient remain in one partition.

The final analytic cohort contains 11,339 paired samples. The patient-exclusive partitions preserve an overall positive prevalence of approximately 13.99%.

<p align="center">
  <img src="docs/assets/cohort_split_balance.png" alt="Cohort class balance across the patient-level splits" width="900">
</p>

## Held-out results

All seven models were selected without using the held-out test labels and evaluated on the same 1,695 test samples, including 237 positive cases. Because the outcome is imbalanced, both AUROC and AUPRC are reported; the no-skill AUPRC is approximately 0.1398.

| Model | Modality | Test AUROC | Test AUPRC |
| --- | --- | ---: | ---: |
| Image-Only CNN | CXR | 0.66625 | 0.25764 |
| MedFuse Uni-CXR | CXR | 0.67019 | 0.23330 |
| MedFuse Uni-EHR | EHR | 0.71035 | 0.35697 |
| Early Fusion | CXR + EHR | 0.72946 | 0.38592 |
| EHR-Only Transformer | EHR | 0.73603 | 0.36742 |
| MedFuse Multimodal LSTM | CXR + EHR | 0.74535 | 0.34305 |
| **RespireTransFuse** | **CXR + EHR** | **0.75919** | **0.40738** |

<p align="center">
  <img src="docs/assets/seven_model_precision_recall.jpg" alt="Held-out precision-recall curves for the seven evaluated models" width="820">
</p>

RespireTransFuse achieved the highest observed AUROC and AUPRC in this internal evaluation. The EHR-only Transformer was the strongest unimodal model and had the best reported calibration, while the multimodal comparison indicates that the radiograph contributes complementary information when it is integrated with the temporal EHR representation.

## Model design

The project contains four thesis-developed models and three adapted MedFuse baselines:

1. **Image-Only CNN:** EfficientNet-B0 with global and focused spatial pooling.
2. **EHR-Only Transformer:** a mask-aware 24-step encoder with local temporal processing and multi-view aggregation.
3. **Early Fusion:** concatenation of projected image and EHR summaries.
4. **RespireTransFuse:** four image tokens and 24 EHR tokens projected to a shared width of 48, followed by one four-head attention block in each direction.
5. **MedFuse Uni-CXR:** adapted ResNet-34 image baseline.
6. **MedFuse Uni-EHR:** adapted recurrent EHR baseline.
7. **MedFuse Multimodal LSTM:** adapted recurrent CXR-EHR fusion baseline.

RespireTransFuse concatenates the projected image summary, projected EHR summary, EHR-conditioned image summary, and image-conditioned EHR summary. The resulting 192-dimensional representation is passed to the fusion prediction head. See [`docs/models.md`](docs/models.md) for a compact comparison.

## External resources

Large data and image-model assets are stored outside the repository:

- [Required MIMIC-IV and MIMIC-CXR project data](https://drive.google.com/drive/folders/1l-LYWFxiVTThrFozhGGk8xNpNA8jbJlt?usp=sharing)
- [CXR image-model resources](https://drive.google.com/drive/folders/1PeLDVnkaq7b-tPSXCB-zGD6pHQpzsqEv?usp=sharing)

Place or mount the downloaded folders so the repository has the following layout:

```text
data/raw/
|-- mimic_cxr/
|   |-- metadata/
|   `-- images/
`-- mimiciv/
    |-- hosp/
    `-- icu/
```

MIMIC data use remains subject to the applicable PhysioNet credentialing and data-use requirements.

## Installation

Python 3.10 or newer is recommended. Create an isolated environment and install the dependencies from the repository root:

```bash
python -m venv .venv
```

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Then install the project packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches the local CUDA environment.

## Quick smoke test

The committed dummy pack contains 100 paired samples, a 24 x 30 EHR tensor, and the corresponding small image set. It checks data loading, all seven model configurations, two-epoch training, checkpoint creation, and output writing without requiring Bash:

```bash
python start_dummy_test.py
```

This is the recommended command on Windows, macOS, and Linux. To inspect the commands without running training:

```bash
python start_dummy_test.py --dry-run
```

Unix-like environments can run the equivalent shell launcher:

```bash
bash data/dummy_100/run_2epoch_7_models.sh
```

Dummy outputs are written to `outputs/dummy_100/`. These short runs verify execution only and are not intended for scientific interpretation.

## Preprocessing

The cross-platform preprocessing entry point is [`start_preprocessing.py`](start_preprocessing.py):

```bash
python start_preprocessing.py
```

<p align="center">
  <img src="docs/assets/preprocessing_pipeline.jpg" alt="RespireTransFuse preprocessing pipeline" width="820">
</p>

The launcher performs and validates the following stages:

1. Check the required MIMIC-CXR and MIMIC-IV source files.
2. Build the temporally eligible, CXR-indexed cohort and patient-level splits.
3. Aggregate chart and laboratory events into 24 hourly EHR bins.
4. Remove variables with no training-partition observations.
5. Rank clinically constrained candidates using training-only evidence.
6. Build and screen a broader candidate tensor.
7. Merge both evidence streams into the final 30-variable registry.
8. Normalize the final tensor with training-only statistics.
9. Verify cohort integrity, tensor alignment, metadata, and output files.

Generated cohorts are written under `data/processed/cohorts/`; candidate and train-ready EHR files are written under `data/processed/ehr/`. The lower-level Unix helper remains available at `scripts/preprocess/run_preprocessing_before_training.sh`.

## Training

Update `configs/paths.yaml` for the local data location, then run an experiment from the repository root:

```bash
python scripts/train/train_ehr.py --config configs/experiments/ehr_only.yaml
python scripts/train/train_image.py --config configs/experiments/image_only.yaml
python scripts/train/train_early_fusion.py --config configs/experiments/early_fusion.yaml
python scripts/train/train_respire_transfuse.py --config configs/experiments/respire_transfuse.yaml
python scripts/train/train_medfuse.py --config configs/experiments/medfuse.yaml
```

On macOS, Linux, WSL, or Git Bash, `bash run_all.sh` launches the five primary training entry points in sequence. Experiment outputs under `outputs/` include checkpoints, prediction tables, histories, calibration artifacts, summaries, and saved configurations.

## Evaluation

Post-training analysis scripts are grouped under `scripts/eval/`. They generate model-specific learning curves, held-out discrimination plots, cohort summaries, calibration comparisons, and predicted-risk analyses. The figure scripts currently use a `BASE` path near the top of each file; set it to the local repository path before running outside the original Colab environment.

## Repository structure

```text
RespireTransFuse/
|-- configs/                 Experiment, path, and preprocessing settings
|-- data/dummy_100/          Cross-platform seven-model smoke-test pack
|-- docs/                    Model, pipeline, experiment, and figure documentation
|-- external/medfuse_original/
|                            Bundled MedFuse baseline implementation
|-- scripts/preprocess/      Cohort and EHR preparation stages
|-- scripts/train/           Training entry points
|-- scripts/eval/            Analysis and plotting scripts
|-- src/respire_transfuse/   Reusable datasets, models, training, and utilities
|-- start_preprocessing.py   Validated preprocessing launcher
`-- start_dummy_test.py      Validated seven-model dummy launcher
```

## Research scope

The reported results are a retrospective internal evaluation on linked MIMIC-IV and MIMIC-CXR data. They do not establish clinical readiness. External, temporal, and prospective validation are required before probability estimates or model outputs could be considered for clinical use.
