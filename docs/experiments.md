# Experiments

Dataset locations are stored in `configs/paths.yaml`. Reported experiment settings are stored under `configs/experiments/`.

## Launching training

Run all seven models in dependency order:

```bash
python start_training.py
```

Run one configured experiment:

```bash
python start_training.py --model ehr_only
python start_training.py --model image_only
python start_training.py --model early_fusion
python start_training.py --model medfuse_uni_cxr
python start_training.py --model medfuse_uni_ehr
python start_training.py --model medfuse_multimodal_lstm
python start_training.py --model respire_transfuse
```

Use `--dry-run` to print commands without starting training. `run_all.sh` invokes the same launcher on Unix-like systems.

## Reported settings

All runs use seed 42 and select checkpoints by validation AUPRC.

| Model | Epochs | Batch | Learning rate | Weight decay | Dropout | Main training state |
| --- | ---: | ---: | --- | --- | --- | --- |
| Image-Only CNN | 20 | 32 | 5e-5 | 5e-4 | 0.20 | Frozen EfficientNet backbone, AMP, EMA 0.997 |
| EHR-Only Transformer | 20 | 32 | 9e-6 | 1e-5 | 0.30 | Unweighted BCE, no balanced batches |
| Early Fusion | 20 | 32 | image head 2e-5; EHR 4e-6; fusion 1e-4 | image 1e-3; EHR/fusion 5e-3 | fusion 0.70 | Frozen image backbone; train EHR, projections, and fusion |
| RespireTransFuse | 20 | 32 | image head 1e-4; EHR/fusion 6e-4 | image 1e-4; EHR 4e-4; fusion 3e-4 | fusion 0.05 | Frozen image backbone; AMP; residual scale 0.40 |
| MedFuse Uni-CXR | 20 | 32 | 3e-5 | 1e-4 | 0.30 | Pretrained frozen ResNet-34 backbone; classifier probe |
| MedFuse Uni-EHR | 20 | 64 | 1e-5 | 3e-4 | 0.70 | Dimension 256; two recurrent layers |
| MedFuse Multimodal LSTM | 20 | 32 | 9e-6 | 1e-3 | 0.30 | Frozen loaded Uni-CXR and Uni-EHR branches |

The image augmentations used by the project models are affine probability 0.15, color-jitter probability 0.03, and horizontal-flip probability 0. The EHR and fusion models use gradient clipping at 1.0; MedFuse Uni-CXR and Uni-EHR use 5.0.

The MedFuse multimodal run expects:

- `outputs/medfuse_ehr/best_checkpoint.pth.tar`
- `external/medfuse_original/checkpoints/cxr_rad_full/best_checkpoint.pth.tar`

The CXR checkpoint is available through the external image-model resource link on the main README.

## Outputs and evaluation

Training outputs under `outputs/` include checkpoints, prediction tables, histories, calibration artifacts, summaries, and configuration snapshots. Analysis scripts under `scripts/eval/` cover:

- seven-model ROC and precision-recall curves;
- bootstrap confidence intervals and pairwise comparisons;
- raw and validation-calibrated probability metrics;
- RespireTransFuse calibration comparison;
- predicted-risk distributions and threshold summaries.

The committed dummy suite uses reduced two-epoch settings under `data/dummy_100/configs/` to test execution rather than reproduce reported metrics.
