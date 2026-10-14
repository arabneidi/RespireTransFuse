# Experiments

Dataset locations are stored in `configs/paths.yaml`. Experiment settings are stored under `configs/experiments/`.

## Launching training

Run all seven models with seeds 40, 41, and 42:

```bash
python start_training.py
```

Run one model or a selected seed set:

```bash
python start_training.py --model early_fusion
python start_training.py --model respire_transfuse --seeds 40 41
```

Use `--dry-run` to print commands without starting training. `run_all.sh` calls the same launcher on Unix-like systems.

## Reported settings

All runs use at most 20 epochs, batch size 32, early-stopping patience 10, and seeds 40, 41, and 42. Checkpoints are selected by validation AUPRC.

| Model | Optimizer | Learning rate | Weight decay | Dropout |
| --- | --- | --- | --- | --- |
| Image-Only CNN | AdamW | 2e-5 | 1e-3 | 0.35 |
| EHR-Only Transformer | AdamW | 4e-4 | 1e-3 | 0.35 |
| Early Fusion | AdamW | image 2e-4; EHR 4e-5; fusion 1e-4 | image 1e-4; EHR 3e-4; fusion 3e-4 | image 0.30; EHR 0.30; fusion 0.20 |
| MedFuse Uni-CXR | Adam | 3e-5 | 3e-4 | 0.30 |
| MedFuse Uni-EHR | Adam | 3e-4 | 3e-4 | 0.20 |
| MedFuse Multimodal LSTM | Adam | 3e-4 | 3e-4 | 0.40 |
| RespireTransFuse | AdamW | image 1e-4; EHR 6e-4; fusion 6e-4 | image 1e-4; EHR 3.5e-4; fusion 2.5e-4 | image 0.40; EHR 0.35; fusion 0.20 |

Project image models use 224 x 224 inputs, no horizontal flip, affine augmentation with probability 0.15 and limits of 4 degrees, 2% translation, and 0.98 to 1.02 scale, plus color jitter with probability 0.03. The EfficientNet-B0 backbone is frozen. RespireTransFuse uses a residual cross-attention scale of 0.40 and auxiliary loss weights of 0.05 for EHR and 0.02 for image.

The MedFuse multimodal run loads the matching seed's Uni-EHR checkpoint and the external ResNet-34 CXR checkpoint:

- `outputs/medfuse_ehr{seed}/best_checkpoint.pth.tar`
- `external/medfuse_original/checkpoints/cxr_rad_full/best_checkpoint.pth.tar`

## Ablations

The matched ablation launcher uses each full-model run's saved configuration:

```bash
python scripts/train/launch_respire_ablations.py
```

| ID | Variant |
| --- | --- |
| A0 | Summary concatenation without cross-attention |
| A1 | Image queries EHR only |
| A2 | EHR queries image only |
| A3 | Reciprocal cross-attention without auxiliary losses |
| A4 | Full RespireTransFuse model |

## Calibration and uncertainty

Each model and seed receives an independent validation-only logistic calibration of the form `z_cal = a + b*z`. Validation labels select the F1 threshold, and the fitted transform and threshold are then applied to held-out test predictions. Test labels are not used for fitting or tuning.

The evaluation scripts report fixed-width 10-bin expected calibration error. Confidence intervals use 2,000 patient-cluster bootstrap repetitions, and pairwise comparisons use 20,000 patient-level permutations.

The seven model-specific `*_three_seed_calibration_ci.py` scripts under `scripts/eval/` implement the same calibration and bootstrap contract. The committed dummy suite uses two epochs to test execution and does not reproduce scientific results.
