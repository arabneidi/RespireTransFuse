# Experiments

Experiment settings are stored under `configs/experiments/`, while dataset locations are stored in `configs/paths.yaml`.

Run a project-specific model directly from the repository root:

```bash
python scripts/train/train_ehr.py --config configs/experiments/ehr_only.yaml
python scripts/train/train_image.py --config configs/experiments/image_only.yaml
python scripts/train/train_early_fusion.py --config configs/experiments/early_fusion.yaml
python scripts/train/train_respire_transfuse.py --config configs/experiments/respire_transfuse.yaml
```

Run the configured MedFuse baseline:

```bash
python scripts/train/train_medfuse.py --config configs/experiments/medfuse.yaml
```

`run_all.sh` launches the five primary training entry points in sequence. The seven-model dummy suite uses the dedicated MedFuse configurations under `data/dummy_100/configs/`.

Training outputs are stored under `outputs/` and include checkpoints, prediction tables, history files, calibration artifacts, summaries, and configuration snapshots. Plotting and comparison scripts are grouped under `scripts/eval/`.
