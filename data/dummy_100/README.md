# Dummy 100 smoke-test pack

This folder contains a committed 100-sample subset for checking the complete training workflow before using the full MIMIC data. It preserves the cohort schema, 24 x 30 EHR tensor shape, patient-level split labels, outcome distribution, and paired chest X-ray paths used by the main experiments.

## Included files

```text
cohort_dummy_100.csv
ehr_dummy_100.npz
real_images/
configs/paths_dummy_100.yaml
configs/medfuse_dummy_2e.yaml
check_dummy_requirements.py
run_medfuse_dummy_yaml.py
run_2epoch_7_models.sh
```

## Run from Python

From the repository root, install the project requirements and run:

```bash
python start_dummy_test.py
```

The Python launcher works on Windows, macOS, and Linux. It verifies the data pack, runs all seven model configurations for two epochs, checks the intermediate MedFuse checkpoints, and stops immediately if a command fails.

Use `--dry-run` to inspect the commands without training:

```bash
python start_dummy_test.py --dry-run
```

## Run from a Unix shell

The equivalent shell launcher is available for macOS, Linux, WSL, and Git Bash:

```bash
bash data/dummy_100/run_2epoch_7_models.sh
```

Outputs are written to `outputs/dummy_100/`. The two-epoch runs are intended only to verify data loading, model construction, optimization, checkpointing, and output generation; their metrics are not scientific results.
