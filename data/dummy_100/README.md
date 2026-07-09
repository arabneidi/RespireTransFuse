# Respire dummy 100-sample pack

This folder is for a fast code-smoke test of the RespireTransFuse training scripts.
It creates a real 100-row subset from your existing cohort CSV and EHR NPZ instead of random synthetic data. The subset keeps:

- the same CSV schema as the original cohort;
- the same EHR NPZ keys, including `X`, `M`, `y`, `sample_id`, `split`, and `feature_names`;
- the same EHR time/feature shape as the original NPZ;
- approximately the same split/label percentages using strata = `split × label`;
- image rows only, so EHR-only, image-only, multimodal, and MedFuse can all read it.

## Created files after running the generator

```text
data/dummy_100/cohort_dummy_100.csv
data/dummy_100/ehr_dummy_100.npz
data/dummy_100/summary_dummy_100.json
data/dummy_100/configs/*.yaml
data/dummy_100/run_2epoch_6_models.sh
```

## Colab start cell

```python
BASE = "/content/drive/MyDrive/respire-transfuse"

!cd "$BASE" && python -u data/dummy_100/make_dummy_100.py   --base "$BASE"   --cohort_csv "$BASE/data/processed/cohorts/cohort.csv"   --ehr_npz "$BASE/data/processed/ehr/ehr_final_24h_train_ready/ehr_24h_final_train_ready_current_split.npz"   --output_dir "$BASE/data/dummy_100"   --n_samples 1000   --seed 42
```

## Run all models for 2 epochs

```python
BASE = "/content/drive/MyDrive/respire-transfuse"
!bash "$BASE/data/dummy_100/run_2epoch_6_models.sh"
```

## Important notes

1. This is a smoke test only. Do not report the metrics as results.
2. The generator subsets your existing real EHR arrays, so the 30-feature structure remains unchanged.
3. The multimodal/image scripts read the image paths from `verified_image_path`; this pack does not copy images.
4. The generated run script uses the current config-style training scripts: paths come from `--config`, and MedFuse paths are passed directly through CLI flags.
5. For MedFuse multimodal, set `MEDFUSE_SRC` if the original MedFuse folder is not at the default path:

```bash
export MEDFUSE_SRC="/content/drive/MyDrive/MedFuse-data/MedFuse-main-unzipped/MedFuse-main"
```
