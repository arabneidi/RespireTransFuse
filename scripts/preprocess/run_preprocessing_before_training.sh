#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/content/drive/MyDrive/respire-transfuse}"

cd "$BASE"

python -u scripts/preprocess/build_cohort.py \
  --repo_root "$BASE"

python -u scripts/preprocess/build_ehr_tensor.py \
  --root "$BASE" \
  --cohort_csv "$BASE/data/processed/cohorts/cohort.csv" \
  --output_dir "$BASE/data/processed/ehr/ehr_feature_selection_24h" \
  --output_name "ehr_24h_current_split.npz" \
  --window_hours 24 \
  --chunksize 750000 \
  --max_chart_features 80 \
  --max_lab_features 80

python -u scripts/preprocess/filter_ehr_features.py \
  --input_npz "$BASE/data/processed/ehr/ehr_feature_selection_24h/tensors/ehr_24h_current_split.npz" \
  --output_npz "$BASE/data/processed/ehr/ehr_feature_selection_24h/tensors/ehr_24h_current_split_nonzero_train.npz" \
  --output_features_csv "$BASE/data/processed/ehr/ehr_feature_selection_24h/features/ehr_24h_current_split_nonzero_train_features.csv" \
  --output_summary_json "$BASE/data/processed/ehr/ehr_feature_selection_24h/audits/ehr_24h_current_split_nonzero_train_filter_summary.json"

python -u scripts/preprocess/select_ehr_features.py \
  --npz_path "$BASE/data/processed/ehr/ehr_feature_selection_24h/tensors/ehr_24h_current_split_nonzero_train.npz" \
  --output_dir "$BASE/data/processed/ehr/ehr_feature_selection_24h/features/selection_clinical" \
  --n_bootstraps 80

python -u scripts/preprocess/build_broad_ehr_tensor.py \
  --cohort_csv "$BASE/data/processed/cohorts/cohort.csv" \
  --min_train_sample_coverage 0.005 \
  --exclude_obvious_admin_or_leakage

python -u scripts/preprocess/select_broad_ehr_features.py \
  --npz_path "$BASE/data/processed/ehr/ehr_broad_feature_selection_24h/ehr_24h_broad_current_split.npz" \
  --output_dir "$BASE/data/processed/ehr/ehr_broad_feature_selection_24h/features/selection_strict_v4" \
  --n_bootstraps 80 \
  --elastic_top_k 120

python -u scripts/preprocess/build_final_ehr_features.py \
  --repo_root "$BASE"

python -u scripts/preprocess/build_final_ehr_train_ready.py \
  --input_npz "$BASE/data/processed/ehr/ehr_final_24h/ehr_24h_final_current_split.npz" \
  --features_csv "$BASE/data/processed/ehr/ehr_final_24h/ehr_24h_final_selected_features.csv" \
  --output_dir "$BASE/data/processed/ehr/ehr_final_24h_train_ready"
