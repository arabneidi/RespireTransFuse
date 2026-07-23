#!/usr/bin/env bash
# Execute the preprocessing stages in their required order for Unix-like systems.

set -euo pipefail

SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"

DEFAULT_BASE="$(
  cd -- "$SCRIPT_DIR/../.." >/dev/null 2>&1
  pwd
)"

BASE="${BASE:-$DEFAULT_BASE}"
SEED="${SEED:-42}"

export PYTHONHASHSEED="$SEED"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$BASE"

python -u scripts/preprocess/build_cohort.py \
  --repo_root "$BASE" \
  --seed "$SEED" \
  --split_search_iterations 50000 \
  --max_prevalence_gap 0.0025 \
  --max_split_size_deviation 0.01

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
  --n_bootstraps 80 \
  --random_state "$SEED"

python -u scripts/preprocess/build_broad_ehr_tensor.py \
  --cohort_csv "$BASE/data/processed/cohorts/cohort.csv" \
  --min_train_sample_coverage 0.005 \
  --exclude_obvious_admin_or_leakage

python -u scripts/preprocess/select_broad_ehr_features.py \
  --npz_path "$BASE/data/processed/ehr/ehr_broad_feature_selection_24h/ehr_24h_broad_current_split.npz" \
  --output_dir "$BASE/data/processed/ehr/ehr_broad_feature_selection_24h/features/selection_strict_v4" \
  --n_bootstraps 80 \
  --elastic_top_k 120 \
  --random_state "$SEED"

python -u scripts/preprocess/build_final_ehr_features.py \
  --repo_root "$BASE"

python -u scripts/preprocess/build_final_ehr_train_ready.py \
  --input_npz "$BASE/data/processed/ehr/ehr_final_24h/ehr_24h_final_current_split.npz" \
  --features_csv "$BASE/data/processed/ehr/ehr_final_24h/ehr_24h_final_selected_features.csv" \
  --output_dir "$BASE/data/processed/ehr/ehr_final_24h_train_ready"
