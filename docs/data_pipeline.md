# Data pipeline

Run the complete preprocessing workflow from the repository root:

```bash
python start_preprocessing.py
```

The launcher checks each stage before continuing and records all derived files under `data/processed/`.

1. Construct the temporally eligible CXR-indexed cohort and assign patients to approximately 70% training, 15% validation, and 15% test partitions.
2. Aggregate MIMIC-IV chart and laboratory events into 24 hourly pre-index bins.
3. Remove candidate variables with no observations in the training partition.
4. Rank the clinically constrained candidates using training-only relevance, redundancy, and stability evidence.
5. Build and screen a broader candidate tensor using the same training-only controls.
6. Merge the clinical and broad evidence into the final 30-variable registry.
7. Apply normalization statistics estimated only from training observations.
8. Verify cohort integrity, tensor alignment, feature metadata, and expected output files.

The final cohort is written under `data/processed/cohorts/`. Candidate, selected, and train-ready EHR files are written under `data/processed/ehr/`.
