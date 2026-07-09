# Data pipeline

Preprocessing order:

1. Build clean CXR-indexed cohort.
2. Build 24-hour pre-index current EHR tensor.
3. Remove EHR features with zero observations in the training split.
4. Perform train-only consensus feature selection.
5. Build the final final selected train-ready EHR tensor.
6. Build image index.
7. Build multimodal manifest for paired CXR-EHR training.

Final EHR feature set: the final selected temporal EHR feature set.
