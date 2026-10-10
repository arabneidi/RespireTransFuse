# Data pipeline

Run the complete workflow from the repository root:

```bash
python start_preprocessing.py
```

The launcher validates each stage before continuing and writes derived files under `data/processed/`.

## Cohort construction

Each candidate sample is a frontal MIMIC-CXR study linked to a MIMIC-IV ICU stay. Respiratory intervention events are identified from procedure events.

- Prior-event exclusion is evaluated within the index ICU stay.
- Future events are searched across all ICU stays in the same hospital admission.
- A positive sample has its first future event during the 0-to-48-hour prediction window.
- A negative sample has no event through 72 hours and remains hospitalized through the complete 72-hour horizon.
- Events from 48 to 72 hours, death before 72 hours, early hospital discharge, and uncertain follow-up are excluded.
- Patients are assigned to approximately 70% training, 15% validation, and 15% test partitions with no cross-partition patient overlap.

The resulting cohort contains 9,591 samples: 6,765 training, 1,396 validation, and 1,430 test samples. Overall prevalence is 17.41%.

## EHR assembly

The EHR observation window is `[index time - 24 hours, index time]` and is divided into 24 hourly bins.

- Chart events must have `charttime <= index_time`.
- Laboratory events must have both `charttime <= index_time` and `storetime <= index_time`.
- An event exactly at the index time is assigned to the final hourly bin.
- Missing values remain masked; normalization statistics are estimated from observed training values only.

Candidate variables are screened through clinical and broad training-only selection branches. The final registry contains 18 clinically selected and 8 data-driven variables:

| Branch | Variables |
| --- | --- |
| Clinical | Heart rate, respiratory rate, oxygen saturation, non-invasive systolic and diastolic blood pressure, temperature, oxygen flow, inspired oxygen fraction, lactate, pH, pO2, pCO2, bicarbonate, hemoglobin, white blood cells, creatinine, magnesium, potassium |
| Data-driven | Differential bands, arterial O2 pressure, arterial CO2 pressure, arterial TCO2, RDW-SD, LDH, functional fibrinogen, whole-blood chloride |

The final train-ready tensor has shape `N x 24 x 26`. Its NPZ metadata includes feature identities, order, masks, labels, patient-level splits, and training-derived normalization statistics.

## Stages

1. Construct the strict CXR-indexed cohort.
2. Build the initial 24-hour clinical candidate tensor.
3. Remove variables without training observations.
4. Run clinical feature selection.
5. Build the broad candidate tensor.
6. Run broad feature selection.
7. Assemble the final 26-variable tensor.
8. Normalize from training data and verify cohort, tensor, and metadata alignment.

Use `--start-step` only when earlier outputs from the same cohort version have already been verified:

```bash
python start_preprocessing.py --start-step 5
```
