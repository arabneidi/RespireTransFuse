# Data pipeline

Run the complete workflow from the repository root:

```bash
python start_preprocessing.py
```

The launcher validates every stage and writes derived files under `data/processed/`.

## Cohort construction

Each candidate is a frontal MIMIC-CXR study linked to a MIMIC-IV ICU stay. Qualifying procedures use item IDs 224385, 225792, and 225794.

- The image must occur during the linked hospital admission and before recorded death.
- A qualifying procedure before or at the image within the index ICU stay excludes the sample.
- A procedure interval active at image time also excludes the sample.
- Future events are searched across ICU stays in the same hospital admission and are limited by discharge and death times.
- A positive sample has its first future event in `(0, 48]` hours.
- A negative sample has no event through 72 hours and complete hospital observation through the 72-hour horizon.
- Events from 48 to 72 hours, early death or discharge, and uncertain follow-up are excluded.
- Patients are assigned to approximately 70% training, 15% validation, and 15% test partitions without patient overlap.

The final cohort contains 9,509 samples from 4,668 patients, including 1,667 positives. Split counts are 6,691 training samples with 1,175 positives, 1,389 validation samples with 243 positives, and 1,429 test samples with 249 positives.

## EHR assembly

The observation window is `[image time - 24 hours, image time]`, divided into 24 hourly bins. Multiple observations of one variable in an hour are averaged.

- Chart events require `charttime <= index_time`.
- Laboratory events require both `charttime <= index_time` and `storetime <= index_time`.
- An event at the image time belongs to the final hourly bin.
- Missing entries remain masked without forward filling.
- Means and standard deviations are estimated from observed training values only.
- Missing entries become zero after standardization while the observation mask is retained.

Clinical and broad candidate branches are screened using training data only. Their evidence is merged into this ordered 26-variable registry:

| Position | Variable | Item ID |
| ---: | --- | ---: |
| 1 | Heart rate | 220045 |
| 2 | Respiratory rate | 220210 |
| 3 | Oxygen saturation | 220277 |
| 4 | Non-invasive systolic blood pressure | 220179 |
| 5 | Non-invasive diastolic blood pressure | 220180 |
| 6 | Temperature Fahrenheit | 223761 |
| 7 | Oxygen flow | 223834 |
| 8 | Inspired oxygen fraction | 223835 |
| 9 | Lactate | 50813 |
| 10 | pH | 50820 |
| 11 | pO2 | 50821 |
| 12 | pCO2 | 50818 |
| 13 | Bicarbonate | 50882 |
| 14 | Hemoglobin | 50811 |
| 15 | White blood cells | 51301 |
| 16 | Creatinine | 50912 |
| 17 | Magnesium | 50960 |
| 18 | Potassium | 50971 |
| 19 | Differential bands | 225638 |
| 20 | Arterial oxygen pressure | 220224 |
| 21 | Arterial carbon dioxide pressure | 220235 |
| 22 | Arterial total carbon dioxide, calculated | 225698 |
| 23 | RDW-SD | 52172 |
| 24 | LDH | 220632 |
| 25 | Functional fibrinogen | 51214 |
| 26 | Whole-blood chloride | 226536 |

The train-ready tensor has shape `N x 24 x 26`. Its NPZ metadata records feature identities and order, masks, labels, patient-level splits, and training-derived normalization statistics.

## Stages

1. Construct the CXR-indexed cohort.
2. Build the 24-hour clinical candidate tensor.
3. Remove variables without training observations.
4. Run clinical feature selection.
5. Build the broad candidate tensor.
6. Run broad feature selection.
7. Assemble the final 26-variable tensor.
8. Normalize from training data and verify cohort, tensor, and metadata alignment.

Use `--start-step` only when earlier outputs from the same cohort version have already been generated:

```bash
python start_preprocessing.py --start-step 5
```
