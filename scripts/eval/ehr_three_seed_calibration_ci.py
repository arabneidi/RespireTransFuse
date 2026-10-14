

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score, roc_auc_score, log_loss, brier_score_loss,
    precision_score, recall_score, f1_score, confusion_matrix,
    precision_recall_curve,
)

BASE = Path(__file__).resolve().parents[2]
SEEDS = (40, 41, 42)
N_BOOT = 2000
BOOTSTRAP_SEED = 20261011
OUT = BASE / 'outputs' / 'ehr_only_calibrated_3seeds'
COHORT = BASE / 'data/processed/cohorts/cohort.csv'
OUT.mkdir(parents=True, exist_ok=True)

def load_predictions(seed, split):
    path = BASE / 'outputs' / f'ehr_only{seed}' / f'{split}_predictions.csv'
    assert path.is_file(), f'Missing: {path}'
    df = pd.read_csv(path, dtype={'sample_id': str})
    required = {'sample_id', 'label', 'logit', 'prob'}
    assert required.issubset(df.columns), f'{path}: missing {required - set(df.columns)}'
    assert df.sample_id.notna().all() and df.sample_id.is_unique
    for col in ['label', 'logit', 'prob']:
        assert np.isfinite(df[col].to_numpy(dtype=float)).all(), f'{path}: non-finite {col}'
    p = expit(df['logit'].to_numpy(dtype=float))
    assert np.allclose(p, df['prob'].to_numpy(dtype=float), atol=1e-5), \
        f'Inconsistent logit and raw probability in {path}'
    return df

def calibrate_fit(y, logits):

    def objective(v):
        z = logits / np.exp(v[0]) + v[1]
        return float(np.mean(np.logaddexp(0., z) - y*z))

    result = minimize(objective, x0=np.array([0., 0.]), method='L-BFGS-B',
                      bounds=[(-np.log(100), np.log(100)), (-12., 12.)])

    if not np.isfinite(result.fun) or result.fun > objective([0., 0.]) + 1e-8:
        warnings.warn('Calibration did not improve validation log-loss; using identity.')
        return 1., 0.
    if not result.success:
        warnings.warn(f'Calibration optimizer status: {result.message}')
    return float(np.exp(result.x[0])), float(result.x[1])

def best_f1_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2*precision[:-1]*recall[:-1] / np.maximum(precision[:-1]+recall[:-1], 1e-15)

    idx = int(np.flatnonzero(np.isclose(f1, f1.max(), atol=1e-12, rtol=0))[-1])
    return float(thresholds[idx])

def ece_10(y, p):

    bucket = np.minimum((p*10).astype(int), 9)
    return float(sum(abs(p[bucket == k].mean() - y[bucket == k].mean()) *
                     np.mean(bucket == k) for k in range(10) if np.any(bucket == k)))

def evaluate(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        AUROC=float(roc_auc_score(y, p)),
        AUPRC=float(average_precision_score(y, p)),
        Brier=float(brier_score_loss(y, p)),
        ECE10=ece_10(y, p),
        LogLoss=float(log_loss(y, np.clip(p, 1e-15, 1-1e-15), labels=[0, 1])),
        Sensitivity=float(recall_score(y, pred, zero_division=0)),
        Specificity=float(tn / (tn+fp)) if (tn+fp) else np.nan,
        Precision=float(precision_score(y, pred, zero_division=0)),
        F1=float(f1_score(y, pred, zero_division=0)),
        TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
    )

assert COHORT.is_file(), f'Missing cohort: {COHORT}'
cohort = pd.read_csv(COHORT, usecols=['sample_id', 'subject_id', 'label', 'split'],
                     dtype={'sample_id': str, 'subject_id': str})
assert cohort.sample_id.is_unique, 'Duplicate sample IDs in cohort'
assert cohort.subject_id.notna().all(), 'Missing subject IDs in cohort'
lookup = cohort.set_index('sample_id')

results, fit_info, test_by_seed = [], [], {}
ref_val_ids, ref_test_ids = None, None
for seed in SEEDS:
    val = load_predictions(seed, 'val').sort_values('sample_id').reset_index(drop=True)
    test = load_predictions(seed, 'test').sort_values('sample_id').reset_index(drop=True)
    assert (len(val), len(test)) == (1389, 1429), 'Unexpected split size'
    assert (int(val.label.sum()), int(test.label.sum())) == (243, 249), 'Unexpected positives'
    for name, df in [('val', val), ('test', test)]:
        assert df.sample_id.isin(lookup.index).all(), f'{name}: sample missing in cohort'
        info = lookup.loc[df.sample_id]
        assert (info['split'].to_numpy() == name).all(), f'{name}: split mismatch'
        assert (info['label'].to_numpy(dtype=int) == df.label.to_numpy(dtype=int)).all(), f'{name}: label mismatch'
    if ref_val_ids is None:
        ref_val_ids, ref_test_ids = val.sample_id.to_numpy(), test.sample_id.to_numpy()
    else:
        assert np.array_equal(ref_val_ids, val.sample_id.to_numpy()), 'Different val samples across seeds'
        assert np.array_equal(ref_test_ids, test.sample_id.to_numpy()), 'Different test samples across seeds'

    yv, yt = val.label.to_numpy(dtype=int), test.label.to_numpy(dtype=int)
    zv, zt = val.logit.to_numpy(dtype=float), test.logit.to_numpy(dtype=float)
    T, bias = calibrate_fit(yv, zv)
    val_raw, test_raw = expit(zv), expit(zt)
    val_cal, test_cal = expit(zv/T + bias), expit(zt/T + bias)
    threshold_raw = best_f1_threshold(yv, val_raw)
    threshold_cal = best_f1_threshold(yv, val_cal)

    fit_info.append(dict(seed=seed, temperature=T, bias=bias,
                         raw_val_f1_threshold=threshold_raw, calibrated_val_f1_threshold=threshold_cal,
                         val_logloss_raw=log_loss(yv, val_raw), val_logloss_calibrated=log_loss(yv, val_cal)))
    for split, y, raw_p, cal_p in [('val', yv, val_raw, val_cal), ('test', yt, test_raw, test_cal)]:
        for stage, p, t in [('raw', raw_p, threshold_raw), ('calibrated', cal_p, threshold_cal)]:
            results.append(dict(seed=seed, split=split, stage=stage, threshold=t,
                                **evaluate(y, p, t)))

    for split, df, raw_p, cal_p in [('val', val, val_raw, val_cal),
                                     ('test', test, test_raw, test_cal)]:
        predictions = df[['sample_id', 'label']].copy()
        predictions['subject_id'] = lookup.loc[df.sample_id, 'subject_id'].to_numpy()
        predictions['raw_prob'] = raw_p
        predictions['calibrated_prob'] = cal_p
        predictions['raw_prediction'] = (raw_p >= threshold_raw).astype(int)
        predictions['calibrated_prediction'] = (cal_p >= threshold_cal).astype(int)
        predictions.to_csv(OUT / f'{split}_calibrated_seed{seed}.csv', index=False)
    test_by_seed[seed] = dict(y=yt, raw=test_raw, calibrated=test_cal,
                              t_raw=threshold_raw, t_cal=threshold_cal)

metrics = pd.DataFrame(results)
metrics.to_csv(OUT / 'metrics_all_seeds_raw_vs_calibrated.csv', index=False)
pd.DataFrame(fit_info).to_csv(OUT / 'calibration_parameters.csv', index=False)

numeric_metrics = ['AUROC', 'AUPRC', 'Brier', 'ECE10', 'LogLoss',
                   'Sensitivity', 'Specificity', 'Precision', 'F1']
summary = (metrics[metrics.split == 'test'].groupby('stage')[numeric_metrics]
           .agg(['mean', 'std']))
summary.to_csv(OUT / 'test_mean_sd_across_seeds.csv')

subjects = lookup.loc[ref_test_ids, 'subject_id'].to_numpy()
unique_subjects, subject_codes = np.unique(subjects, return_inverse=True)
clusters = [np.flatnonzero(subject_codes == k) for k in range(len(unique_subjects))]
y_test = test_by_seed[SEEDS[0]]['y']
assert all(np.array_equal(y_test, test_by_seed[s]['y']) for s in SEEDS)
rng = np.random.default_rng(BOOTSTRAP_SEED)

boot_rows = []
for b in range(N_BOOT):
    picked = rng.integers(0, len(clusters), size=len(clusters))
    ii = np.concatenate([clusters[i] for i in picked])
    yy = y_test[ii]
    if len(np.unique(yy)) < 2:
        continue
    for stage in ('raw', 'calibrated'):
        per_seed = []
        for seed in SEEDS:
            d = test_by_seed[seed]
            pp = d[stage][ii]
            tt = d['t_raw' if stage == 'raw' else 't_cal']
            vals = evaluate(yy, pp, tt)
            per_seed.append(vals)
            boot_rows.append(dict(rep=b, stage=stage, seed=str(seed), **
                                  {m: vals[m] for m in numeric_metrics}))
        boot_rows.append(dict(rep=b, stage=stage, seed='mean_of_3', **
                              {m: float(np.mean([v[m] for v in per_seed])) for m in numeric_metrics}))

boots = pd.DataFrame(boot_rows)
boots.to_csv(OUT / 'patient_bootstrap_replicates.csv', index=False)
ci_rows = []
for (stage, seed), group in boots.groupby(['stage', 'seed']):
    for name in numeric_metrics:
        lo, hi = np.percentile(group[name].dropna(), [2.5, 97.5])
        if seed == 'mean_of_3':
            center = float(summary.loc[stage, (name, 'mean')])
        else:
            center = float(metrics.loc[(metrics.seed == int(seed)) &
                                       (metrics.split == 'test') &
                                       (metrics.stage == stage), name].iloc[0])
        ci_rows.append(dict(stage=stage, seed=seed, metric=name,
                            estimate=center, CI95_low=lo, CI95_high=hi))
ci = pd.DataFrame(ci_rows)
ci.to_csv(OUT / 'test_patient_bootstrap_CI95.csv', index=False)

print('\n=== Calibration fit (validation only) ===')
print(pd.DataFrame(fit_info).round(5).to_string(index=False))
print('\n=== Test results (before / after calibration) ===')
print(metrics.loc[metrics.split == 'test', ['seed', 'stage'] + numeric_metrics].round(5).to_string(index=False))
print('\n=== Test mean +/- sample SD across seeds ===')
print(summary.round(5).to_string())
print('\n=== Patient-bootstrap 95% CI for three-seed mean (calibrated) ===')
print(ci[(ci.seed == 'mean_of_3') & (ci.stage == 'calibrated')].round(5).to_string(index=False))
print(f'\nSubjects in test: {len(unique_subjects)}, test samples: {len(y_test)}')
print(f'Bootstrap repetitions: {N_BOOT}; saved results in: {OUT}')
