import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def _clean_arrays(df):
    y_true = np.asarray(df["y_truth"]).reshape(-1)
    y_pred = np.asarray(df["y_pred"]).reshape(-1)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    return y_true, y_pred


def evaluate_new(df):
    y_true, y_pred = _clean_arrays(df)

    if y_true.size == 0:
        return np.nan, np.nan

    if np.sum(y_true == 1) == 0:
        auprc = np.nan
    else:
        auprc = average_precision_score(y_true, y_pred)

    if np.unique(y_true).size < 2:
        auroc = np.nan
    else:
        auroc = roc_auc_score(y_true, y_pred)

    return auprc, auroc


def bootstraping_eval(df, num_iter):
    auroc_list = []
    auprc_list = []

    if len(df) == 0:
        return auprc_list, auroc_list

    for _ in range(num_iter):
        sample = df.sample(frac=1, replace=True)
        y_true, _ = _clean_arrays(sample)

        if y_true.size == 0:
            continue

        if np.unique(y_true).size < 2:
            continue

        if np.sum(y_true == 1) == 0:
            continue

        auprc, auroc = evaluate_new(sample)

        if np.isfinite(auprc) and np.isfinite(auroc):
            auprc_list.append(float(auprc))
            auroc_list.append(float(auroc))

    return auprc_list, auroc_list


def computing_confidence_intervals(list_, true_value):
    true_value = float(true_value) if np.isfinite(true_value) else np.nan

    if len(list_) == 0:
        return true_value, true_value

    values = np.asarray(list_, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return true_value, true_value

    lower = float(np.percentile(values, 2.5))
    upper = float(np.percentile(values, 97.5))

    return upper, lower


def get_model_performance(df):
    test_auprc, test_auroc = evaluate_new(df)
    auprc_list, auroc_list = bootstraping_eval(df, num_iter=1000)

    upper_auprc, lower_auprc = computing_confidence_intervals(auprc_list, test_auprc)
    upper_auroc, lower_auroc = computing_confidence_intervals(auroc_list, test_auroc)

    return (test_auprc, upper_auprc, lower_auprc), (test_auroc, upper_auroc, lower_auroc)
