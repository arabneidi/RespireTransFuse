
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_consensus_ehr_feature_selection.py

Master train-only EHR feature-selection pipeline for RespireTransFuse.

This script is designed for thesis-quality feature selection.

It DOES NOT hard-code top-K.
It DOES NOT use validation/test rows for selection.
It DOES NOT modify the NPZ tensor.
It creates ranked evidence tables so the final feature count can be chosen defensibly.

Input:
    outputs/tensors/ehr_48h_clean_source_paired.npz
or:
    outputs/tensors/ehr_48h_clean_source_ehr_full.npz

Expected NPZ keys:
    X_raw:     [N, 48, F], NaN where missing
    mask:      [N, 48, F]
    y:         [N]
    split:     [N]
    variables: [F]
    labels:    [F]
    sources:   [F]
    itemids:   [F]

Methods:
    1. Leakage/static/noise flags
    2. Temporal summary extraction:
        mean, last, max, min, std, observed_fraction, last_minus_first, slope
    3. Train-only univariate relevance:
        point-biserial correlation
        AUROC
        AUPRC
        AUPRC lift over prevalence
        mutual information
        mean difference
    4. Train-only redundancy analysis:
        feature-feature correlation matrix
        covariance matrix diagonal = variance
        correlation clusters
    5. mRMR-style ranking:
        maximize relevance, minimize redundancy
    6. Elastic Net stability selection:
        repeated train-only subsampling
        class-weighted logistic regression
    7. Optional RFECV confirmation:
        class-weighted logistic regression
        average precision scoring
    8. Consensus score and final recommendation labels:
        core
        strong
        review
        reject_or_noise

Outputs:
    outputs/features/<stem>_all_modes_train_only.csv
    outputs/features/<stem>_best_summary_per_feature_train_only.csv
    outputs/features/<stem>_correlation_clusters_train_only.csv
    outputs/features/<stem>_mrmr_ranked_train_only.csv
    outputs/features/<stem>_elasticnet_stability_train_only.csv
    outputs/features/<stem>_consensus_feature_evidence_train_only.csv
    outputs/features/<stem>_recommended_core_features.csv
    outputs/features/<stem>_recommended_strong_features.csv
    outputs/features/<stem>_review_features.csv
    outputs/features/<stem>_feature_selection_summary.json

Important:
    This is feature selection for the EHR temporal branch.
    Static demographic features should be built separately later.
"""

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.feature_selection import mutual_info_classif, RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# General utilities
# ============================================================

def print_section(title: str):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def safe_str_array(arr, length=None):
    if arr is None:
        if length is None:
            return None
        return np.array([""] * length)
    return np.array([str(x) for x in arr])


def safe_numeric_corr(x, y):
    """
    Pearson correlation between numeric x and binary y.
    Equivalent to point-biserial correlation for binary y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 20:
        return np.nan

    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def safe_auroc_ap(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    out = {
        "auroc": np.nan,
        "auroc_abs": np.nan,
        "auprc": np.nan,
    }

    if len(x) < 20:
        return out

    if len(np.unique(y)) < 2:
        return out

    try:
        auc = float(roc_auc_score(y, x))
        out["auroc"] = auc
        out["auroc_abs"] = float(max(auc, 1.0 - auc))
    except Exception:
        pass

    try:
        ap = float(average_precision_score(y, x))
        out["auprc"] = ap
    except Exception:
        pass

    return out


def safe_mutual_info(x, y, random_state=42):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 50:
        return np.nan

    if len(np.unique(y)) < 2:
        return np.nan

    if np.nanstd(x) == 0:
        return np.nan

    try:
        xi = x.reshape(-1, 1)
        mi = mutual_info_classif(
            xi,
            y,
            discrete_features=False,
            random_state=random_state,
        )
        return float(mi[0])
    except Exception:
        return np.nan


def sigmoid01(x, center, scale):
    """
    Smoothly map a metric into 0..1.
    """
    if x is None or not np.isfinite(x):
        return 0.0
    return float(1.0 / (1.0 + np.exp(-(x - center) / scale)))


# ============================================================
# Clinical/leakage categorization
# ============================================================

def possible_leakage_flag(variable, label, source):
    import re

    label_text = str(label).lower().strip()
    variable_text = str(variable).lower().strip()

    parts = variable_text.split("::")
    if len(parts) >= 3:
        clean_variable_text = parts[2].lower().strip()
    else:
        clean_variable_text = variable_text

    text = f"{clean_variable_text} {label_text}".lower()
    compact_label = re.sub(r"[^a-z0-9]", "", label_text)

    if len(compact_label) <= 1:
        return True

    exact_reject_labels = {
        "acuity workload question 1",
        "acuity workload question 2",
        "alarms on",
        "any fear in relationships",
        "art lumen volume",
        "baedp",
        "bipap epap",
        "bipap ipap",
        "bipap o2 flow",
        "epithelial cells",
        "etoh",
        "gcs - eye opening",
        "gcs - motor response",
        "gcs - verbal response",
        "granular casts",
        "high risk (>51) interventions",
        "hyaline casts",
        "iabp alarms activated",
        "iabp dressing occlusive",
        "iabp mean",
        "iabp placed in outside facility",
        "iabp volume",
        "iabp zero/calibrate",
        "inspired gas temp.",
        "ketone",
        "nausea and vomiting (ciwa)",
        "ocat - lips tongue gums palate",
        "ocat - saliva secretions, voice quality",
        "ocat - swallow",
        "ocat - teeth",
        "pa line cm mark",
        "paedp",
        "par-activity",
        "par-circulation",
        "par-consciousness",
        "par-oxygen saturation",
        "par-remain sedated",
        "par-respiration",
        "pca 1 hour limit",
        "pca attempt",
        "pca basal rate (ml/hour)",
        "pca bolus",
        "pca dose",
        "pca inject",
        "pca lockout (min)",
        "pca total dose",
        "pcwp",
        "ph (dipstick)",
        "see chart for initial patient assessment",
        "specific gravity",
        "specific gravity (urine)",
        "svo2",
        "urobilinogen",
        "ven lumen volume",
        "visual / hearing deficit",
    }

    if label_text in exact_reject_labels:
        return True

    phrase_reject = [
        "alarms on",
        "parameters checked",
        "spo2 desat limit",
        "desat limit",
        "high risk",
        "interventions",
        "secondary diagnosis",
        "history of falling",
        "history of slips",
        "visual / hearing deficit",
        "emotional / physical / sexual harm",
        "health care proxy",
        "legal guardian",
        "social work",
        "discharge needs",
        "home tf",
        "special diet",
        "recreational drug use",
        "sexuality / reproductive",
        "unable to assess",
        "current dyspnea assessment",
        "intravenous",
        "iv access",
        "iv/saline",
        "saline lock",
        "outside facility",
        "placed in the field",
        "dressing occlusive",
        "tip cultured",
        "zero/calibrate",
        "line tip",
        "line placed",
        "catheter placed",
        "catheter dressing",
        "lumen volume",
        "blood transfusion consent",
        "icu consent signed",
        "called out",
        "bladder scan",
        "guaiac",
        "insulin pump",
        "pca dose",
        "pca lockout",
        "pca 1 hour limit",
        "pca inject",
        "pca attempt",
        "pca total dose",
        "pca basal rate",
        "pca bolus",
        "epidural infusion",
        "epidural bolus",
        "epidural total dose",
        "heparin dose",
        "vancomycin",
        "gentamicin",
        "tobramycin",
        "amikacin",
        "digoxin",
        "phenytoin",
        "dilantin",
        "lithium",
        "theophylline",
        "tacro",
        "fk506",
        "anti-xa",
        "warfarin",
        "gcs",
        "braden",
        "richmond",
        "rass",
        "cam-icu",
        "ciwa",
        "ocat",
        "par-",
        "pain",
        "mental status",
        "orientation",
        "orient/clouding",
        "strength",
        "gait",
        "transferring",
        "ambulatory",
        "mobility",
        "activity",
        "self adl",
        "fall",
        "skin",
        "wound",
        "pressure ulcer",
        "impaired skin",
        "ulcer",
        "eye care",
        "skin care",
        "back care",
        "collar care",
        "bed bath",
        "chg bath",
        "mouth care",
        "ventilator",
        "ventilation",
        "vent mode",
        "peep",
        "fio2",
        "inspired o2",
        "inspired gas",
        "fraction inspired",
        "tidal",
        "minute volume",
        "airway",
        "peak insp",
        "plateau",
        "respiratory rate (set)",
        "respiratory rate (total)",
        "respiratory rate (spontaneous)",
        "apnea",
        "inspiratory time",
        "inspiratory ratio",
        "expiratory",
        "fspn",
        "paw",
        "vti",
        "bipap",
        "ipap",
        "epap",
        "psv",
        "cuff pressure",
        "intub",
        "extub",
        "ett",
        "trach",
        "tube",
        "subglottal",
        "suctioning",
        "incentive spirometry",
        "cough/deep breath",
        "st segment",
        "monitoring on",
        "pulmonary artery",
        "pa catheter",
        "pa line",
        "paedp",
        "baedp",
        "pcwp",
        "central venous pressure",
        "central venous o2",
        "mixed venous",
        "svo2",
        "iabp",
        "assisted systole",
        "assisted diastole",
        "unassisted systole",
        "augmented diastole",
        "balloon pump",
        "impella",
        "ecmo",
        "nicom",
        "cardiac index",
        "cardiac output",
        "stroke volume",
        "svv",
        "svi",
        "co / ci change",
        "dialysis patient",
        "dialysis catheter",
        "hemodialysis",
        "dialysate",
        "replacement rate",
        "ultrafiltrate",
        "fluid removal",
        "blood flow",
        "prefilter",
        "post filter",
        "citrate",
        "access pressure",
        "filter pressure",
        "effluent pressure",
        "return pressure",
        "pacemaker",
        "temporary ventricular",
        "venticular",
        "stim setting",
        "sens setting",
        "threshold",
        "intra cranial pressure",
        "intracranial pressure",
        "icp",
        "cerebral perfusion",
        "saliva",
        "voice quality",
        "swallow",
        "paroxysmal sweats",
        "auditory disturbance",
        "visual disturbance",
        "tactile disturbance",
        "anxiety",
        "agitation",
        "tremor",
        "headache",
        "urine",
        "pleural",
        "ascites",
        "body fluid",
        "csf",
        "sputum",
        "stool",
        "pericardial",
        "synovial",
    ]

    if any(term in text for term in phrase_reject):
        return True

    regex_reject = [
        r"\balarm\b",
        r"\balarms\b",
        r"\bparameter\b",
        r"\bparameters\b",
        r"\bscore\b",
        r"\bapache\b",
        r"\bdevice\b",
        r"\bmode\b",
        r"\bprocedure\b",
        r"\bgauge\b",
        r"\bdressing\b",
        r"\bocclusive\b",
        r"\bcatheter\b",
        r"\bpicc\b",
        r"\bcordis\b",
        r"\bintroducer\b",
        r"\bsheath\b",
        r"\bhickman\b",
        r"\btunneled\b",
        r"\bport\b",
        r"\blumen\b",
        r"\bcalibrate\b",
        r"\bflush\b",
        r"\baspiration\b",
        r"\bconsent\b",
        r"\bpca\b",
        r"\biabp\b",
        r"\bpcwp\b",
        r"\bsvo2\b",
        r"\bpaedp\b",
        r"\bbaedp\b",
        r"\bpa\s+line\b",
        r"\bpa\s+catheter\b",
        r"\bicp\b",
        r"\bbipap\b",
        r"\bipap\b",
        r"\bepap\b",
        r"\bpsv\b",
        r"\bpaw\b",
        r"\bvti\b",
        r"\bfspn\b",
    ]

    return any(re.search(pattern, text) for pattern in regex_reject)

def static_like_flag(variable, label, source):
    text = f"{variable} {label} {source}".lower()

    static_terms = [
        "height",
        "weight",
        "admission weight",
        "daily weight",
        "feeding weight",
    ]

    return any(term in text for term in static_terms)


def clinical_category(variable, label, source):
    text = f"{variable} {label} {source}".lower()

    if source == "chartevents":
        if "heart rate" in text:
            return "vital_heart_rate"
        if "respiratory rate" in text:
            return "vital_respiratory_rate"
        if "o2 saturation" in text or "oxygen saturation" in text:
            return "vital_oxygen_saturation"
        if "blood pressure" in text or "bp " in text:
            return "vital_blood_pressure"
        if "temperature" in text:
            return "vital_temperature"
        if "glucose" in text:
            return "bedside_glucose"
        return "chart_other"

    if source == "labevents":
        if any(k in text for k in ["ph", "po2", "pco2", "bicarbonate", "base excess", "total co2", "carbon dioxide", "oxygen"]):
            return "lab_blood_gas_respiration"
        if any(k in text for k in ["lactate", "anion gap", "glucose"]):
            return "lab_metabolic"
        if any(k in text for k in ["hemoglobin", "hematocrit", "white blood", "wbc", "platelet"]):
            return "lab_cbc"
        if any(k in text for k in ["sodium", "potassium", "chloride", "calcium", "magnesium", "phosphate"]):
            return "lab_electrolyte"
        if any(k in text for k in ["creatinine", "urea nitrogen", "bun"]):
            return "lab_renal"
        if any(k in text for k in ["bilirubin", "albumin"]):
            return "lab_liver_protein"
        if any(k in text for k in ["inr", "ptt", "pt", "fibrinogen"]):
            return "lab_coagulation"
        return "lab_other"

    return "unknown"


def physiological_direction_hint(label):
    text = str(label).lower().strip()

    if any(k in text for k in ["o2 saturation", "oxygen saturation", "po2"]):
        return "lower_may_indicate_worse_oxygenation"

    if any(k in text for k in ["pco2", "carbon dioxide"]):
        return "higher_may_indicate_hypercapnia"

    if text == "ph" or text.startswith("ph "):
        return "lower_may_indicate_acidosis"

    if "lactate" in text:
        return "higher_may_indicate_shock_or_stress"

    if "respiratory rate" in text:
        return "higher_may_indicate_respiratory_distress"

    if "heart rate" in text:
        return "higher_may_indicate_physiologic_stress"

    if "blood pressure" in text:
        return "low_or_high_may_indicate_instability"

    return ""


# ============================================================
# Temporal summaries
# ============================================================

def temporal_summary_matrix(X, mode):
    """
    X shape: [N, T, F]
    Return M: [N, F]
    """
    if mode == "mean":
        return np.nanmean(X, axis=1)

    if mode == "max":
        return np.nanmax(X, axis=1)

    if mode == "min":
        return np.nanmin(X, axis=1)

    if mode == "std":
        return np.nanstd(X, axis=1)

    if mode == "observed_fraction":
        return np.isfinite(X).mean(axis=1).astype(np.float32)

    if mode == "last":
        N, T, F = X.shape
        valid = np.isfinite(X)
        # reverse time and find first valid in reversed direction
        rev_valid = valid[:, ::-1, :]
        rev_idx = np.argmax(rev_valid, axis=1)
        has_valid = rev_valid.any(axis=1)

        out = np.full((N, F), np.nan, dtype=np.float32)
        for j in range(F):
            rows = np.where(has_valid[:, j])[0]
            if len(rows) > 0:
                original_idx = T - 1 - rev_idx[rows, j]
                out[rows, j] = X[rows, original_idx, j]
        return out

    if mode == "first":
        N, T, F = X.shape
        valid = np.isfinite(X)
        idx = np.argmax(valid, axis=1)
        has_valid = valid.any(axis=1)

        out = np.full((N, F), np.nan, dtype=np.float32)
        for j in range(F):
            rows = np.where(has_valid[:, j])[0]
            if len(rows) > 0:
                out[rows, j] = X[rows, idx[rows, j], j]
        return out

    if mode == "last_minus_first":
        last = temporal_summary_matrix(X, "last")
        first = temporal_summary_matrix(X, "first")
        return last - first

    if mode == "slope":
        """
        Simple per-sample linear slope over observed time points.
        For each feature and sample:
            slope = Cov(t, x) / Var(t)
        """
        N, T, F = X.shape
        t = np.arange(T, dtype=np.float32)
        out = np.full((N, F), np.nan, dtype=np.float32)

        for j in range(F):
            xj = X[:, :, j]
            for i in range(N):
                xi = xj[i]
                valid = np.isfinite(xi)
                if valid.sum() >= 3:
                    tv = t[valid]
                    xv = xi[valid]
                    tv_centered = tv - tv.mean()
                    denom = np.sum(tv_centered ** 2)
                    if denom > 0:
                        out[i, j] = np.sum(tv_centered * (xv - xv.mean())) / denom

        return out

    raise ValueError(f"Unknown summary mode: {mode}")


# ============================================================
# Univariate audit
# ============================================================

def univariate_score(row):
    """
    Train-only relevance score for a feature-summary pair.

    This score is not final feature selection.
    It combines complementary views:
        - correlation with binary label
        - AUROC distance from random
        - AUPRC lift over prevalence
        - mutual information
        - valid sample fraction
    """
    abs_corr = row.get("abs_point_biserial_corr", np.nan)
    auroc_abs = row.get("auroc_abs", np.nan)
    ap_lift = row.get("ap_lift_over_prevalence", np.nan)
    mi = row.get("mutual_information", np.nan)
    valid_fraction = row.get("valid_fraction", 0.0)
    coverage = row.get("hourly_coverage", 0.0)

    corr_component = min(abs_corr / 0.20, 1.0) if np.isfinite(abs_corr) else 0.0
    auroc_component = min(max(auroc_abs - 0.50, 0.0) / 0.25, 1.0) if np.isfinite(auroc_abs) else 0.0
    ap_component = min(ap_lift / 3.0, 1.0) if np.isfinite(ap_lift) else 0.0
    mi_component = min(mi / 0.05, 1.0) if np.isfinite(mi) else 0.0
    valid_component = min(valid_fraction, 1.0)
    coverage_component = min(coverage / 0.05, 1.0) if coverage < 0.05 else 1.0

    score = (
        0.25 * corr_component
        + 0.25 * ap_component
        + 0.20 * auroc_component
        + 0.15 * mi_component
        + 0.10 * valid_component
        + 0.05 * coverage_component
    )

    return float(score)


def audit_all_summary_modes(X_train, y_train, feature_meta, summary_modes, random_state=42):
    print_section("TRAIN-ONLY UNIVARIATE AUDIT ACROSS TEMPORAL SUMMARIES")

    prevalence = float(np.mean(y_train))
    all_rows = []

    for mode in summary_modes:
        print(f"Summarizing mode: {mode}", flush=True)

        M = temporal_summary_matrix(X_train, mode)

        for j in range(M.shape[1]):
            x = M[:, j]
            valid = np.isfinite(x)
            valid_n = int(valid.sum())
            valid_fraction = float(valid.mean())

            yv = y_train[valid]
            xv = x[valid]

            pos_n = int((yv == 1).sum()) if valid_n > 0 else 0
            neg_n = int((yv == 0).sum()) if valid_n > 0 else 0

            mean_pos = np.nan
            mean_neg = np.nan
            abs_mean_difference = np.nan
            variance = np.nan

            if valid_n >= 20 and pos_n > 0 and neg_n > 0:
                mean_pos = float(np.nanmean(xv[yv == 1]))
                mean_neg = float(np.nanmean(xv[yv == 0]))
                abs_mean_difference = float(abs(mean_pos - mean_neg))
                variance = float(np.nanvar(xv))

            corr = safe_numeric_corr(x, y_train)
            abs_corr = abs(corr) if np.isfinite(corr) else np.nan

            auc_ap = safe_auroc_ap(x, y_train)
            auroc = auc_ap["auroc"]
            auroc_abs = auc_ap["auroc_abs"]
            auprc = auc_ap["auprc"]

            ap_lift = float(auprc / prevalence) if np.isfinite(auprc) and prevalence > 0 else np.nan
            mi = safe_mutual_info(x, y_train, random_state=random_state)

            meta = feature_meta.iloc[j].to_dict()

            row = {
                "feature_index": int(j),
                "variable": meta["variable"],
                "label": meta["label"],
                "source": meta["source"],
                "itemid": meta["itemid"],
                "value_source": meta["value_source"],
                "clinical_category": meta["clinical_category"],
                "summary_mode": mode,
                "hourly_coverage": meta["hourly_coverage"],
                "valid_n": valid_n,
                "valid_fraction": valid_fraction,
                "positive_n": pos_n,
                "negative_n": neg_n,
                "variance": variance,
                "mean_positive": mean_pos,
                "mean_negative": mean_neg,
                "abs_mean_difference": abs_mean_difference,
                "point_biserial_corr": corr,
                "abs_point_biserial_corr": abs_corr,
                "auroc": auroc,
                "auroc_abs": auroc_abs,
                "auprc": auprc,
                "ap_lift_over_prevalence": ap_lift,
                "mutual_information": mi,
                "possible_leakage_flag": meta["possible_leakage_flag"],
                "static_like_flag": meta["static_like_flag"],
                "physiology_direction_hint": meta["physiology_direction_hint"],
            }

            row["univariate_relevance_score"] = univariate_score(row)

            all_rows.append(row)

    all_modes = pd.DataFrame(all_rows)

    best = (
        all_modes
        .sort_values("univariate_relevance_score", ascending=False)
        .groupby("feature_index", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    best = best.sort_values("univariate_relevance_score", ascending=False).reset_index(drop=True)

    print("All feature-summary rows:", all_modes.shape)
    print("Best summary rows:", best.shape)

    return all_modes, best


# ============================================================
# Matrix construction, correlation, clustering
# ============================================================

def build_best_summary_matrix(X_train, best_df):
    """
    Build matrix M [N_train, F] where each feature uses its best temporal summary mode.
    Feature columns are ordered by feature_index ascending.
    """
    best_sorted = best_df.sort_values("feature_index").reset_index(drop=True)
    feature_indices = best_sorted["feature_index"].astype(int).tolist()

    cols = []

    for _, row in best_sorted.iterrows():
        j = int(row["feature_index"])
        mode = str(row["summary_mode"])
        M_mode = temporal_summary_matrix(X_train[:, :, [j]], mode)
        cols.append(M_mode[:, 0])

    M = np.vstack(cols).T
    return M, best_sorted


def impute_and_scale_train_matrix(M):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    M_imp = imputer.fit_transform(M)
    M_scaled = scaler.fit_transform(M_imp)

    M_scaled = np.nan_to_num(M_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    return M_imp, M_scaled, imputer, scaler


def union_find_clusters(abs_corr, threshold):
    """
    Simple graph clustering:
        connect features if abs(corr) >= threshold
        clusters are connected components.
    """
    n = abs_corr.shape[0]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if abs_corr[i, j] >= threshold:
                union(i, j)

    roots = [find(i) for i in range(n)]
    root_to_cluster = {}
    cluster_ids = []

    for r in roots:
        if r not in root_to_cluster:
            root_to_cluster[r] = len(root_to_cluster) + 1
        cluster_ids.append(root_to_cluster[r])

    return np.array(cluster_ids, dtype=int)


def redundancy_analysis(M_scaled, best_sorted, corr_threshold):
    print_section("TRAIN-ONLY REDUNDANCY ANALYSIS")

    corr_matrix = np.corrcoef(M_scaled, rowvar=False)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    abs_corr = np.abs(corr_matrix)

    cov_matrix = np.cov(M_scaled, rowvar=False)
    cov_matrix = np.nan_to_num(cov_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    variance_diagonal = np.diag(cov_matrix)

    cluster_ids = union_find_clusters(abs_corr, threshold=corr_threshold)

    cluster_df = best_sorted.copy()
    cluster_df["correlation_cluster_id"] = cluster_ids
    cluster_df["standardized_variance_diagonal"] = variance_diagonal

    cluster_sizes = (
        cluster_df.groupby("correlation_cluster_id")
        .size()
        .rename("correlation_cluster_size")
        .reset_index()
    )

    cluster_df = cluster_df.merge(cluster_sizes, on="correlation_cluster_id", how="left")

    cluster_df = cluster_df.sort_values(
        ["correlation_cluster_id", "univariate_relevance_score"],
        ascending=[True, False],
    ).reset_index(drop=True)

    print("Correlation matrix:", corr_matrix.shape)
    print("Number of correlation clusters:", int(cluster_df["correlation_cluster_id"].nunique()))
    print("Largest cluster size:", int(cluster_df["correlation_cluster_size"].max()))

    return corr_matrix, cov_matrix, cluster_df


# ============================================================
# mRMR ranking
# ============================================================

def mrmr_rank(best_sorted, corr_matrix, lambda_redundancy=0.50):
    print_section("mRMR-STYLE TRAIN-ONLY RANKING")

    df = best_sorted.copy().reset_index(drop=True)

    eligible = (
        (df["possible_leakage_flag"] == False)
        & (df["static_like_flag"] == False)
        & (df["valid_n"] >= 100)
        & (df["valid_fraction"] > 0)
        & (df["univariate_relevance_score"] > 0)
    )

    df["mrmr_eligible"] = eligible

    eligible_positions = np.where(eligible.to_numpy())[0].tolist()

    relevance = df["univariate_relevance_score"].to_numpy(dtype=float)

    selected = []
    remaining = eligible_positions.copy()

    while len(remaining) > 0:
        best_pos = None
        best_score = -np.inf

        for pos in remaining:
            rel = relevance[pos]

            if len(selected) == 0:
                redundancy = 0.0
            else:
                redundancy = float(np.mean(np.abs(corr_matrix[pos, selected])))

            score = rel - lambda_redundancy * redundancy

            if score > best_score:
                best_score = score
                best_pos = pos

        selected.append(best_pos)
        remaining.remove(best_pos)

    mrmr_rank_map = {}
    mrmr_score_map = {}
    redundancy_map = {}

    for rank_idx, pos in enumerate(selected):
        rel = relevance[pos]

        if rank_idx == 0:
            redundancy = 0.0
        else:
            previous = selected[:rank_idx]
            redundancy = float(np.mean(np.abs(corr_matrix[pos, previous])))

        score = rel - lambda_redundancy * redundancy

        mrmr_rank_map[pos] = rank_idx + 1
        mrmr_score_map[pos] = score
        redundancy_map[pos] = redundancy

    df["mrmr_rank"] = np.nan
    df["mrmr_score"] = np.nan
    df["mean_abs_corr_with_previous_selected"] = np.nan
    df["lambda_redundancy"] = lambda_redundancy

    for pos in selected:
        df.loc[pos, "mrmr_rank"] = mrmr_rank_map[pos]
        df.loc[pos, "mrmr_score"] = mrmr_score_map[pos]
        df.loc[pos, "mean_abs_corr_with_previous_selected"] = redundancy_map[pos]

    ranked = df[df["mrmr_eligible"]].copy()
    ranked = ranked.sort_values("mrmr_rank").reset_index(drop=True)

    print("mRMR eligible features:", len(ranked))

    return ranked, df


# ============================================================
# Elastic Net stability selection
# ============================================================

def elasticnet_stability_selection(
    M_scaled,
    y_train,
    best_sorted,
    n_bootstraps=80,
    subsample_fraction=0.75,
    C_grid=None,
    l1_ratio_grid=None,
    random_state=42,
    max_iter=5000,
    elastic_top_k=0,
):
    print_section("ELASTIC NET STABILITY SELECTION")

    if C_grid is None:
        C_grid = [0.03, 0.06, 0.1, 0.2, 0.5, 1.0]

    if l1_ratio_grid is None:
        l1_ratio_grid = [0.2, 0.5, 0.8]

    rng = np.random.default_rng(random_state)

    df = best_sorted.copy().reset_index(drop=True)

    eligible_mask = (
        (df["possible_leakage_flag"] == False)
        & (df["static_like_flag"] == False)
        & (df["valid_n"] >= 100)
        & (df["valid_fraction"] > 0)
        & (df["univariate_relevance_score"] > 0)
    ).to_numpy()

    eligible_positions = np.where(eligible_mask)[0]

    original_eligible_count = len(eligible_positions)

    if elastic_top_k is not None and int(elastic_top_k) > 0 and len(eligible_positions) > int(elastic_top_k):
        tmp = df.iloc[eligible_positions].copy()

        tmp["_rel_sort"] = tmp["univariate_relevance_score"].fillna(0.0)
        tmp["_valid_sort"] = tmp["valid_fraction"].fillna(0.0)
        tmp["_coverage_sort"] = tmp["hourly_coverage"].fillna(0.0)

        tmp = tmp.sort_values(
            ["_rel_sort", "_valid_sort", "_coverage_sort"],
            ascending=[False, False, False],
        )

        eligible_positions = tmp.index.to_numpy()[:int(elastic_top_k)]

    print(f"ElasticNet original eligible features: {original_eligible_count}", flush=True)
    print(f"ElasticNet top-K requested: {elastic_top_k}", flush=True)
    print(f"ElasticNet eligible features after top-K filter: {len(eligible_positions)}", flush=True)

    if len(eligible_positions) == 0:
        raise ValueError("No features eligible for Elastic Net stability selection.")

    X = M_scaled[:, eligible_positions]
    y = y_train.astype(int)

    n = X.shape[0]
    p = X.shape[1]

    selection_counts = np.zeros(p, dtype=np.float64)
    coefficient_abs_sum = np.zeros(p, dtype=np.float64)
    model_count = 0

    positive_indices = np.where(y == 1)[0]
    negative_indices = np.where(y == 0)[0]

    if len(positive_indices) < 5 or len(negative_indices) < 5:
        raise ValueError("Not enough positive/negative samples for stability selection.")

    print(f"ElasticNet train rows: {X.shape[0]}", flush=True)
    print(f"ElasticNet eligible features: {X.shape[1]}", flush=True)
    print(f"ElasticNet requested bootstraps: {n_bootstraps}", flush=True)

    for b in tqdm(range(n_bootstraps), desc="ElasticNet bootstraps", unit="bootstrap"):
        # Stratified subsample
        pos_size = max(2, int(len(positive_indices) * subsample_fraction))
        neg_size = max(2, int(len(negative_indices) * subsample_fraction))

        pos_sample = rng.choice(positive_indices, size=pos_size, replace=False)
        neg_sample = rng.choice(negative_indices, size=neg_size, replace=False)
        sample_idx = np.concatenate([pos_sample, neg_sample])
        rng.shuffle(sample_idx)

        Xb = X[sample_idx]
        yb = y[sample_idx]

        # Randomize regularization settings across bootstraps.
        C = float(rng.choice(C_grid))
        l1_ratio = float(rng.choice(l1_ratio_grid))

        clf = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=C,
            l1_ratio=l1_ratio,
            class_weight="balanced",
            max_iter=max_iter,
            random_state=random_state + b,
            n_jobs=-1,
        )

        try:
            clf.fit(Xb, yb)
            coef = clf.coef_.reshape(-1)
            selected = np.abs(coef) > 1e-8

            selection_counts += selected.astype(float)
            coefficient_abs_sum += np.abs(coef)
            model_count += 1

        except Exception as e:
            print(f"ElasticNet bootstrap {b} failed: {e}", flush=True)
            continue

    if model_count == 0:
        raise RuntimeError("All Elastic Net stability models failed.")

    stability_df = df.iloc[eligible_positions].copy().reset_index(drop=True)
    stability_df["elasticnet_selection_probability"] = selection_counts / model_count
    stability_df["elasticnet_mean_abs_coefficient"] = coefficient_abs_sum / model_count
    stability_df["elasticnet_successful_bootstraps"] = model_count
    stability_df["elasticnet_n_requested_bootstraps"] = n_bootstraps

    stability_df = stability_df.sort_values(
        ["elasticnet_selection_probability", "elasticnet_mean_abs_coefficient", "univariate_relevance_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    print("Successful stability models:", model_count)
    print("Features with selection probability >= 0.70:", int((stability_df["elasticnet_selection_probability"] >= 0.70).sum()))
    print("Features with selection probability >= 0.50:", int((stability_df["elasticnet_selection_probability"] >= 0.50).sum()))
    print("Features with selection probability >= 0.30:", int((stability_df["elasticnet_selection_probability"] >= 0.30).sum()))

    return stability_df


# ============================================================
# Optional RFECV confirmation
# ============================================================

def rfecv_confirmation(M_scaled, y_train, best_sorted, min_features_to_select=5, step=0.10, cv_folds=5, random_state=42):
    print_section("OPTIONAL RFECV CONFIRMATION")

    df = best_sorted.copy().reset_index(drop=True)

    eligible_mask = (
        (df["possible_leakage_flag"] == False)
        & (df["static_like_flag"] == False)
        & (df["valid_n"] >= 100)
        & (df["valid_fraction"] > 0)
        & (df["univariate_relevance_score"] > 0)
    ).to_numpy()

    eligible_positions = np.where(eligible_mask)[0]

    if len(eligible_positions) < min_features_to_select:
        print("Not enough eligible features for RFECV.")
        out = df.copy()
        out["rfecv_selected"] = False
        out["rfecv_rank"] = np.nan
        return out

    X = M_scaled[:, eligible_positions]
    y = y_train.astype(int)

    estimator = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=3000,
        random_state=random_state,
    )

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    selector = RFECV(
        estimator=estimator,
        step=step,
        cv=cv,
        scoring="average_precision",
        min_features_to_select=min_features_to_select,
        n_jobs=-1,
    )

    try:
        selector.fit(X, y)
    except Exception as e:
        print(f"RFECV failed: {e}")
        out = df.copy()
        out["rfecv_selected"] = False
        out["rfecv_rank"] = np.nan
        return out

    out = df.iloc[eligible_positions].copy().reset_index(drop=True)
    out["rfecv_selected"] = selector.support_.astype(bool)
    out["rfecv_rank"] = selector.ranking_.astype(int)
    out["rfecv_n_features_selected"] = int(selector.n_features_)

    print("RFECV selected features:", int(selector.n_features_))

    return out


# ============================================================
# Consensus evidence and recommendation
# ============================================================

def build_consensus(
    best_sorted,
    cluster_df,
    mrmr_ranked,
    stability_df,
    rfecv_df=None,
):
    print_section("BUILDING CONSENSUS FEATURE EVIDENCE TABLE")

    df = best_sorted.copy().reset_index(drop=True)

    keep_cols_cluster = [
        "feature_index",
        "correlation_cluster_id",
        "correlation_cluster_size",
        "standardized_variance_diagonal",
    ]

    cluster_small = cluster_df[keep_cols_cluster].drop_duplicates("feature_index")
    df = df.merge(cluster_small, on="feature_index", how="left")

    mrmr_small = mrmr_ranked[
        [
            "feature_index",
            "mrmr_rank",
            "mrmr_score",
            "mean_abs_corr_with_previous_selected",
            "lambda_redundancy",
        ]
    ].drop_duplicates("feature_index")

    df = df.merge(mrmr_small, on="feature_index", how="left")

    stability_small = stability_df[
        [
            "feature_index",
            "elasticnet_selection_probability",
            "elasticnet_mean_abs_coefficient",
            "elasticnet_successful_bootstraps",
            "elasticnet_n_requested_bootstraps",
        ]
    ].drop_duplicates("feature_index")

    df = df.merge(stability_small, on="feature_index", how="left")

    if rfecv_df is not None:
        rfecv_cols = ["feature_index", "rfecv_selected", "rfecv_rank"]
        if "rfecv_n_features_selected" in rfecv_df.columns:
            rfecv_cols.append("rfecv_n_features_selected")

        rfecv_small = rfecv_df[rfecv_cols].drop_duplicates("feature_index")
        df = df.merge(rfecv_small, on="feature_index", how="left")
    else:
        df["rfecv_selected"] = False
        df["rfecv_rank"] = np.nan

    # Fill missing numeric evidence.
    df["elasticnet_selection_probability"] = df["elasticnet_selection_probability"].fillna(0.0)
    df["elasticnet_mean_abs_coefficient"] = df["elasticnet_mean_abs_coefficient"].fillna(0.0)

    df["rfecv_selected"] = df["rfecv_selected"].fillna(False).astype(bool)

    # Rank-derived scores.
    max_mrmr_rank = df["mrmr_rank"].max()
    if not np.isfinite(max_mrmr_rank):
        max_mrmr_rank = len(df)

    df["mrmr_rank_score"] = 0.0
    valid_mrmr = df["mrmr_rank"].notna()
    df.loc[valid_mrmr, "mrmr_rank_score"] = 1.0 - ((df.loc[valid_mrmr, "mrmr_rank"] - 1.0) / max(max_mrmr_rank, 1.0))

    # Components
    df["component_relevance"] = df["univariate_relevance_score"].clip(0, 1)
    df["component_stability"] = df["elasticnet_selection_probability"].clip(0, 1)
    df["component_mrmr"] = df["mrmr_rank_score"].clip(0, 1)
    df["component_coverage"] = df["valid_fraction"].clip(0, 1)
    df["component_rfecv"] = df["rfecv_selected"].astype(float)

    # Penalties
    df["penalty_leakage"] = df["possible_leakage_flag"].astype(float)
    df["penalty_static"] = df["static_like_flag"].astype(float)
    df["penalty_sparse"] = (df["valid_n"] < 100).astype(float)
    df["penalty_low_relevance"] = (df["univariate_relevance_score"] <= 0).astype(float)

    df["consensus_score"] = (
        0.30 * df["component_relevance"]
        + 0.30 * df["component_stability"]
        + 0.20 * df["component_mrmr"]
        + 0.10 * df["component_coverage"]
        + 0.10 * df["component_rfecv"]
        - 0.50 * df["penalty_leakage"]
        - 0.30 * df["penalty_static"]
        - 0.20 * df["penalty_sparse"]
        - 0.20 * df["penalty_low_relevance"]
    )

    df["consensus_score"] = df["consensus_score"].clip(lower=0.0)

    # Recommendation labels.
    def decision(row):
        if row["possible_leakage_flag"]:
            return "reject_possible_leakage"
        if row["static_like_flag"]:
            return "reject_static_like_temporal"
        if row["valid_n"] < 100 or row["valid_fraction"] < 0.02:
            return "reject_or_review_sparse"
        if row["univariate_relevance_score"] <= 0:
            return "reject_low_signal"

        stable = row["elasticnet_selection_probability"]
        mrmr_rank = row["mrmr_rank"]
        rel = row["univariate_relevance_score"]
        score = row["consensus_score"]

        if stable >= 0.70 and pd.notna(mrmr_rank) and mrmr_rank <= 30 and score >= 0.55:
            return "core"

        if stable >= 0.50 and pd.notna(mrmr_rank) and mrmr_rank <= 50 and score >= 0.45:
            return "strong"

        if stable >= 0.30 or rel >= 0.35 or row["rfecv_selected"]:
            return "review"

        return "weak_or_noise"

    df["recommendation"] = df.apply(decision, axis=1)

    # Sort by final evidence.
    df = df.sort_values(
        ["recommendation", "consensus_score", "elasticnet_selection_probability", "mrmr_rank"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)

    # Better human order for recommendation categories.
    order = {
        "core": 1,
        "strong": 2,
        "review": 3,
        "weak_or_noise": 4,
        "reject_or_review_sparse": 5,
        "reject_low_signal": 6,
        "reject_static_like_temporal": 7,
        "reject_possible_leakage": 8,
    }

    df["recommendation_order"] = df["recommendation"].map(order).fillna(99).astype(int)

    df = df.sort_values(
        ["recommendation_order", "consensus_score", "elasticnet_selection_probability", "univariate_relevance_score"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    return df


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--npz_path",
        type=str,
        required=True,
        help="Path to clean source-built EHR NPZ.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/Respiratory-deterioration/outputs/features",
    )

    parser.add_argument(
        "--summary_modes",
        type=str,
        default="mean,last,max,min,std,observed_fraction,last_minus_first,slope",
        help="Comma-separated temporal summary modes.",
    )

    parser.add_argument(
        "--corr_threshold",
        type=float,
        default=0.85,
        help="Absolute correlation threshold for redundancy clusters.",
    )

    parser.add_argument(
        "--lambda_redundancy",
        type=float,
        default=0.50,
        help="mRMR redundancy penalty.",
    )

    parser.add_argument(
        "--n_bootstraps",
        type=int,
        default=80,
        help="Elastic Net stability-selection bootstraps.",
    )

    parser.add_argument(
        "--elastic_top_k",
        type=int,
        default=0,
        help="Run Elastic Net only on top-K eligible features ranked by train-only univariate evidence. Use 0 for all eligible features.",
    )

    parser.add_argument(
        "--subsample_fraction",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--run_rfecv",
        action="store_true",
        help="Run RFECV confirmation. Slower but useful.",
    )

    parser.add_argument(
        "--rfecv_min_features",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--rfecv_cv_folds",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    npz_path = Path(args.npz_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = npz_path.stem

    summary_modes = [x.strip() for x in args.summary_modes.split(",") if x.strip()]

    print_section("LOADING NPZ")
    z = np.load(npz_path, allow_pickle=True)

    required = ["X_raw", "mask", "y", "split", "variables"]
    for key in required:
        if key not in z.files:
            raise KeyError(f"Missing key '{key}'. Available keys: {z.files}")

    X_raw = z["X_raw"].astype(np.float32)
    mask = z["mask"].astype(np.float32)
    y = z["y"].astype(int)
    split = safe_str_array(z["split"])

    variables = safe_str_array(z["variables"])
    labels = safe_str_array(z["labels"] if "labels" in z.files else None, length=len(variables))
    sources = safe_str_array(z["sources"] if "sources" in z.files else None, length=len(variables))
    value_sources = safe_str_array(z["value_sources"] if "value_sources" in z.files else None, length=len(variables))

    if "itemids" in z.files:
        itemids = z["itemids"]
    else:
        itemids = np.array([-1] * len(variables))

    if X_raw.ndim != 3:
        raise ValueError(f"Expected X_raw [N,T,F], got {X_raw.shape}")

    N, T, F = X_raw.shape

    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    if train_mask.sum() == 0:
        raise ValueError("No train rows found.")

    X_train = X_raw[train_mask]
    mask_train = mask[train_mask]
    y_train = y[train_mask]

    print("NPZ:", npz_path)
    print("X_raw:", X_raw.shape)
    print("mask:", mask.shape)
    print("Feature count:", F)
    print("Full label counts:", dict(zip(*np.unique(y, return_counts=True))))
    print("Split counts:", dict(zip(*np.unique(split, return_counts=True))))
    print("Train label counts:", dict(zip(*np.unique(y_train, return_counts=True))))
    print("Train prevalence:", float(np.mean(y_train)))

    # ------------------------------------------------------------
    # Feature metadata
    # ------------------------------------------------------------
    print_section("BUILDING FEATURE METADATA")

    hourly_coverage = mask_train.mean(axis=(0, 1))

    feature_meta = pd.DataFrame({
        "feature_index": np.arange(F, dtype=int),
        "variable": variables,
        "label": labels,
        "source": sources,
        "itemid": itemids,
        "value_source": value_sources,
        "hourly_coverage": hourly_coverage,
    })

    feature_meta["possible_leakage_flag"] = feature_meta.apply(
        lambda r: possible_leakage_flag(r["variable"], r["label"], r["source"]),
        axis=1,
    )

    feature_meta["static_like_flag"] = feature_meta.apply(
        lambda r: static_like_flag(r["variable"], r["label"], r["source"]),
        axis=1,
    )

    feature_meta["clinical_category"] = feature_meta.apply(
        lambda r: clinical_category(r["variable"], r["label"], r["source"]),
        axis=1,
    )

    feature_meta["physiology_direction_hint"] = feature_meta["label"].apply(physiological_direction_hint)

    metadata_path = output_dir / f"{stem}_feature_metadata_train_only.csv"
    feature_meta.to_csv(metadata_path, index=False)

    print("feature metadata rows:", len(feature_meta))
    print("possible leakage/proxy:", int(feature_meta["possible_leakage_flag"].sum()))
    print("static-like:", int(feature_meta["static_like_flag"].sum()))
    print("saved feature metadata:", metadata_path)

    # ------------------------------------------------------------
    # Univariate audit
    # ------------------------------------------------------------
    all_modes, best = audit_all_summary_modes(
        X_train=X_train,
        y_train=y_train,
        feature_meta=feature_meta,
        summary_modes=summary_modes,
        random_state=args.random_state,
    )

    all_modes_path = output_dir / f"{stem}_all_modes_train_only.csv"
    best_path = output_dir / f"{stem}_best_summary_per_feature_train_only.csv"

    all_modes.to_csv(all_modes_path, index=False)
    best.to_csv(best_path, index=False)

    # ------------------------------------------------------------
    # Best-summary matrix and redundancy analysis
    # ------------------------------------------------------------
    M_train, best_sorted = build_best_summary_matrix(X_train, best)

    M_imp, M_scaled, imputer, scaler = impute_and_scale_train_matrix(M_train)

    corr_matrix, cov_matrix, cluster_df = redundancy_analysis(
        M_scaled=M_scaled,
        best_sorted=best_sorted,
        corr_threshold=args.corr_threshold,
    )

    corr_path = output_dir / f"{stem}_feature_correlation_matrix_train_only.npy"
    cov_path = output_dir / f"{stem}_feature_covariance_matrix_train_only.npy"
    cluster_path = output_dir / f"{stem}_correlation_clusters_train_only.csv"

    np.save(corr_path, corr_matrix)
    np.save(cov_path, cov_matrix)
    cluster_df.to_csv(cluster_path, index=False)

    # ------------------------------------------------------------
    # mRMR ranking
    # ------------------------------------------------------------
    mrmr_ranked, mrmr_all = mrmr_rank(
        best_sorted=cluster_df.sort_values("feature_index").reset_index(drop=True),
        corr_matrix=corr_matrix,
        lambda_redundancy=args.lambda_redundancy,
    )

    mrmr_ranked_path = output_dir / f"{stem}_mrmr_ranked_train_only.csv"
    mrmr_all_path = output_dir / f"{stem}_mrmr_all_features_train_only.csv"

    mrmr_ranked.to_csv(mrmr_ranked_path, index=False)
    mrmr_all.to_csv(mrmr_all_path, index=False)

    # ------------------------------------------------------------
    # Elastic Net stability selection
    # ------------------------------------------------------------
    stability_df = elasticnet_stability_selection(
        M_scaled=M_scaled,
        y_train=y_train,
        best_sorted=cluster_df.sort_values("feature_index").reset_index(drop=True),
        n_bootstraps=args.n_bootstraps,
        subsample_fraction=args.subsample_fraction,
        random_state=args.random_state,
        elastic_top_k=args.elastic_top_k,
    )

    stability_path = output_dir / f"{stem}_elasticnet_stability_train_only.csv"
    stability_df.to_csv(stability_path, index=False)

    # ------------------------------------------------------------
    # Optional RFECV
    # ------------------------------------------------------------
    rfecv_df = None
    rfecv_path = None

    if args.run_rfecv:
        rfecv_df = rfecv_confirmation(
            M_scaled=M_scaled,
            y_train=y_train,
            best_sorted=cluster_df.sort_values("feature_index").reset_index(drop=True),
            min_features_to_select=args.rfecv_min_features,
            cv_folds=args.rfecv_cv_folds,
            random_state=args.random_state,
        )

        rfecv_path = output_dir / f"{stem}_rfecv_confirmation_train_only.csv"
        rfecv_df.to_csv(rfecv_path, index=False)

    # ------------------------------------------------------------
    # Consensus table
    # ------------------------------------------------------------
    consensus = build_consensus(
        best_sorted=best,
        cluster_df=cluster_df,
        mrmr_ranked=mrmr_ranked,
        stability_df=stability_df,
        rfecv_df=rfecv_df,
    )

    consensus_path = output_dir / f"{stem}_consensus_feature_evidence_train_only.csv"
    consensus.to_csv(consensus_path, index=False)

    core = consensus[consensus["recommendation"] == "core"].copy()
    strong = consensus[consensus["recommendation"].isin(["core", "strong"])].copy()
    review = consensus[consensus["recommendation"] == "review"].copy()

    core_path = output_dir / f"{stem}_recommended_core_features.csv"
    strong_path = output_dir / f"{stem}_recommended_core_plus_strong_features.csv"
    review_path = output_dir / f"{stem}_review_features.csv"

    core.to_csv(core_path, index=False)
    strong.to_csv(strong_path, index=False)
    review.to_csv(review_path, index=False)

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    recommendation_counts = consensus["recommendation"].value_counts().to_dict()

    category_summary = (
        consensus.groupby(["recommendation", "clinical_category"])
        .size()
        .rename("n_features")
        .reset_index()
        .sort_values(["recommendation", "n_features"], ascending=[True, False])
    )

    category_summary_path = output_dir / f"{stem}_recommendation_by_category_summary.csv"
    category_summary.to_csv(category_summary_path, index=False)

    summary = {
        "npz_path": str(npz_path),
        "shape": list(X_raw.shape),
        "train_n": int(train_mask.sum()),
        "val_n": int(val_mask.sum()),
        "test_n": int(test_mask.sum()),
        "train_label_counts": {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
        "train_prevalence": float(np.mean(y_train)),
        "summary_modes": summary_modes,
        "corr_threshold": args.corr_threshold,
        "lambda_redundancy": args.lambda_redundancy,
        "n_bootstraps": args.n_bootstraps,
        "random_state": int(args.random_state),
        "elastic_top_k": args.elastic_top_k,
        "subsample_fraction": args.subsample_fraction,
        "run_rfecv": bool(args.run_rfecv),
        "recommendation_counts": {str(k): int(v) for k, v in recommendation_counts.items()},
        "outputs": {
            "all_modes": str(all_modes_path),
            "best_summary": str(best_path),
            "correlation_matrix_npy": str(corr_path),
            "covariance_matrix_npy": str(cov_path),
            "correlation_clusters": str(cluster_path),
            "mrmr_ranked": str(mrmr_ranked_path),
            "mrmr_all": str(mrmr_all_path),
            "elasticnet_stability": str(stability_path),
            "rfecv": str(rfecv_path) if rfecv_path is not None else None,
            "consensus": str(consensus_path),
            "recommended_core": str(core_path),
            "recommended_core_plus_strong": str(strong_path),
            "review": str(review_path),
            "recommendation_by_category": str(category_summary_path),
        },
    }

    summary_path = output_dir / f"{stem}_feature_selection_summary.json"
    save_json(summary, summary_path)

    # ------------------------------------------------------------
    # Print useful outputs
    # ------------------------------------------------------------
    print_section("RECOMMENDATION COUNTS")
    print(consensus["recommendation"].value_counts().to_string())

    print_section("TOP CONSENSUS FEATURES")
    show_cols = [
        "recommendation",
        "feature_index",
        "variable",
        "label",
        "source",
        "clinical_category",
        "summary_mode",
        "valid_fraction",
        "hourly_coverage",
        "point_biserial_corr",
        "auroc",
        "auprc",
        "ap_lift_over_prevalence",
        "mutual_information",
        "univariate_relevance_score",
        "correlation_cluster_id",
        "correlation_cluster_size",
        "mrmr_rank",
        "mrmr_score",
        "elasticnet_selection_probability",
        "elasticnet_mean_abs_coefficient",
        "rfecv_selected",
        "consensus_score",
        "physiology_direction_hint",
    ]

    existing_show_cols = [c for c in show_cols if c in consensus.columns]
    top_preview_path = output_dir / f"{stem}_top_consensus_preview.csv"
    consensus[existing_show_cols].head(80).to_csv(top_preview_path, index=False)
    print("saved top consensus preview:", top_preview_path)

    print_section("SAVED OUTPUTS")
    for k, v in summary["outputs"].items():
        print(f"{k}: {v}")
if __name__ == "__main__":
    main()
