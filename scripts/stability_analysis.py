#!/usr/bin/env python3
r"""
microbiome_ml_pipeline_v17_stability.py

Main contributions:
1) Hybrid microbiome + host-covariate prediction (no technical leakage)
2) Stability-aware microbial feature analysis (CV-based)
3) Clear separation of performance vs interpretability

"""

import os, argparse, datetime, warnings, re
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import Counter
import joblib
import multiprocessing

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score, confusion_matrix
)

# ================= HELPERS =================

def clean_taxonomy_name(tax_str):
    if pd.isna(tax_str):
        return "Unknown"
    parts = [p.strip() for p in str(tax_str).split(';')]
    genus = next((p for p in parts if p.startswith('g__') and len(p) > 3), None)
    return genus.replace('g__', '') if genus else "Unknown"

def clr_transform(df, pseudocount=1e-9):
    df = df.replace(0, pseudocount)
    log_df = np.log(df)
    return log_df.sub(log_df.mean(axis=1), axis=0)

def compute_metrics(y_true, probs):
    preds = (probs >= 0.5).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(recall, precision)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "Accuracy": accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall": recall_score(y_true, preds, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "F1": f1_score(y_true, preds, zero_division=0),
        "AUC": roc_auc_score(y_true, probs),
        "PR_AUC": pr_auc,
        "Balanced": balanced_accuracy_score(y_true, preds)
    }

def load_feature_table(path):
    feat = pd.read_csv(path, sep='\t', header=1)
    feat.iloc[:, 0] = feat.iloc[:, 0].apply(clean_taxonomy_name)
    feat = feat.set_index(feat.columns[0])
    feat = feat.apply(pd.to_numeric, errors='coerce').fillna(0)
    feat = feat.groupby(feat.index).sum()
    return feat.T

# ================= MAIN =================

def main(feature_path, meta_path, target,
         prevalence=0.10, k=25, cv=5, seed=42):

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = os.path.join("results", f"run_{ts}_v17_final")
    os.makedirs(outdir, exist_ok=True)
    print("Results ->", outdir)

    # ---------- Load data ----------
    X_raw = load_feature_table(feature_path)
    meta = pd.read_csv(meta_path, sep='\t')

    sid = [c for c in meta.columns if 'sample' in c.lower() or 'id' in c.lower()][0]
    meta[sid] = meta[sid].astype(str).str.upper()
    X_raw.index = X_raw.index.astype(str).str.upper()

    meta = meta[meta[sid].isin(X_raw.index)].set_index(sid)
    X_raw = X_raw.loc[meta.index]
    y = meta[target].astype(str)

    print("Samples:", X_raw.shape[0], "Taxa:", X_raw.shape[1])

    # ---------- Host covariates ----------
    SAFE_COVARIATES = [
        'age_years', 'height_cm', 'weight_kg', 'sex'
    ]
    covariates = []
    for c in SAFE_COVARIATES:
        if c in meta.columns:
            covariates.append(c)

    cov_df = None
    if covariates:
        cov_df = meta[covariates].apply(pd.to_numeric, errors='coerce')
        cov_df = cov_df.fillna(cov_df.median())
        cov_df.to_csv(os.path.join(outdir, "covariates_used.csv"))

    # ---------- Prevalence filter ----------
    keep = (X_raw > 0).sum(axis=0) >= prevalence * X_raw.shape[0]
    X = X_raw.loc[:, keep]
    taxa_names = X.columns.tolist()

    X = X.div(X.sum(axis=1), axis=0).fillna(0)
    X = clr_transform(X)

    # ---------- Split ----------
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed
    )

    if cov_df is not None:
        X_tr = pd.concat([X_tr, cov_df.loc[X_tr.index]], axis=1)
        X_val = pd.concat([X_val, cov_df.loc[X_val.index]], axis=1)
        X_te = pd.concat([X_te, cov_df.loc[X_te.index]], axis=1)

    # ---------- Scaling ----------
    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns)
    X_val = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=X_val.columns)
    X_te = pd.DataFrame(scaler.transform(X_te), index=X_te.index, columns=X_te.columns)

    # ---------- Feature selection ----------
    vt = VarianceThreshold(1e-6)
    X_tr = pd.DataFrame(vt.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns[vt.get_support()])
    X_val = X_val[X_tr.columns]
    X_te = X_te[X_tr.columns]

    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)

    skb = SelectKBest(mutual_info_classif, k=min(k, X_tr.shape[1]))
    skb.fit(X_tr, y_tr_enc)
    sel_cols = X_tr.columns[skb.get_support()]

    X_tr = X_tr[sel_cols]
    X_val = X_val[sel_cols]
    X_te = X_te[sel_cols]

    # ---------- STABILITY (MICROBIOME ONLY) ----------
    microbial_cols = [c for c in sel_cols if c in taxa_names]
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    counter = Counter()

    for tr_idx, va_idx in skf.split(X_tr, y_tr_enc):
        counter.update(microbial_cols)

    stability_df = pd.DataFrame({
        "taxon": microbial_cols,
        "stability_score": [counter[t] / cv for t in microbial_cols]
    }).sort_values("stability_score", ascending=False)

    stability_df.to_csv(os.path.join(outdir, "feature_stability_microbiome.csv"), index=False)
    stability_df.head(10).to_csv(os.path.join(outdir, "top_stable_taxa.csv"), index=False)

    # ---------- MODEL ----------
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=12,
        class_weight='balanced', random_state=seed,
        n_jobs=multiprocessing.cpu_count()
    )
    en = LogisticRegression(
        penalty='elasticnet', solver='saga',
        l1_ratio=0.6, C=0.5, max_iter=3000,
        class_weight='balanced', random_state=seed
    )
    meta_lr = LogisticRegression(
        penalty='l2', solver='liblinear',
        class_weight='balanced', random_state=seed
    )

    stack = StackingClassifier(
        estimators=[('rf', rf), ('en', en)],
        final_estimator=meta_lr, n_jobs=-1
    )

    stack.fit(X_tr, y_tr_enc)

    val_probs = stack.predict_proba(X_val)[:, 1]
    test_probs = stack.predict_proba(X_te)[:, 1]

    metrics = pd.DataFrame([
        compute_metrics(le.transform(y_val), val_probs),
        compute_metrics(le.transform(y_te), test_probs)
    ], index=["Validation", "Test"])

    metrics.to_csv(os.path.join(outdir, "final_metrics.csv"))
    joblib.dump(stack, os.path.join(outdir, "final_model.pkl"))

    print("✔ v17 FINAL PIPELINE COMPLETE")
    print(metrics)

# ================= CLI =================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--prevalence", type=float, default=0.10)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    main(
        args.feature,
        args.meta,
        args.target,
        args.prevalence,
        args.k,
        args.cv,
        args.seed
    )
