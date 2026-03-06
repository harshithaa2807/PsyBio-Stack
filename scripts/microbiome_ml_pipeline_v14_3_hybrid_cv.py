#!/usr/bin/env python3
"""
microbiome_ml_pipeline_v14_3_hybrid_cv.py

v14.3: Hybrid microbiome + clinical covariate pipeline with:
 - global prevalence filter
 - automatic covariate cleaning & encoding
 - CLR / log1p transforms
 - VarianceThreshold -> SelectKBest(k=25) -> optional PCA (default 0.90)
 - Stratified K-Fold CV on training set with bagged stacking (RF + ElasticNet)
 - Final model retrained on full training set and evaluated on validation & test
 - Saves metrics, plots, selected features, and pipeline bundle

Usage:
  python scripts\microbiome_ml_pipeline_v14_3_hybrid_cv.py --feature data\feature-table.tsv --meta data\metadata.tsv --target depression_bipolar_schizophrenia
"""
import os, argparse, datetime, joblib, warnings, re
warnings.filterwarnings("ignore")
import multiprocessing
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_curve
)

# optional XGBoost (not required)
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# plotting headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# ------------------ helpers ------------------
def clean_taxonomy_name(tax_str):
    if pd.isna(tax_str):
        return "Unknown"
    parts = [p.strip() for p in str(tax_str).split(';')]
    genus = next((p for p in parts if p.startswith('g__') and len(p) > 3), None)
    family = next((p for p in parts if p.startswith('f__') and len(p) > 3), None)
    if genus and genus != 'g__':
        return re.sub(r'^g__', '', genus)
    if family and family != 'f__':
        return re.sub(r'^f__', '', family)
    return "Unknown"

def clr_transform(df, pseudocount=1e-9):
    df2 = df.replace(0, pseudocount)
    log_df = np.log(df2)
    gm = log_df.mean(axis=1)
    clr = log_df.sub(gm, axis=0)
    return clr

def try_parse_numeric_series(s):
    s = s.astype(str).replace({'nan': np.nan, 'None': np.nan})
    s_lower = s.str.lower().fillna('')
    bool_mask = s_lower.isin(['true','false','yes','no','y','n'])
    out = pd.Series(index=s.index, dtype=float)
    out[bool_mask] = s_lower[bool_mask].map(lambda v: 1.0 if v in ('true','yes','y') else 0.0)
    others = s[~bool_mask].dropna()
    num_extracted = others.str.extract(r'([-+]?\d*\.?\d+)', expand=False)
    coerced = pd.to_numeric(num_extracted, errors='coerce')
    out.loc[coerced.index] = coerced
    return out

def encode_categorical(df_cat, min_fraction=0.02):
    if df_cat.shape[1] == 0:
        return pd.DataFrame(index=df_cat.index)
    enc_frames = []
    n = df_cat.shape[0]
    for col in df_cat.columns:
        ser = df_cat[col].astype(str).fillna('NA')
        counts = ser.value_counts(dropna=False)
        keep_levels = counts[counts >= (min_fraction * n)].index.tolist()
        ser_filtered = ser.where(ser.isin(keep_levels), other='OTHER')
        oh = pd.get_dummies(ser_filtered, prefix=col)
        enc_frames.append(oh)
    if enc_frames:
        df_enc = pd.concat(enc_frames, axis=1)
    else:
        df_enc = pd.DataFrame(index=df_cat.index)
    return df_enc

def save_confusion_and_plots(y_true, probs, name, outdir):
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"{name} Confusion Matrix")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name}_confusion_matrix.png"), dpi=200)
    plt.close()
    pd.DataFrame(cm, index=["Actual_No","Actual_Yes"], columns=["Pred_No","Pred_Yes"]).to_csv(os.path.join(outdir, f"{name}_confusion_matrix.csv"))

    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}"); plt.plot([0,1],[0,1],'k--'); plt.title(f"{name} ROC"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name}_roc.png"), dpi=200); plt.close()

    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(recall, precision)
    plt.figure(); plt.plot(recall, precision, label=f"PR AUC={pr_auc:.3f}"); plt.title(f"{name} PR"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name}_pr.png"), dpi=200); plt.close()

def compute_metrics(y_true, probs):
    preds = (probs >= 0.5).astype(int)
    try:
        roc = roc_auc_score(y_true, probs)
    except Exception:
        roc = float('nan')
    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(recall, precision)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        "Accuracy": accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall": recall_score(y_true, preds, zero_division=0),
        "Specificity": spec,
        "F1": f1_score(y_true, preds, zero_division=0),
        "AUC": roc,
        "PR_AUC": pr_auc,
        "Balanced": balanced_accuracy_score(y_true, preds)
    }

# ------------------ load features ------------------
def load_feature_table(feature_path):
    feat = pd.read_csv(feature_path, sep='\t', header=1, dtype=str)
    first_col = feat.columns[0]
    feat[first_col] = feat[first_col].apply(clean_taxonomy_name)
    feat = feat.set_index(first_col)
    feat = feat.apply(pd.to_numeric, errors='coerce').fillna(0)
    feat = feat.groupby(feat.index).sum()
    otu = feat.T
    return otu

# ------------------ main pipeline ------------------
def main(feature_path, meta_path, target_col,
         prevalence_cutoff=0.10, k=25, pca_energy=0.90, transform_choice='auto',
         bagged_runs=5, cv_folds=5, random_state=42):
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join("results", f"run_{ts}_v14_3_hybrid_cv")
    os.makedirs(results_dir, exist_ok=True)
    print("Results ->", results_dir)

    # load data
    X_raw = load_feature_table(feature_path)  # samples x taxa
    meta = pd.read_csv(meta_path, sep='\t', dtype=str)
    sample_cols = [c for c in meta.columns if 'sample' in c.lower() or 'id' in c.lower()]
    if not sample_cols:
        raise ValueError("No sample-id column in metadata.tsv")
    sid = sample_cols[0]
    meta[sid] = meta[sid].astype(str).str.strip().str.upper()
    X_raw.index = X_raw.index.astype(str).str.strip().str.upper()

    # align samples
    common = sorted(list(set(meta[sid]) & set(X_raw.index)))
    if len(common) == 0:
        raise ValueError("No overlapping sample IDs")
    meta = meta[meta[sid].isin(common)].set_index(sid)
    X_raw = X_raw.loc[common]
    y = meta.loc[common, target_col].astype(str).str.strip()
    print("Samples:", X_raw.shape[0], "Taxa:", X_raw.shape[1])
    print("Target distribution:\n", y.value_counts())

    # detect covariates
    candidate_covs = [c for c in meta.columns if c != target_col]
    numeric_cols, categorical_cols = [], []
    for c in candidate_covs:
        parsed = try_parse_numeric_series(meta[c])
        if parsed.notnull().sum() >= 0.10 * meta.shape[0]:
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    print("Detected numeric covariates:", numeric_cols)
    print("Detected categorical covariates (candidates):", categorical_cols)

    cov_numeric = pd.DataFrame(index=meta.index)
    for c in numeric_cols:
        parsed = try_parse_numeric_series(meta[c])
        parsed = parsed.fillna(parsed.median())
        cov_numeric[c] = parsed

    cov_categorical_raw = meta[categorical_cols].astype(str).replace('nan', np.nan).fillna('NA')
    cov_categorical_raw = cov_categorical_raw.applymap(lambda v: str(v).strip())
    cov_categorical_encoded = encode_categorical(cov_categorical_raw, min_fraction=0.02)
    print("Categorical encoded shape:", cov_categorical_encoded.shape)

    if cov_numeric.shape[1] == 0 and cov_categorical_encoded.shape[1] == 0:
        covariates_df = None
        print("No usable clinical covariates found.")
    else:
        covariates_df = pd.concat([cov_numeric, cov_categorical_encoded], axis=1)
        covariates_df = covariates_df.loc[X_raw.index]
        print("Final covariates shape:", covariates_df.shape)
        covariates_df.to_csv(os.path.join(results_dir, "covariates_used.csv"))

    # global prevalence filter
    prevalence = (X_raw > 0).sum(axis=0)
    keep_mask = prevalence >= (prevalence_cutoff * X_raw.shape[0])
    kept_taxa = X_raw.columns[keep_mask].tolist()
    print(f"Kept {len(kept_taxa)} taxa after global prevalence filter (cutoff={prevalence_cutoff})")
    with open(os.path.join(results_dir, "kept_taxa_global.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(kept_taxa))

    X_kept = X_raw.loc[:, kept_taxa]
    X_rel = X_kept.div(X_kept.sum(axis=1), axis=0).fillna(0)

    # splits 70/15/15
    X_train_rel, X_temp_rel, y_train, y_temp = train_test_split(X_rel, y, test_size=0.3, stratify=y, random_state=random_state)
    X_val_rel, X_test_rel, y_val, y_test = train_test_split(X_temp_rel, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state)
    print("Splits: train/val/test =", X_train_rel.shape[0], X_val_rel.shape[0], X_test_rel.shape[0])

    # decide transforms to try
    transforms = []
    if transform_choice == 'auto':
        transforms = ['clr', 'log1p']
    else:
        transforms = [transform_choice]

    variant_summaries = []

    # fixed random seeds for bagging
    seeds = [random_state + i*7 for i in range(bagged_runs)]

    # For each transform, perform CV on training set to estimate performance
    for transform_name in transforms:
        print(f"\n--- Transform: {transform_name} ---")
        # Transform data (microbiome only)
        if transform_name == 'clr':
            X_train_proc_micro = clr_transform(X_train_rel.replace(0, 1e-9))
            X_val_proc_micro = clr_transform(X_val_rel.replace(0, 1e-9))
            X_test_proc_micro = clr_transform(X_test_rel.replace(0, 1e-9))
        else:
            X_train_proc_micro = np.log1p(X_train_rel.replace(0, 1e-6))
            X_val_proc_micro = np.log1p(X_val_rel.replace(0, 1e-6))
            X_test_proc_micro = np.log1p(X_test_rel.replace(0, 1e-6))

        # attach covariates
        if covariates_df is not None:
            X_train_full = pd.concat([X_train_proc_micro, covariates_df.loc[X_train_proc_micro.index]], axis=1)
            X_val_full = pd.concat([X_val_proc_micro, covariates_df.loc[X_val_proc_micro.index]], axis=1)
            X_test_full = pd.concat([X_test_proc_micro, covariates_df.loc[X_test_proc_micro.index]], axis=1)
        else:
            X_train_full = X_train_proc_micro.copy()
            X_val_full = X_val_proc_micro.copy()
            X_test_full = X_test_proc_micro.copy()

        # standardize
        scaler = StandardScaler()
        X_train_s = pd.DataFrame(scaler.fit_transform(X_train_full), index=X_train_full.index, columns=X_train_full.columns)
        X_val_s = pd.DataFrame(scaler.transform(X_val_full), index=X_val_full.index, columns=X_val_full.columns)
        X_test_s = pd.DataFrame(scaler.transform(X_test_full), index=X_test_full.index, columns=X_test_full.columns)

        # variance threshold
        vt = VarianceThreshold(threshold=1e-6)
        X_train_v = pd.DataFrame(vt.fit_transform(X_train_s), index=X_train_s.index, columns=X_train_s.columns[vt.get_support()])
        X_val_v = pd.DataFrame(vt.transform(X_val_s), index=X_val_s.index, columns=X_train_s.columns[vt.get_support()])
        X_test_v = pd.DataFrame(vt.transform(X_test_s), index=X_test_s.index, columns=X_train_s.columns[vt.get_support()])

        # SelectKBest
        k_actual = min(k, X_train_v.shape[1])
        skb = SelectKBest(mutual_info_classif, k=k_actual)
        le_sel = LabelEncoder(); y_train_enc = le_sel.fit_transform(y_train.astype(str))
        skb.fit(X_train_v.values, y_train_enc)
        sel_cols = X_train_v.columns[skb.get_support()]
        X_train_k = pd.DataFrame(skb.transform(X_train_v), index=X_train_v.index, columns=sel_cols)
        X_val_k = pd.DataFrame(skb.transform(X_val_v), index=X_val_v.index, columns=sel_cols)
        X_test_k = pd.DataFrame(skb.transform(X_test_v), index=X_test_v.index, columns=sel_cols)
        print(f"After SelectKBest: {X_train_k.shape[1]} features")

        # PCA optional
        if pca_energy is not None and 0 < pca_energy < 1:
            pca = PCA(n_components=pca_energy, svd_solver='full', random_state=random_state)
            X_train_final = pca.fit_transform(X_train_k.values)
            X_val_final = pca.transform(X_val_k.values)
            X_test_final = pca.transform(X_test_k.values)
            print("PCA applied. Reduced dim ->", X_train_final.shape[1])
        else:
            pca = None
            X_train_final = X_train_k.values
            X_val_final = X_val_k.values
            X_test_final = X_test_k.values

        # prepare learners
        n_cores = multiprocessing.cpu_count()
        rf = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=12, class_weight='balanced', random_state=random_state, n_jobs=n_cores)
        enet = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.6, C=0.5, max_iter=3000, class_weight='balanced', random_state=random_state)

        estimators = [('rf', rf), ('en', enet)]
        meta = LogisticRegression(penalty='l2', solver='liblinear', C=1.0, class_weight='balanced', max_iter=2000, random_state=random_state)
        stacking = StackingClassifier(estimators=estimators, final_estimator=meta, passthrough=False, n_jobs=n_cores)

        # STRATIFIED K-FOLD CV on training set -> get out-of-fold val probs (cv_val_probs)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        oof_val_probs = np.zeros(X_train_final.shape[0], dtype=float)
        oof_val_idxs = np.zeros(X_train_final.shape[0], dtype=bool)
        fold_index = 0
        for train_idx, val_idx in skf.split(X_train_final, y_train_enc):
            fold_index += 1
            X_tr = X_train_final[train_idx]
            X_va = X_train_final[val_idx]
            y_tr = y_train_enc[train_idx]
            # Train bagged stacking on this fold and average
            val_probs_runs = []
            for i,seed in enumerate(seeds):
                rf.set_params(random_state=seed + fold_index)
                enet.set_params(random_state=seed + fold_index)
                meta.set_params(random_state=seed + fold_index)
                stacking.fit(X_tr, y_tr)
                val_probs_runs.append(stacking.predict_proba(X_va)[:,1])
            val_probs_fold = np.mean(np.vstack(val_probs_runs), axis=0)
            oof_val_probs[val_idx] = val_probs_fold
            oof_val_idxs[val_idx] = True
            print(f"  CV fold {fold_index} done")
        # estimate CV AUC on training set (out-of-fold)
        try:
            cv_auc = roc_auc_score(y_train_enc[oof_val_idxs], oof_val_probs[oof_val_idxs])
        except Exception:
            cv_auc = float('nan')
        print("CV (oof) AUC on training set:", cv_auc)

        # Now train final stacking on full training set (bagged runs) and evaluate on val & test
        train_probs_runs = []
        val_probs_runs = []
        test_probs_runs = []
        le_final = LabelEncoder(); le_final.fit(y_train.astype(str))
        for i,seed in enumerate(seeds):
            rf.set_params(random_state=seed)
            enet.set_params(random_state=seed)
            meta.set_params(random_state=seed)
            stacking.fit(X_train_final, le_final.transform(y_train.astype(str)))
            train_probs_runs.append(stacking.predict_proba(X_train_final)[:,1])
            val_probs_runs.append(stacking.predict_proba(X_val_final)[:,1])
            test_probs_runs.append(stacking.predict_proba(X_test_final)[:,1])
            joblib.dump(stacking, os.path.join(results_dir, f"stacking_{transform_name}_run{i+1}_seed{seed}.pkl"), compress=3)

        train_proba = np.mean(np.vstack(train_probs_runs), axis=0)
        val_proba = np.mean(np.vstack(val_probs_runs), axis=0)
        test_proba = np.mean(np.vstack(test_probs_runs), axis=0)

        le_enc = LabelEncoder(); le_enc.fit(y_train.astype(str))
        y_train_enc_final = le_enc.transform(y_train.astype(str))
        y_val_enc_final = le_enc.transform(y_val.astype(str))
        y_test_enc_final = le_enc.transform(y_test.astype(str))

        train_metrics = compute_metrics(y_train_enc_final, train_proba)
        val_metrics = compute_metrics(y_val_enc_final, val_proba)
        test_metrics = compute_metrics(y_test_enc_final, test_proba)

        # Save selected features & pipeline bundle
        with open(os.path.join(results_dir, f"selected_features_{transform_name}.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(sel_cols.tolist()))
        bundle = {
            'transform': transform_name,
            'kept_taxa': kept_taxa,
            'covariate_columns': covariates_df.columns.tolist() if covariates_df is not None else [],
            'scaler': scaler,
            'vt_support': vt.get_support(),
            'skb': skb,
            'sel_cols': sel_cols.tolist(),
            'pca': pca,
            'rf_params': rf.get_params(),
            'enet_params': enet.get_params(),
            'meta_params': meta.get_params(),
            'seeds': seeds
        }
        joblib.dump(bundle, os.path.join(results_dir, f"pipeline_bundle_{transform_name}.pkl"), compress=3)

        # save metrics & plots
        pd.DataFrame([train_metrics, val_metrics, test_metrics], index=['Training','Validation','Test']).to_csv(os.path.join(results_dir, f"metrics_{transform_name}.csv"))
        save_confusion_and_plots(y_train_enc_final, train_proba, f"{transform_name}_Training", results_dir)
        save_confusion_and_plots(y_val_enc_final, val_proba, f"{transform_name}_Validation", results_dir)
        save_confusion_and_plots(y_test_enc_final, test_proba, f"{transform_name}_Test", results_dir)

        variant_summaries.append({
            'transform': transform_name,
            'cv_oof_auc': cv_auc,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'bundle_path': os.path.join(results_dir, f"pipeline_bundle_{transform_name}.pkl"),
            'selected_features': sel_cols.tolist()
        })
        # Save final trained model for SHAP interpretation
        model_save_path = os.path.join(results_dir, f"final_model_{transform_name}.pkl")
        joblib.dump(stacking, model_save_path, compress=3)
        print(f"Saved final trained model → {model_save_path}")


    # choose best variant by validation AUC
    best = max(variant_summaries, key=lambda x: (x['val_metrics']['AUC'] if x['val_metrics']['AUC'] is not None else -1))
    print("\nBest transform chosen:", best['transform'], "Validation AUC:", best['val_metrics']['AUC'])
    pd.DataFrame([best['train_metrics'], best['val_metrics'], best['test_metrics']], index=['Training','Validation','Test']).to_csv(os.path.join(results_dir, "final_metrics_v14_3.csv"))
    with open(os.path.join(results_dir, "final_selected_features_v14_3.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(best['selected_features']))
    # copy bundle as best
    joblib.dump(joblib.load(best['bundle_path']), os.path.join(results_dir, "pipeline_bundle_best_v14_3.pkl"), compress=3)

    print("Done. Results & models saved in:", results_dir)
    print("Final validation metrics:", best['val_metrics'])
    print("Final test metrics:", best['test_metrics'])

# ------------------ CLI ------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--feature", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--prevalence", type=float, default=0.10)
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--pca", type=float, default=0.90)
    p.add_argument("--transform", choices=['auto','clr','log1p'], default='auto')
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args.feature, args.meta, args.target, prevalence_cutoff=args.prevalence, k=args.k, pca_energy=args.pca, transform_choice=args.transform, bagged_runs=args.runs, cv_folds=args.cv, random_state=args.seed)

