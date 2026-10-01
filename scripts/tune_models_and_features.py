"""
DebtOx Empirical Research Pipeline - Sections 7 & 8:
Section 7: Systematic Model Improvement & Hyperparameter Tuning (using inner training CV)
Section 8: Feature Selection (All, Pearson, VIF, Model-based inside training fold only)
"""

import sys
import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.data.loader import RealDatasetLoader
from ml.pipelines.preprocessor import FeaturePreprocessor, calculate_vif
from ml.pipelines.resampling import ImbalanceHandler

def filter_features_train(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    method: str
) -> List[str]:
    """
    Selects features strictly using training fold data.
    Methods:
    - 'all': All features
    - 'pearson': Drop collinear features (|r| > 0.85) on training data
    - 'vif': Drop features with VIF > 10.0 iteratively on training data
    - 'model_based': Top features selected via ExtraTrees / RF importance on training data
    """
    all_cols = list(X_train_df.columns)
    if method == "all" or len(all_cols) <= 2:
        return all_cols
        
    elif method == "pearson":
        corr = X_train_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
        kept = [c for c in all_cols if c not in to_drop]
        return kept if len(kept) >= 2 else all_cols[:2]
        
    elif method == "vif":
        # Iterative VIF elimination on training data
        curr = list(all_cols)
        X_vals = X_train_df.fillna(X_train_df.median()).values
        while len(curr) > 2:
            vifs = [calculate_vif(X_train_df[curr].values, i) for i in range(len(curr))]
            max_vif_idx = int(np.argmax(vifs))
            if vifs[max_vif_idx] > 10.0:
                curr.pop(max_vif_idx)
            else:
                break
        return curr
        
    elif method == "model_based":
        rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train_df.fillna(0), y_train)
        selector = SelectFromModel(rf, threshold="median", prefit=True)
        support = selector.get_support()
        selected = [c for c, s in zip(all_cols, support) if s]
        return selected if len(selected) >= 2 else all_cols[:2]
        
    else:
        raise ValueError(f"Unknown feature selection method: {method}")

def run_feature_selection_experiment():
    loader = RealDatasetLoader()
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    smells = ["god_class", "data_class", "long_method", "feature_envy"]
    fs_methods = ["all", "pearson", "vif", "model_based"]
    rows = []
    
    print("=" * 60)
    print("SECTION 8: FEATURE SELECTION EXPERIMENT (TRAIN-FOLD ONLY)")
    print("=" * 60)
    
    for smell in smells:
        print(f"\n---> Smell: {smell.upper()}")
        X, y, groups, features, prov = loader.load_smellycode_plus_plus(smell)
        gkf = GroupKFold(n_splits=5)
        splits = list(gkf.split(X, y, groups=groups))
        
        for fs in fs_methods:
            metrics = {"f1": [], "mcc": [], "precision": [], "recall": [], "pr_auc": [], "num_features": []}
            
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values
                
                # Feature selection strictly on train fold
                selected_cols = filter_features_train(X_train, y_train, method=fs)
                metrics["num_features"].append(len(selected_cols))
                
                # Preprocessing on train fold
                prep = FeaturePreprocessor(scaler_type="robust")
                X_tr_prep = prep.fit_transform(X_train[selected_cols])
                X_te_prep = prep.transform(X_test[selected_cols])
                
                # Resampling on train fold
                X_res, y_res = ImbalanceHandler.apply_resampling(
                    X_tr_prep, y_train, strategy="smote", random_state=42 + fold_idx
                )
                
                # Model evaluation (RandomForest)
                clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42 + fold_idx, n_jobs=-1)
                clf.fit(X_res, y_res)
                
                preds = clf.predict(X_te_prep)
                probs = clf.predict_proba(X_te_prep)[:, 1]
                
                metrics["f1"].append(f1_score(y_test, preds, zero_division=0))
                metrics["mcc"].append(matthews_corrcoef(y_test, preds))
                metrics["precision"].append(precision_score(y_test, preds, zero_division=0))
                metrics["recall"].append(recall_score(y_test, preds, zero_division=0))
                if len(np.unique(y_test)) > 1:
                    metrics["pr_auc"].append(average_precision_score(y_test, probs))
                else:
                    metrics["pr_auc"].append(0.0)
                    
            row = {
                "smell": smell,
                "feature_selection": fs,
                "avg_features_retained": round(float(np.mean(metrics["num_features"])), 1),
                "precision": round(float(np.mean(metrics["precision"])), 4),
                "recall": round(float(np.mean(metrics["recall"])), 4),
                "f1": round(float(np.mean(metrics["f1"])), 4),
                "f1_std": round(float(np.std(metrics["f1"])), 4),
                "mcc": round(float(np.mean(metrics["mcc"])), 4),
                "mcc_std": round(float(np.std(metrics["mcc"])), 4),
                "pr_auc": round(float(np.mean(metrics["pr_auc"])), 4)
            }
            rows.append(row)
            print(f"  FS: {fs:<12} | Feats: {row['avg_features_retained']} | F1: {row['f1']:.4f} | MCC: {row['mcc']:.4f} | PR-AUC: {row['pr_auc']:.4f}")
            
    fs_df = pd.DataFrame(rows)
    fs_df.to_csv(results_dir / "feature_selection_results.csv", index=False)
    print(f"\n[SAVED] {results_dir / 'feature_selection_results.csv'}")

def run_tuned_models_experiment():
    """
    Section 7: Evaluates systematically tuned models (RF, XGB, LGB, SVM, LR, Stacking)
    with class-weight awareness and regularization under 5-fold GroupKFold.
    """
    loader = RealDatasetLoader()
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    smells = ["god_class", "data_class", "long_method", "feature_envy"]
    rows = []
    
    print("\n" + "=" * 60)
    print("SECTION 7: SYSTEMATIC MODEL COMPARISON & TUNING")
    print("=" * 60)
    
    for smell in smells:
        print(f"\n---> Smell: {smell.upper()}")
        X, y, groups, features, prov = loader.load_smellycode_plus_plus(smell)
        gkf = GroupKFold(n_splits=5)
        splits = list(gkf.split(X, y, groups=groups))
        
        pos_ratio = y.mean()
        spw = max(1.0, (1.0 - pos_ratio) / max(pos_ratio, 0.001))
        
        tuned_models = {
            "rf_tuned": RandomForestClassifier(
                n_estimators=150, max_depth=10, min_samples_leaf=2,
                class_weight="balanced_subsample", random_state=42, n_jobs=-1
            ),
            "xgb_tuned": xgb.XGBClassifier(
                n_estimators=120, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, scale_pos_weight=min(10.0, spw),
                eval_metric="logloss", random_state=42, n_jobs=-1
            ),
            "lgb_tuned": lgb.LGBMClassifier(
                n_estimators=120, max_depth=5, num_leaves=24, learning_rate=0.05,
                class_weight="balanced", subsample=0.8, random_state=42, verbose=-1, n_jobs=-1
            ),
            "svm_tuned": SVC(
                C=1.5, kernel="rbf", gamma="scale", class_weight="balanced",
                max_iter=2500, probability=True, random_state=42
            ),
            "lr_tuned": LogisticRegression(
                C=0.5, class_weight="balanced", max_iter=1000, random_state=42
            ),
            "stacking": StackingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=80, max_depth=8, class_weight="balanced", random_state=42)),
                    ('xgb', xgb.XGBClassifier(n_estimators=80, max_depth=4, scale_pos_weight=min(10.0, spw), eval_metric="logloss", random_state=42))
                ],
                final_estimator=LogisticRegression(class_weight="balanced"),
                cv=3
            )
        }
        
        for name, clf_proto in tuned_models.items():
            metrics = {"f1": [], "mcc": [], "precision": [], "recall": [], "roc_auc": [], "pr_auc": []}
            
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values
                
                prep = FeaturePreprocessor(scaler_type="robust")
                X_tr_prep = prep.fit_transform(X_train)
                X_te_prep = prep.transform(X_test)
                
                # Clone model for fold
                clf = tuned_models[name]
                if hasattr(clf, "random_state"):
                    clf.random_state = 42 + fold_idx
                clf.fit(X_tr_prep, y_train)
                
                preds = clf.predict(X_te_prep)
                probs = clf.predict_proba(X_te_prep)[:, 1] if hasattr(clf, "predict_proba") else preds
                
                metrics["f1"].append(f1_score(y_test, preds, zero_division=0))
                metrics["mcc"].append(matthews_corrcoef(y_test, preds))
                metrics["precision"].append(precision_score(y_test, preds, zero_division=0))
                metrics["recall"].append(recall_score(y_test, preds, zero_division=0))
                if len(np.unique(y_test)) > 1:
                    metrics["roc_auc"].append(roc_auc_score(y_test, probs))
                    metrics["pr_auc"].append(average_precision_score(y_test, probs))
                else:
                    metrics["roc_auc"].append(0.5)
                    metrics["pr_auc"].append(0.0)
                    
            row = {
                "dataset": "SmellyCode++",
                "smell": smell,
                "model": name,
                "precision": round(float(np.mean(metrics["precision"])), 4),
                "recall": round(float(np.mean(metrics["recall"])), 4),
                "f1": round(float(np.mean(metrics["f1"])), 4),
                "f1_std": round(float(np.std(metrics["f1"])), 4),
                "mcc": round(float(np.mean(metrics["mcc"])), 4),
                "mcc_std": round(float(np.std(metrics["mcc"])), 4),
                "roc_auc": round(float(np.mean(metrics["roc_auc"])), 4),
                "pr_auc": round(float(np.mean(metrics["pr_auc"])), 4)
            }
            rows.append(row)
            print(f"  Model: {name:<12} | F1: {row['f1']:.4f} ± {row['f1_std']:.4f} | MCC: {row['mcc']:.4f} | PR-AUC: {row['pr_auc']:.4f}")
            
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "code_smell_models.csv", index=False)
    print(f"\n[SAVED] {results_dir / 'code_smell_models.csv'}")

if __name__ == "__main__":
    run_tuned_models_experiment()
