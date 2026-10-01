"""
DebtOx Empirical Research Pipeline - Sections 5 & 6:
Section 5: Systematic Class Imbalance Evaluation (No balancing, class weights, RUS, SMOTE, Borderline-SMOTE, NearMiss)
Section 6: Inner-Validation Threshold Optimization (Optimizing F1 and MCC on validation only, evaluating on untouched test folds)
"""

import sys
import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.data.loader import RealDatasetLoader
from ml.pipelines.preprocessor import FeaturePreprocessor
from ml.pipelines.resampling import ImbalanceHandler

def get_base_classifier(model_name: str, use_class_weight: bool = False, random_state: int = 42):
    cw = "balanced" if use_class_weight else None
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_split=4,
            class_weight=cw, random_state=random_state, n_jobs=-1
        )
    elif model_name == "xgb":
        spw = 5.0 if use_class_weight else 1.0
        return xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.08,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=random_state, n_jobs=-1
        )
    elif model_name == "lgb":
        return lgb.LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.08,
            class_weight=cw, random_state=random_state, verbose=-1, n_jobs=-1
        )
    elif model_name == "lr":
        return LogisticRegression(max_iter=1000, class_weight=cw, random_state=random_state)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def optimize_threshold_inner(
    X_train_prep: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
    resampling_strategy: str,
    use_class_weight: bool,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Performs 3-fold inner cross-validation strictly on the training fold to find:
    1) Best threshold for F1
    2) Best threshold for MCC
    Returns (best_thresh_f1, best_thresh_mcc).
    """
    from sklearn.model_selection import StratifiedKFold
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    
    val_probs = []
    val_targets = []
    
    for in_train_idx, in_val_idx in inner_cv.split(X_train_prep, y_train):
        X_in_tr, X_in_val = X_train_prep[in_train_idx], X_train_prep[in_val_idx]
        y_in_tr, y_in_val = y_train[in_train_idx], y_train[in_val_idx]
        
        # Apply resampling strictly on inner train
        if not use_class_weight and resampling_strategy != "none":
            X_in_tr_res, y_in_tr_res = ImbalanceHandler.apply_resampling(
                X_in_tr, y_in_tr, strategy=resampling_strategy, random_state=random_state
            )
        else:
            X_in_tr_res, y_in_tr_res = X_in_tr, y_in_tr
            
        clf = get_base_classifier(model_name, use_class_weight=use_class_weight, random_state=random_state)
        clf.fit(X_in_tr_res, y_in_tr_res)
        
        if hasattr(clf, "predict_proba"):
            p = clf.predict_proba(X_in_val)[:, 1]
        else:
            p = clf.predict(X_in_val).astype(float)
            
        val_probs.extend(p)
        val_targets.extend(y_in_val)
        
    val_probs = np.array(val_probs)
    val_targets = np.array(val_targets)
    
    threshold_grid = np.linspace(0.05, 0.90, 35)
    best_f1 = -1.0
    best_thresh_f1 = 0.50
    best_mcc = -2.0
    best_thresh_mcc = 0.50
    
    for t in threshold_grid:
        preds = (val_probs >= t).astype(int)
        score_f1 = f1_score(val_targets, preds, zero_division=0)
        score_mcc = matthews_corrcoef(val_targets, preds)
        
        if score_f1 > best_f1:
            best_f1 = score_f1
            best_thresh_f1 = float(t)
        if score_mcc > best_mcc:
            best_mcc = score_mcc
            best_thresh_mcc = float(t)
            
    return round(best_thresh_f1, 3), round(best_thresh_mcc, 3)

def run_resampling_and_threshold_experiments():
    loader = RealDatasetLoader()
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    smells = ["god_class", "data_class", "long_method", "feature_envy"]
    strategies = [
        ("no_balancing", "none", False),
        ("class_weight", "none", True),
        ("random_undersampling", "random_under", False),
        ("smote", "smote", False),
        ("borderline_smote", "borderline_smote", False),
        ("nearmiss", "nearmiss", False)
    ]
    
    resampling_rows = []
    threshold_rows = []
    
    print("=" * 60)
    print("RUNNING IMBALANCE STRATEGIES & THRESHOLD OPTIMIZATION")
    print("=" * 60)
    
    for smell in smells:
        print(f"\n---> Smell: {smell.upper()}")
        X, y, groups, features, prov = loader.load_smellycode_plus_plus(smell)
        gkf = GroupKFold(n_splits=5)
        splits = list(gkf.split(X, y, groups=groups))
        
        # Test primarily with RF and XGBoost
        test_models = ["rf", "xgb"]
        
        for model_name in test_models:
            for strat_label, strat_val, use_cw in strategies:
                metrics_default = {"precision": [], "recall": [], "f1": [], "mcc": [], "pr_auc": [], "roc_auc": []}
                metrics_f1_opt = {"threshold": [], "precision": [], "recall": [], "f1": [], "mcc": []}
                metrics_mcc_opt = {"threshold": [], "precision": [], "recall": [], "f1": [], "mcc": []}
                
                for fold_idx, (train_idx, test_idx) in enumerate(splits):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values
                    
                    # Preprocess strictly on train fold
                    prep = FeaturePreprocessor(scaler_type="robust")
                    X_tr_prep = prep.fit_transform(X_train)
                    X_te_prep = prep.transform(X_test)
                    
                    # Resample strictly on train fold
                    if not use_cw and strat_val != "none":
                        X_res, y_res = ImbalanceHandler.apply_resampling(
                            X_tr_prep, y_train, strategy=strat_val, random_state=42 + fold_idx
                        )
                    else:
                        X_res, y_res = X_tr_prep, y_train
                        
                    clf = get_base_classifier(model_name, use_class_weight=use_cw, random_state=42 + fold_idx)
                    clf.fit(X_res, y_res)
                    
                    test_probs = clf.predict_proba(X_te_prep)[:, 1]
                    
                    # 1. Default threshold (0.50)
                    preds_def = (test_probs >= 0.50).astype(int)
                    metrics_default["precision"].append(precision_score(y_test, preds_def, zero_division=0))
                    metrics_default["recall"].append(recall_score(y_test, preds_def, zero_division=0))
                    metrics_default["f1"].append(f1_score(y_test, preds_def, zero_division=0))
                    metrics_default["mcc"].append(matthews_corrcoef(y_test, preds_def))
                    if len(np.unique(y_test)) > 1:
                        metrics_default["pr_auc"].append(average_precision_score(y_test, test_probs))
                        metrics_default["roc_auc"].append(roc_auc_score(y_test, test_probs))
                    else:
                        metrics_default["pr_auc"].append(0.0)
                        metrics_default["roc_auc"].append(0.5)
                        
                    # 2. Inner-validation threshold search (optimizing on train fold only)
                    opt_t_f1, opt_t_mcc = optimize_threshold_inner(
                        X_tr_prep, y_train, model_name=model_name,
                        resampling_strategy=strat_val, use_class_weight=use_cw,
                        random_state=42 + fold_idx
                    )
                    
                    # Evaluate tuned threshold on untouched test fold
                    preds_f1_opt = (test_probs >= opt_t_f1).astype(int)
                    metrics_f1_opt["threshold"].append(opt_t_f1)
                    metrics_f1_opt["precision"].append(precision_score(y_test, preds_f1_opt, zero_division=0))
                    metrics_f1_opt["recall"].append(recall_score(y_test, preds_f1_opt, zero_division=0))
                    metrics_f1_opt["f1"].append(f1_score(y_test, preds_f1_opt, zero_division=0))
                    metrics_f1_opt["mcc"].append(matthews_corrcoef(y_test, preds_f1_opt))
                    
                    preds_mcc_opt = (test_probs >= opt_t_mcc).astype(int)
                    metrics_mcc_opt["threshold"].append(opt_t_mcc)
                    metrics_mcc_opt["precision"].append(precision_score(y_test, preds_mcc_opt, zero_division=0))
                    metrics_mcc_opt["recall"].append(recall_score(y_test, preds_mcc_opt, zero_division=0))
                    metrics_mcc_opt["f1"].append(f1_score(y_test, preds_mcc_opt, zero_division=0))
                    metrics_mcc_opt["mcc"].append(matthews_corrcoef(y_test, preds_mcc_opt))

                # Section 5 row
                resampling_rows.append({
                    "smell": smell,
                    "model": model_name,
                    "strategy": strat_label,
                    "precision": round(float(np.mean(metrics_default["precision"])), 4),
                    "precision_std": round(float(np.std(metrics_default["precision"])), 4),
                    "recall": round(float(np.mean(metrics_default["recall"])), 4),
                    "recall_std": round(float(np.std(metrics_default["recall"])), 4),
                    "f1": round(float(np.mean(metrics_default["f1"])), 4),
                    "f1_std": round(float(np.std(metrics_default["f1"])), 4),
                    "mcc": round(float(np.mean(metrics_default["mcc"])), 4),
                    "mcc_std": round(float(np.std(metrics_default["mcc"])), 4),
                    "pr_auc": round(float(np.mean(metrics_default["pr_auc"])), 4),
                    "roc_auc": round(float(np.mean(metrics_default["roc_auc"])), 4)
                })
                
                # Section 6 row (threshold comparison)
                threshold_rows.append({
                    "smell": smell,
                    "model": model_name,
                    "imbalance_handling": strat_label,
                    "default_threshold": 0.50,
                    "default_precision": round(float(np.mean(metrics_default["precision"])), 4),
                    "default_recall": round(float(np.mean(metrics_default["recall"])), 4),
                    "default_f1": round(float(np.mean(metrics_default["f1"])), 4),
                    "default_mcc": round(float(np.mean(metrics_default["mcc"])), 4),
                    "val_opt_f1_threshold": round(float(np.mean(metrics_f1_opt["threshold"])), 3),
                    "val_opt_f1_precision": round(float(np.mean(metrics_f1_opt["precision"])), 4),
                    "val_opt_f1_recall": round(float(np.mean(metrics_f1_opt["recall"])), 4),
                    "val_opt_f1_score": round(float(np.mean(metrics_f1_opt["f1"])), 4),
                    "val_opt_mcc_threshold": round(float(np.mean(metrics_mcc_opt["threshold"])), 3),
                    "val_opt_mcc_score": round(float(np.mean(metrics_mcc_opt["mcc"])), 4)
                })
                print(f"  {model_name.upper()} | {strat_label:<20} | Def F1: {resampling_rows[-1]['f1']:.4f} | Opt F1: {threshold_rows[-1]['val_opt_f1_score']:.4f} (t={threshold_rows[-1]['val_opt_f1_threshold']:.2f})")
                
    res_df = pd.DataFrame(resampling_rows)
    res_df.to_csv(results_dir / "resampling_comparison.csv", index=False)
    print(f"\n[SAVED] {results_dir / 'resampling_comparison.csv'}")
    
    thresh_df = pd.DataFrame(threshold_rows)
    thresh_df.to_csv(results_dir / "threshold_optimization.csv", index=False)
    print(f"[SAVED] {results_dir / 'threshold_optimization.csv'}")
    
    # Save manifest
    runs_dir = Path("experiments/runs/imbalance_and_threshold")
    runs_dir.mkdir(parents=True, exist_ok=True)
    with open(runs_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "experiment_id": "imbalance_and_threshold_v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategies_evaluated": [s[0] for s in strategies],
            "models_evaluated": test_models,
            "threshold_selection": "Inner StratifiedKFold validation only, untouched outer test evaluation"
        }, f, indent=2)

if __name__ == "__main__":
    run_resampling_and_threshold_experiments()
