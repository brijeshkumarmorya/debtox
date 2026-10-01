"""
DebtOx Empirical Research Pipeline - Section 4: Clean Baseline
Establishes and freezes the baseline experiment using 5-fold Project-level GroupKFold.
No data leakage: Feature preprocessing strictly inside training folds.
Default decision threshold = 0.50.
"""

import sys
import os
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

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

def get_classifiers(random_state: int = 42) -> Dict[str, Any]:
    return {
        "rf": RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_split=4,
            random_state=random_state, n_jobs=-1
        ),
        "xgb": xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.08,
            eval_metric="logloss", random_state=random_state, n_jobs=-1
        ),
        "lgb": lgb.LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.08,
            random_state=random_state, verbose=-1, n_jobs=-1
        ),
        "dt": DecisionTreeClassifier(
            max_depth=8, min_samples_split=5, random_state=random_state
        ),
        "lr": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "svm": SVC(
            probability=True, kernel="rbf", max_iter=2500, random_state=random_state
        )
    }

def run_baseline():
    loader = RealDatasetLoader()
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    baseline_csv = results_dir / "baseline_results.csv"
    
    if baseline_csv.exists():
        print(f"[INFO] Baseline results file already exists at {baseline_csv}. Reading existing baseline...")
        return pd.read_csv(baseline_csv)
    
    runs_dir = Path("experiments/runs/baseline")
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    smells = ["god_class", "data_class", "long_method", "feature_envy"]
    rows = []
    fold_details = []
    
    print("=" * 60)
    print("RUNNING LEAKAGE-SAFE PROJECT-LEVEL GROUPKFOLD BASELINE")
    print("=" * 60)
    
    for smell in smells:
        print(f"\n---> Evaluating Smell: {smell.upper()}")
        X, y, groups, features, prov = loader.load_smellycode_plus_plus(smell)
        print(f"Dataset: SmellyCode++ | Total samples: {len(X)} | Positives: {y.sum()} ({y.mean()*100:.2f}%) | Projects: {groups.nunique()}")
        
        gkf = GroupKFold(n_splits=5)
        splits = list(gkf.split(X, y, groups=groups))
        
        models = get_classifiers(random_state=42)
        
        for model_name, clf_proto in models.items():
            print(f"  Model: {model_name.upper():<5} ... ", end="", flush=True)
            fold_metrics = {
                "accuracy": [], "precision": [], "recall": [],
                "f1": [], "mcc": [], "roc_auc": [], "pr_auc": []
            }
            
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values
                
                # Fit preprocessor strictly on train fold
                prep = FeaturePreprocessor(scaler_type="robust")
                X_train_prep = prep.fit_transform(X_train)
                X_test_prep = prep.transform(X_test)
                
                # Resampling strictly on train fold (SMOTE default)
                X_res, y_res = ImbalanceHandler.apply_resampling(
                    X_train_prep, y_train, strategy="smote", random_state=42 + fold_idx
                )
                
                # Clone/init classifier
                clf = get_classifiers(random_state=42 + fold_idx)[model_name]
                clf.fit(X_res, y_res)
                
                # Predictions on untouched test fold with default threshold 0.50
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(X_test_prep)[:, 1]
                    preds = (probs >= 0.50).astype(int)
                else:
                    preds = clf.predict(X_test_prep)
                    probs = preds.astype(float)
                
                acc = accuracy_score(y_test, preds)
                prec = precision_score(y_test, preds, zero_division=0)
                rec = recall_score(y_test, preds, zero_division=0)
                f1 = f1_score(y_test, preds, zero_division=0)
                mcc = matthews_corrcoef(y_test, preds)
                
                fold_metrics["accuracy"].append(acc)
                fold_metrics["precision"].append(prec)
                fold_metrics["recall"].append(rec)
                fold_metrics["f1"].append(f1)
                fold_metrics["mcc"].append(mcc)
                
                if len(np.unique(y_test)) > 1:
                    roc_auc = roc_auc_score(y_test, probs)
                    pr_auc = average_precision_score(y_test, probs)
                    fold_metrics["roc_auc"].append(roc_auc)
                    fold_metrics["pr_auc"].append(pr_auc)
                else:
                    fold_metrics["roc_auc"].append(0.5)
                    fold_metrics["pr_auc"].append(0.0)
                
                cm = confusion_matrix(y_test, preds)
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                fold_details.append({
                    "smell": smell,
                    "model": model_name,
                    "fold": fold_idx + 1,
                    "test_projects": list(groups.iloc[test_idx].unique()),
                    "test_samples": len(test_idx),
                    "test_positives": int(y_test.sum()),
                    "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                    "precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(f1, 4), "mcc": round(mcc, 4)
                })
            
            row = {
                "dataset": "SmellyCode++",
                "smell": smell,
                "model": model_name,
                "split_strategy": "GroupKFold (5-fold, by Project)",
                "preprocessing": "RobustScaler (train-fold only)",
                "resampling": "SMOTE (train-fold only)",
                "threshold": 0.50,
                "accuracy": round(float(np.mean(fold_metrics["accuracy"])), 4),
                "accuracy_std": round(float(np.std(fold_metrics["accuracy"])), 4),
                "precision": round(float(np.mean(fold_metrics["precision"])), 4),
                "precision_std": round(float(np.std(fold_metrics["precision"])), 4),
                "recall": round(float(np.mean(fold_metrics["recall"])), 4),
                "recall_std": round(float(np.std(fold_metrics["recall"])), 4),
                "f1": round(float(np.mean(fold_metrics["f1"])), 4),
                "f1_std": round(float(np.std(fold_metrics["f1"])), 4),
                "mcc": round(float(np.mean(fold_metrics["mcc"])), 4),
                "mcc_std": round(float(np.std(fold_metrics["mcc"])), 4),
                "roc_auc": round(float(np.mean(fold_metrics["roc_auc"])), 4),
                "roc_auc_std": round(float(np.std(fold_metrics["roc_auc"])), 4),
                "pr_auc": round(float(np.mean(fold_metrics["pr_auc"])), 4),
                "pr_auc_std": round(float(np.std(fold_metrics["pr_auc"])), 4)
            }
            rows.append(row)
            print(f"F1: {row['f1']:.4f} ± {row['f1_std']:.4f} | MCC: {row['mcc']:.4f} | PR-AUC: {row['pr_auc']:.4f}")
    
    baseline_df = pd.DataFrame(rows)
    baseline_df.to_csv(baseline_csv, index=False)
    print(f"\n[SUCCESS] Baseline results frozen and saved to {baseline_csv}")
    
    # Save manifest for reproducibility (Section 20)
    manifest = {
        "experiment_id": "baseline_groupkfold_v1",
        "description": "Baseline Project-Level GroupKFold experiment with default threshold 0.50",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": "SmellyCode++",
        "dataset_hash": prov.checksum_sha256,
        "features": features,
        "models": list(models.keys()),
        "threshold": 0.50,
        "folds": 5,
        "split_strategy": "GroupKFold by Project",
        "fold_details": fold_details
    }
    with open(runs_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[SUCCESS] Run manifest saved to {runs_dir / 'run_manifest.json'}")
    return baseline_df

if __name__ == "__main__":
    run_baseline()
