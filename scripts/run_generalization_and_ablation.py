"""
DebtOx Empirical Research Pipeline - Sections 9, 10, 12, 15:
Section 9: Within-Project vs Cross-Project Generalization (StratifiedKFold vs GroupKFold)
Section 10: External Cross-Dataset Validation (SmellyCode++ <-> Crowdsmelling using compatible features)
Section 12: Feature Group Ablation Study (Complexity+Size vs Halstead vs All)
Section 15: Probability Calibration & Brier Score Evaluation
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score, brier_score_loss
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.data.loader import RealDatasetLoader
from ml.pipelines.preprocessor import FeaturePreprocessor
from ml.pipelines.resampling import ImbalanceHandler

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (np.sum(mask) / n) * np.abs(bin_acc - bin_conf)
    return round(float(ece), 4)

def run_section_9_within_vs_cross():
    """
    Compares Within-Project (StratifiedKFold) vs Cross-Project (GroupKFold).
    Reports mean, std, median, 95% CI.
    """
    loader = RealDatasetLoader()
    results_dir = Path("experiments/results")
    if (results_dir / "within_vs_cross_project.csv").exists():
        print("\n[INFO] within_vs_cross_project.csv already exists. Skipping Section 9 recalculation...")
        return
    smells = ["god_class", "data_class", "long_method", "feature_envy"]
    rows = []
    
    print("\n" + "=" * 60)
    print("SECTION 9: WITHIN-PROJECT VS CROSS-PROJECT GENERALIZATION")
    print("=" * 60)
    
    for smell in smells:
        X, y, groups, features, prov = loader.load_smellycode_plus_plus(smell)
        
        # 1. Cross-Project: GroupKFold
        gkf = GroupKFold(n_splits=5)
        cross_f1, cross_mcc, cross_pr_auc = [], [], []
        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            prep = FeaturePreprocessor(scaler_type="robust")
            X_tr = prep.fit_transform(X.iloc[train_idx])
            X_te = prep.transform(X.iloc[test_idx])
            y_tr, y_te = y.iloc[train_idx].values, y.iloc[test_idx].values
            
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
            probs = clf.predict_proba(X_te)[:, 1]
            
            cross_f1.append(f1_score(y_te, preds, zero_division=0))
            cross_mcc.append(matthews_corrcoef(y_te, preds))
            if len(np.unique(y_te)) > 1:
                cross_pr_auc.append(average_precision_score(y_te, probs))
            else:
                cross_pr_auc.append(0.0)
                
        # 2. Within-Project: StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        within_f1, within_mcc, within_pr_auc = [], [], []
        for train_idx, test_idx in skf.split(X, y):
            prep = FeaturePreprocessor(scaler_type="robust")
            X_tr = prep.fit_transform(X.iloc[train_idx])
            X_te = prep.transform(X.iloc[test_idx])
            y_tr, y_te = y.iloc[train_idx].values, y.iloc[test_idx].values
            
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
            probs = clf.predict_proba(X_te)[:, 1]
            
            within_f1.append(f1_score(y_te, preds, zero_division=0))
            within_mcc.append(matthews_corrcoef(y_te, preds))
            if len(np.unique(y_te)) > 1:
                within_pr_auc.append(average_precision_score(y_te, probs))
            else:
                within_pr_auc.append(0.0)
                
        rows.append({
            "smell": smell,
            "within_f1_mean": round(float(np.mean(within_f1)), 4),
            "within_f1_std": round(float(np.std(within_f1)), 4),
            "within_f1_median": round(float(np.median(within_f1)), 4),
            "within_f1_ci95_low": round(float(np.mean(within_f1) - 1.96 * np.std(within_f1)/np.sqrt(5)), 4),
            "within_f1_ci95_high": round(float(np.mean(within_f1) + 1.96 * np.std(within_f1)/np.sqrt(5)), 4),
            "within_mcc_mean": round(float(np.mean(within_mcc)), 4),
            "within_prauc_mean": round(float(np.mean(within_pr_auc)), 4),
            "cross_f1_mean": round(float(np.mean(cross_f1)), 4),
            "cross_f1_std": round(float(np.std(cross_f1)), 4),
            "cross_f1_median": round(float(np.median(cross_f1)), 4),
            "cross_f1_ci95_low": round(float(np.mean(cross_f1) - 1.96 * np.std(cross_f1)/np.sqrt(5)), 4),
            "cross_f1_ci95_high": round(float(np.mean(cross_f1) + 1.96 * np.std(cross_f1)/np.sqrt(5)), 4),
            "cross_mcc_mean": round(float(np.mean(cross_mcc)), 4),
            "cross_prauc_mean": round(float(np.mean(cross_pr_auc)), 4),
            "generalization_gap_f1": round(float(np.mean(within_f1) - np.mean(cross_f1)), 4)
        })
        print(f"Smell: {smell:<15} | Within F1: {rows[-1]['within_f1_mean']:.4f} | Cross F1: {rows[-1]['cross_f1_mean']:.4f} | Gap: {rows[-1]['generalization_gap_f1']:.4f}")
        
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "within_vs_cross_project.csv", index=False)
    print(f"[SAVED] {results_dir / 'within_vs_cross_project.csv'}")

def run_section_10_cross_dataset():
    """
    Section 10: External Cross-Dataset Validation
    Train: SmellyCode++ -> Test: Crowdsmelling (compatible features: SLOC, cyclomatic_complexity)
    Train: Crowdsmelling -> Test: SmellyCode++
    """
    loader = RealDatasetLoader()
    results_dir = Path("experiments/results")
    smells = ["god_class", "long_method", "feature_envy"]  # Smells in Crowdsmelling
    shared_features = ["SLOC", "cyclomatic_complexity"]
    rows = []
    
    print("\n" + "=" * 60)
    print("SECTION 10: EXTERNAL CROSS-DATASET VALIDATION")
    print("=" * 60)
    
    for smell in smells:
        # Load SmellyCode++
        X_sc, y_sc, _, _, _ = loader.load_smellycode_plus_plus(smell)
        # Load Crowdsmelling
        X_cs, y_cs, _, _, _ = loader.load_crowdsmelling(smell)
        
        # Filter to shared compatible features
        if smell == "god_class" and "WMC" in X_cs.columns:
            X_cs = X_cs.rename(columns={"WMC": "cyclomatic_complexity"})
        X_sc_sub = X_sc[shared_features]
        X_cs_sub = X_cs[shared_features]
        
        # Direction 1: Train SmellyCode++ -> Test Crowdsmelling
        prep1 = FeaturePreprocessor(scaler_type="robust")
        X_sc_prep = prep1.fit_transform(X_sc_sub)
        X_cs_prep = prep1.transform(X_cs_sub)
        
        clf1 = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1)
        clf1.fit(X_sc_prep, y_sc.values)
        preds1 = clf1.predict(X_cs_prep)
        probs1 = clf1.predict_proba(X_cs_prep)[:, 1]
        
        row1 = {
            "smell": smell,
            "train_dataset": "SmellyCode++",
            "test_dataset": "Crowdsmelling (Human GT)",
            "shared_features": "SLOC, cyclomatic_complexity",
            "train_samples": len(X_sc),
            "test_samples": len(X_cs),
            "test_prevalence": round(float(y_cs.mean()), 4),
            "precision": round(float(precision_score(y_cs, preds1, zero_division=0)), 4),
            "recall": round(float(recall_score(y_cs, preds1, zero_division=0)), 4),
            "f1": round(float(f1_score(y_cs, preds1, zero_division=0)), 4),
            "mcc": round(float(matthews_corrcoef(y_cs, preds1)), 4),
            "roc_auc": round(float(roc_auc_score(y_cs, probs1)), 4),
            "pr_auc": round(float(average_precision_score(y_cs, probs1)), 4)
        }
        rows.append(row1)
        print(f"Dir 1: SmellyCode++ -> Crowdsmelling | {smell:<15} | F1: {row1['f1']:.4f} | MCC: {row1['mcc']:.4f} | PR-AUC: {row1['pr_auc']:.4f}")
        
        # Direction 2: Train Crowdsmelling -> Test SmellyCode++
        prep2 = FeaturePreprocessor(scaler_type="robust")
        X_cs_prep2 = prep2.fit_transform(X_cs_sub)
        X_sc_prep2 = prep2.transform(X_sc_sub)
        
        clf2 = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1)
        clf2.fit(X_cs_prep2, y_cs.values)
        preds2 = clf2.predict(X_sc_prep2)
        probs2 = clf2.predict_proba(X_sc_prep2)[:, 1]
        
        row2 = {
            "smell": smell,
            "train_dataset": "Crowdsmelling (Human GT)",
            "test_dataset": "SmellyCode++",
            "shared_features": "SLOC, cyclomatic_complexity",
            "train_samples": len(X_cs),
            "test_samples": len(X_sc),
            "test_prevalence": round(float(y_sc.mean()), 4),
            "precision": round(float(precision_score(y_sc, preds2, zero_division=0)), 4),
            "recall": round(float(recall_score(y_sc, preds2, zero_division=0)), 4),
            "f1": round(float(f1_score(y_sc, preds2, zero_division=0)), 4),
            "mcc": round(float(matthews_corrcoef(y_sc, preds2)), 4),
            "roc_auc": round(float(roc_auc_score(y_sc, probs2)), 4),
            "pr_auc": round(float(average_precision_score(y_sc, probs2)), 4)
        }
        rows.append(row2)
        print(f"Dir 2: Crowdsmelling -> SmellyCode++ | {smell:<15} | F1: {row2['f1']:.4f} | MCC: {row2['mcc']:.4f} | PR-AUC: {row2['pr_auc']:.4f}")
        
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "cross_dataset_results.csv", index=False)
    print(f"[SAVED] {results_dir / 'cross_dataset_results.csv'}")

def run_section_12_ablation():
    """
    Section 12: Ablation Study across Feature Groups:
    A. Complexity + size only (SLOC, cyclomatic_complexity)
    B. Halstead only (12 Halstead features)
    C. All available features (14 features)
    """
    loader = RealDatasetLoader()
    results_dir = Path("experiments/results")
    smells = ["god_class", "data_class", "long_method", "feature_envy"]
    rows = []
    
    print("\n" + "=" * 60)
    print("SECTION 12: FEATURE GROUP ABLATION STUDY")
    print("=" * 60)
    
    for smell in smells:
        X, y, groups, features, prov = loader.load_smellycode_plus_plus(smell)
        gkf = GroupKFold(n_splits=5)
        splits = list(gkf.split(X, y, groups=groups))
        
        halstead_cols = [c for c in features if c not in ["SLOC", "cyclomatic_complexity"]]
        feature_sets = [
            ("complexity_size_only", ["SLOC", "cyclomatic_complexity"]),
            ("halstead_only", halstead_cols),
            ("all_features", features)
        ]
        
        for set_name, cols in feature_sets:
            metrics = {"f1": [], "mcc": [], "precision": [], "recall": [], "pr_auc": []}
            
            for train_idx, test_idx in splits:
                X_train, X_test = X[cols].iloc[train_idx], X[cols].iloc[test_idx]
                y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values
                
                prep = FeaturePreprocessor(scaler_type="robust")
                X_tr = prep.fit_transform(X_train)
                X_te = prep.transform(X_test)
                
                clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
                clf.fit(X_tr, y_train)
                preds = clf.predict(X_te)
                probs = clf.predict_proba(X_te)[:, 1]
                
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
                "feature_set": set_name,
                "feature_count": len(cols),
                "precision": round(float(np.mean(metrics["precision"])), 4),
                "recall": round(float(np.mean(metrics["recall"])), 4),
                "f1": round(float(np.mean(metrics["f1"])), 4),
                "f1_std": round(float(np.std(metrics["f1"])), 4),
                "mcc": round(float(np.mean(metrics["mcc"])), 4),
                "mcc_std": round(float(np.std(metrics["mcc"])), 4),
                "pr_auc": round(float(np.mean(metrics["pr_auc"])), 4)
            }
            rows.append(row)
            print(f"Smell: {smell:<15} | Set: {set_name:<20} | F1: {row['f1']:.4f} | MCC: {row['mcc']:.4f} | PR-AUC: {row['pr_auc']:.4f}")
            
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "ablation_results.csv", index=False)
    print(f"[SAVED] {results_dir / 'ablation_results.csv'}")

def run_section_15_calibration():
    """
    Section 15: Probability Calibration & Brier Score Evaluation
    Evaluates uncalibrated vs Platt-scaled (sigmoid) calibration using GroupKFold.
    """
    loader = RealDatasetLoader()
    results_dir = Path("experiments/results")
    smells = ["god_class", "data_class", "long_method", "feature_envy"]
    rows = []
    
    print("\n" + "=" * 60)
    print("SECTION 15: PROBABILITY CALIBRATION & BRIER SCORE")
    print("=" * 60)
    
    for smell in smells:
        X, y, groups, features, prov = loader.load_smellycode_plus_plus(smell)
        gkf = GroupKFold(n_splits=5)
        splits = list(gkf.split(X, y, groups=groups))
        
        uncal_brier, cal_brier = [], []
        uncal_ece, cal_ece = [], []
        uncal_prauc, cal_prauc = [], []
        
        for train_idx, test_idx in splits:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values
            
            prep = FeaturePreprocessor(scaler_type="robust")
            X_tr = prep.fit_transform(X_train)
            X_te = prep.transform(X_test)
            
            # Base uncalibrated model
            base_clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
            base_clf.fit(X_tr, y_train)
            uncal_prob = base_clf.predict_proba(X_te)[:, 1]
            
            # Calibrated model (Platt scaling with internal 3-fold CV on train fold)
            cal_clf = CalibratedClassifierCV(
                estimator=RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1),
                method="sigmoid", cv=3
            )
            cal_clf.fit(X_tr, y_train)
            cal_prob = cal_clf.predict_proba(X_te)[:, 1]
            
            uncal_brier.append(brier_score_loss(y_test, uncal_prob))
            cal_brier.append(brier_score_loss(y_test, cal_prob))
            uncal_ece.append(expected_calibration_error(y_test, uncal_prob))
            cal_ece.append(expected_calibration_error(y_test, cal_prob))
            
            if len(np.unique(y_test)) > 1:
                uncal_prauc.append(average_precision_score(y_test, uncal_prob))
                cal_prauc.append(average_precision_score(y_test, cal_prob))
            else:
                uncal_prauc.append(0.0)
                cal_prauc.append(0.0)
                
        row = {
            "smell": smell,
            "uncalibrated_brier": round(float(np.mean(uncal_brier)), 4),
            "calibrated_brier": round(float(np.mean(cal_brier)), 4),
            "brier_improvement": round(float(np.mean(uncal_brier) - np.mean(cal_brier)), 4),
            "uncalibrated_ece": round(float(np.mean(uncal_ece)), 4),
            "calibrated_ece": round(float(np.mean(cal_ece)), 4),
            "uncalibrated_pr_auc": round(float(np.mean(uncal_prauc)), 4),
            "calibrated_pr_auc": round(float(np.mean(cal_prauc)), 4)
        }
        rows.append(row)
        print(f"Smell: {smell:<15} | Brier: {row['uncalibrated_brier']} -> {row['calibrated_brier']} | ECE: {row['uncalibrated_ece']} -> {row['calibrated_ece']}")
        
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "calibration_results.csv", index=False)
    print(f"[SAVED] {results_dir / 'calibration_results.csv'}")

if __name__ == "__main__":
    run_section_9_within_vs_cross()
    run_section_10_cross_dataset()
    run_section_12_ablation()
    run_section_15_calibration()
