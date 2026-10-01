import os
import json
import joblib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb

from ml.pipelines.preprocessor import FeaturePreprocessor
from ml.pipelines.resampling import ImbalanceHandler
from backend.app.core.config import settings

class ModelTrainer:
    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or settings.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _get_classifier(self, model_type: str, random_state: int = 42):
        model_type = model_type.lower()
        if model_type == "rf":
            return RandomForestClassifier(
                n_estimators=100, max_depth=12, min_samples_split=4,
                random_state=random_state, n_jobs=-1
            )
        elif model_type == "xgb":
            return xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.08,
                eval_metric="logloss", random_state=random_state, n_jobs=-1
            )
        elif model_type == "lgb":
            return lgb.LGBMClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.08,
                random_state=random_state, verbose=-1, n_jobs=-1
            )
        elif model_type == "dt":
            return DecisionTreeClassifier(max_depth=8, min_samples_split=5, random_state=random_state)
        elif model_type == "lr":
            return LogisticRegression(max_iter=1000, random_state=random_state)
        elif model_type == "svm":
            return SVC(probability=True, kernel="rbf", max_iter=2500, random_state=random_state)
        elif model_type == "stacking":
            estimators = [
                ('rf', RandomForestClassifier(n_estimators=50, max_depth=6, random_state=random_state)),
                ('xgb', xgb.XGBClassifier(n_estimators=50, max_depth=4, eval_metric="logloss", random_state=random_state))
            ]
            return StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(),
                cv=3
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def train_smell_model(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        smell_name: str,
        granularity: str,
        model_type: str = "rf",
        resampling_strategy: str = "smote",
        random_state: int = 42,
        cv_folds: int = 5,
        groups: Optional[pd.Series] = None,
        dataset_name: str = "smellycode_plus_plus"
    ) -> Dict[str, Any]:
        """
        Trains and validates a code smell prediction model using cross-validation.
        Applies feature preprocessing and resampling strictly on training folds.
        Supports project-level GroupKFold to guarantee zero data leakage across projects.
        Saves the final trained model artifact with complete metadata and provenance.
        """
        X = df[feature_cols]
        y = df[target_col].values
        
        if groups is not None and groups.nunique() >= cv_folds:
            from sklearn.model_selection import GroupKFold
            splitter = GroupKFold(n_splits=cv_folds)
            splits = list(splitter.split(X, y, groups=groups))
        else:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            splits = list(skf.split(X, y))
        
        cv_metrics = {
            "accuracy": [], "precision": [], "recall": [],
            "f1": [], "mcc": [], "roc_auc": [], "pr_auc": []
        }
        cms = []
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_train_df, X_val_df = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Preprocess on train fold only
            prep = FeaturePreprocessor(scaler_type="robust")
            X_train_prep = prep.fit_transform(X_train_df)
            X_val_prep = prep.transform(X_val_df)
            
            # Resample on train fold only
            X_res, y_res = ImbalanceHandler.apply_resampling(
                X_train_prep, y_train, strategy=resampling_strategy, random_state=random_state
            )
            
            clf = self._get_classifier(model_type, random_state=random_state + fold)
            clf.fit(X_res, y_res)
            
            preds = clf.predict(X_val_prep)
            probs = clf.predict_proba(X_val_prep)[:, 1] if hasattr(clf, "predict_proba") else preds
            
            cv_metrics["accuracy"].append(accuracy_score(y_val, preds))
            cv_metrics["precision"].append(precision_score(y_val, preds, zero_division=0))
            cv_metrics["recall"].append(recall_score(y_val, preds, zero_division=0))
            cv_metrics["f1"].append(f1_score(y_val, preds, zero_division=0))
            cv_metrics["mcc"].append(matthews_corrcoef(y_val, preds))
            
            if len(np.unique(y_val)) > 1:
                cv_metrics["roc_auc"].append(roc_auc_score(y_val, probs))
                cv_metrics["pr_auc"].append(average_precision_score(y_val, probs))
            cms.append(confusion_matrix(y_val, preds).tolist())

        # Fit final production model on full dataset with final preprocessor
        final_prep = FeaturePreprocessor(scaler_type="robust")
        X_full_prep = final_prep.fit_transform(X)
        X_full_res, y_full_res = ImbalanceHandler.apply_resampling(
            X_full_prep, y, strategy=resampling_strategy, random_state=random_state
        )
        
        final_clf = self._get_classifier(model_type, random_state=random_state)
        final_clf.fit(X_full_res, y_full_res)
        
        eval_summary = {
            metric: round(float(np.mean(vals)), 4) for metric, vals in cv_metrics.items() if vals
        }
        
        model_id = f"{smell_name.lower().replace(' ', '_')}_{model_type}"
        artifact_path = self.models_dir / f"{model_id}.joblib"
        
        metadata = {
            "model_id": model_id,
            "smell_name": smell_name,
            "granularity": granularity,
            "model_type": model_type,
            "dataset_provenance": dataset_name,
            "resampling_strategy": resampling_strategy,
            "feature_schema": feature_cols,
            "evaluation_metrics": eval_summary,
            "cv_folds": cv_folds,
            "training_timestamp": datetime.now(timezone.utc).isoformat(),
            "random_seed": random_state,
            "confusion_matrices": cms
        }
        
        artifact = {
            "classifier": final_clf,
            "preprocessor": final_prep,
            "metadata": metadata
        }
        
        joblib.dump(artifact, artifact_path)
        
        # Save separate metadata JSON for easy discovery
        meta_path = self.models_dir / f"{model_id}_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            
        return metadata
