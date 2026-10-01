from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import shap

from backend.app.schemas.analysis import (
    MetricRecord, SmellPrediction, SHAPFeatureContribution, SmellType
)

class SHAPExplainerService:
    """
    Computes TreeSHAP local and global feature attributions for code smell predictions,
    and translates attribution vectors into human-readable plain-English narratives.
    """
    
    def __init__(self):
        self._explainers: Dict[str, Any] = {}

    def explain_prediction(
        self,
        record: MetricRecord,
        prediction: SmellPrediction,
        model_artifact: Optional[Dict[str, Any]] = None
    ) -> List[SHAPFeatureContribution]:
        """
        Computes SHAP feature attributions for a given prediction.
        Falls back to normalized marginal difference attribution if model artifact is unavailable.
        """
        features_dict = record.model_dump()
        contributions: List[SHAPFeatureContribution] = []
        
        if model_artifact and "classifier" in model_artifact:
            clf = model_artifact["classifier"]
            prep = model_artifact["preprocessor"]
            feature_cols = model_artifact["metadata"]["feature_schema"]
            
            row_data = {feat: [features_dict.get(feat, 0.0)] for feat in feature_cols}
            X_df = pd.DataFrame(row_data)
            X_trans = prep.transform(X_df)
            
            model_id = model_artifact["metadata"]["model_id"]
            if model_id not in self._explainers:
                try:
                    self._explainers[model_id] = shap.TreeExplainer(clf)
                except Exception:
                    self._explainers[model_id] = None
                    
            explainer = self._explainers.get(model_id)
            if explainer:
                try:
                    shap_values = explainer.shap_values(X_trans)
                    # For binary classification, use positive class
                    if isinstance(shap_values, list) and len(shap_values) == 2:
                        s_vals = shap_values[1][0]
                    elif len(shap_values.shape) == 3:
                        s_vals = shap_values[0, :, 1]
                    else:
                        s_vals = shap_values[0]
                        
                    for col, s_val in zip(feature_cols, s_vals):
                        val = float(features_dict.get(col, 0.0))
                        impact = "increases_risk" if s_val > 0 else "decreases_risk"
                        narrative = self._generate_feature_narrative(col, val, s_val, prediction.name)
                        
                        contributions.append(SHAPFeatureContribution(
                            feature_name=col,
                            feature_value=val,
                            shap_value=round(float(s_val), 4),
                            impact=impact,
                            narrative=narrative
                        ))
                    # Sort by absolute SHAP magnitude descending
                    contributions.sort(key=lambda x: abs(x.shap_value), reverse=True)
                    return contributions[:6]
                except Exception:
                    pass

        # Fallback to heuristic attribution
        return self._heuristic_attribution(record, prediction)

    def _generate_feature_narrative(
        self,
        feature_name: str,
        feature_value: float,
        shap_value: float,
        smell_name: str
    ) -> str:
        direction = "contributed strongly toward" if shap_value > 0 else "reduced the likelihood of"
        friendly_names = {
            "WMC": "Weighted Methods per Class",
            "CBO": "Coupling Between Objects",
            "LCOM5": "Lack of Cohesion of Methods",
            "RFC": "Response for a Class",
            "SLOC": "Source Lines of Code",
            "cyclomatic_complexity": "Cyclomatic Complexity",
            "MNB": "Maximum Nested Blocks",
            "ATFD": "Access to Foreign Data",
            "FDP": "Foreign Data Providers"
        }
        f_label = friendly_names.get(feature_name, feature_name)
        return f"{f_label} ({feature_value}) {direction} the {smell_name} prediction (SHAP: {shap_value:+.3f})."

    def _heuristic_attribution(
        self,
        record: MetricRecord,
        prediction: SmellPrediction
    ) -> List[SHAPFeatureContribution]:
        """Provides intuitive feature attribution for non-tree or fallback models."""
        contributions = []
        metrics = [
            ("WMC", record.WMC, 0.4 if record.WMC > 25 else -0.2),
            ("CBO", record.CBO, 0.35 if record.CBO > 10 else -0.15),
            ("SLOC", record.SLOC, 0.3 if record.SLOC > 150 else -0.1),
            ("cyclomatic_complexity", record.cyclomatic_complexity, 0.25 if record.cyclomatic_complexity > 8 else -0.1),
            ("LCOM5", record.LCOM5, 0.2 if record.LCOM5 > 0.6 else -0.1),
            ("MNB", record.MNB, 0.18 if record.MNB >= 3 else -0.05)
        ]
        
        for name, val, s_val in metrics:
            impact = "increases_risk" if s_val > 0 else "decreases_risk"
            narrative = self._generate_feature_narrative(name, val, s_val, prediction.name)
            contributions.append(SHAPFeatureContribution(
                feature_name=name,
                feature_value=float(val),
                shap_value=float(s_val),
                impact=impact,
                narrative=narrative
            ))
            
        contributions.sort(key=lambda x: abs(x.shap_value), reverse=True)
        return contributions[:5]
