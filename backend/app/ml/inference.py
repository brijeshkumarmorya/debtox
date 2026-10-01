import json
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from backend.app.core.config import settings
from backend.app.schemas.analysis import (
    MetricRecord, SmellPrediction, EntityGranularity, SmellType
)

class SmellInferenceEngine:
    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or settings.MODELS_DIR
        self._loaded_models: Dict[str, Any] = {}
        self._load_available_models()

    def _load_available_models(self):
        """Loads all .joblib model artifacts from the models directory."""
        if not self.models_dir.exists():
            return
        for file in self.models_dir.glob("*.joblib"):
            try:
                artifact = joblib.load(file)
                model_id = artifact["metadata"]["model_id"]
                self._loaded_models[model_id] = artifact
            except Exception:
                continue

    def get_loaded_models_metadata(self) -> List[Dict[str, Any]]:
        return [m["metadata"] for m in self._loaded_models.values()]

    def predict_component_smells(self, record: MetricRecord) -> List[SmellPrediction]:
        """
        Runs ML prediction for all supported code smells relevant to the component's granularity.
        """
        predictions: List[SmellPrediction] = []
        
        if record.granularity == EntityGranularity.CLASS:
            target_smells = [SmellType.GOD_CLASS, SmellType.DATA_CLASS, SmellType.BRAIN_CLASS]
        else:
            target_smells = [SmellType.LONG_METHOD, SmellType.FEATURE_ENVY, SmellType.BRAIN_METHOD]
            
        for smell in target_smells:
            pred = self._predict_single_smell(record, smell)
            predictions.append(pred)
            
        return predictions

    def _predict_single_smell(self, record: MetricRecord, smell: SmellType) -> SmellPrediction:
        smell_key = smell.value.lower().replace(' ', '_')
        # Look for model artifact (e.g., god_class_rf or god_class_xgb)
        matched_model = None
        for m_id, artifact in self._loaded_models.items():
            if m_id.startswith(smell_key):
                matched_model = artifact
                break
                
        metrics_dict = record.model_dump()
        
        if matched_model:
            clf = matched_model["classifier"]
            prep = matched_model["preprocessor"]
            features = matched_model["metadata"]["feature_schema"]
            
            # Construct 1-row DataFrame
            row_data = {feat: [metrics_dict.get(feat, 0.0)] for feat in features}
            X_df = pd.DataFrame(row_data)
            
            try:
                X_trans = prep.transform(X_df)
                if hasattr(clf, "predict_proba"):
                    prob = float(clf.predict_proba(X_trans)[0][1])
                else:
                    prob = float(clf.predict(X_trans)[0])
                is_smelly = prob >= 0.50
            except Exception:
                prob, is_smelly = self._heuristic_fallback(record, smell)
        else:
            # Deterministic heuristic fallback when model has not yet been trained
            prob, is_smelly = self._heuristic_fallback(record, smell)
            
        severity = "low"
        if prob >= 0.85:
            severity = "critical"
        elif prob >= 0.70:
            severity = "high"
        elif prob >= 0.50:
            severity = "medium"

        explanation = self._build_explanation(record, smell, prob, is_smelly)
        recommendation = self._build_recommendation(smell)

        return SmellPrediction(
            smell_id=smell_key,
            name=smell.value,
            granularity=record.granularity,
            is_smelly=is_smelly,
            probability=round(prob, 4),
            threshold=0.50,
            severity=severity,
            explanation=explanation,
            refactoring_recommendation=recommendation
        )

    def _heuristic_fallback(self, record: MetricRecord, smell: SmellType) -> (float, bool):
        """Rule-backed deterministic heuristic baseline."""
        if smell == SmellType.GOD_CLASS:
            is_smelly = (record.WMC > 35 and record.CBO > 12 and record.SLOC > 250)
            score = (record.WMC / 40.0 + record.CBO / 15.0 + record.SLOC / 300.0) / 3.0
            return float(min(0.99, max(0.05, score))), is_smelly
            
        elif smell == SmellType.DATA_CLASS:
            is_smelly = (record.WMC < 12 and record.LCOM5 > 0.70 and record.CBO < 6)
            score = (record.LCOM5 + (12 - min(12, record.WMC)) / 12.0) / 2.0
            return float(min(0.95, max(0.05, score))), is_smelly
            
        elif smell == SmellType.BRAIN_CLASS:
            is_smelly = (record.WMC > 45 and record.MNB >= 4 and record.SLOC > 300)
            score = (record.WMC / 50.0 + record.MNB / 4.0) / 2.0
            return float(min(0.99, max(0.05, score))), is_smelly
            
        elif smell == SmellType.LONG_METHOD:
            is_smelly = (record.SLOC > 45 and record.cyclomatic_complexity > 8)
            score = (record.SLOC / 50.0 + record.cyclomatic_complexity / 10.0) / 2.0
            return float(min(0.99, max(0.05, score))), is_smelly
            
        elif smell == SmellType.FEATURE_ENVY:
            is_smelly = (record.ATFD >= 4 and record.FDP <= 2)
            score = (record.ATFD / 5.0)
            return float(min(0.95, max(0.05, score))), is_smelly
            
        elif smell == SmellType.BRAIN_METHOD:
            is_smelly = (record.cyclomatic_complexity > 14 and record.MNB >= 3)
            score = (record.cyclomatic_complexity / 15.0 + record.MNB / 3.0) / 2.0
            return float(min(0.99, max(0.05, score))), is_smelly
            
        return 0.10, False

    def _build_explanation(self, record: MetricRecord, smell: SmellType, prob: float, is_smelly: bool) -> str:
        if not is_smelly:
            return f"Metrics for {record.class_name} are within healthy thresholds (smell probability: {prob*100:.1f}%)."
            
        if smell == SmellType.GOD_CLASS:
            return f"Class exhibits centralized control with elevated WMC ({record.WMC}), CBO ({record.CBO}), and {int(record.SLOC)} SLOC."
        elif smell == SmellType.DATA_CLASS:
            return f"Class functions primarily as a passive data carrier with low method complexity ({record.WMC}) and high lack of cohesion ({record.LCOM5:.2f})."
        elif smell == SmellType.BRAIN_CLASS:
            return f"Class concentrates intense coordination logic with WMC ({record.WMC}) and deeply nested control structures (MNB: {record.MNB})."
        elif smell == SmellType.LONG_METHOD:
            return f"Method spans {int(record.SLOC)} lines with cyclomatic complexity of {int(record.cyclomatic_complexity)}, exceeding modular readability bounds."
        elif smell == SmellType.FEATURE_ENVY:
            return f"Method accesses {int(record.ATFD)} foreign attributes across {int(record.FDP)} external providers, indicating misplaced responsibility."
        elif smell == SmellType.BRAIN_METHOD:
            return f"Method contains complex decision branching (CC: {int(record.cyclomatic_complexity)}) and deeply nested blocks ({int(record.MNB)} levels)."
        return "Model detected elevated design defect risk based on structural metric interactions."

    def _build_recommendation(self, smell: SmellType) -> str:
        if smell == SmellType.GOD_CLASS:
            return "Apply 'Extract Class' or 'Extract Subclass' refactorings to divide distinct responsibilities into cohesive components."
        elif smell == SmellType.DATA_CLASS:
            return "Consider 'Move Method' to relocate data-manipulating operations closer to this class, encapsulating behavior."
        elif smell == SmellType.BRAIN_CLASS:
            return "Decompose centralized algorithmic logic into strategy objects or delegator classes to lower cognitive coupling."
        elif smell == SmellType.LONG_METHOD:
            return "Apply 'Extract Method' to decompose independent execution chunks into well-named helper routines."
        elif smell == SmellType.FEATURE_ENVY:
            return "Consider 'Move Method' to relocate this method to the foreign class it interacts with most heavily."
        elif smell == SmellType.BRAIN_METHOD:
            return "Apply 'Replace Nested Conditional with Guard Clauses' or extract complex loops into separate processing functions."
        return "Inspect and refactor to adhere to single responsibility principles."
