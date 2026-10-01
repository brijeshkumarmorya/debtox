from typing import Dict, List, Optional
from backend.app.schemas.analysis import MetricRecord, SmellPrediction

class RiskPrioritizer:
    """
    Computes a normalized multi-factor Risk Score [0.0 - 100.0]
    for components based on smell probabilities, TDP, TDI, and severity.
    """
    
    @staticmethod
    def calculate_component_risk(
        predictions: List[SmellPrediction],
        tdp_hours: float,
        tdi_hours: float = 0.0
    ) -> float:
        smelly_preds = [p for p in predictions if p.is_smelly]
        if not smelly_preds:
            return 0.0
            
        max_prob = max(p.probability for p in smelly_preds)
        
        severity_map = {"low": 0.25, "medium": 0.50, "high": 0.75, "critical": 1.0}
        max_severity = max(severity_map.get(p.severity, 0.5) for p in smelly_preds)
        
        # 1. Smell Probability Factor (Weight: 35)
        f_prob = max_prob * 35.0
        
        # 2. TDP Factor (Weight: 30) - 20 hours considered high TDP ceiling
        f_tdp = min(1.0, tdp_hours / 20.0) * 30.0
        
        # 3. TDI Factor (Weight: 20) - 30 hours considered high TDI ceiling
        f_tdi = min(1.0, tdi_hours / 30.0) * 20.0
        
        # 4. Severity Factor (Weight: 15)
        f_sev = max_severity * 15.0
        
        total_risk = f_prob + f_tdp + f_tdi + f_sev
        return round(float(min(100.0, max(0.0, total_risk))), 1)
