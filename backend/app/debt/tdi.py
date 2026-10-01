from typing import Dict, List, Optional, Any
import numpy as np
from backend.app.core.config import settings
from backend.app.schemas.analysis import MetricRecord, EvolutionRecord

class TDIEngine:
    """
    Technical Debt Interest (TDI) Engine.
    Quantifies the extra ongoing maintenance effort accumulated over time
    due to lingering code smells, based on excess code churn relative to clean baselines.
    """
    
    def __init__(self, hourly_rate: float = settings.DEFAULT_DEVELOPER_HOURLY_RATE):
        self.hourly_rate = hourly_rate

    def estimate_tdi(
        self,
        component_records: List[MetricRecord],
        evolution_records: List[EvolutionRecord],
        smelly_entity_ids: set
    ) -> Dict[str, float]:
        """
        Calculates TDI hours for each smelly entity.
        TDI = sum_over_revisions max(0, Churn(smelly) - BaselineChurn(clean_peers)) * C_maint
        """
        tdi_results: Dict[str, float] = {}
        if not evolution_records or len(evolution_records) < 3:
            # Insufficient longitudinal history
            return {eid: 0.0 for eid in smelly_entity_ids}

        # Step 1: Calculate average clean component churn baseline
        clean_churns = [
            r.churn for r in component_records 
            if r.entity_id not in smelly_entity_ids and r.churn is not None and r.churn > 0
        ]
        
        # Median churn per clean component (fallback to 25 LOC if sparse)
        baseline_churn = float(np.median(clean_churns)) if clean_churns else 25.0
        
        # Maintenance cost multiplier: approx 0.08 hours per churned line of code
        churn_to_hours_ratio = 0.08
        
        for record in component_records:
            if record.entity_id in smelly_entity_ids:
                actual_churn = record.churn or 0.0
                commits_touching = record.commits_touching_component or 1
                
                # Excess churn per revision over baseline
                excess_churn = max(0.0, actual_churn - (baseline_churn * (commits_touching / 2.0)))
                
                # If component is very complex, friction multiplier increases
                friction_multiplier = 1.0 + min(1.0, (record.cyclomatic_complexity / 20.0))
                
                tdi_hours = excess_churn * churn_to_hours_ratio * friction_multiplier
                tdi_results[record.entity_id] = round(float(tdi_hours), 2)
            else:
                tdi_results[record.entity_id] = 0.0
                
        return tdi_results
