from typing import Dict, Any, Optional
from backend.app.core.config import settings
from backend.app.schemas.analysis import MetricRecord, SmellType, EntityGranularity

class TDPEngine:
    """
    Technical Debt Principal (TDP) Estimation Engine.
    Estimates the one-time developer effort (in hours and USD) required to remediate code smells.
    Implements a calibrated COCOMO II software maintenance formulation alongside SQALE baselines.
    """
    
    def __init__(
        self,
        cocomo_a: float = settings.COCOMO_A,
        cocomo_e: float = settings.COCOMO_E,
        hourly_rate: float = settings.DEFAULT_DEVELOPER_HOURLY_RATE
    ):
        self.A = cocomo_a
        self.E = cocomo_e
        self.hourly_rate = hourly_rate

    def estimate_tdp(
        self,
        record: MetricRecord,
        smell_type: SmellType,
        probability: float
    ) -> Dict[str, Any]:
        """
        Estimates TDP hours using parametric COCOMO II maintenance formulation.
        Effort = 152 * A * [ (KSLOC_smell) * (1 + SU/100) ]^E * prod(EM)
        """
        if probability < 0.50:
            return {
                "tdp_hours": 0.0,
                "tdp_cost_usd": 0.0,
                "sqale_hours": 0.0,
                "estimation_method": "Clean module / below threshold"
            }
            
        sloc = max(5.0, float(record.SLOC))
        ksloc = sloc / 1000.0
        
        # Software Understanding (SU) penalty (0-50%) based on complexity
        # Higher cyclomatic complexity and nesting increase cognitive comprehension overhead
        su = min(50.0, 10.0 + (record.cyclomatic_complexity * 1.5) + (record.MNB * 3.0))
        
        # Effort Multiplier (EM) based on smell severity and coupling
        # High CBO creates ripple effects during refactoring
        em_coupling = 1.0 + min(0.5, (record.CBO / 25.0) * 0.3)
        em_smell = 1.0
        
        # Base refactoring scope and SQALE baseline
        if smell_type == SmellType.GOD_CLASS:
            refactor_ratio = 0.40  # Refactoring touches approx 40% of the monolithic class
            sqale_hours = settings.SQALE_GOD_CLASS_MINUTES / 60.0
            em_smell = 1.25
        elif smell_type == SmellType.DATA_CLASS:
            refactor_ratio = 0.25
            sqale_hours = 1.0
            em_smell = 0.90
        elif smell_type == SmellType.BRAIN_CLASS:
            refactor_ratio = 0.35
            sqale_hours = 2.5
            em_smell = 1.30
        elif smell_type == SmellType.LONG_METHOD:
            refactor_ratio = 0.60  # Extracting methods restructures a major part of the method
            sqale_hours = settings.SQALE_LONG_METHOD_MINUTES / 60.0
            em_smell = 1.10
        elif smell_type == SmellType.FEATURE_ENVY:
            refactor_ratio = 0.50
            sqale_hours = 0.75
            em_smell = 1.15
        elif smell_type == SmellType.BRAIN_METHOD:
            refactor_ratio = 0.55
            sqale_hours = 1.5
            em_smell = 1.25
        else:
            refactor_ratio = 0.30
            sqale_hours = 1.0

        effective_ksloc = max(0.015, ksloc * refactor_ratio)
        size_with_su = effective_ksloc * (1.0 + (su / 100.0))
        
        # COCOMO II maintenance person-months converted to hours (152 hours/person-month)
        person_months = self.A * (size_with_su ** self.E) * em_coupling * em_smell
        raw_hours = person_months * 152.0
        
        # Scale smoothly with prediction probability
        prob_weight = 0.6 + (0.4 * probability)
        final_hours = max(0.5, round(raw_hours * prob_weight, 2))
        cost_usd = round(final_hours * self.hourly_rate, 2)
        
        return {
            "tdp_hours": final_hours,
            "tdp_cost_usd": cost_usd,
            "sqale_hours": sqale_hours,
            "estimation_method": "Calibrated COCOMO II Maintenance Model",
            "assumptions": {
                "software_understanding_penalty": round(su, 1),
                "refactor_scope_ratio": refactor_ratio,
                "hourly_rate_usd": self.hourly_rate
            }
        }
