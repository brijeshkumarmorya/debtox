import pytest
from backend.app.debt.tdp import TDPEngine
from backend.app.debt.tdi import TDIEngine
from backend.app.debt.prioritization import RiskPrioritizer
from backend.app.schemas.analysis import MetricRecord, EntityGranularity, SmellType, SmellPrediction

def test_tdp_estimation_clean():
    engine = TDPEngine()
    rec = MetricRecord(
        entity_id="test:1", file_path="Test.java", class_name="Test",
        granularity=EntityGranularity.CLASS, start_line=1, end_line=50,
        SLOC=40, cyclomatic_complexity=3
    )
    res = engine.estimate_tdp(rec, SmellType.GOD_CLASS, probability=0.20)
    assert res["tdp_hours"] == 0.0

def test_tdp_estimation_smelly():
    engine = TDPEngine()
    rec = MetricRecord(
        entity_id="test:1", file_path="Test.java", class_name="Test",
        granularity=EntityGranularity.CLASS, start_line=1, end_line=300,
        SLOC=350, cyclomatic_complexity=25, WMC=40, CBO=15
    )
    res = engine.estimate_tdp(rec, SmellType.GOD_CLASS, probability=0.90)
    assert res["tdp_hours"] > 0
    assert res["tdp_cost_usd"] > 0
    assert "COCOMO II" in res["estimation_method"]

def test_risk_prioritizer():
    preds = [
        SmellPrediction(
            smell_id="god_class", name="God Class", granularity=EntityGranularity.CLASS,
            is_smelly=True, probability=0.92, severity="high", tdp_hours=14.5,
            explanation="test", refactoring_recommendation="test"
        )
    ]
    risk = RiskPrioritizer.calculate_component_risk(preds, tdp_hours=14.5, tdi_hours=5.0)
    assert 50.0 <= risk <= 100.0
