from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from typing import Dict, List, Optional, Any

from backend.app.schemas.analysis import (
    AnalysisRequest, AnalysisSummary, ComponentDetail, EntityGranularity
)
from backend.app.services.analysis_service import AnalysisService
from backend.app.core.config import settings

router = APIRouter()
analysis_service = AnalysisService()

@router.get("/health")
def get_health() -> Dict[str, Any]:
    models_meta = analysis_service.inference_engine.get_loaded_models_metadata()
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "loaded_models_count": len(models_meta)
    }

@router.get("/version")
def get_version() -> Dict[str, str]:
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "api_version": "v1"
    }

@router.post("/analyze", response_model=AnalysisSummary)
def start_analysis(
    req: AnalysisRequest,
    background_tasks: BackgroundTasks
) -> AnalysisSummary:
    """
    Submits a repository for asynchronous analysis.
    Returns immediately with an analysis_id and QUEUED status.
    """
    summary = analysis_service.create_analysis_job(req)
    background_tasks.add_task(analysis_service.execute_analysis, summary.analysis_id, req)
    return summary

@router.get("/analyses")
def list_analyses() -> List[Dict[str, Any]]:
    return analysis_service.list_analyses()

@router.get("/analyses/{analysis_id}", response_model=AnalysisSummary)
def get_analysis(analysis_id: str) -> AnalysisSummary:
    summary = analysis_service.get_analysis(analysis_id)
    if not summary:
        raise HTTPException(status_code=404, detail=f"Analysis ID {analysis_id} not found")
    return summary

@router.get("/analyses/{analysis_id}/summary")
def get_analysis_summary_only(analysis_id: str) -> Dict[str, Any]:
    summary = analysis_service.get_analysis(analysis_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Analysis not found")
    data = summary.model_dump()
    data.pop("components", None)
    return data

@router.get("/analyses/{analysis_id}/components", response_model=List[ComponentDetail])
def get_analysis_components(
    analysis_id: str,
    smell: Optional[str] = Query(None, description="Filter by smell name"),
    granularity: Optional[EntityGranularity] = Query(None, description="Filter by class or method"),
    min_risk: Optional[float] = Query(0.0, description="Minimum risk score (0-100)")
) -> List[ComponentDetail]:
    summary = analysis_service.get_analysis(analysis_id)
    if not summary or not summary.components:
        raise HTTPException(status_code=404, detail="Analysis components not found")
        
    filtered = summary.components
    if granularity:
        filtered = [c for c in filtered if c.granularity == granularity]
    if min_risk > 0:
        filtered = [c for c in filtered if c.risk_score >= min_risk]
    if smell:
        filtered = [c for c in filtered if smell in c.detected_smells]
        
    return filtered

@router.get("/analyses/{analysis_id}/smells")
def get_analysis_smells(analysis_id: str) -> Dict[str, Any]:
    summary = analysis_service.get_analysis(analysis_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {
        "analysis_id": summary.analysis_id,
        "total_smells": summary.total_smells,
        "smell_counts": summary.smell_counts,
        "affected_components_count": summary.affected_components_count
    }

@router.get("/analyses/{analysis_id}/debt")
def get_analysis_debt(analysis_id: str) -> Dict[str, Any]:
    summary = analysis_service.get_analysis(analysis_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {
        "analysis_id": summary.analysis_id,
        "total_tdp_hours": summary.total_tdp_hours,
        "total_tdi_hours": summary.total_tdi_hours,
        "total_debt_hours": summary.total_debt_hours,
        "total_estimated_cost_usd": summary.total_estimated_cost_usd,
        "hourly_rate_usd": settings.DEFAULT_DEVELOPER_HOURLY_RATE
    }

@router.get("/analyses/{analysis_id}/explanations")
def get_analysis_explanations(analysis_id: str) -> List[Dict[str, Any]]:
    summary = analysis_service.get_analysis(analysis_id)
    if not summary or not summary.components:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return [
        {
            "entity_id": c.entity_id,
            "class_name": c.class_name,
            "method_name": c.method_name,
            "detected_smells": c.detected_smells,
            "risk_score": c.risk_score,
            "shap_contributions": [s.model_dump() for s in (c.shap_contributions or [])]
        }
        for c in summary.components if c.shap_contributions
    ]

@router.get("/models")
def get_models() -> List[Dict[str, Any]]:
    return analysis_service.inference_engine.get_loaded_models_metadata()
