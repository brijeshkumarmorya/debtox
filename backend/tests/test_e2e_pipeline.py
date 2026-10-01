import pytest
from backend.app.services.analysis_service import AnalysisService
from backend.app.schemas.analysis import AnalysisRequest, AnalysisMode, AnalysisStatus

def test_full_e2e_analysis_on_sample_repo():
    service = AnalysisService()
    req = AnalysisRequest(local_path="data/sample_repo", mode=AnalysisMode.RESEARCH)
    
    # 1. Job Creation
    job = service.create_analysis_job(req)
    assert job.status == AnalysisStatus.QUEUED
    assert job.analysis_id is not None
    
    # 2. Execution
    service.execute_analysis(job.analysis_id, req)
    
    # 3. Verification
    res = service.get_analysis(job.analysis_id)
    assert res is not None
    assert res.status == AnalysisStatus.COMPLETED
    assert res.total_files >= 3
    assert res.total_classes >= 3
    assert res.total_methods >= 10
    assert res.total_smells >= 1
    assert res.total_tdp_hours > 0
    assert len(res.components) > 0
    assert len(res.evolution) >= 1
