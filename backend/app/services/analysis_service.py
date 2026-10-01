import json
import shutil
import time
import uuid
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

from backend.app.core.config import settings
from backend.app.schemas.analysis import (
    AnalysisRequest, AnalysisSummary, AnalysisStatus, AnalysisMode,
    ComponentDetail, EvolutionRecord, MetricRecord, SmellType
)
from analyzer.extraction.engine import MetricExtractionEngine
from analyzer.git.miner import GitEvolutionMiner
from backend.app.ml.inference import SmellInferenceEngine
from backend.app.debt.tdp import TDPEngine
from backend.app.debt.tdi import TDIEngine
from backend.app.debt.prioritization import RiskPrioritizer
from backend.app.explainability.shap_explainer import SHAPExplainerService

class AnalysisService:
    def __init__(self):
        self.metric_engine = MetricExtractionEngine()
        self.git_miner = GitEvolutionMiner()
        self.inference_engine = SmellInferenceEngine()
        self.tdp_engine = TDPEngine()
        self.tdi_engine = TDIEngine()
        self.shap_explainer = SHAPExplainerService()
        self.storage_dir = settings.STORAGE_DIR / "analyses"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._in_memory_jobs: Dict[str, AnalysisSummary] = {}

    def get_analysis(self, analysis_id: str) -> Optional[AnalysisSummary]:
        if analysis_id in self._in_memory_jobs:
            return self._in_memory_jobs[analysis_id]
        file_path = self.storage_dir / f"{analysis_id}.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            summary = AnalysisSummary(**data)
            self._in_memory_jobs[analysis_id] = summary
            return summary
        return None

    def list_analyses(self) -> List[Dict[str, Any]]:
        results = []
        for file in self.storage_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results.append({
                    "analysis_id": data["analysis_id"],
                    "repository_name": data["repository_name"],
                    "status": data["status"],
                    "created_at": data["created_at"],
                    "total_files": data.get("total_files", 0),
                    "total_smells": data.get("total_smells", 0),
                    "total_debt_hours": data.get("total_debt_hours", 0.0)
                })
            except Exception:
                continue
        # Also include any active in-memory running jobs
        for job_id, job in self._in_memory_jobs.items():
            if not any(r["analysis_id"] == job_id for r in results):
                results.append({
                    "analysis_id": job.analysis_id,
                    "repository_name": job.repository_name,
                    "status": job.status,
                    "created_at": job.created_at,
                    "total_files": job.total_files,
                    "total_smells": job.total_smells,
                    "total_debt_hours": job.total_debt_hours
                })
        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return results

    def create_analysis_job(self, req: AnalysisRequest) -> AnalysisSummary:
        analysis_id = str(uuid.uuid4())
        repo_name = "local_repository"
        repo_target = req.local_path or req.repository_url or "unknown"
        if req.repository_url:
            repo_name = req.repository_url.rstrip("/").split("/")[-1].replace(".git", "")
        elif req.local_path:
            repo_name = Path(req.local_path).name

        summary = AnalysisSummary(
            analysis_id=analysis_id,
            repository_name=repo_name,
            repository_path_or_url=repo_target,
            branch=req.branch,
            commit_hash=req.commit_hash,
            status=AnalysisStatus.QUEUED,
            mode=req.mode,
            progress_percentage=5,
            current_stage="Queued for analysis",
            created_at=datetime.now(timezone.utc).isoformat(),
            components=[],
            evolution=[]
        )
        self._in_memory_jobs[analysis_id] = summary
        return summary

    def execute_analysis(self, analysis_id: str, req: AnalysisRequest):
        summary = self._in_memory_jobs.get(analysis_id)
        if not summary:
            return

        start_time = time.time()
        temp_dir: Optional[Path] = None
        repo_obj = None
        try:
            # Stage 1: Ingestion
            summary.status = AnalysisStatus.RUNNING
            summary.progress_percentage = 15
            summary.current_stage = "Ingesting Repository & Git Mining"
            
            repo_target = req.local_path or req.repository_url
            if not repo_target:
                raise ValueError("Neither repository_url nor local_path was supplied.")
                
            repo_obj, repo_dir, is_temp = self.git_miner.clone_or_open_repository(
                repo_target, branch=req.branch
            )
            if is_temp:
                temp_dir = repo_dir
                
            summary.commit_hash = repo_obj.head.commit.hexsha[:8] if repo_obj and not repo_obj.bare else "HEAD"
            
            # Stage 2: Git Evolution History
            evolution_records: List[EvolutionRecord] = []
            if req.mode != AnalysisMode.QUICK and repo_obj:
                evolution_records = self.git_miner.extract_commit_history(
                    repo_obj, max_commits=req.max_history_commits or 100
                )
            summary.evolution = evolution_records

            # Stage 3: Static Code Metrics
            summary.progress_percentage = 40
            summary.current_stage = "Parsing Java Source Files & Extracting Metrics"
            metric_records = self.metric_engine.analyze_repository_path(
                repo_dir=repo_dir,
                repo_id=summary.repository_name,
                commit_hash=summary.commit_hash
            )
            
            java_files = set(r.file_path for r in metric_records)
            classes_count = sum(1 for r in metric_records if r.granularity.value == "class")
            methods_count = sum(1 for r in metric_records if r.granularity.value == "method")
            
            summary.total_files = len(java_files)
            summary.total_classes = classes_count
            summary.total_methods = methods_count

            # Stage 4: Smell Prediction & TDP
            summary.progress_percentage = 65
            summary.current_stage = "Predicting Code Smells & Estimating Principal (TDP)"
            
            components: List[ComponentDetail] = []
            smelly_entity_ids = set()
            smell_counts: Dict[str, int] = {}
            total_tdp = 0.0
            
            for m_rec in metric_records:
                # Augment evolutionary churn if Git is active
                if req.mode != AnalysisMode.QUICK and repo_obj:
                    churn_info = self.git_miner.compute_component_churn(repo_obj, m_rec.file_path)
                    m_rec.churn = churn_info["churn"]
                    m_rec.added_loc = churn_info["added_loc"]
                    m_rec.deleted_loc = churn_info["deleted_loc"]
                    m_rec.commits_touching_component = churn_info["commits_touching_component"]
                    m_rec.commit_frequency = churn_info["commit_frequency"]
                    m_rec.age_days = churn_info["age_days"]
                    
                preds = self.inference_engine.predict_component_smells(m_rec)
                detected = [p.name for p in preds if p.is_smelly]
                
                comp_tdp = 0.0
                for p in preds:
                    if p.is_smelly:
                        smelly_entity_ids.add(m_rec.entity_id)
                        smell_counts[p.name] = smell_counts.get(p.name, 0) + 1
                        tdp_res = self.tdp_engine.estimate_tdp(m_rec, SmellType(p.name), p.probability)
                        p.tdp_hours = tdp_res["tdp_hours"]
                        p.tdp_cost_usd = tdp_res["tdp_cost_usd"]
                        comp_tdp += p.tdp_hours
                        
                total_tdp += comp_tdp
                
                # SHAP explanations for smelly components or top candidates
                shap_contribs = None
                if detected and req.mode == AnalysisMode.RESEARCH:
                    # Explain the primary smell
                    primary_pred = next(p for p in preds if p.is_smelly)
                    shap_contribs = self.shap_explainer.explain_prediction(m_rec, primary_pred)

                primary_rec = None
                for p in preds:
                    if p.is_smelly:
                        primary_rec = p.refactoring_recommendation
                        break
                        
                comp = ComponentDetail(
                    entity_id=m_rec.entity_id,
                    file_path=m_rec.file_path,
                    class_name=m_rec.class_name,
                    method_name=m_rec.method_name,
                    granularity=m_rec.granularity,
                    start_line=m_rec.start_line,
                    end_line=m_rec.end_line,
                    metrics=m_rec.model_dump(),
                    smell_predictions=preds,
                    detected_smells=detected,
                    tdp_hours=round(comp_tdp, 2),
                    tdi_hours=0.0,
                    total_debt_hours=round(comp_tdp, 2),
                    risk_score=0.0,
                    shap_contributions=shap_contribs,
                    primary_recommendation=primary_rec
                )
                components.append(comp)

            # Stage 5: TDI & Risk Prioritization
            summary.progress_percentage = 85
            summary.current_stage = "Computing Technical Debt Interest (TDI) & Risk Scoring"
            
            tdi_map = self.tdi_engine.estimate_tdi(metric_records, evolution_records, smelly_entity_ids)
            total_tdi = 0.0
            
            for comp in components:
                comp_tdi = tdi_map.get(comp.entity_id, 0.0)
                comp.tdi_hours = comp_tdi
                comp.total_debt_hours = round(comp.tdp_hours + comp_tdi, 2)
                comp.risk_score = RiskPrioritizer.calculate_component_risk(
                    comp.smell_predictions, comp.tdp_hours, comp_tdi
                )
                total_tdi += comp_tdi

            # Sort components by risk score descending
            components.sort(key=lambda c: c.risk_score, reverse=True)
            summary.components = components
            
            # Map aggregated stats to summary
            total_smells = sum(smell_counts.values())
            summary.total_smells = total_smells
            summary.affected_components_count = len(smelly_entity_ids)
            summary.smell_counts = smell_counts
            summary.total_tdp_hours = round(total_tdp, 2)
            summary.total_tdi_hours = round(total_tdi, 2)
            summary.total_debt_hours = round(total_tdp + total_tdi, 2)
            summary.total_estimated_cost_usd = round(summary.total_debt_hours * settings.DEFAULT_DEVELOPER_HOURLY_RATE, 2)
            
            avg_risk = float(np.mean([c.risk_score for c in components if c.risk_score > 0])) if components else 0.0
            summary.average_risk_score = round(avg_risk, 1)

            # Finalize
            summary.status = AnalysisStatus.COMPLETED
            summary.progress_percentage = 100
            summary.current_stage = "Analysis Completed Successfully"
            summary.completed_at = datetime.now(timezone.utc).isoformat()
            summary.runtime_seconds = round(time.time() - start_time, 2)

            # Persist to disk
            out_file = self.storage_dir / f"{analysis_id}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(summary.model_dump(), f, indent=2)

        except Exception as e:
            summary.status = AnalysisStatus.FAILED
            summary.error_message = str(e)
            summary.current_stage = f"Failed: {str(e)}"
            summary.completed_at = datetime.now(timezone.utc).isoformat()
            summary.runtime_seconds = round(time.time() - start_time, 2)
        finally:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
