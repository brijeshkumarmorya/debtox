"""
DebtOx Empirical Research Pipeline - Section 13:
Practical DebtOx Analysis Validation on Real Java Open-Source Repositories
Runs DebtOx end-to-end analysis on:
1. Google Gson (data/demo_repos/gson)
2. Apache Commons-CLI (data/demo_repos/commons-cli)
3. Sample Reference Repository (data/sample_repo)
Outputs:
- experiments/results/real_repository_demo_results.csv
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.services.analysis_service import AnalysisService
from backend.app.schemas.analysis import AnalysisRequest, AnalysisMode

def run_real_repository_demo():
    print("=" * 60)
    print("SECTION 13: PRACTICAL DEBTOX ANALYSIS ON REAL REPOSITORIES")
    print("=" * 60)
    
    service = AnalysisService()
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    target_repos = [
        {
            "name": "google-gson",
            "category": "External Real Open-Source Demonstration",
            "path": Path("data/demo_repos/gson").resolve(),
            "description": "Google Gson JSON serialization/deserialization library"
        },
        {
            "name": "apache-commons-cli",
            "category": "External Real Open-Source Demonstration",
            "path": Path("data/demo_repos/commons-cli").resolve(),
            "description": "Apache Commons CLI command-line argument parser"
        },
        {
            "name": "debtox-sample-repo",
            "category": "Reference Validation Benchmark",
            "path": Path("data/sample_repo").resolve(),
            "description": "Reference Java benchmark repository with synthetic git history"
        }
    ]
    
    demo_rows = []
    
    for repo_info in target_repos:
        repo_path = repo_info["path"]
        if not repo_path.exists():
            print(f"[WARN] Repository path {repo_path} does not exist. Skipping...")
            continue
            
        print(f"\n---> Analyzing Repository: {repo_info['name']} ({repo_path})")
        
        req = AnalysisRequest(
            local_path=str(repo_path),
            mode=AnalysisMode.FULL,
            enable_shap=True
        )
        
        job = service.create_analysis_job(req)
        service.execute_analysis(job.analysis_id, req)
        
        summary = service.get_analysis(job.analysis_id)
        if not summary:
            print(f"[ERROR] Analysis failed for {repo_info['name']}")
            continue
            
        est_cost = summary.total_estimated_cost_usd if hasattr(summary, "total_estimated_cost_usd") and summary.total_estimated_cost_usd else round(summary.total_debt_hours * 75.0, 2)
        print(f"  Detected Smells: {summary.total_smells} | Debt Effort: {summary.total_debt_hours:.2f} hrs | Est. Cost: ${est_cost:,.2f}")
        
        # Smell counts breakdown
        smell_counts = {
            "god_class": 0,
            "data_class": 0,
            "long_method": 0,
            "feature_envy": 0
        }
        for comp in (summary.components or []):
            for s in comp.detected_smells:
                s_key = str(s).lower().replace(" ", "_")
                if s_key in smell_counts:
                    smell_counts[s_key] += 1
                    
        # Extract representative SHAP explanation top feature if available
        top_shap_feature = "SLOC"
        for comp in (summary.components or []):
            if comp.shap_contributions and len(comp.shap_contributions) > 0:
                top_shap_feature = comp.shap_contributions[0].feature_name
                break
                
        demo_rows.append({
            "repository_name": repo_info["name"],
            "repository_category": repo_info["category"],
            "description": repo_info["description"],
            "total_java_files": summary.total_files,
            "total_classes": summary.total_classes,
            "total_methods": summary.total_methods,
            "total_smells_detected": summary.total_smells,
            "god_classes_count": smell_counts["god_class"],
            "data_classes_count": smell_counts["data_class"],
            "long_methods_count": smell_counts["long_method"],
            "feature_envy_count": smell_counts["feature_envy"],
            "total_tdp_hours": round(summary.total_tdp_hours or summary.total_debt_hours, 2),
            "total_tdp_cost_usd": round(est_cost, 2),
            "total_tdi_hours": round(summary.total_tdi_hours or 0.0, 2),
            "total_tdi_cost_usd": round((summary.total_tdi_hours or 0.0) * 75.0, 2),
            "dominant_shap_attribution_feature": top_shap_feature,
            "analysis_status": summary.status.value,
        })
        
    df_demo = pd.DataFrame(demo_rows)
    out_csv = results_dir / "real_repository_demo_results.csv"
    df_demo.to_csv(out_csv, index=False)
    print(f"\n[SUCCESS] Real repository demo results saved to {out_csv}")
    return df_demo

if __name__ == "__main__":
    run_real_repository_demo()
