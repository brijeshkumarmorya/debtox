from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class EntityGranularity(str, Enum):
    CLASS = "class"
    METHOD = "method"

class SmellType(str, Enum):
    GOD_CLASS = "God Class"
    DATA_CLASS = "Data Class"
    BRAIN_CLASS = "Brain Class"
    LONG_METHOD = "Long Method"
    FEATURE_ENVY = "Feature Envy"
    BRAIN_METHOD = "Brain Method"

class AnalysisMode(str, Enum):
    QUICK = "quick"         # Snapshot only, no deep Git history
    FULL = "full"           # Snapshot + Git history & TDI
    RESEARCH = "research"   # Snapshot + Git history + full SHAP attributions + CSV export

class AnalysisStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class MetricRecord(BaseModel):
    # Identity
    entity_id: str
    file_path: str
    class_name: str
    method_name: Optional[str] = None
    granularity: EntityGranularity
    start_line: int
    end_line: int
    
    # Structural / OO
    WMC: float = 0.0
    CBO: float = 0.0
    LCOM: float = 0.0
    LCOM5: float = 0.0
    RFC: float = 0.0
    DIT: float = 0.0
    NOC: float = 0.0
    ATFD: float = 0.0
    FDP: float = 0.0
    
    # Complexity & Size
    SLOC: float = 0.0
    LLOC: float = 0.0
    cyclomatic_complexity: float = 1.0
    MNB: float = 0.0  # Max nested blocks
    
    # Halstead
    halstead_length: float = 0.0
    halstead_volume: float = 0.0
    halstead_difficulty: float = 0.0
    halstead_effort: float = 0.0
    
    # Evolutionary (when available)
    churn: Optional[float] = 0.0
    added_loc: Optional[float] = 0.0
    deleted_loc: Optional[float] = 0.0
    num_hunks: Optional[float] = 0.0
    commit_frequency: Optional[float] = 0.0
    commits_touching_component: Optional[int] = 0
    age_days: Optional[float] = 0.0
    historical_refactor_count: Optional[int] = 0

class SmellPrediction(BaseModel):
    smell_id: str
    name: str
    granularity: EntityGranularity
    is_smelly: bool
    probability: float
    threshold: float = 0.50
    severity: str = "medium"  # low, medium, high, critical
    tdp_hours: float = 0.0
    tdp_cost_usd: float = 0.0
    explanation: str
    refactoring_recommendation: str

class SHAPFeatureContribution(BaseModel):
    feature_name: str
    feature_value: float
    shap_value: float
    impact: str  # "increases_risk", "decreases_risk"
    narrative: str

class ComponentDetail(BaseModel):
    entity_id: str
    file_path: str
    class_name: str
    method_name: Optional[str] = None
    granularity: EntityGranularity
    start_line: int
    end_line: int
    metrics: Dict[str, Any]
    smell_predictions: List[SmellPrediction] = []
    detected_smells: List[str] = []
    tdp_hours: float = 0.0
    tdi_hours: Optional[float] = 0.0
    total_debt_hours: float = 0.0
    risk_score: float = 0.0  # 0.0 to 100.0
    shap_contributions: Optional[List[SHAPFeatureContribution]] = None
    primary_recommendation: Optional[str] = None

class EvolutionRecord(BaseModel):
    commit_hash: str
    short_hash: str
    timestamp: str
    author: str
    message: str
    added_lines: int
    deleted_lines: int
    total_churn: int
    changed_files: int
    num_hunks: int
    smell_count: Optional[int] = None
    tdp_hours: Optional[float] = None
    tdi_hours: Optional[float] = None

class AnalysisRequest(BaseModel):
    repository_url: Optional[str] = None
    local_path: Optional[str] = None
    branch: Optional[str] = None
    commit_hash: Optional[str] = None
    mode: AnalysisMode = AnalysisMode.FULL
    max_history_commits: Optional[int] = 100

class AnalysisSummary(BaseModel):
    analysis_id: str
    repository_name: str
    repository_path_or_url: str
    branch: Optional[str] = None
    commit_hash: Optional[str] = None
    status: AnalysisStatus
    mode: AnalysisMode
    progress_percentage: int = 0
    current_stage: str = "Initialized"
    error_message: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    runtime_seconds: Optional[float] = 0.0
    
    total_files: int = 0
    total_classes: int = 0
    total_methods: int = 0
    total_smells: int = 0
    affected_components_count: int = 0
    smell_counts: Dict[str, int] = {}
    
    total_tdp_hours: float = 0.0
    total_tdi_hours: float = 0.0
    total_debt_hours: float = 0.0
    total_estimated_cost_usd: float = 0.0
    average_risk_score: float = 0.0
    
    components: Optional[List[ComponentDetail]] = None
    evolution: Optional[List[EvolutionRecord]] = None
