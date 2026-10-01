export type SupportedSmell = 'God Class' | 'Data Class' | 'Long Method' | 'Feature Envy';

export type NavigationTab = 'home' | 'analyze' | 'issues' | 'projects';

export interface SmellPrediction {
  smell_id: string;
  name: string;
  granularity: 'class' | 'method';
  is_smelly: boolean;
  probability: number;
  threshold: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  tdp_hours: number;
  tdp_cost_usd: number;
  explanation: string;
  refactoring_recommendation: string;
}

export interface SHAPFeatureContribution {
  feature_name: string;
  feature_value: number;
  shap_value: number;
  impact: 'increases_risk' | 'decreases_risk';
  narrative: string;
}

export interface ComponentDetail {
  entity_id: string;
  file_path: string;
  class_name: string;
  method_name?: string;
  granularity: 'class' | 'method';
  start_line: number;
  end_line: number;
  metrics: Record<string, any>;
  smell_predictions: SmellPrediction[];
  detected_smells: string[];
  tdp_hours: number;
  tdi_hours?: number;
  total_debt_hours: number;
  risk_score: number;
  shap_contributions?: SHAPFeatureContribution[];
  primary_recommendation?: string;
}

export interface EvolutionRecord {
  commit_hash: string;
  short_hash: string;
  timestamp: string;
  author: string;
  message: string;
  added_lines: number;
  deleted_lines: number;
  total_churn: number;
  changed_files: number;
  num_hunks: number;
  smell_count?: number;
  tdp_hours?: number;
  tdi_hours?: number;
}

export interface AnalysisSummary {
  analysis_id: string;
  repository_name: string;
  repository_path_or_url: string;
  branch?: string;
  commit_hash?: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  mode: 'quick' | 'full' | 'research';
  progress_percentage: number;
  current_stage: string;
  error_message?: string;
  created_at: string;
  completed_at?: string;
  runtime_seconds?: number;
  total_files: number;
  total_classes: number;
  total_methods: number;
  total_smells: number;
  affected_components_count: number;
  smell_counts: Record<string, number>;
  total_tdp_hours: number;
  total_tdi_hours: number;
  total_debt_hours: number;
  total_estimated_cost_usd: number;
  average_risk_score: number;
  components?: ComponentDetail[];
  evolution?: EvolutionRecord[];
}

export interface AnalysisRequest {
  repository_url?: string;
  local_path?: string;
  branch?: string;
  commit_hash?: string;
  mode: 'quick' | 'full' | 'research';
  max_history_commits?: number;
}

export interface ModelMetadata {
  model_id: string;
  smell_name: string;
  granularity: 'class' | 'method';
  model_type: string;
  resampling_strategy: string;
  evaluation_metrics: {
    precision: number;
    recall: number;
    f1: number;
    mcc: number;
    roc_auc: number;
    pr_auc: number;
  };
  tuned_hyperparameters?: Record<string, any>;
  optimal_threshold?: number;
  feature_names?: string[];
}
