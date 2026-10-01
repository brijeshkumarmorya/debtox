import { AnalysisRequest, AnalysisSummary, ComponentDetail, ModelMetadata } from '../types';

const API_BASE = '/api/v1';

export async function startAnalysis(req: AnalysisRequest): Promise<AnalysisSummary> {
  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to start analysis');
  }
  return res.json();
}

export async function getAnalysis(analysisId: string): Promise<AnalysisSummary> {
  const res = await fetch(`${API_BASE}/analyses/${analysisId}`);
  if (!res.ok) {
    throw new Error('Analysis not found');
  }
  return res.json();
}

export async function getAnalysisSummaryOnly(analysisId: string): Promise<Partial<AnalysisSummary>> {
  const res = await fetch(`${API_BASE}/analyses/${analysisId}/summary`);
  if (!res.ok) {
    throw new Error('Analysis summary not found');
  }
  return res.json();
}

export async function getAnalysisComponents(
  analysisId: string,
  params?: { smell?: string; granularity?: string; min_risk?: number }
): Promise<ComponentDetail[]> {
  const query = new URLSearchParams();
  if (params?.smell) query.append('smell', params.smell);
  if (params?.granularity) query.append('granularity', params.granularity);
  if (params?.min_risk !== undefined) query.append('min_risk', params.min_risk.toString());

  const res = await fetch(`${API_BASE}/analyses/${analysisId}/components?${query.toString()}`);
  if (!res.ok) return [];
  return res.json();
}

export async function getAnalysisSmells(analysisId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/analyses/${analysisId}/smells`);
  if (!res.ok) return null;
  return res.json();
}

export async function getAnalysisDebt(analysisId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/analyses/${analysisId}/debt`);
  if (!res.ok) return null;
  return res.json();
}

export async function getAnalysisExplanations(analysisId: string): Promise<any[]> {
  const res = await fetch(`${API_BASE}/analyses/${analysisId}/explanations`);
  if (!res.ok) return [];
  return res.json();
}

export async function listAnalyses(): Promise<any[]> {
  const res = await fetch(`${API_BASE}/analyses`);
  if (!res.ok) return [];
  return res.json();
}

export async function getModels(): Promise<ModelMetadata[]> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) return [];
  return res.json();
}

export async function getHealth(): Promise<{ status: string; service?: string; version?: string; loaded_models_count: number }> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) return { status: 'offline', loaded_models_count: 0 };
    return res.json();
  } catch {
    return { status: 'offline', loaded_models_count: 0 };
  }
}
