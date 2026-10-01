import React, { useState } from 'react';
import { 
  Play, 
  CheckCircle2, 
  ArrowRight, 
  FolderGit2, 
  GitBranch, 
  Loader2, 
  AlertCircle, 
  ChevronDown, 
  ChevronUp,
  ShieldAlert,
  Sparkles
} from 'lucide-react';
import { startAnalysis, getAnalysis } from '../services/api';
import { AnalysisSummary, NavigationTab } from '../types';

interface AnalyzePageProps {
  onAnalysisLoaded: (analysis: AnalysisSummary) => void;
  onNavigate: (tab: NavigationTab) => void;
}

export const AnalyzePage: React.FC<AnalyzePageProps> = ({ 
  onAnalysisLoaded, 
  onNavigate 
}) => {
  const [sourceType, setSourceType] = useState<'local' | 'github'>('local');
  const [repoPath, setRepoPath] = useState<string>('data/sample_repo');
  const [githubUrl, setGithubUrl] = useState<string>('');
  const [branch, setBranch] = useState<string>('');
  
  // Advanced options toggle
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [mode, setMode] = useState<'quick' | 'full' | 'research'>('research');
  const [maxCommits, setMaxCommits] = useState<number>(100);

  // Execution states
  const [status, setStatus] = useState<'idle' | 'running' | 'completed'>('idle');
  const [currentJob, setCurrentJob] = useState<AnalysisSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleStartAnalysis = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    const target = sourceType === 'local' ? repoPath.trim() : githubUrl.trim();
    if (!target) return;

    setStatus('running');

    try {
      const isUrl = sourceType === 'github' || target.startsWith('http') || target.startsWith('git@');
      const req = {
        repository_url: isUrl ? target : undefined,
        local_path: !isUrl ? target : undefined,
        branch: branch.trim() || undefined,
        mode,
        max_history_commits: maxCommits,
      };

      const job = await startAnalysis(req);
      setCurrentJob(job);

      // Poll until finished
      const interval = setInterval(async () => {
        try {
          const updated = await getAnalysis(job.analysis_id);
          setCurrentJob(updated);

          if (updated.status === 'completed') {
            clearInterval(interval);
            setStatus('completed');
            onAnalysisLoaded(updated);
          } else if (updated.status === 'failed') {
            clearInterval(interval);
            setStatus('idle');
            setError(updated.error_message || 'Analysis failed. Please check the path and repository structure.');
          }
        } catch (err: any) {
          clearInterval(interval);
          setStatus('idle');
          setError(err.message || 'Error communicating with analysis service');
        }
      }, 1500);

    } catch (err: any) {
      setStatus('idle');
      setError(err.message || 'Failed to submit analysis');
    }
  };

  // Determine stage checklist status
  const getStageState = (stageIndex: number) => {
    const progress = currentJob?.progress_percentage || 0;
    const stageThresholds = [15, 35, 60, 85, 95];

    if (progress >= stageThresholds[stageIndex]) {
      return 'done';
    } else if (stageIndex === 0 || progress >= stageThresholds[stageIndex - 1]) {
      return 'active';
    }
    return 'pending';
  };

  const checklist = [
    'Repository loaded & verified',
    'Parsing Java Abstract Syntax Tree',
    'Extracting software complexity metrics',
    'Detecting code smells (God Class, Data Class, etc.)',
    'Estimating technical debt & remediation effort',
  ];

  return (
    <div className="max-w-2xl mx-auto px-4 sm:px-6 py-10 sm:py-16 space-y-8">
      {/* STEP 1: START ANALYSIS (Idle State) */}
      {status === 'idle' && (
        <div className="bg-surface rounded-3xl border border-debtox-border p-6 sm:p-10 shadow-subtle space-y-8">
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold text-debtox-ink font-sans">
              Analyze a Repository
            </h1>
            <p className="text-xs sm:text-sm text-debtox-ink-muted mt-1.5 leading-relaxed">
              DebtOx scans your Java project for architectural code smells, estimates remediation effort, and tells you what to fix first.
            </p>
          </div>

          <form onSubmit={handleStartAnalysis} className="space-y-6">
            {/* Source Type Selector */}
            <div className="grid grid-cols-2 gap-2 p-1 rounded-xl bg-surface-secondary">
              <button
                type="button"
                onClick={() => setSourceType('local')}
                className={`py-2 text-xs font-semibold rounded-lg transition cursor-pointer ${
                  sourceType === 'local'
                    ? 'bg-surface text-debtox-ink shadow-subtle'
                    : 'text-debtox-ink-muted hover:text-debtox-ink'
                }`}
              >
                Local / Existing Folder
              </button>
              <button
                type="button"
                onClick={() => setSourceType('github')}
                className={`py-2 text-xs font-semibold rounded-lg transition cursor-pointer ${
                  sourceType === 'github'
                    ? 'bg-surface text-debtox-ink shadow-subtle'
                    : 'text-debtox-ink-muted hover:text-debtox-ink'
                }`}
              >
                GitHub Repository
              </button>
            </div>

            {/* Input by Source Type */}
            {sourceType === 'local' ? (
              <div className="space-y-2">
                <label className="block text-xs font-medium text-debtox-ink">
                  Project Directory Path
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={repoPath}
                    onChange={(e) => setRepoPath(e.target.value)}
                    placeholder="e.g. data/sample_repo or /path/to/java/project"
                    className="flex-1 bg-surface-recessed border border-debtox-border rounded-xl px-4 py-3 text-xs sm:text-sm text-debtox-ink placeholder-debtox-ink-subtle focus:outline-none focus:border-debtox-primary font-mono"
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setRepoPath('data/sample_repo')}
                    className="px-3 py-2 text-xs font-medium bg-surface-secondary hover:bg-surface-secondary/80 text-debtox-ink rounded-xl border border-debtox-border transition cursor-pointer whitespace-nowrap"
                  >
                    Use Sample Repo
                  </button>
                </div>
                <p className="text-[11px] text-debtox-ink-muted">
                  Path to a folder containing Java source code files.
                </p>
              </div>
            ) : (
              <div className="space-y-2">
                <label className="block text-xs font-medium text-debtox-ink">
                  Git Clone URL
                </label>
                <input
                  type="text"
                  value={githubUrl}
                  onChange={(e) => setGithubUrl(e.target.value)}
                  placeholder="https://github.com/organization/repository.git"
                  className="w-full bg-surface-recessed border border-debtox-border rounded-xl px-4 py-3 text-xs sm:text-sm text-debtox-ink placeholder-debtox-ink-subtle focus:outline-none focus:border-debtox-primary font-mono"
                  required
                />
                <div className="pt-2">
                  <label className="block text-[11px] font-medium text-debtox-ink-muted mb-1">
                    Branch (Optional)
                  </label>
                  <input
                    type="text"
                    value={branch}
                    onChange={(e) => setBranch(e.target.value)}
                    placeholder="main / master"
                    className="w-full bg-surface-recessed border border-debtox-border rounded-xl px-3 py-2 text-xs text-debtox-ink font-mono"
                  />
                </div>
              </div>
            )}

            {/* Error Message */}
            {error && (
              <div className="p-3.5 rounded-xl bg-risk-high-bg border border-risk-high/30 text-risk-high text-xs flex items-start space-x-2">
                <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                <span>{error}</span>
              </div>
            )}

            {/* Collapsible Advanced Settings */}
            <div className="border-t border-debtox-border-subtle pt-3">
              <button
                type="button"
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="flex items-center space-x-1.5 text-xs text-debtox-ink-muted hover:text-debtox-ink transition font-medium cursor-pointer"
              >
                {showAdvanced ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                <span>Advanced options</span>
              </button>

              {showAdvanced && (
                <div className="mt-3 p-4 rounded-xl bg-surface-secondary/50 border border-debtox-border space-y-3 text-xs">
                  <div>
                    <label className="block font-medium text-debtox-ink mb-1">Analysis Mode</label>
                    <select
                      value={mode}
                      onChange={(e) => setMode(e.target.value as any)}
                      className="w-full bg-surface border border-debtox-border rounded-lg px-3 py-2 text-xs text-debtox-ink"
                    >
                      <option value="research">Standard (Code smells + Technical debt + Explanations)</option>
                      <option value="full">Full (Code smells + Git history churn)</option>
                      <option value="quick">Quick (Code snapshot only)</option>
                    </select>
                  </div>
                  <div>
                    <label className="block font-medium text-debtox-ink mb-1">Commit History Limit</label>
                    <input
                      type="number"
                      min={10}
                      max={500}
                      value={maxCommits}
                      onChange={(e) => setMaxCommits(Number(e.target.value))}
                      className="w-full bg-surface border border-debtox-border rounded-lg px-3 py-2 text-xs text-debtox-ink font-mono"
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Submit Action */}
            <button
              type="submit"
              className="w-full py-3.5 px-6 rounded-xl font-semibold text-sm bg-debtox-primary hover:bg-debtox-primary-hover text-white shadow-subtle flex items-center justify-center space-x-2 transition cursor-pointer"
            >
              <Play className="w-4 h-4 fill-current" />
              <span>Start Analysis</span>
            </button>
          </form>
        </div>
      )}

      {/* STEP 2: CLEAN PROGRESS SCREEN (Running State) */}
      {status === 'running' && (
        <div className="bg-surface rounded-3xl border border-debtox-border p-8 sm:p-10 shadow-subtle space-y-8 text-center sm:text-left">
          <div className="space-y-2">
            <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-surface-secondary border border-debtox-border text-xs text-debtox-ink">
              <Loader2 className="w-3.5 h-3.5 animate-spin text-debtox-primary" />
              <span>Diagnostic in progress</span>
            </div>
            <h2 className="text-2xl font-bold text-debtox-ink">
              Analyzing repository...
            </h2>
            <p className="text-xs text-debtox-ink-muted">
              {currentJob?.current_stage || 'Preparing analysis pipeline...'}
            </p>
          </div>

          {/* Simple Clean Progress Bar */}
          <div className="space-y-1.5">
            <div className="w-full bg-surface-secondary rounded-full h-2 overflow-hidden">
              <div 
                className="bg-debtox-primary h-2 rounded-full transition-all duration-500"
                style={{ width: `${Math.max(5, currentJob?.progress_percentage || 0)}%` }}
              />
            </div>
            <div className="flex justify-between text-[11px] text-debtox-ink-muted font-mono">
              <span>Progress</span>
              <span>{currentJob?.progress_percentage || 0}%</span>
            </div>
          </div>

          {/* Progressive Checklist */}
          <div className="space-y-3 pt-2 text-xs text-left">
            {checklist.map((item, idx) => {
              const itemState = getStageState(idx);
              return (
                <div key={idx} className="flex items-center space-x-3">
                  {itemState === 'done' ? (
                    <span className="w-4 h-4 rounded-full bg-debtox-primary text-white flex items-center justify-center text-[10px] font-bold">
                      ✓
                    </span>
                  ) : itemState === 'active' ? (
                    <span className="w-4 h-4 text-debtox-primary font-bold text-sm flex items-center justify-center animate-pulse">
                      →
                    </span>
                  ) : (
                    <span className="w-4 h-4 rounded-full border border-debtox-border text-transparent flex items-center justify-center text-[10px]">
                      •
                    </span>
                  )}
                  <span className={itemState === 'done' ? 'text-debtox-ink font-medium' : itemState === 'active' ? 'text-debtox-primary font-semibold' : 'text-debtox-ink-subtle'}>
                    {item}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* STEP 3: RESULT SUMMARY (Completed State) */}
      {status === 'completed' && currentJob && (
        <div className="bg-surface rounded-3xl border border-debtox-border p-8 sm:p-10 shadow-subtle space-y-8 animate-in fade-in duration-300">
          <div className="space-y-2">
            <div className="inline-flex items-center space-x-1.5 text-xs font-semibold text-debtox-primary">
              <CheckCircle2 className="w-4 h-4" />
              <span>Analysis Complete</span>
            </div>
            <h2 className="text-2xl sm:text-3xl font-bold text-debtox-ink font-sans">
              Your project has {currentJob.total_smells} issues.
            </h2>
            <p className="text-xs text-debtox-ink-muted">
              {currentJob.affected_components_count} components require architectural attention in <b className="text-debtox-ink">{currentJob.repository_name}</b>.
            </p>
          </div>

          {/* Smell Count Breakdown Card */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-4 rounded-2xl bg-surface-secondary/70 border border-debtox-border">
            <div className="p-3 bg-surface rounded-xl border border-debtox-border/60">
              <div className="text-[11px] font-medium text-debtox-ink-muted">God Class</div>
              <div className="text-xl font-bold text-debtox-ink mt-0.5">
                {currentJob.smell_counts?.['God Class'] || 0}
              </div>
            </div>
            <div className="p-3 bg-surface rounded-xl border border-debtox-border/60">
              <div className="text-[11px] font-medium text-debtox-ink-muted">Data Class</div>
              <div className="text-xl font-bold text-debtox-ink mt-0.5">
                {currentJob.smell_counts?.['Data Class'] || 0}
              </div>
            </div>
            <div className="p-3 bg-surface rounded-xl border border-debtox-border/60">
              <div className="text-[11px] font-medium text-debtox-ink-muted">Long Method</div>
              <div className="text-xl font-bold text-debtox-ink mt-0.5">
                {currentJob.smell_counts?.['Long Method'] || 0}
              </div>
            </div>
            <div className="p-3 bg-surface rounded-xl border border-debtox-border/60">
              <div className="text-[11px] font-medium text-debtox-ink-muted">Feature Envy</div>
              <div className="text-xl font-bold text-debtox-ink mt-0.5">
                {currentJob.smell_counts?.['Feature Envy'] || 0}
              </div>
            </div>
          </div>

          {/* Obvious Primary CTA */}
          <div className="flex flex-col sm:flex-row items-center gap-3 pt-2">
            <button
              onClick={() => onNavigate('issues')}
              className="w-full sm:w-auto px-6 py-3.5 rounded-xl font-semibold text-sm bg-debtox-primary hover:bg-debtox-primary-hover text-white shadow-subtle flex items-center justify-center space-x-2 transition cursor-pointer"
            >
              <span>View Issues</span>
              <ArrowRight className="w-4 h-4" />
            </button>

            <button
              onClick={() => setStatus('idle')}
              className="w-full sm:w-auto px-4 py-3 rounded-xl text-xs font-medium text-debtox-ink-muted hover:text-debtox-ink transition text-center cursor-pointer"
            >
              Analyze Another Repository
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
