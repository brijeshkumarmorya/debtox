import React, { useEffect, useState } from 'react';
import { FolderGit2, ArrowRight, Clock, ShieldAlert, CheckCircle2, Play, RefreshCw } from 'lucide-react';
import { listAnalyses, getAnalysis } from '../services/api';
import { AnalysisSummary, NavigationTab } from '../types';

interface ProjectsPageProps {
  onSelectProject: (analysis: AnalysisSummary) => void;
  onNavigate: (tab: NavigationTab) => void;
}

export const ProjectsPage: React.FC<ProjectsPageProps> = ({ 
  onSelectProject, 
  onNavigate 
}) => {
  const [projects, setProjects] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    loadProjects();
  }, []);

  const loadProjects = async () => {
    setLoading(true);
    try {
      const list = await listAnalyses();
      setProjects(list || []);
    } catch {
      setProjects([]);
    } finally {
      setLoading(false);
    }
  };

  const handleOpen = async (id: string) => {
    try {
      const full = await getAnalysis(id);
      onSelectProject(full);
      onNavigate('issues');
    } catch {
      // Ignore
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8 sm:py-12 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-debtox-border pb-6">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold text-debtox-ink font-sans">
            Analyzed Projects
          </h1>
          <p className="text-xs text-debtox-ink-muted mt-1">
            History of scanned repositories and their architectural health audits.
          </p>
        </div>

        <button
          onClick={loadProjects}
          className="p-2 rounded-xl bg-surface border border-debtox-border hover:bg-surface-secondary text-debtox-ink-muted hover:text-debtox-ink transition cursor-pointer"
          title="Refresh list"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {/* Projects List */}
      {loading ? (
        <div className="py-16 text-center text-xs text-debtox-ink-muted font-mono">
          Loading past analyses...
        </div>
      ) : projects.length === 0 ? (
        <div className="p-12 text-center bg-surface rounded-3xl border border-debtox-border shadow-subtle space-y-4">
          <div className="w-12 h-12 rounded-2xl bg-surface-secondary flex items-center justify-center mx-auto text-debtox-ink-muted">
            <FolderGit2 className="w-6 h-6" />
          </div>
          <div className="space-y-1">
            <h3 className="text-base font-bold text-debtox-ink">No projects analyzed yet</h3>
            <p className="text-xs text-debtox-ink-muted max-w-sm mx-auto">
              Scan your first Java repository to detect code smells and estimate technical debt.
            </p>
          </div>
          <button
            onClick={() => onNavigate('analyze')}
            className="px-5 py-2.5 rounded-xl font-semibold text-xs bg-debtox-primary hover:bg-debtox-primary-hover text-white shadow-subtle inline-flex items-center space-x-1.5 transition cursor-pointer"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>Analyze a Repository</span>
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          {projects.map((item) => (
            <div
              key={item.analysis_id}
              onClick={() => handleOpen(item.analysis_id)}
              className="p-5 rounded-2xl bg-surface hover:bg-surface-secondary/40 border border-debtox-border transition cursor-pointer flex flex-col sm:flex-row sm:items-center justify-between gap-4 group shadow-subtle"
            >
              <div className="space-y-1 min-w-0">
                <div className="flex items-center space-x-2">
                  <span className="font-bold text-debtox-ink group-hover:text-debtox-primary transition font-sans text-sm truncate">
                    {item.repository_name}
                  </span>
                  <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                    item.status === 'completed'
                      ? 'bg-risk-low-bg text-risk-low border border-risk-low/30'
                      : 'bg-risk-high-bg text-risk-high'
                  }`}>
                    {item.status}
                  </span>
                </div>
                <p className="text-xs text-debtox-ink-muted font-mono truncate">
                  {item.repository_path_or_url}
                </p>
                <div className="text-[11px] text-debtox-ink-muted flex items-center space-x-3 pt-0.5 font-mono">
                  <span>{new Date(item.created_at).toLocaleDateString()}</span>
                  <span>•</span>
                  <span>{item.total_files || 0} files</span>
                </div>
              </div>

              <div className="flex sm:flex-col items-center sm:items-end justify-between sm:justify-center border-t sm:border-t-0 pt-2 sm:pt-0 border-debtox-border-subtle gap-2 flex-shrink-0">
                <div className="text-xs font-mono text-debtox-ink-muted">
                  Issues: <b className="text-debtox-ink font-bold">{item.total_smells || 0}</b>
                </div>
                <span className="text-xs font-semibold text-debtox-primary flex items-center space-x-1 group-hover:translate-x-0.5 transition-transform">
                  <span>Open Analysis</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
