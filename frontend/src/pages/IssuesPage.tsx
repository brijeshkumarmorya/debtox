import React, { useState } from 'react';
import { 
  Search, 
  Filter, 
  ShieldAlert, 
  ChevronRight, 
  FileCode2, 
  ArrowUpDown,
  Download,
  AlertCircle
} from 'lucide-react';
import { AnalysisSummary, ComponentDetail, SupportedSmell } from '../types';

interface IssuesPageProps {
  analysis: AnalysisSummary;
  onSelectComponent: (component: ComponentDetail) => void;
}

export const IssuesPage: React.FC<IssuesPageProps> = ({ 
  analysis, 
  onSelectComponent 
}) => {
  const [selectedSmell, setSelectedSmell] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [sortBy, setSortBy] = useState<'risk' | 'effort' | 'name'>('risk');

  const components = analysis.components || [];
  const smellyComponents = components.filter(c => c.detected_smells.length > 0);

  const supportedSmells: SupportedSmell[] = [
    'God Class',
    'Data Class',
    'Long Method',
    'Feature Envy'
  ];

  const filtered = smellyComponents.filter(c => {
    if (selectedSmell !== 'all' && !c.detected_smells.includes(selectedSmell)) return false;

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      const matchFile = c.file_path.toLowerCase().includes(q);
      const matchClass = c.class_name.toLowerCase().includes(q);
      const matchMethod = (c.method_name || '').toLowerCase().includes(q);
      if (!matchFile && !matchClass && !matchMethod) return false;
    }
    return true;
  }).sort((a, b) => {
    if (sortBy === 'risk') return b.risk_score - a.risk_score;
    if (sortBy === 'effort') return b.total_debt_hours - a.total_debt_hours;
    return a.class_name.localeCompare(b.class_name);
  });

  const getRiskBadge = (score: number) => {
    if (score >= 65) return { label: 'High', style: 'bg-risk-high-bg text-risk-high border-risk-high/30' };
    if (score >= 35) return { label: 'Medium', style: 'bg-risk-medium-bg text-risk-medium border-risk-medium/30' };
    return { label: 'Low', style: 'bg-risk-low-bg text-risk-low border-risk-low/30' };
  };

  const formatEffort = (hours: number) => {
    if (hours >= 8) {
      const days = (hours / 8).toFixed(1);
      return `~${days}d (${hours}h)`;
    }
    return `~${hours}h`;
  };

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-8 sm:py-12 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-debtox-border pb-6">
        <div>
          <span className="text-xs font-mono uppercase tracking-wider text-debtox-primary font-semibold">
            {analysis.repository_name}
          </span>
          <h1 className="text-2xl sm:text-3xl font-bold text-debtox-ink font-sans mt-0.5">
            Detected Code Issues
          </h1>
          <p className="text-xs text-debtox-ink-muted mt-1">
            {smellyComponents.length} components have architectural maintainability concerns.
          </p>
        </div>

        <div className="text-right">
          <div className="text-xs font-mono text-debtox-ink-muted">Total Remediation Effort</div>
          <div className="text-xl font-bold text-debtox-ink font-mono mt-0.5">
            {analysis.total_debt_hours} hours
          </div>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="bg-surface rounded-2xl border border-debtox-border p-4 shadow-subtle space-y-3">
        {/* Smell Type Tabs */}
        <div className="flex items-center space-x-1.5 overflow-x-auto pb-1 scrollbar-none text-xs">
          <button
            onClick={() => setSelectedSmell('all')}
            className={`px-3 py-1.5 rounded-lg font-medium transition whitespace-nowrap cursor-pointer ${
              selectedSmell === 'all'
                ? 'bg-debtox-primary text-white shadow-subtle'
                : 'bg-surface-secondary text-debtox-ink-muted hover:text-debtox-ink'
            }`}
          >
            All Issues ({smellyComponents.length})
          </button>
          {supportedSmells.map((smell) => {
            const count = analysis.smell_counts?.[smell] || 0;
            const isSelected = selectedSmell === smell;
            return (
              <button
                key={smell}
                onClick={() => setSelectedSmell(smell)}
                className={`px-3 py-1.5 rounded-lg font-medium transition whitespace-nowrap cursor-pointer flex items-center space-x-1.5 ${
                  isSelected
                    ? 'bg-debtox-primary text-white shadow-subtle'
                    : 'bg-surface-secondary text-debtox-ink-muted hover:text-debtox-ink'
                }`}
              >
                <span>{smell}</span>
                <span className={`text-[10px] px-1.5 py-0.2 rounded-full font-mono ${
                  isSelected ? 'bg-white/25 text-white' : 'bg-surface text-debtox-ink'
                }`}>
                  {count}
                </span>
              </button>
            );
          })}
        </div>

        {/* Search & Sort Controls */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-2 border-t border-debtox-border-subtle">
          <div className="relative flex-1 max-w-sm">
            <Search className="w-4 h-4 text-debtox-ink-muted absolute left-3 top-2.5" />
            <input
              type="text"
              placeholder="Search component or file..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-surface-recessed border border-debtox-border rounded-xl pl-9 pr-3 py-2 text-xs text-debtox-ink placeholder-debtox-ink-subtle focus:outline-none focus:border-debtox-primary font-mono"
            />
          </div>

          <div className="flex items-center space-x-2 text-xs text-debtox-ink-muted font-medium">
            <span>Sort:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="bg-surface border border-debtox-border rounded-lg px-2.5 py-1.5 text-xs text-debtox-ink focus:outline-none"
            >
              <option value="risk">Highest Risk</option>
              <option value="effort">Remediation Effort</option>
              <option value="name">Component Name (A-Z)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Issues Table */}
      <div className="bg-surface rounded-2xl border border-debtox-border overflow-hidden shadow-subtle">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-debtox-ink">
            <thead className="bg-surface-secondary/70 border-b border-debtox-border text-[11px] font-semibold text-debtox-ink-muted uppercase tracking-wider">
              <tr>
                <th className="py-3.5 px-4">Component</th>
                <th className="py-3.5 px-3">Smell</th>
                <th className="py-3.5 px-3">Risk Level</th>
                <th className="py-3.5 px-3 text-right">Confidence</th>
                <th className="py-3.5 px-3 text-right">Est. Effort</th>
                <th className="py-3.5 px-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-debtox-border-subtle">
              {filtered.length === 0 ? (
                <tr>
                  <td colSpan={6} className="py-12 text-center text-debtox-ink-muted text-xs">
                    No components match your search or filter criteria.
                  </td>
                </tr>
              ) : (
                filtered.map((comp) => {
                  const risk = getRiskBadge(comp.risk_score);
                  const primarySmell = comp.detected_smells[0] || 'Code Smell';
                  const maxConfidence = comp.smell_predictions.length > 0
                    ? Math.max(...comp.smell_predictions.map(p => p.probability))
                    : 0.8;

                  return (
                    <tr
                      key={comp.entity_id}
                      onClick={() => onSelectComponent(comp)}
                      className="hover:bg-surface-secondary/40 transition cursor-pointer group"
                    >
                      <td className="py-3 px-4">
                        <div className="font-semibold text-debtox-ink group-hover:text-debtox-primary transition font-sans text-xs">
                          {comp.class_name}{comp.method_name ? `.${comp.method_name}()` : ''}
                        </div>
                        <div className="text-[11px] text-debtox-ink-muted font-mono truncate max-w-sm">
                          {comp.file_path}
                        </div>
                      </td>

                      <td className="py-3 px-3">
                        <span className="font-medium text-debtox-ink">
                          {primarySmell}
                        </span>
                      </td>

                      <td className="py-3 px-3">
                        <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${risk.style}`}>
                          {risk.label}
                        </span>
                      </td>

                      <td className="py-3 px-3 text-right font-mono text-debtox-ink-muted">
                        {(maxConfidence * 100).toFixed(0)}%
                      </td>

                      <td className="py-3 px-3 text-right font-mono font-medium text-debtox-ink">
                        {formatEffort(comp.total_debt_hours)}
                      </td>

                      <td className="py-3 px-4 text-right">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onSelectComponent(comp);
                          }}
                          className="px-2.5 py-1 rounded-lg text-xs font-medium text-debtox-primary hover:bg-debtox-primary-light transition"
                        >
                          Diagnose →
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
