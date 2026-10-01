import React, { useState } from 'react';
import { 
  X, 
  ChevronDown, 
  ChevronUp, 
  AlertCircle, 
  CheckCircle2, 
  Clock, 
  Scale, 
  Sparkles, 
  FileCode2, 
  Lightbulb, 
  HelpCircle,
  GitCommit
} from 'lucide-react';
import { ComponentDetail, SmellPrediction } from '../types';

interface ComponentModalProps {
  component: ComponentDetail | null;
  onClose: () => void;
}

export const ComponentModal: React.FC<ComponentModalProps> = ({ 
  component, 
  onClose 
}) => {
  const [showTechnicalMetrics, setShowTechnicalMetrics] = useState(false);
  const [showShapExplanation, setShowShapExplanation] = useState(false);
  const [showDebtDetails, setShowDebtDetails] = useState(false);

  if (!component) return null;

  const smells = component.smell_predictions.filter(p => p.is_smelly);
  const primarySmell = smells[0] || component.smell_predictions[0];

  const getRiskLabel = (score: number) => {
    if (score >= 65) return { label: 'High Risk', badge: 'bg-risk-high-bg text-risk-high border-risk-high/30' };
    if (score >= 35) return { label: 'Medium Risk', badge: 'bg-risk-medium-bg text-risk-medium border-risk-medium/30' };
    return { label: 'Low Risk', badge: 'bg-risk-low-bg text-risk-low border-risk-low/30' };
  };

  const risk = getRiskLabel(component.risk_score);

  // Friendly explanations for beginner users
  const getSmellDescription = (name: string) => {
    switch (name) {
      case 'God Class':
        return 'This class is doing too much. It has grown into a bloated central controller with multiple distinct responsibilities that should be split into smaller, focused modules.';
      case 'Data Class':
        return 'This class holds fields with getters and setters, but lacks business logic. The behavior that operates on this data is scattered across other classes.';
      case 'Long Method':
        return 'This method contains too much code and convoluted control flow. It should be decomposed into smaller helper methods.';
      case 'Feature Envy':
        return 'This method calls methods and accesses fields of another class more than its own. It should likely be moved to that class.';
      default:
        return 'An architectural maintainability concern was detected in this component.';
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-6 bg-[#242720]/40 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="bg-surface rounded-3xl border border-debtox-border w-full max-w-2xl max-h-[90vh] overflow-y-auto shadow-modal space-y-6 p-6 sm:p-8">
        {/* Header */}
        <div className="flex items-start justify-between pb-4 border-b border-debtox-border gap-4">
          <div className="space-y-1 min-w-0">
            <div className="flex items-center space-x-2">
              <span className={`text-[11px] font-semibold px-2.5 py-0.5 rounded-full border ${risk.badge}`}>
                {risk.label}
              </span>
              <span className="text-[11px] font-mono text-debtox-ink-muted uppercase">
                {component.granularity}
              </span>
            </div>
            <h2 className="text-xl sm:text-2xl font-bold text-debtox-ink font-sans break-all mt-1">
              {component.class_name}{component.method_name ? `.${component.method_name}()` : ''}
            </h2>
            <p className="text-xs text-debtox-ink-muted font-mono flex items-center space-x-1.5 truncate">
              <FileCode2 className="w-3.5 h-3.5 flex-shrink-0" />
              <span>{component.file_path} (Lines {component.start_line}–{component.end_line})</span>
            </p>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-xl text-debtox-ink-muted hover:text-debtox-ink hover:bg-surface-secondary transition cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* PRIMARY: What was detected */}
        <div className="space-y-3">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-debtox-ink-muted">
            What Was Detected
          </h3>

          <div className="p-4 rounded-2xl bg-surface-secondary/60 border border-debtox-border space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-bold text-debtox-ink">
                {primarySmell?.name || 'Code Smell'}
              </span>
              <span className="text-xs font-mono text-debtox-ink-muted">
                Confidence: {((primarySmell?.probability || 0.8) * 100).toFixed(0)}%
              </span>
            </div>
            <p className="text-xs text-debtox-ink leading-relaxed">
              {getSmellDescription(primarySmell?.name || '')}
            </p>
          </div>
        </div>

        {/* ACTION: What should I do? */}
        <div className="space-y-3">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-debtox-ink-muted">
            Recommended Action
          </h3>

          <div className="p-4 rounded-2xl bg-debtox-primary-light border border-debtox-primary/30 space-y-2">
            <div className="flex items-start space-x-2 text-debtox-ink">
              <Lightbulb className="w-4 h-4 text-debtox-primary flex-shrink-0 mt-0.5" />
              <div>
                <span className="text-xs font-bold text-debtox-ink block">
                  Refactor Recommended
                </span>
                <p className="text-xs text-debtox-ink leading-relaxed mt-0.5">
                  {primarySmell?.refactoring_recommendation || component.primary_recommendation || 'Split this component into smaller, focused modules.'}
                </p>
              </div>
            </div>

            <div className="pt-2 border-t border-debtox-primary/20 flex flex-wrap items-center justify-between gap-2 text-[11px] text-debtox-ink-muted font-mono">
              <span>Estimated fix effort: <b className="text-debtox-ink">{component.tdp_hours} hours</b></span>
              <span>Maintenance drag: <b className="text-debtox-ink">{component.tdi_hours || 0} hrs/cycle</b></span>
            </div>
          </div>
        </div>

        {/* PROGRESSIVE DISCLOSURE: Expandable Advanced Sections */}
        <div className="space-y-2 border-t border-debtox-border pt-4 text-xs">
          {/* 1. Why did DebtOx detect this? (TreeSHAP Explanation) */}
          <div className="rounded-xl border border-debtox-border overflow-hidden bg-surface">
            <button
              onClick={() => setShowShapExplanation(!showShapExplanation)}
              className="w-full p-3.5 flex items-center justify-between text-left font-medium text-debtox-ink hover:bg-surface-secondary/40 transition cursor-pointer"
            >
              <span className="flex items-center space-x-2">
                <Sparkles className="w-4 h-4 text-debtox-primary" />
                <span>Why did DebtOx detect this?</span>
              </span>
              {showShapExplanation ? <ChevronUp className="w-4 h-4 text-debtox-ink-muted" /> : <ChevronDown className="w-4 h-4 text-debtox-ink-muted" />}
            </button>

            {showShapExplanation && (
              <div className="p-4 pt-1 border-t border-debtox-border bg-surface-secondary/30 space-y-2.5">
                <p className="text-xs text-debtox-ink-muted">
                  The machine learning model evaluated this component against 107,554 real Java samples. These factors contributed most to the prediction:
                </p>

                {component.shap_contributions && component.shap_contributions.length > 0 ? (
                  <div className="space-y-2 pt-1">
                    {component.shap_contributions.map((s) => (
                      <div key={s.feature_name} className="p-2.5 rounded-lg bg-surface border border-debtox-border text-xs space-y-1">
                        <div className="flex justify-between font-mono">
                          <span className="font-semibold text-debtox-ink">{s.feature_name} = {s.feature_value}</span>
                          <span className={s.impact === 'increases_risk' ? 'text-risk-high font-semibold' : 'text-debtox-primary font-semibold'}>
                            {s.impact === 'increases_risk' ? 'Increases risk' : 'Reduces risk'}
                          </span>
                        </div>
                        <p className="text-[11px] text-debtox-ink-muted">{s.narrative}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-debtox-ink-muted italic">
                    Class size (SLOC) and method complexity were the primary contributors.
                  </p>
                )}
              </div>
            )}
          </div>

          {/* 2. Technical Metrics */}
          <div className="rounded-xl border border-debtox-border overflow-hidden bg-surface">
            <button
              onClick={() => setShowTechnicalMetrics(!showTechnicalMetrics)}
              className="w-full p-3.5 flex items-center justify-between text-left font-medium text-debtox-ink hover:bg-surface-secondary/40 transition cursor-pointer"
            >
              <span className="flex items-center space-x-2">
                <FileCode2 className="w-4 h-4 text-debtox-ink-muted" />
                <span>Technical Code Metrics</span>
              </span>
              {showTechnicalMetrics ? <ChevronUp className="w-4 h-4 text-debtox-ink-muted" /> : <ChevronDown className="w-4 h-4 text-debtox-ink-muted" />}
            </button>

            {showTechnicalMetrics && (
              <div className="p-4 pt-1 border-t border-debtox-border bg-surface-secondary/30 space-y-3">
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 font-mono text-xs">
                  {Object.entries(component.metrics || {}).map(([key, val]) => {
                    if (['entity_id', 'file_path', 'class_name', 'method_name', 'granularity'].includes(key)) return null;
                    return (
                      <div key={key} className="p-2 bg-surface rounded-lg border border-debtox-border">
                        <div className="text-[10px] text-debtox-ink-muted truncate uppercase">{key}</div>
                        <div className="font-bold text-debtox-ink mt-0.5">{String(val)}</div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          {/* 3. Technical Debt Details */}
          <div className="rounded-xl border border-debtox-border overflow-hidden bg-surface">
            <button
              onClick={() => setShowDebtDetails(!showDebtDetails)}
              className="w-full p-3.5 flex items-center justify-between text-left font-medium text-debtox-ink hover:bg-surface-secondary/40 transition cursor-pointer"
            >
              <span className="flex items-center space-x-2">
                <Scale className="w-4 h-4 text-debtox-ink-muted" />
                <span>Technical Debt Calculation</span>
              </span>
              {showDebtDetails ? <ChevronUp className="w-4 h-4 text-debtox-ink-muted" /> : <ChevronDown className="w-4 h-4 text-debtox-ink-muted" />}
            </button>

            {showDebtDetails && (
              <div className="p-4 pt-1 border-t border-debtox-border bg-surface-secondary/30 space-y-2 text-xs leading-relaxed text-debtox-ink-muted">
                <p>
                  <b className="text-debtox-ink">Principal (TDP = {component.tdp_hours}h):</b> Estimated developer effort to refactor the code smell now using standard COCOMO II software effort equations.
                </p>
                <p>
                  <b className="text-debtox-ink">Interest (TDI = {component.tdi_hours || 0}h):</b> Extra friction drag accumulated each time this file is modified in Git commits due to high coupling and complexity.
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="pt-2 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl text-xs font-semibold bg-surface-secondary hover:bg-surface-secondary/80 text-debtox-ink border border-debtox-border transition cursor-pointer"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
