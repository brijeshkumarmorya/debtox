import React from 'react';
import { ArrowRight, Play, CheckCircle2, ChevronRight, Sparkles, FolderGit2 } from 'lucide-react';
import { NavigationTab, AnalysisSummary } from '../types';

interface HomePageProps {
  onNavigate: (tab: NavigationTab) => void;
  onLoadDemo: () => void;
  currentAnalysis: AnalysisSummary | null;
  loadingDemo: boolean;
}

export const HomePage: React.FC<HomePageProps> = ({
  onNavigate,
  onLoadDemo,
  currentAnalysis,
  loadingDemo
}) => {
  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-12 sm:py-20 space-y-16">
      {/* Minimalist Hero Section */}
      <div className="text-center space-y-6">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-surface-secondary border border-debtox-border text-xs text-debtox-ink font-medium">
          <span className="w-2 h-2 rounded-full bg-debtox-primary" />
          <span>Software Health & Technical Debt Diagnosis</span>
        </div>

        <div className="space-y-3">
          <h1 className="text-3xl sm:text-5xl font-bold tracking-tight text-debtox-ink font-sans leading-tight">
            Find code smells.<br />
            Understand technical debt.<br />
            <span className="text-debtox-primary">Know what to fix first.</span>
          </h1>

          <p className="text-base sm:text-lg text-debtox-ink-muted max-w-xl mx-auto leading-relaxed">
            DebtOx inspects Java code structure and Git history to highlight maintainability bottlenecks before they slow down your team.
          </p>
        </div>

        {/* Primary Action Buttons */}
        <div className="flex flex-wrap items-center justify-center gap-3 pt-2">
          <button
            onClick={() => onNavigate('analyze')}
            className="px-6 py-3 rounded-xl font-semibold text-sm bg-debtox-primary hover:bg-debtox-primary-hover text-white shadow-subtle flex items-center space-x-2 transition cursor-pointer"
          >
            <span>Analyze a Repository</span>
            <ArrowRight className="w-4 h-4" />
          </button>

          <button
            onClick={onLoadDemo}
            disabled={loadingDemo}
            className="px-6 py-3 rounded-xl font-medium text-sm bg-surface hover:bg-surface-secondary/60 text-debtox-ink border border-debtox-border transition shadow-subtle flex items-center space-x-2 cursor-pointer disabled:opacity-50"
          >
            <FolderGit2 className="w-4 h-4 text-debtox-ink-muted" />
            <span>{loadingDemo ? 'Loading Sample...' : 'Explore Demo Project'}</span>
          </button>
        </div>

        {/* Active Project Quick Card (if one is loaded) */}
        {currentAnalysis && (
          <div className="pt-4">
            <div 
              onClick={() => onNavigate('issues')}
              className="inline-flex items-center space-x-3 p-3 px-5 rounded-2xl bg-surface border border-debtox-border hover:border-debtox-primary text-xs cursor-pointer shadow-subtle transition text-left"
            >
              <div className="w-2 h-2 rounded-full bg-debtox-primary animate-pulse" />
              <div>
                <span className="font-semibold text-debtox-ink">{currentAnalysis.repository_name}</span>
                <span className="text-debtox-ink-muted mx-1.5">•</span>
                <span className="text-debtox-ink-muted">{currentAnalysis.total_smells} issues detected</span>
              </div>
              <span className="text-debtox-primary font-semibold flex items-center ml-2">
                <span>View Issues</span>
                <ChevronRight className="w-3.5 h-3.5" />
              </span>
            </div>
          </div>
        )}
      </div>

      {/* 3-Step Clear Guided Explanation */}
      <div className="border-t border-debtox-border pt-12">
        <h2 className="text-center text-xs font-semibold uppercase tracking-wider text-debtox-ink-muted mb-8">
          How DebtOx Guides You
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Step 1 */}
          <div className="p-6 rounded-2xl bg-surface border border-debtox-border shadow-subtle space-y-2">
            <span className="text-xs font-mono font-bold text-debtox-primary">01</span>
            <h3 className="text-base font-bold text-debtox-ink">1. Analyze</h3>
            <p className="text-xs text-debtox-ink-muted leading-relaxed">
              Connect a GitHub URL or local Java directory. DebtOx reads your code structure and commit history automatically.
            </p>
          </div>

          {/* Step 2 */}
          <div className="p-6 rounded-2xl bg-surface border border-debtox-border shadow-subtle space-y-2">
            <span className="text-xs font-mono font-bold text-debtox-primary">02</span>
            <h3 className="text-base font-bold text-debtox-ink">2. Understand</h3>
            <p className="text-xs text-debtox-ink-muted leading-relaxed">
              Clear plain-language explanations show exactly what was detected (God Class, Data Class, Long Method, Feature Envy) and why.
            </p>
          </div>

          {/* Step 3 */}
          <div className="p-6 rounded-2xl bg-surface border border-debtox-border shadow-subtle space-y-2">
            <span className="text-xs font-mono font-bold text-debtox-primary">03</span>
            <h3 className="text-base font-bold text-debtox-ink">3. Prioritize</h3>
            <p className="text-xs text-debtox-ink-muted leading-relaxed">
              See estimated remediation effort and ongoing maintenance drag so your team fixes high-ROI problems first.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
