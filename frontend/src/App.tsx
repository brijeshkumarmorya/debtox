import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { HomePage } from './pages/HomePage';
import { AnalyzePage } from './pages/AnalyzePage';
import { IssuesPage } from './pages/IssuesPage';
import { ProjectsPage } from './pages/ProjectsPage';
import { ComponentModal } from './components/ComponentModal';
import { AnalysisSummary, ComponentDetail, NavigationTab } from './types';
import { listAnalyses, getAnalysis, startAnalysis } from './services/api';
import { ArrowRight, FolderGit2, Play } from 'lucide-react';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<NavigationTab>('home');
  const [currentAnalysis, setCurrentAnalysis] = useState<AnalysisSummary | null>(null);
  const [selectedComponent, setSelectedComponent] = useState<ComponentDetail | null>(null);
  const [loadingDemo, setLoadingDemo] = useState<boolean>(false);

  useEffect(() => {
    // Attempt to load the most recent completed analysis on initial load
    listAnalyses().then(async (list) => {
      if (list && list.length > 0) {
        const completed = list.find((a: any) => a.status === 'completed') || list[0];
        const latest = await getAnalysis(completed.analysis_id);
        setCurrentAnalysis(latest);
      }
    }).catch(() => {});
  }, []);

  const handleAnalysisLoaded = (analysis: AnalysisSummary) => {
    setCurrentAnalysis(analysis);
  };

  const handleLoadDemo = async () => {
    setLoadingDemo(true);
    try {
      const list = await listAnalyses();
      const existing = list.find((a: any) => 
        (a.repository_name === 'sample_repo' || a.repository_path_or_url?.includes('sample_repo')) &&
        a.status === 'completed'
      );

      if (existing) {
        const full = await getAnalysis(existing.analysis_id);
        setCurrentAnalysis(full);
        setActiveTab('issues');
      } else {
        // Run analysis on data/sample_repo
        setActiveTab('analyze');
      }
    } catch {
      setActiveTab('analyze');
    } finally {
      setLoadingDemo(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-canvas text-debtox-ink font-sans selection:bg-debtox-primary selection:text-white">
      {/* Minimal Top Navigation */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        currentAnalysis={currentAnalysis}
      />

      {/* Main Content Area */}
      <main className="flex-1 pb-16">
        {activeTab === 'home' && (
          <HomePage
            onNavigate={(tab) => setActiveTab(tab)}
            onLoadDemo={handleLoadDemo}
            currentAnalysis={currentAnalysis}
            loadingDemo={loadingDemo}
          />
        )}

        {activeTab === 'analyze' && (
          <AnalyzePage
            onAnalysisLoaded={handleAnalysisLoaded}
            onNavigate={(tab) => setActiveTab(tab)}
          />
        )}

        {activeTab === 'issues' && (
          currentAnalysis ? (
            <IssuesPage
              analysis={currentAnalysis}
              onSelectComponent={(c) => setSelectedComponent(c)}
            />
          ) : (
            <div className="max-w-md mx-auto px-4 py-20 text-center space-y-4">
              <div className="w-12 h-12 rounded-2xl bg-surface-secondary flex items-center justify-center mx-auto text-debtox-ink-muted">
                <FolderGit2 className="w-6 h-6" />
              </div>
              <h2 className="text-xl font-bold text-debtox-ink">No Active Project</h2>
              <p className="text-xs text-debtox-ink-muted leading-relaxed">
                Analyze a repository first or explore our sample dataset to view detected code issues.
              </p>
              <div className="pt-2 flex flex-col sm:flex-row items-center justify-center gap-2">
                <button
                  onClick={() => setActiveTab('analyze')}
                  className="w-full sm:w-auto px-5 py-2.5 rounded-xl font-semibold text-xs bg-debtox-primary hover:bg-debtox-primary-hover text-white shadow-subtle flex items-center justify-center space-x-1.5 transition cursor-pointer"
                >
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>Analyze Repository</span>
                </button>
                <button
                  onClick={handleLoadDemo}
                  className="w-full sm:w-auto px-4 py-2.5 rounded-xl font-medium text-xs bg-surface border border-debtox-border text-debtox-ink hover:bg-surface-secondary transition cursor-pointer"
                >
                  Load Sample
                </button>
              </div>
            </div>
          )
        )}

        {activeTab === 'projects' && (
          <ProjectsPage
            onSelectProject={(analysis) => setCurrentAnalysis(analysis)}
            onNavigate={(tab) => setActiveTab(tab)}
          />
        )}
      </main>

      {/* Component Diagnosis Modal */}
      <ComponentModal
        component={selectedComponent}
        onClose={() => setSelectedComponent(null)}
      />

      {/* Clean Minimalist Footer */}
      <footer className="border-t border-debtox-border bg-surface/50 py-6 text-xs text-debtox-ink-muted font-sans">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-2">
          <div className="flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-debtox-primary" />
            <span className="font-semibold text-debtox-ink">DebtOx</span>
            <span>— Software Code Smell & Technical Debt Diagnosis</span>
          </div>
          <div className="text-[11px] text-debtox-ink-subtle">
            Focus on what to fix first.
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
