import React from 'react';
import { Activity, Play, CheckCircle2, FolderGit2 } from 'lucide-react';
import { NavigationTab, AnalysisSummary } from '../types';

interface NavbarProps {
  activeTab: NavigationTab;
  setActiveTab: (tab: NavigationTab) => void;
  currentAnalysis: AnalysisSummary | null;
}

export const Navbar: React.FC<NavbarProps> = ({ 
  activeTab, 
  setActiveTab, 
  currentAnalysis 
}) => {
  const navItems: { id: NavigationTab; label: string; count?: number }[] = [
    { id: 'home', label: 'Home' },
    { id: 'analyze', label: 'Analyze' },
    { 
      id: 'issues', 
      label: 'Issues', 
      count: currentAnalysis ? currentAnalysis.total_smells : undefined 
    },
    { id: 'projects', label: 'Projects' },
  ];

  return (
    <header className="sticky top-0 z-40 bg-[#F7F2EB]/95 backdrop-blur border-b border-debtox-border">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
        {/* Brand */}
        <div 
          onClick={() => setActiveTab('home')}
          className="flex items-center space-x-2.5 cursor-pointer group"
        >
          <div className="w-8 h-8 rounded-lg bg-debtox-primary flex items-center justify-center text-white shadow-sm">
            <Activity className="w-4 h-4 stroke-[2.5]" />
          </div>
          <div className="flex items-baseline space-x-1.5">
            <span className="text-lg font-bold tracking-tight text-debtox-ink font-sans">
              DebtOx
            </span>
            <span className="text-[11px] text-debtox-ink-muted hidden sm:inline">
              diagnostic
            </span>
          </div>
        </div>

        {/* Clean Navigation Links */}
        <nav className="flex items-center space-x-1 sm:space-x-2">
          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition flex items-center space-x-1.5 cursor-pointer ${
                  isActive
                    ? 'bg-debtox-primary text-white shadow-subtle'
                    : 'text-debtox-ink-muted hover:text-debtox-ink hover:bg-surface-secondary/70'
                }`}
              >
                <span>{item.label}</span>
                {item.count !== undefined && item.count > 0 && (
                  <span className={`text-[10px] px-1.5 py-0.2 rounded-full font-mono font-semibold ${
                    isActive ? 'bg-white/25 text-white' : 'bg-surface-secondary text-debtox-ink'
                  }`}>
                    {item.count}
                  </span>
                )}
              </button>
            );
          })}
        </nav>

        {/* Active Project Pill & Quick Action */}
        <div className="flex items-center space-x-2">
          {currentAnalysis && (
            <div 
              onClick={() => setActiveTab('issues')}
              className="hidden md:flex items-center space-x-1.5 px-2.5 py-1 rounded-lg bg-surface border border-debtox-border text-xs text-debtox-ink cursor-pointer hover:border-debtox-primary/50 transition"
              title="View current project issues"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-debtox-primary" />
              <span className="font-medium truncate max-w-[120px]">{currentAnalysis.repository_name}</span>
            </div>
          )}

          {activeTab !== 'analyze' && (
            <button
              onClick={() => setActiveTab('analyze')}
              className="hidden sm:flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-surface border border-debtox-border hover:border-debtox-primary text-debtox-ink hover:text-debtox-primary transition shadow-subtle cursor-pointer"
            >
              <Play className="w-3 h-3 fill-current" />
              <span>Analyze</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
};
