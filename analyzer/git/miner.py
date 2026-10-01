import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import git
from backend.app.schemas.analysis import EvolutionRecord

class GitEvolutionMiner:
    """
    Mines Git repository histories, extracts commit metadata, diffs,
    line additions, deletions, hunks, and maps churn to components.
    """
    
    def clone_or_open_repository(
        self,
        repo_path_or_url: str,
        target_dir: Optional[Path] = None,
        branch: Optional[str] = None
    ) -> Tuple[git.Repo, Path, bool]:
        """
        Clones a remote repository URL or opens an existing local directory.
        Returns: (git.Repo instance, working_directory, is_temporary)
        """
        is_temp = False
        if repo_path_or_url.startswith(("http://", "https://", "git@")):
            is_temp = True
            if not target_dir:
                import tempfile
                target_dir = Path(tempfile.mkdtemp(prefix="debtox_repo_"))
            repo = git.Repo.clone_from(repo_path_or_url, str(target_dir), branch=branch, depth=500)
        else:
            repo_path = Path(repo_path_or_url).resolve()
            if not repo_path.exists():
                raise ValueError(f"Repository path does not exist: {repo_path}")
            repo = git.Repo(str(repo_path))
            target_dir = repo_path
            if branch and branch != repo.active_branch.name:
                repo.git.checkout(branch)
                
        return repo, target_dir, is_temp

    def extract_commit_history(
        self,
        repo: git.Repo,
        max_commits: int = 100
    ) -> List[EvolutionRecord]:
        """
        Extracts up to max_commits from the current HEAD backwards,
        computing file changes, added lines, deleted lines, hunks, and total churn.
        """
        records: List[EvolutionRecord] = []
        try:
            commits = list(repo.iter_commits('HEAD', max_count=max_commits))
        except Exception:
            return records

        for idx, commit in enumerate(commits):
            added_lines = 0
            deleted_lines = 0
            changed_files_count = 0
            num_hunks = 0
            
            try:
                # Compare commit with parent
                if commit.parents:
                    parent = commit.parents[0]
                    diffs = parent.diff(commit, create_patch=True)
                    changed_files_count = len(diffs)
                    
                    for d in diffs:
                        # Skip binary files
                        if d.b_blob and d.b_blob.mime_type and 'text' not in d.b_blob.mime_type:
                            continue
                        if d.diff:
                            diff_text = d.diff.decode('utf-8', errors='ignore')
                            # Count additions and deletions
                            add_count = len([line for line in diff_text.splitlines() if line.startswith('+') and not line.startswith('+++')])
                            del_count = len([line for line in diff_text.splitlines() if line.startswith('-') and not line.startswith('---')])
                            hunk_count = len([line for line in diff_text.splitlines() if line.startswith('@@')])
                            
                            added_lines += add_count
                            deleted_lines += del_count
                            num_hunks += max(1, hunk_count)
                else:
                    # Initial commit: estimate from tree stats
                    stats = commit.stats.total
                    added_lines = stats.get('lines', 0)
                    changed_files_count = stats.get('files', 1)
                    num_hunks = changed_files_count
            except Exception:
                pass
                
            total_churn = added_lines + deleted_lines
            dt = datetime.fromtimestamp(commit.committed_date)
            
            records.append(EvolutionRecord(
                commit_hash=commit.hexsha,
                short_hash=commit.hexsha[:7],
                timestamp=dt.isoformat(),
                author=commit.author.name or "Unknown",
                message=commit.message.strip().split('\n')[0][:120],
                added_lines=added_lines,
                deleted_lines=deleted_lines,
                total_churn=total_churn,
                changed_files=changed_files_count,
                num_hunks=num_hunks
            ))
            
        return records

    def compute_component_churn(
        self,
        repo: git.Repo,
        file_path_relative: str,
        max_commits: int = 50
    ) -> Dict[str, Any]:
        """
        Computes historical churn for a specific file/component over past commits.
        """
        commits_touching = 0
        total_churn = 0
        added_lines = 0
        deleted_lines = 0
        first_commit_date = None
        last_commit_date = None
        
        try:
            commits = list(repo.iter_commits(paths=file_path_relative, max_count=max_commits))
            commits_touching = len(commits)
            if commits:
                last_commit_date = datetime.fromtimestamp(commits[0].committed_date)
                first_commit_date = datetime.fromtimestamp(commits[-1].committed_date)
                
                for c in commits:
                    if c.parents:
                        diffs = c.parents[0].diff(c, paths=file_path_relative, create_patch=True)
                        for d in diffs:
                            if d.diff:
                                text = d.diff.decode('utf-8', errors='ignore')
                                add = len([l for l in text.splitlines() if l.startswith('+') and not l.startswith('+++')])
                                sub = len([l for l in text.splitlines() if l.startswith('-') and not l.startswith('---')])
                                added_lines += add
                                deleted_lines += sub
                                total_churn += (add + sub)
        except Exception:
            pass

        age_days = 0.0
        if first_commit_date and last_commit_date:
            age_days = (last_commit_date - first_commit_date).total_seconds() / 86400.0

        return {
            "churn": float(total_churn),
            "added_loc": float(added_lines),
            "deleted_loc": float(deleted_lines),
            "commits_touching_component": commits_touching,
            "commit_frequency": round(commits_touching / max(1.0, age_days / 30.0), 2) if age_days > 0 else float(commits_touching),
            "age_days": round(age_days, 1)
        }
