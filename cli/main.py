import sys
import json
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.app.core.config import settings
from backend.app.schemas.analysis import AnalysisRequest, AnalysisMode
from backend.app.services.analysis_service import AnalysisService

app = typer.Typer(
    name="debt-ox",
    help="DebtOx — Code Smell Prediction and Technical Debt Estimation CLI",
    add_completion=False
)
console = Console()

@app.command()
def version():
    """Display DebtOx system version."""
    console.print(Panel(f"[bold cyan]DebtOx[/bold cyan] Version [green]{settings.VERSION}[/green]\nAPI Version: [yellow]v1[/yellow]\nPlatform: Production-grade Code Smell & Technical Debt Engine", title="DebtOx Info"))

@app.command()
def analyze(
    target: str = typer.Argument(..., help="Local directory path or Git repository URL"),
    branch: Optional[str] = typer.Option(None, "--branch", "-b", help="Git branch to checkout"),
    mode: str = typer.Option("full", "--mode", "-m", help="Analysis mode: quick, full, research"),
    max_commits: int = typer.Option(100, "--commits", "-c", help="Max history commits to mine"),
    output_format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv")
):
    """Analyze a Java repository for code smells and technical debt."""
    console.print(f"[bold green]Starting DebtOx Analysis on:[/bold green] {target}")
    
    analysis_mode = AnalysisMode.FULL
    if mode.lower() == "quick":
        analysis_mode = AnalysisMode.QUICK
    elif mode.lower() == "research":
        analysis_mode = AnalysisMode.RESEARCH

    req = AnalysisRequest(
        local_path=target if not target.startswith(("http://", "https://", "git@")) else None,
        repository_url=target if target.startswith(("http://", "https://", "git@")) else None,
        branch=branch,
        mode=analysis_mode,
        max_history_commits=max_commits
    )
    
    service = AnalysisService()
    job = service.create_analysis_job(req)
    console.print(f"[cyan]Analysis Job ID:[/cyan] [bold]{job.analysis_id}[/bold]")
    
    with console.status("[bold green]Executing AST parsing, metric extraction, and ML predictions...[/bold green]"):
        service.execute_analysis(job.analysis_id, req)
        res = service.get_analysis(job.analysis_id)
        
    if not res or res.status.value == "failed":
        console.print(f"[bold red]Analysis Failed:[/bold red] {res.error_message if res else 'Unknown error'}")
        raise typer.Exit(code=1)

    if output_format.lower() == "json":
        console.print(json.dumps(res.model_dump(), indent=2))
        return

    # Print Summary Table
    summary_table = Table(title="DebtOx Analysis Summary", show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="bold white")
    
    summary_table.add_row("Repository", res.repository_name)
    summary_table.add_row("Java Files", str(res.total_files))
    summary_table.add_row("Classes", str(res.total_classes))
    summary_table.add_row("Methods", str(res.total_methods))
    summary_table.add_row("Total Code Smells", f"[red]{res.total_smells}[/red]")
    summary_table.add_row("Affected Components", str(res.affected_components_count))
    summary_table.add_row("TD Principal (TDP)", f"[yellow]{res.total_tdp_hours:.1f} hours[/yellow]")
    summary_table.add_row("TD Interest (TDI)", f"[yellow]{res.total_tdi_hours:.1f} hours[/yellow]")
    summary_table.add_row("Total Debt", f"[bold red]{res.total_debt_hours:.1f} hours[/bold red]")
    summary_table.add_row("Estimated Cost", f"[bold green]${res.total_estimated_cost_usd:,.2f}[/bold green]")
    summary_table.add_row("Average Risk Score", f"{res.average_risk_score:.1f} / 100")
    console.print(summary_table)

    # Print Top High-Risk Components
    if res.components:
        comp_table = Table(title="Top High-Risk Smelly Components", show_header=True, header_style="bold yellow")
        comp_table.add_column("Granularity", style="cyan")
        comp_table.add_column("Component Name", style="white")
        comp_table.add_column("Detected Smells", style="red")
        comp_table.add_column("TDP (h)", justify="right")
        comp_table.add_column("Risk Score", justify="right", style="bold red")
        
        top_smelly = [c for c in res.components if c.detected_smells][:10]
        for c in top_smelly:
            name = c.class_name if not c.method_name else f"{c.class_name}.{c.method_name}()"
            comp_table.add_row(
                c.granularity.value.upper(),
                name,
                ", ".join(c.detected_smells),
                f"{c.tdp_hours:.1f}",
                f"{c.risk_score:.1f}"
            )
        console.print(comp_table)

@app.command()
def train():
    """Run model training and cross-validation across all supported smells."""
    from ml.experiments.runner import ResearchExperimentRunner
    runner = ResearchExperimentRunner()
    with console.status("[bold green]Executing full model training and evaluation suite...[/bold green]"):
        res = runner.run_all_experiments()
    console.print(Panel(f"[bold green]Model Training Completed![/bold green]\nOutput Directory: {res['output_directory']}\nRuntime: {res['runtime_seconds']}s", title="DebtOx ML Engine"))

@app.command()
def explain(
    analysis_id: str = typer.Argument(..., help="Analysis ID to explain"),
    top_n: int = typer.Option(5, "--top", "-t", help="Number of components to explain")
):
    """View SHAP explainability insights for an analysis."""
    service = AnalysisService()
    res = service.get_analysis(analysis_id)
    if not res:
        console.print(f"[bold red]Analysis not found:[/bold red] {analysis_id}")
        raise typer.Exit(code=1)

    smelly = [c for c in res.components if c.detected_smells][:top_n]
    if not smelly:
        console.print("[green]No code smells detected in this analysis.[/green]")
        return

    for c in smelly:
        name = c.class_name if not c.method_name else f"{c.class_name}.{c.method_name}()"
        console.print(Panel(
            f"[bold]Detected Smells:[/bold] {', '.join(c.detected_smells)}\n"
            f"[bold]TDP Hours:[/bold] {c.tdp_hours}h | [bold]Risk Score:[/bold] {c.risk_score}\n"
            f"[bold]Primary Recommendation:[/bold] [italic]{c.primary_recommendation or 'N/A'}[/italic]",
            title=f"Component: {name}"
        ))
        if c.shap_contributions:
            table = Table(title="Top Feature Attributions (SHAP)", show_header=True)
            table.add_column("Feature")
            table.add_column("Value", justify="right")
            table.add_column("SHAP Value", justify="right")
            table.add_column("Impact")
            table.add_column("Explanation")
            for s in c.shap_contributions:
                color = "red" if s.impact == "increases_risk" else "green"
                table.add_row(
                    s.feature_name,
                    f"{s.feature_value}",
                    f"[{color}]{s.shap_value:+.3f}[/{color}]",
                    f"[{color}]{s.impact}[/{color}]",
                    s.narrative
                )
            console.print(table)

@app.command()
def report(
    analysis_id: str = typer.Argument(..., help="Analysis ID to export"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path (defaults to report_<id>.json)")
):
    """Export analysis report to JSON."""
    service = AnalysisService()
    res = service.get_analysis(analysis_id)
    if not res:
        console.print(f"[bold red]Analysis not found:[/bold red] {analysis_id}")
        raise typer.Exit(code=1)
    out_path = Path(output_file or f"report_{analysis_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res.model_dump(), f, indent=2)
    console.print(f"[bold green]Report exported successfully to:[/bold green] {out_path.resolve()}")

if __name__ == "__main__":
    app()
