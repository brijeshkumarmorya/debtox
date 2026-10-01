"""
DebtOx Empirical Research Pipeline - Section 21:
Generates the comprehensive research paper tables markdown (experiments/results/paper_tables.md)
strictly and automatically from real experiment CSV files.
Zero manual data entry, zero fabricated figures.
"""

import os
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone

def generate_paper_tables():
    res_dir = Path("experiments/results")
    md_path = res_dir / "paper_tables.md"
    
    lines = [
        "# DebtOx: Empirical Evaluation & Validation Result Tables\n",
        f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "**Provenance**: 100% computed from real empirical datasets (*SmellyCode++* Nature Scientific Data 2025 and *Crowdsmelling* EMSE 2022).",
        "**Leakage Prevention**: All cross-validation utilizes 5-fold Project-Level GroupKFold. All transformations, resampling, and threshold calibrations occur strictly inside training folds.\n",
        "---\n"
    ]
    
    # Table 1: Baseline Project-Level GroupKFold
    baseline_csv = res_dir / "baseline_results.csv"
    if baseline_csv.exists():
        df_base = pd.read_csv(baseline_csv)
        lines.append("## Table 1: Baseline Code Smell Prediction Performance (5-Fold Project-Level GroupKFold, Default Threshold = 0.50)")
        lines.append("*Evaluated on SmellyCode++ using un-tuned classifiers with default 0.50 threshold and training-only SMOTE.*")
        lines.append("")
        lines.append("| Smell Type | Granularity | Model | Precision | Recall | F1-Score | MCC | PR-AUC | ROC-AUC | Status |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, r in df_base.iterrows():
            gran = "Class" if r["smell"] in ["god_class", "data_class"] else "Method"
            smell_label = r["smell"].replace("_", " ").title()
            lines.append(
                f"| {smell_label} | {gran} | {r['model'].upper()} | {r['precision']:.4f} | {r['recall']:.4f} | "
                f"**{r['f1']:.4f} ± {r['f1_std']:.4f}** | {r['mcc']:.4f} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} | VALIDATED |"
            )
        lines.append("| Brain Class | Class | - | - | - | - | - | - | - | **UNAVAILABLE (Excluded)** |")
        lines.append("| Brain Method | Method | - | - | - | - | - | - | - | **UNAVAILABLE (Excluded)** |")
        lines.append("\n---\n")

    # Table 2: Tuned / Improved Models
    models_csv = res_dir / "code_smell_models.csv"
    if models_csv.exists():
        df_mod = pd.read_csv(models_csv)
        lines.append("## Table 2: Systematically Tuned Code Smell Models (Project-Level GroupKFold)")
        lines.append("*Evaluated with cost-sensitive class weights, tree regularization, and train-fold optimization.*")
        lines.append("")
        lines.append("| Smell Type | Model Architecture | Precision | Recall | F1-Score (Mean ± Std) | MCC | ROC-AUC | PR-AUC |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in df_mod.iterrows():
            smell_label = r["smell"].replace("_", " ").title()
            lines.append(
                f"| {smell_label} | {r['model']} | {r['precision']:.4f} | {r['recall']:.4f} | "
                f"**{r['f1']:.4f} ± {r['f1_std']:.4f}** | {r['mcc']:.4f} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} |"
            )
        lines.append("\n---\n")

    # Table 3: Class Imbalance Resampling Strategies
    resamp_csv = res_dir / "resampling_comparison.csv"
    if resamp_csv.exists():
        df_res = pd.read_csv(resamp_csv)
        lines.append("## Table 3: Systematic Class Imbalance Strategy Comparison (Train-Fold Only)")
        lines.append("*Comparing No Balancing, Class Weights, RUS, SMOTE, Borderline-SMOTE, and NearMiss under 5-Fold GroupKFold.*")
        lines.append("")
        lines.append("| Smell | Model | Resampling Strategy | Precision | Recall | F1-Score | MCC | PR-AUC | ROC-AUC |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in df_res.iterrows():
            lines.append(
                f"| {r['smell'].replace('_', ' ').title()} | {r['model'].upper()} | {r['strategy']} | {r['precision']:.4f} | "
                f"{r['recall']:.4f} | **{r['f1']:.4f}** | {r['mcc']:.4f} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} |"
            )
        lines.append("\n---\n")

    # Table 4: Threshold Optimization
    thresh_csv = res_dir / "threshold_optimization.csv"
    if thresh_csv.exists():
        df_thr = pd.read_csv(thresh_csv)
        lines.append("## Table 4: Inner-Validation Decision Threshold Optimization")
        lines.append("*Decision thresholds selected exclusively via 3-fold inner validation on training folds and evaluated on untouched test folds.*")
        lines.append("")
        lines.append("| Smell | Model | Strategy | Def Thresh (0.50) F1 | Opt Val F1 Thresh | Test F1 (Tuned) | Opt Val MCC Thresh | Test MCC (Tuned) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in df_thr.iterrows():
            lines.append(
                f"| {r['smell'].replace('_', ' ').title()} | {r['model'].upper()} | {r['imbalance_handling']} | {r['default_f1']:.4f} | "
                f"{r['val_opt_f1_threshold']:.2f} | **{r['val_opt_f1_score']:.4f}** | {r['val_opt_mcc_threshold']:.2f} | **{r['val_opt_mcc_score']:.4f}** |"
            )
        lines.append("\n---\n")

    # Table 5: Feature Selection
    fs_csv = res_dir / "feature_selection_results.csv"
    if fs_csv.exists():
        df_fs = pd.read_csv(fs_csv)
        lines.append("## Table 5: Train-Fold Feature Selection Strategy Comparison")
        lines.append("*Feature reduction evaluated independently inside each training fold to avoid target/collinearity leakage.*")
        lines.append("")
        lines.append("| Smell | Selection Method | Retained Features | Precision | Recall | F1-Score | MCC | PR-AUC |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in df_fs.iterrows():
            lines.append(
                f"| {r['smell'].replace('_', ' ').title()} | {r['feature_selection']} | {r['avg_features_retained']} | {r['precision']:.4f} | "
                f"{r['recall']:.4f} | **{r['f1']:.4f} ± {r['f1_std']:.4f}** | {r['mcc']:.4f} | {r['pr_auc']:.4f} |"
            )
        lines.append("\n---\n")

    # Table 6: Within-Project vs Cross-Project Generalization Gap
    gen_csv = res_dir / "within_vs_cross_project.csv"
    if gen_csv.exists():
        df_gen = pd.read_csv(gen_csv)
        lines.append("## Table 6: Within-Project vs. Cross-Project Generalization Gap")
        lines.append("*Demonstrating distribution shift and inter-project variance (StratifiedKFold vs GroupKFold).*")
        lines.append("")
        lines.append("| Smell | Within-Project F1 (Mean ± Std) | Within 95% CI | Cross-Project F1 (Mean ± Std) | Cross 95% CI | Generalization Gap (Δ F1) |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in df_gen.iterrows():
            lines.append(
                f"| {r['smell'].replace('_', ' ').title()} | {r['within_f1_mean']:.4f} ± {r['within_f1_std']:.4f} | "
                f"[{r['within_f1_ci95_low']:.4f}, {r['within_f1_ci95_high']:.4f}] | {r['cross_f1_mean']:.4f} ± {r['cross_f1_std']:.4f} | "
                f"[{r['cross_f1_ci95_low']:.4f}, {r['cross_f1_ci95_high']:.4f}] | **{r['generalization_gap_f1']:.4f}** |"
            )
        lines.append("\n---\n")

    # Table 7: External Cross-Dataset Validation
    cross_csv = res_dir / "cross_dataset_results.csv"
    if cross_csv.exists():
        df_cross = pd.read_csv(cross_csv)
        lines.append("## Table 7: External Cross-Dataset Generalization (SmellyCode++ vs. Crowdsmelling)")
        lines.append("*Strictly using mathematically harmonized feature intersection (SLOC and Cyclomatic Complexity/WMC).*")
        lines.append("")
        lines.append("| Smell | Training Dataset | Testing Dataset | Shared Metrics | Test Samples | Prevalence | F1-Score | MCC | PR-AUC | ROC-AUC |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, r in df_cross.iterrows():
            lines.append(
                f"| {r['smell'].replace('_', ' ').title()} | {r['train_dataset']} | {r['test_dataset']} | {r['shared_features']} | "
                f"{r['test_samples']} | {r['test_prevalence']:.4f} | **{r['f1']:.4f}** | {r['mcc']:.4f} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} |"
            )
        lines.append("\n---\n")

    # Table 8: Ablation Study
    abl_csv = res_dir / "ablation_results.csv"
    if abl_csv.exists():
        df_abl = pd.read_csv(abl_csv)
        lines.append("## Table 8: Controlled Feature-Group Ablation Study")
        lines.append("*Evaluating individual contribution of Size+Complexity vs Halstead token metrics under 5-Fold GroupKFold.*")
        lines.append("")
        lines.append("| Smell | Feature Group | Feature Count | Precision | Recall | F1-Score | MCC | PR-AUC |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in df_abl.iterrows():
            lines.append(
                f"| {r['smell'].replace('_', ' ').title()} | {r['feature_set']} | {r['feature_count']} | {r['precision']:.4f} | "
                f"{r['recall']:.4f} | **{r['f1']:.4f} ± {r['f1_std']:.4f}** | {r['mcc']:.4f} | {r['pr_auc']:.4f} |"
            )
        lines.append("\n---\n")

    # Table 9: Probability Calibration
    cal_csv = res_dir / "calibration_results.csv"
    if cal_csv.exists():
        df_cal = pd.read_csv(cal_csv)
        lines.append("## Table 9: Probability Calibration & Brier Score Improvement")
        lines.append("*Assessing probability reliability before and after Platt Sigmoid Calibration.*")
        lines.append("")
        lines.append("| Smell | Uncalibrated Brier | Calibrated Brier | Brier Improvement | Uncalibrated ECE | Calibrated ECE | Calibrated PR-AUC |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in df_cal.iterrows():
            lines.append(
                f"| {r['smell'].replace('_', ' ').title()} | {r['uncalibrated_brier']:.4f} | {r['calibrated_brier']:.4f} | "
                f"**{r['brier_improvement']:.4f}** | {r['uncalibrated_ece']:.4f} | {r['calibrated_ece']:.4f} | {r['calibrated_pr_auc']:.4f} |"
            )
        lines.append("\n---\n")

    # Table 10: Explainability / SHAP
    shap_csv = res_dir / "shap_features.csv"
    if shap_csv.exists():
        df_shap = pd.read_csv(shap_csv)
        lines.append("## Table 10: Global Feature Explainability (TreeSHAP Importance & Empirical Cutoffs)")
        lines.append("*Mean absolute SHAP value indicating empirical feature contribution to prediction across real samples.*")
        lines.append("")
        lines.append("| Smell | Importance Rank | Feature Name | Mean |SHAP| | Empirical Threshold (Median) | Semantic Role |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in df_shap.iterrows():
            lines.append(
                f"| {r['smell'].replace('_', ' ').title()} | Rank {r['importance_rank']} | `{r['feature']}` | {r['mean_abs_shap']:.4f} | "
                f"{r['empirical_threshold_median']} | {r['explanation_role']} |"
            )
        lines.append("\n---\n")

    # Table 11: TDP Comparison
    tdp_csv = res_dir / "tdp_comparison.csv"
    if tdp_csv.exists():
        df_tdp = pd.read_csv(tdp_csv)
        lines.append("## Table 11: Technical Debt Principal (TDP) Estimation Benchmark")
        lines.append("*Comparative accuracy of model-based technical debt effort estimation across industry and academic baselines.*")
        lines.append("")
        lines.append("| Estimation Model | MMRE | MdMRE | PRED(0.20) | PRED(0.25) | PRED(0.30) | MAE (Hours) | RMSE (Hours) | R² |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in df_tdp.iterrows():
            lines.append(
                f"| {r['model']} | {r['mmre']:.4f} | {r['mdmre']:.4f} | {r['pred_20']:.4f} | "
                f"**{r['pred_25']:.4f}** | {r['pred_30']:.4f} | {r['mae_hours']:.2f} | {r['rmse_hours']:.2f} | **{r['r_squared']:.4f}** |"
            )
        lines.append("\n---\n")

    # Table 12: TDI Analysis
    tdi_csv = res_dir / "tdi_analysis.csv"
    if tdi_csv.exists():
        df_tdi = pd.read_csv(tdi_csv)
        lines.append("## Table 12: Technical Debt Interest (TDI) Longitudinal Churn & Economic Friction")
        lines.append("*Longitudinal Git commit tracking separating observed code churn from estimated maintenance interest.*")
        lines.append("")
        lines.append("| Cohort Group | Sample Size | Avg Revisions | Observed Churn (LOC) | Excess Churn (%) | Friction Multiplier | Est. Interest (Hours) | Est. Economic Cost (USD) | Verification Status |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in df_tdi.iterrows():
            lines.append(
                f"| {r['cohort']} | {r['sample_size']} | {r['avg_revisions']} | {r['avg_observed_churn_loc']:.1f} | "
                f"{r['excess_churn_pct']:.1f}% | {r['friction_multiplier']:.2f}x | {r['estimated_interest_hours']:.2f} hrs | "
                f"**${r['estimated_economic_interest_usd']:,.2f}** | `{r['longitudinal_status']}` |"
            )
        lines.append("\n---\n")

    # Table 13: Real Open-Source Repository Demonstration
    repo_csv = res_dir / "real_repository_demo_results.csv"
    if repo_csv.exists():
        df_repo = pd.read_csv(repo_csv)
        lines.append("## Table 13: Practical Open-Source Repository Demonstration")
        lines.append("*End-to-end AST parsing, metric extraction, smell prediction, and parametric TDP estimation on real-world Java repositories.*")
        lines.append("")
        lines.append("| Target Repository | Category | Java Files | Classes | Methods | Total Smells | God Classes | Feature Envy | TDP (Hours) | TDP Cost ($65/hr) | Dominant SHAP Metric | Status |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in df_repo.iterrows():
            lines.append(
                f"| `{r['repository_name']}` | {r['repository_category']} | {int(r['total_java_files']):,} | "
                f"{int(r['total_classes']):,} | {int(r['total_methods']):,} | {int(r['total_smells_detected']):,} | "
                f"{int(r['god_classes_count']):,} | {int(r['feature_envy_count']):,} | **{r['total_tdp_hours']:,.1f} hrs** | "
                f"**${r['total_tdp_cost_usd']:,.2f}** | `{r['dominant_shap_attribution_feature']}` | `{r['analysis_status'].upper()}` |"
            )
        lines.append("\n---\n")

    # Table 14: TDP Multi-Dimensional Sensitivity Analysis
    tdp_sens_csv = res_dir / "tdp_sensitivity.csv"
    if tdp_sens_csv.exists():
        df_sens = pd.read_csv(tdp_sens_csv)
        lines.append("## Table 14: Technical Debt Principal (TDP) Multi-Dimensional Sensitivity Scenarios")
        lines.append("*Comparative evaluation of non-linear parametric effort estimation against flat industry heuristics across complexity and labor rate boundaries.*")
        lines.append("")
        lines.append("| Smell Type | SLOC | Cyclomatic Complexity | SU Penalty (%) | Est. TDP Effort (Hours) | Cost @ $50/hr | Cost @ $75/hr | Cost @ $100/hr | Cost @ $125/hr | SonarQube Flat Heuristic |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        # Representative slice of scenarios
        rep_indices = [
            (df_sens['smell'] == 'God Class') & (df_sens['sloc'] == 500) & (df_sens['cyclomatic_complexity'] == 15),
            (df_sens['smell'] == 'God Class') & (df_sens['sloc'] == 1000) & (df_sens['cyclomatic_complexity'] == 30),
            (df_sens['smell'] == 'God Class') & (df_sens['sloc'] == 2500) & (df_sens['cyclomatic_complexity'] == 50),
            (df_sens['smell'] == 'God Class') & (df_sens['sloc'] == 2500) & (df_sens['cyclomatic_complexity'] == 80),
            (df_sens['smell'] == 'Data Class') & (df_sens['sloc'] == 100) & (df_sens['cyclomatic_complexity'] == 5),
            (df_sens['smell'] == 'Data Class') & (df_sens['sloc'] == 500) & (df_sens['cyclomatic_complexity'] == 15),
            (df_sens['smell'] == 'Long Method') & (df_sens['sloc'] == 100) & (df_sens['cyclomatic_complexity'] == 15),
            (df_sens['smell'] == 'Long Method') & (df_sens['sloc'] == 250) & (df_sens['cyclomatic_complexity'] == 30),
            (df_sens['smell'] == 'Long Method') & (df_sens['sloc'] == 500) & (df_sens['cyclomatic_complexity'] == 50),
            (df_sens['smell'] == 'Feature Envy') & (df_sens['sloc'] == 100) & (df_sens['cyclomatic_complexity'] == 15),
            (df_sens['smell'] == 'Feature Envy') & (df_sens['sloc'] == 500) & (df_sens['cyclomatic_complexity'] == 30),
        ]
        for cond in rep_indices:
            sub = df_sens[cond]
            if len(sub) >= 4:
                first = sub.iloc[0]
                h_50 = sub[sub['hourly_rate_usd'] == 50.0]['debtox_estimated_tdp_cost_usd'].values[0]
                h_75 = sub[sub['hourly_rate_usd'] == 75.0]['debtox_estimated_tdp_cost_usd'].values[0]
                h_100 = sub[sub['hourly_rate_usd'] == 100.0]['debtox_estimated_tdp_cost_usd'].values[0]
                h_125 = sub[sub['hourly_rate_usd'] == 125.0]['debtox_estimated_tdp_cost_usd'].values[0]
                lines.append(
                    f"| **{first['smell']}** | {int(first['sloc'])} | {int(first['cyclomatic_complexity'])} | "
                    f"{first['software_understanding_penalty_pct']:.1f}% | **{first['debtox_estimated_tdp_hours']:.1f} hrs** | "
                    f"${h_50:,.2f} | **${h_75:,.2f}** | ${h_100:,.2f} | ${h_125:,.2f} | {first['sqale_rule_hours']:.1f} hrs |"
                )
        lines.append("\n---\n")

    # Table 15: Statistical Hypothesis Testing
    stat_csv = res_dir / "final_statistics.csv"
    if stat_csv.exists():
        df_stat = pd.read_csv(stat_csv)
        lines.append("## Table 15: Non-Parametric Statistical Hypothesis Testing & Effect Size Analysis")
        lines.append("*Paired Wilcoxon signed-rank tests across 5-fold cross-validation, Cliff's delta effect sizes, and Holm-Bonferroni FWER control.*")
        lines.append("")
        lines.append("| Code Smell | Hypothesis Comparison | Baseline Mean F1 | Tuned Mean F1 | Δ F1 | Tuned 95% Confidence Interval | Wilcoxon W | Raw p-value | Cliff's δ | Effect Magnitude | Holm-Bonferroni α | Significant (FWER α=0.05) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in df_stat.iterrows():
            lines.append(
                f"| `{r['smell'].replace('_', ' ').title()}` | {r['comparison']} | {r['baseline_mean']:.4f} | "
                f"**{r['tuned_mean']:.4f}** | {r['delta_mean']:+.4f} | `{r['tuned_95_ci']}` | "
                f"{r['wilcoxon_stat']:.1f} | {r['raw_p_value']:.4f} | {r['cliffs_delta']:+.2f} | "
                f"**{r['effect_size'].upper()}** | {r['holm_bonferroni_threshold']:.4f} | `{r['statistically_significant_after_correction']}` |"
            )
        lines.append("\n---\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[SUCCESS] Paper tables markdown regenerated at {md_path}")

if __name__ == "__main__":
    generate_paper_tables()

