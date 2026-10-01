import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import skew

def generate_feature_audit():
    print(">>> Generating Feature Audit and Dataset Feature Compatibility...")
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load SmellyCode++
    df = pd.read_csv("data/raw/smellycode_plus_plus.csv")
    
    # Categorize DebtOx & Dataset Features
    feature_meta = [
        {"feature": "Logical Lines", "category": "A. Size", "unit": "Lines", "type": "int"},
        {"feature": "Distinct Operators", "category": "D. Halstead", "unit": "Count", "type": "int"},
        {"feature": "Distinct Operands", "category": "D. Halstead", "unit": "Count", "type": "int"},
        {"feature": "Total Operators", "category": "D. Halstead", "unit": "Count", "type": "int"},
        {"feature": "Total Operands", "category": "D. Halstead", "unit": "Count", "type": "int"},
        {"feature": "Vocabulary", "category": "D. Halstead", "unit": "Count", "type": "int"},
        {"feature": "Length", "category": "D. Halstead", "unit": "Count", "type": "int"},
        {"feature": "Calculated Length", "category": "D. Halstead", "unit": "Value", "type": "float"},
        {"feature": "Volume", "category": "D. Halstead", "unit": "Bits", "type": "float"},
        {"feature": "Difficulty", "category": "D. Halstead", "unit": "Ratio", "type": "float"},
        {"feature": "Effort", "category": "D. Halstead", "unit": "Mental Effort", "type": "float"},
        {"feature": "Time Required", "category": "D. Halstead", "unit": "Seconds", "type": "float"},
        {"feature": "Bugs", "category": "D. Halstead", "unit": "Estimated Bugs", "type": "float"},
        {"feature": "Cyclomatic Complexity", "category": "B. Complexity", "unit": "McCabe v(G)", "type": "int"}
    ]
    
    audit_rows = []
    for item in feature_meta:
        col = item["feature"]
        series = df[col].dropna()
        sk = float(skew(series))
        audit_rows.append({
            "Feature": col,
            "Category": item["category"],
            "Missing %": round((df[col].isna().sum() / len(df)) * 100, 2),
            "Mean": round(float(series.mean()), 2),
            "Median": round(float(series.median()), 2),
            "Std": round(float(series.std()), 2),
            "Min": round(float(series.min()), 2),
            "Max": round(float(series.max()), 2),
            "Skewness": round(sk, 2),
            "Requires Log1p": "YES (Highly Skewed)" if abs(sk) > 3.0 else "NO (Moderate/Low)"
        })
        
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(results_dir / "feature_audit.csv", index=False)
    print("Wrote experiments/results/feature_audit.csv")
    
    # 2. Dataset Feature Compatibility Table (Section 11)
    compatibility_data = [
        {
            "Feature_Name": "Source Lines of Code (SLOC / LOC)",
            "Category": "Size",
            "In_SmellyCode++": "Yes ('Logical Lines')",
            "In_Crowdsmelling": "Yes ('LOC_type' / 'LOC_method')",
            "In_DebtOx_AST": "Yes ('SLOC', 'LLOC')",
            "Same_Definition": "Yes",
            "Same_Unit": "Count of non-blank lines",
            "Compatible_for_Cross_Dataset": "YES",
            "Notes": "Fully compatible size metric across all sources."
        },
        {
            "Feature_Name": "Cyclomatic Complexity (v(G) / WMC)",
            "Category": "Complexity",
            "In_SmellyCode++": "Yes ('Cyclomatic Complexity')",
            "In_Crowdsmelling": "Yes ('CYCLO_method' / 'WMC_type')",
            "In_DebtOx_AST": "Yes ('cyclomatic_complexity', 'WMC')",
            "Same_Definition": "Yes",
            "Same_Unit": "McCabe decision points",
            "Compatible_for_Cross_Dataset": "YES",
            "Notes": "McCabe control flow complexity is mathematically identical."
        },
        {
            "Feature_Name": "Coupling Between Objects (CBO)",
            "Category": "OO Structural",
            "In_SmellyCode++": "No",
            "In_Crowdsmelling": "Yes ('CBO_type')",
            "In_DebtOx_AST": "Yes ('CBO')",
            "Same_Definition": "Yes",
            "Same_Unit": "Unique foreign class count",
            "Compatible_for_Cross_Dataset": "NO",
            "Notes": "Absent in SmellyCode++; present in Crowdsmelling & DebtOx."
        },
        {
            "Feature_Name": "Response For a Class (RFC)",
            "Category": "OO Structural",
            "In_SmellyCode++": "No",
            "In_Crowdsmelling": "Yes ('RFC_type')",
            "In_DebtOx_AST": "Yes ('RFC')",
            "Same_Definition": "Yes",
            "Same_Unit": "Callable method count",
            "Compatible_for_Cross_Dataset": "NO",
            "Notes": "Absent in SmellyCode++; present in Crowdsmelling & DebtOx."
        },
        {
            "Feature_Name": "Lack of Cohesion in Methods (LCOM5)",
            "Category": "OO Structural",
            "In_SmellyCode++": "No",
            "In_Crowdsmelling": "Yes ('LCOM5_type')",
            "In_DebtOx_AST": "Yes ('LCOM5')",
            "Same_Definition": "Yes",
            "Same_Unit": "Henderson-Sellers metric [0, 1]",
            "Compatible_for_Cross_Dataset": "NO",
            "Notes": "Absent in SmellyCode++; present in Crowdsmelling & DebtOx."
        },
        {
            "Feature_Name": "Access to Foreign Data (ATFD)",
            "Category": "OO Structural",
            "In_SmellyCode++": "No",
            "In_Crowdsmelling": "Yes ('ATFD_type' / 'ATFD_method')",
            "In_DebtOx_AST": "Yes ('ATFD')",
            "Same_Definition": "Yes",
            "Same_Unit": "Foreign variable access count",
            "Compatible_for_Cross_Dataset": "NO",
            "Notes": "Absent in SmellyCode++; present in Crowdsmelling & DebtOx."
        },
        {
            "Feature_Name": "Foreign Data Providers (FDP)",
            "Category": "OO Structural",
            "In_SmellyCode++": "No",
            "In_Crowdsmelling": "Yes ('FDP_method')",
            "In_DebtOx_AST": "Yes ('FDP')",
            "Same_Definition": "Yes",
            "Same_Unit": "Distinct provider classes",
            "Compatible_for_Cross_Dataset": "NO",
            "Notes": "Absent in SmellyCode++; present in Crowdsmelling & DebtOx."
        },
        {
            "Feature_Name": "Max Nested Blocks (MNB)",
            "Category": "Complexity",
            "In_SmellyCode++": "No",
            "In_Crowdsmelling": "Yes ('MAXNESTING_method')",
            "In_DebtOx_AST": "Yes ('MNB')",
            "Same_Definition": "Yes",
            "Same_Unit": "Max block depth integer",
            "Compatible_for_Cross_Dataset": "NO",
            "Notes": "Absent in SmellyCode++; present in Crowdsmelling & DebtOx."
        },
        {
            "Feature_Name": "Halstead Volume & Effort Suite",
            "Category": "Halstead",
            "In_SmellyCode++": "Yes (12 distinct Halstead metrics)",
            "In_Crowdsmelling": "No",
            "In_DebtOx_AST": "Yes (Volume, Effort, Difficulty)",
            "Same_Definition": "Yes",
            "Same_Unit": "Standard Halstead formulas",
            "Compatible_for_Cross_Dataset": "NO",
            "Notes": "Present in SmellyCode++ & DebtOx; omitted in Crowdsmelling."
        },
        {
            "Feature_Name": "Evolutionary Churn & Commits",
            "Category": "Evolutionary",
            "In_SmellyCode++": "No",
            "In_Crowdsmelling": "No",
            "In_DebtOx_AST": "Yes (via GitMiner)",
            "Same_Definition": "Yes",
            "Same_Unit": "LOC churned, commit count",
            "Compatible_for_Cross_Dataset": "NO",
            "Notes": "Static datasets lack longitudinal Git history; mined dynamically by DebtOx."
        }
    ]
    
    comp_df = pd.DataFrame(compatibility_data)
    comp_df.to_csv(results_dir / "dataset_feature_compatibility.csv", index=False)
    print("Wrote experiments/results/dataset_feature_compatibility.csv")

if __name__ == "__main__":
    generate_feature_audit()
