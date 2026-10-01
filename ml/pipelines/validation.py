import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

class DataValidator:
    """
    Automated data validation verifying schema integrity, range boundaries,
    missing values, duplicates, and train/test leakage.
    """
    
    @staticmethod
    def validate_dataset(df: pd.DataFrame, expected_features: List[str], label_col: Optional[str] = None) -> Dict[str, Any]:
        report = {
            "is_valid": True,
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "missing_values": {},
            "infinite_values": {},
            "out_of_bounds": {},
            "duplicate_rows": int(df.duplicated().sum()),
            "errors": []
        }
        
        # Check expected features
        missing_cols = [col for col in expected_features if col not in df.columns]
        if missing_cols:
            report["is_valid"] = False
            report["errors"].append(f"Missing required feature columns: {missing_cols}")
            
        for col in expected_features:
            if col in df.columns:
                n_missing = int(df[col].isna().sum())
                if n_missing > 0:
                    report["missing_values"][col] = n_missing
                    
                if np.issubdtype(df[col].dtype, np.number):
                    n_inf = int(np.isinf(df[col]).sum())
                    if n_inf > 0:
                        report["infinite_values"][col] = n_inf
                        report["is_valid"] = False
                        
                    # Non-negativity check for software metrics
                    n_neg = int((df[col] < 0).sum())
                    if n_neg > 0:
                        report["out_of_bounds"][col] = f"{n_neg} negative values"
                        
        if label_col and label_col in df.columns:
            unique_labels = set(df[label_col].dropna().unique())
            if not unique_labels.issubset({0, 1, False, True}):
                report["is_valid"] = False
                report["errors"].append(f"Label column {label_col} contains non-binary values: {unique_labels}")
                
        return report

    @staticmethod
    def check_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame, group_col: str = "repository_id") -> bool:
        """
        Verifies that no groups (e.g. repositories or commits) appear in both train and test partitions.
        """
        if group_col in train_df.columns and group_col in test_df.columns:
            train_groups = set(train_df[group_col].dropna().unique())
            test_groups = set(test_df[group_col].dropna().unique())
            overlap = train_groups.intersection(test_groups)
            return len(overlap) == 0
        return True
