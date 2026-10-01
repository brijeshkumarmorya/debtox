from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from ml.data.provenance import DATASET_REGISTRY, DatasetProvenanceRecord

class SmellUnavailableError(Exception):
    """Raised when an empirical evaluation is attempted on a smell that is absent in real datasets."""
    pass

class RealDatasetLoader:
    """
    Production loader for verified empirical software engineering datasets.
    Strictly prohibits synthetic or fabricated data in research experiment flows.
    """

    def __init__(self, data_root: Optional[Path] = None):
        self.data_root = data_root or Path(__file__).resolve().parent.parent.parent / "data"
        self.raw_dir = self.data_root / "raw"
        self.external_dir = self.data_root / "external"

    def get_provenance(self, dataset_name: str) -> DatasetProvenanceRecord:
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_REGISTRY.keys())}")
        return DATASET_REGISTRY[dataset_name]

    def load_smellycode_plus_plus(
        self,
        smell_type: str,
        max_rows: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str], DatasetProvenanceRecord]:
        """
        Loads the primary SmellyCode++ dataset (Nature Scientific Data 2025).
        Returns (X, y, groups, feature_names, provenance).
        """
        provenance = self.get_provenance("smellycode_plus_plus")
        smell_key = smell_type.lower().replace(" ", "_")

        if smell_key in provenance.unavailable_smells:
            raise SmellUnavailableError(
                f"Smell '{smell_type}' is unavailable in the SmellyCode++ dataset. "
                f"In accordance with empirical research requirements, synthetic labels or "
                f"heuristics are strictly disallowed. Exclude this smell from empirical claims."
            )

        smell_col_map = {
            "god_class": "God class",
            "data_class": "Data class",
            "long_method": "Long method",
            "feature_envy": "Feature envy"
        }

        if smell_key not in smell_col_map:
            raise ValueError(f"Smell '{smell_type}' not supported by SmellyCode++. Supported: {list(smell_col_map.keys())}")

        target_col = smell_col_map[smell_key]
        csv_path = self.raw_dir / "smellycode_plus_plus.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"SmellyCode++ raw dataset not found at {csv_path}")

        # Stream or read safely with on_bad_lines='skip'
        read_kwargs = {"on_bad_lines": "skip"}
        if max_rows:
            read_kwargs["nrows"] = max_rows

        df = pd.read_csv(csv_path, **read_kwargs)

        # Standard numeric features supplied by SmellyCode++
        feature_cols = [
            "Logical Lines", "Distinct Operators", "Distinct Operands",
            "Total Operators", "Total Operands", "Vocabulary", "Length",
            "Calculated Length", "Volume", "Difficulty", "Effort",
            "Time Required", "Bugs", "Cyclomatic Complexity"
        ]

        # Verify all feature columns and target column exist
        missing = [c for c in feature_cols + [target_col, "Project"] if c not in df.columns]
        if missing:
            raise ValueError(f"Corrupted or incomplete CSV. Missing columns: {missing}")

        # Drop rows where target is NaN or non-numeric
        df = df.dropna(subset=[target_col, "Project"] + feature_cols)
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)

        for fc in feature_cols:
            df[fc] = pd.to_numeric(df[fc], errors="coerce").fillna(0.0)

        # Rename to normalized DebtOx feature names
        rename_map = {
            "Logical Lines": "SLOC",
            "Distinct Operators": "distinct_operators",
            "Distinct Operands": "distinct_operands",
            "Total Operators": "total_operators",
            "Total Operands": "total_operands",
            "Vocabulary": "vocabulary",
            "Length": "length",
            "Calculated Length": "calculated_length",
            "Volume": "halstead_volume",
            "Difficulty": "halstead_difficulty",
            "Effort": "halstead_effort",
            "Time Required": "halstead_time",
            "Bugs": "halstead_bugs",
            "Cyclomatic Complexity": "cyclomatic_complexity"
        }

        df_renamed = df.rename(columns=rename_map)
        normalized_features = [rename_map[c] for c in feature_cols]

        X = df_renamed[normalized_features]
        y = df_renamed[target_col]
        groups = df_renamed["Project"]

        return X, y, groups, normalized_features, provenance

    def load_crowdsmelling(
        self,
        smell_type: str
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str], DatasetProvenanceRecord]:
        """
        Loads the secondary human-validated Crowdsmelling dataset (EMSE 2022).
        Returns (X, y, groups, feature_names, provenance).
        """
        provenance = self.get_provenance("crowdsmelling")
        smell_key = smell_type.lower().replace(" ", "_")

        if smell_key in provenance.unavailable_smells:
            raise SmellUnavailableError(
                f"Smell '{smell_type}' is unavailable in Crowdsmelling dataset. "
                f"Empirical claims for this smell cannot be made with Crowdsmelling."
            )

        datasets_dir = self.external_dir / "Crowdsmelling" / "Datasets 2020"
        if not datasets_dir.exists():
            raise FileNotFoundError(f"Crowdsmelling dataset not found at {datasets_dir}")

        if smell_key == "god_class":
            csv_file = datasets_dir / "god-class-2020+2019+2018.csv"
            label_col = "is_god_class"
            feature_cols = [
                "LOC_type", "CBO_type", "RFC_type", "NOC_type", "WMC_type",
                "LCOM5_type", "DIT_type", "NOM_type", "NOA_type", "ATFD_type"
            ]
            rename_map = {
                "LOC_type": "SLOC", "CBO_type": "CBO", "RFC_type": "RFC",
                "NOC_type": "NOC", "WMC_type": "WMC", "LCOM5_type": "LCOM5",
                "DIT_type": "DIT", "NOM_type": "NOM", "NOA_type": "NOA",
                "ATFD_type": "ATFD"
            }
        elif smell_key == "long_method":
            csv_file = datasets_dir / "long-method-2020+2019+2018.csv"
            label_col = "is_long_method"
            feature_cols = [
                "LOC_method", "CYCLO_method", "MAXNESTING_method", "ATFD_method",
                "FDP_method", "NOP_method", "WMC_type", "CBO_type", "RFC_type"
            ]
            rename_map = {
                "LOC_method": "SLOC", "CYCLO_method": "cyclomatic_complexity",
                "MAXNESTING_method": "MNB", "ATFD_method": "ATFD",
                "FDP_method": "FDP", "NOP_method": "parameter_count",
                "WMC_type": "WMC", "CBO_type": "CBO", "RFC_type": "RFC"
            }
        elif smell_key == "feature_envy":
            csv_file = datasets_dir / "feature-envy-2020+2019+2018.csv"
            label_col = "is_feature_envy"
            feature_cols = [
                "LOC_method", "CYCLO_method", "MAXNESTING_method", "ATFD_method",
                "FDP_method", "LAA_method", "WMC_type", "CBO_type"
            ]
            rename_map = {
                "LOC_method": "SLOC", "CYCLO_method": "cyclomatic_complexity",
                "MAXNESTING_method": "MNB", "ATFD_method": "ATFD",
                "FDP_method": "FDP", "LAA_method": "LAA",
                "WMC_type": "WMC", "CBO_type": "CBO"
            }
        else:
            raise ValueError(f"Unsupported smell {smell_type} for Crowdsmelling")

        df = pd.read_csv(csv_file)
        df[label_col] = df[label_col].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0}).fillna(0).astype(int)

        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        df_renamed = df.rename(columns=rename_map)
        normalized_features = [rename_map[c] for c in feature_cols]

        X = df_renamed[normalized_features]
        y = df_renamed[label_col]
        groups = df_renamed["project"]

        return X, y, groups, normalized_features, provenance

    @staticmethod
    def get_project_level_splits(
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        n_splits: int = 5,
        random_seed: int = 42
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Creates strictly leakage-free project-level splits.
        Guarantees that all samples from a given project reside either entirely
        in the training set or entirely in the test set.
        """
        # If number of unique projects is >= n_splits, use GroupKFold
        n_groups = groups.nunique()
        actual_splits = min(n_splits, n_groups)

        if actual_splits < 2:
            # Fall back to GroupShuffleSplit with 80/20 train/test
            gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_seed)
            return list(gss.split(X, y, groups=groups))

        gkf = GroupKFold(n_splits=actual_splits)
        return list(gkf.split(X, y, groups=groups))
