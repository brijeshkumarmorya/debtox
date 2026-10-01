import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler

def calculate_vif(X: np.ndarray, feature_idx: int) -> float:
    """
    Computes Variance Inflation Factor (VIF) for feature_idx using Ordinary Least Squares:
    VIF = 1 / (1 - R^2).
    """
    y = X[:, feature_idx]
    x_others = np.delete(X, feature_idx, axis=1)
    # Add intercept column
    A = np.column_stack([x_others, np.ones(len(x_others))])
    try:
        # Solve least squares
        coeffs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
        y_pred = A @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 1.0
        r_squared = 1.0 - (ss_res / ss_tot)
        if r_squared >= 0.9999:
            return 100.0
        return float(1.0 / (1.0 - r_squared))
    except Exception:
        return 1.0

class FeaturePreprocessor:
    """
    Handles feature preprocessing, scaling, collinearity analysis,
    and Variance Inflation Factor (VIF) filtering on training data only.
    """
    
    def __init__(self, use_vif_filter: bool = False, vif_threshold: float = 5.0, scaler_type: str = "robust"):
        self.use_vif_filter = use_vif_filter
        self.vif_threshold = vif_threshold
        self.scaler_type = scaler_type
        self.scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
        self.selected_features: List[str] = []
        self.impute_values: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame) -> "FeaturePreprocessor":
        # Step 1: Compute median imputation values
        self.impute_values = X.median().to_dict()
        X_imputed = X.fillna(self.impute_values)
        
        # Step 2: Optional VIF filtering
        current_features = list(X.columns)
        if self.use_vif_filter and len(current_features) > 2:
            dropped = True
            while dropped and len(current_features) > 2:
                dropped = False
                X_const = X_imputed[current_features].values
                vifs = [calculate_vif(X_const, i) for i in range(len(current_features))]
                max_vif_idx = int(np.argmax(vifs))
                if vifs[max_vif_idx] > self.vif_threshold:
                    dropped_feature = current_features.pop(max_vif_idx)
                    dropped = True
                    
        self.selected_features = current_features
        self.scaler.fit(X_imputed[self.selected_features])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_imputed = X.copy().fillna(self.impute_values)
        cols = self.selected_features if self.selected_features else list(X.columns)
        # Ensure all columns present
        for col in cols:
            if col not in X_imputed.columns:
                X_imputed[col] = self.impute_values.get(col, 0.0)
        return self.scaler.transform(X_imputed[cols])

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    @staticmethod
    def compute_vif_report(df: pd.DataFrame, features: List[str]) -> Dict[str, float]:
        """
        Computes VIF for each feature to diagnose multicollinearity.
        """
        X = df[features].dropna()
        if len(X) < len(features) + 1:
            return {f: 1.0 for f in features}
        vifs = {}
        for i, col in enumerate(features):
            try:
                vifs[col] = round(float(calculate_vif(X.values, i)), 3)
            except Exception:
                vifs[col] = 1.0
        return vifs
