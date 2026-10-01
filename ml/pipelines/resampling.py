import numpy as np
from typing import Tuple, Optional
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss

class ImbalanceHandler:
    """
    Configurable imbalance strategies applied strictly inside training folds.
    Never applied to validation or test data.
    """
    
    @staticmethod
    def apply_resampling(
        X: np.ndarray,
        y: np.ndarray,
        strategy: str = "smote",
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        strategy = strategy.lower().strip()
        
        # Check minority class count
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2 or min(counts) < 6:
            # Not enough minority samples to safely oversample via SMOTE
            return X, y
            
        k_neighbors = min(5, min(counts) - 1)
        
        if strategy in ("none", "raw"):
            return X, y
            
        elif strategy in ("random_under", "under_sampling"):
            rus = RandomUnderSampler(random_state=random_state)
            return rus.fit_resample(X, y)
            
        elif strategy in ("nearmiss", "near_miss"):
            nm = NearMiss(version=1, n_neighbors=min(3, min(counts) - 1))
            return nm.fit_resample(X, y)
            
        elif strategy == "borderline_smote":
            bsmote = BorderlineSMOTE(random_state=random_state, k_neighbors=k_neighbors)
            return bsmote.fit_resample(X, y)
            
        elif strategy in ("smote", "standard_smote"):
            smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
            return smote.fit_resample(X, y)
            
        else:
            raise ValueError(f"Unsupported imbalance strategy: {strategy}. Choose from 'none', 'random_under', 'nearmiss', 'smote', 'borderline_smote'")
