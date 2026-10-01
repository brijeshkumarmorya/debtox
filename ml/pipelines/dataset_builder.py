import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from backend.app.schemas.analysis import SmellType, EntityGranularity

CLASS_FEATURES = [
    "WMC", "CBO", "LCOM5", "RFC", "DIT", "NOC",
    "SLOC", "LLOC", "cyclomatic_complexity", "MNB",
    "halstead_volume", "halstead_difficulty", "halstead_effort"
]

METHOD_FEATURES = [
    "WMC", "CBO", "RFC", "ATFD", "FDP",
    "SLOC", "LLOC", "cyclomatic_complexity", "MNB",
    "halstead_volume", "halstead_difficulty", "halstead_effort"
]

class DatasetBuilder:
    """
    Constructs, validates, and manages normalized machine learning datasets
    for class-level and method-level code smells, reflecting the correlated
    empirical distributions of open-source Java projects (ml-Codesmell / Fontana).
    """
    
    @staticmethod
    def generate_benchmark_dataset(
        granularity: str = "class",
        n_samples: int = 2000,
        random_seed: int = 42,
        test_or_demo_mode: bool = False
    ) -> pd.DataFrame:
        if not test_or_demo_mode:
            raise PermissionError(
                "CRITICAL RESEARCH POLICY VIOLATION: Synthetic/generated datasets are strictly "
                "forbidden for empirical research experiments. Research claims must exclusively "
                "use real public datasets with verified provenance (SmellyCode++ or Crowdsmelling). "
                "This generator is strictly reserved for unit tests and local demo mode."
            )
        np.random.seed(random_seed)
        
        if str(granularity).lower() in ("class", "entitygranularity.class"):
            # Latent architectural factors:
            # z_god: monolithic bloating factor (~10% prevalence)
            # z_data: passive data structure factor (~12% prevalence)
            # z_brain: complex logic concentration factor (~8% prevalence)
            z_god = np.random.normal(0, 1, size=n_samples) > 1.30
            z_data = (~z_god) & (np.random.normal(0, 1, size=n_samples) > 1.25)
            z_brain = np.random.normal(0, 1, size=n_samples) > 1.45
            
            # SLOC
            sloc = np.random.lognormal(4.0, 0.7, size=n_samples) + 15
            sloc = np.where(z_god, np.random.uniform(280, 850, size=n_samples), sloc)
            sloc = np.where(z_brain, np.maximum(sloc, np.random.uniform(320, 900, size=n_samples)), sloc)
            sloc = np.where(z_data, np.random.uniform(40, 140, size=n_samples), sloc)
            
            # LLOC
            lloc = sloc * np.random.uniform(0.65, 0.85, size=n_samples)
            
            # WMC
            wmc = np.random.lognormal(2.3, 0.6, size=n_samples) + 3
            wmc = np.where(z_god, np.random.uniform(38, 95, size=n_samples), wmc)
            wmc = np.where(z_brain, np.random.uniform(48, 110, size=n_samples), wmc)
            wmc = np.where(z_data, np.random.uniform(2, 9, size=n_samples), wmc)
            
            # CBO
            cbo = np.random.lognormal(1.8, 0.5, size=n_samples) + 1
            cbo = np.where(z_god, np.random.uniform(14, 38, size=n_samples), cbo)
            cbo = np.where(z_data, np.random.uniform(1, 5, size=n_samples), cbo)
            
            # LCOM5
            lcom5 = np.random.beta(2.0, 5.0, size=n_samples)
            lcom5 = np.where(z_god, np.random.uniform(0.68, 0.95, size=n_samples), lcom5)
            lcom5 = np.where(z_data, np.random.uniform(0.72, 0.98, size=n_samples), lcom5)
            
            # RFC
            rfc = wmc + cbo + np.random.poisson(5, size=n_samples)
            dit = np.random.choice([0, 1, 2, 3, 4], p=[0.45, 0.35, 0.12, 0.06, 0.02], size=n_samples).astype(float)
            noc = np.random.choice([0, 1, 2, 3], p=[0.70, 0.20, 0.07, 0.03], size=n_samples).astype(float)
            
            # MNB
            mnb = np.random.choice([1, 2, 3, 4], p=[0.50, 0.35, 0.12, 0.03], size=n_samples).astype(float)
            mnb = np.where(z_brain, np.random.choice([4, 5, 6], size=n_samples), mnb)
            
            # Cyclomatic Complexity
            cc = wmc * np.random.uniform(0.85, 1.15, size=n_samples)
            
            # Halstead
            hal_vol = sloc * np.random.uniform(25.0, 45.0, size=n_samples)
            hal_diff = np.random.uniform(5.0, 35.0, size=n_samples)
            hal_eff = hal_vol * hal_diff
            
            df = pd.DataFrame({
                "WMC": np.round(wmc, 1),
                "CBO": np.round(cbo, 1),
                "LCOM5": np.round(lcom5, 3),
                "RFC": np.round(rfc, 1),
                "DIT": dit,
                "NOC": noc,
                "SLOC": np.round(sloc, 1),
                "LLOC": np.round(lloc, 1),
                "cyclomatic_complexity": np.round(cc, 1),
                "MNB": mnb,
                "halstead_volume": np.round(hal_vol, 1),
                "halstead_difficulty": np.round(hal_diff, 1),
                "halstead_effort": np.round(hal_eff, 1)
            })
            
            # Assign labels
            df["smell_god_class"] = ((df["WMC"] > 35) & (df["CBO"] > 12) & (df["LCOM5"] > 0.65) & (df["SLOC"] > 250)).astype(int)
            df["smell_data_class"] = ((df["WMC"] < 12) & (df["LCOM5"] > 0.70) & (df["CBO"] < 6)).astype(int)
            df["smell_brain_class"] = ((df["WMC"] > 45) & (df["MNB"] >= 4) & (df["SLOC"] > 300)).astype(int)
            
        else: # METHOD
            # Latent factors:
            # z_long: long method (~12%)
            # z_fe: feature envy (~10%)
            # z_bm: brain method (~8%)
            z_long = np.random.normal(0, 1, size=n_samples) > 1.22
            z_fe = np.random.normal(0, 1, size=n_samples) > 1.30
            z_bm = np.random.normal(0, 1, size=n_samples) > 1.42
            
            sloc = np.random.lognormal(2.6, 0.65, size=n_samples) + 4
            sloc = np.where(z_long, np.random.uniform(55, 180, size=n_samples), sloc)
            sloc = np.where(z_bm, np.maximum(sloc, np.random.uniform(60, 210, size=n_samples)), sloc)
            
            lloc = sloc * np.random.uniform(0.70, 0.90, size=n_samples)
            
            cc = np.random.lognormal(1.3, 0.5, size=n_samples) + 1
            cc = np.where(z_long, np.random.uniform(9, 22, size=n_samples), cc)
            cc = np.where(z_bm, np.random.uniform(15, 38, size=n_samples), cc)
            
            wmc = cc
            cbo = np.random.choice([0, 1, 2, 3], p=[0.40, 0.35, 0.20, 0.05], size=n_samples).astype(float)
            rfc = np.random.poisson(3, size=n_samples).astype(float)
            
            atfd = np.random.poisson(1.2, size=n_samples).astype(float)
            atfd = np.where(z_fe, np.random.uniform(4.0, 12.0, size=n_samples), atfd)
            
            fdp = np.where(atfd > 0, np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1], size=n_samples), 0.0)
            
            mnb = np.random.choice([1, 2, 3], p=[0.60, 0.30, 0.10], size=n_samples).astype(float)
            mnb = np.where(z_bm, np.random.choice([3, 4, 5], size=n_samples), mnb)
            
            hal_vol = sloc * np.random.uniform(18.0, 32.0, size=n_samples)
            hal_diff = np.random.uniform(3.0, 20.0, size=n_samples)
            hal_eff = hal_vol * hal_diff
            
            df = pd.DataFrame({
                "WMC": np.round(wmc, 1),
                "CBO": cbo,
                "RFC": rfc,
                "ATFD": np.round(atfd, 1),
                "FDP": fdp.astype(float),
                "SLOC": np.round(sloc, 1),
                "LLOC": np.round(lloc, 1),
                "cyclomatic_complexity": np.round(cc, 1),
                "MNB": mnb,
                "halstead_volume": np.round(hal_vol, 1),
                "halstead_difficulty": np.round(hal_diff, 1),
                "halstead_effort": np.round(hal_eff, 1)
            })
            
            df["smell_long_method"] = ((df["SLOC"] > 45) & (df["cyclomatic_complexity"] > 8)).astype(int)
            df["smell_feature_envy"] = ((df["ATFD"] >= 4) & (df["FDP"] <= 2)).astype(int)
            df["smell_brain_method"] = ((df["cyclomatic_complexity"] > 14) & (df["MNB"] >= 3)).astype(int)
            
        return df
