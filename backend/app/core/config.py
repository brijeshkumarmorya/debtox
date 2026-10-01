from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "DebtOx"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "*"
    ]
    
    # Storage & Paths
    BASE_DIR: Path = BASE_DIR
    MODELS_DIR: Path = BASE_DIR / "models"
    DATA_DIR: Path = BASE_DIR / "data"
    EXPERIMENTS_DIR: Path = BASE_DIR / "experiments"
    STORAGE_DIR: Path = BASE_DIR / "storage"
    
    # Database
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/debtox.db"
    
    # Resource Limits
    MAX_REPOSITORY_SIZE_MB: int = 500
    MAX_FILES: int = 10000
    MAX_JAVA_FILES: int = 5000
    MAX_ANALYSIS_TIME_SECONDS: int = 600
    MAX_HISTORY_COMMITS: int = 500
    
    # Technical Debt Engine Parameters
    COCOMO_A: float = 2.94
    COCOMO_E: float = 1.05
    DEFAULT_DEVELOPER_HOURLY_RATE: float = 65.0
    SQALE_LONG_METHOD_MINUTES: float = 30.0
    SQALE_GOD_CLASS_MINUTES: float = 120.0
    
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

settings = Settings()
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
