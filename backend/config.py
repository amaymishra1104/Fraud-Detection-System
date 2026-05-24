"""
Configuration settings for the fraud detection backend API.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""
    
    # API Configuration
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", 8000))
    api_workers: int = int(os.getenv("API_WORKERS", 4))
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Model Configuration
    model_path: str = os.getenv("MODEL_PATH", "./models/fraud_model.pkl")
    scaler_path: str = os.getenv("SCALER_PATH", "./models/scaler.pkl")
    decision_threshold: float = float(os.getenv("DECISION_THRESHOLD", 0.5))
    
    # Database Configuration
    database_url: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://user:password@localhost/fraud_detection"
    )
    database_pool_size: int = int(os.getenv("DATABASE_POOL_SIZE", 20))
    
    # Logging Configuration
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "./logs/app.log")
    
    # Security Configuration
    secret_key: str = os.getenv("SECRET_KEY", "change-me-in-production")
    allowed_origins: list = [
        "http://localhost:3000",
        "http://localhost:8501",
        "http://localhost:8000"
    ]
    
    class Config:
        """Pydantic config."""
        env_file = ".env"


settings = Settings()
