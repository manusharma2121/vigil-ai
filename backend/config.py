"""
VIGIL AI – Application Configuration
Loads settings from environment variables or .env file.
"""
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    # Application
    app_name: str = "VIGIL AI"
    app_version: str = "1.0.0"
    debug: bool = True

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    frontend_url: str = "http://localhost:5173"

    # Model
    model_path: str = "../models/weights/vigil_model.pth"
    model_arch: str = "cnn_lstm"
    confidence_threshold: float = 0.5
    demo_mode: bool = True

    # Video Processing
    max_frames: int = 16
    frame_size: int = 112
    max_upload_size_mb: int = 100

    # Database
    database_url: str = "sqlite:///./vigil.db"

    # Datasets
    rwf2000_path: str = "../datasets/rwf2000"
    hockey_path: str = "../datasets/hockey"
    annotations_path: str = "../datasets/annotations/category_annotations.csv"

    # Training
    batch_size: int = 8
    epochs: int = 30
    learning_rate: float = 0.001

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
