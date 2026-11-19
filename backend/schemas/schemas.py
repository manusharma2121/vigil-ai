"""
VIGIL AI – Pydantic Schemas
Request/response models for all API endpoints.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ── Prediction ──────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    id: int
    filename: str
    prediction: str
    violence_prob: float
    nonviolence_prob: float
    confidence: float
    processing_time: float
    model_used: str
    is_demo: bool
    ground_truth: Optional[str] = None
    activity_category: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class AnnotateRequest(BaseModel):
    prediction_id: int
    activity_category: str
    ground_truth: str
    notes: Optional[str] = None


class AnnotateResponse(BaseModel):
    success: bool
    message: str


# ── Dashboard ────────────────────────────────────────────────────────────────

class DashboardStats(BaseModel):
    total_videos: int
    violent_count: int
    nonviolent_count: int
    false_positive_count: int
    overall_accuracy: Optional[float] = None
    avg_confidence: float
    recent_predictions: List[PredictionResponse]
    confidence_distribution: List[Dict[str, Any]]
    prediction_distribution: List[Dict[str, Any]]


# ── False Positive Analysis ──────────────────────────────────────────────────

class CategoryFPStats(BaseModel):
    category: str
    total_clips: int
    false_positives: int
    fp_rate: float
    contribution_pct: float


class FalsePositiveReport(BaseModel):
    total_false_positives: int
    total_evaluated: int
    overall_fp_rate: float
    most_misclassified: Optional[str] = None
    categories: List[CategoryFPStats]
    has_annotations: bool
    message: str


# ── Dataset ──────────────────────────────────────────────────────────────────

class DatasetInfo(BaseModel):
    name: str
    path: str
    available: bool
    total_videos: int
    violent_count: int
    nonviolent_count: int
    train_count: int
    val_count: int
    test_count: int
    has_annotations: bool
    message: str


class LoadDatasetRequest(BaseModel):
    dataset: str  # "rwf2000" | "hockey" | "both"


class LoadDatasetResponse(BaseModel):
    success: bool
    message: str
    datasets: List[DatasetInfo]


# ── Training ─────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    model_arch: str = "cnn_lstm"
    epochs: int = Field(default=10, ge=1, le=100)
    batch_size: int = Field(default=8, ge=1, le=64)
    learning_rate: float = Field(default=0.001, gt=0)
    dataset: str = "both"


class TrainStatus(BaseModel):
    is_training: bool
    current_epoch: int
    total_epochs: int
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float
    status: str
    history: List[Dict[str, Any]]
    error: Optional[str] = None
    # Enriched fields for the frontend
    state: Optional[str] = None
    message: Optional[str] = None
    latest_metrics: Optional[Dict[str, Any]] = None
    demo_mode_active: Optional[bool] = None


# ── Model Info ───────────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    architecture: str
    is_loaded: bool
    is_demo: bool
    weights_loaded: bool = False
    model_path: str
    input_frames: int
    frame_size: int
    confidence_threshold: float
    num_parameters: Optional[int] = None
    parameters: Optional[int] = None
    device: Optional[str] = None
    trained_epoch: Optional[int] = None
    val_accuracy: Optional[float] = None
    val_loss: Optional[float] = None
    training_history: Optional[List[Dict[str, Any]]] = None
    evaluation: Optional[Dict[str, Any]] = None
