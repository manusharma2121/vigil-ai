"""
VIGIL AI – Predict Router
POST /api/predict       – Upload a video and get a violence prediction
POST /api/predict/annotate – Set ground-truth / category for a prediction
"""
import os
import uuid
import aiofiles
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database import get_db, Prediction, CategoryAnnotation
from schemas.schemas import PredictionResponse, AnnotateRequest, AnnotateResponse
from services.inference import predict
from config import settings

router = APIRouter(prefix="/api/predict", tags=["Predict"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


@router.post("", response_model=PredictionResponse)
async def predict_video(
    file: UploadFile = File(...),
    ground_truth: str = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """Upload a video file and receive a violence detection prediction."""

    # Validate file type
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Validate file size
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    contents = await file.read()
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed: {settings.max_upload_size_mb} MB",
        )

    # Save to uploads directory with unique name
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / unique_name

    async with aiofiles.open(save_path, "wb") as f:
        await f.write(contents)

    # Run inference
    try:
        result = predict(str(save_path))
    except Exception as e:
        # Clean up file on error
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # Store result in database
    record = Prediction(
        filename=file.filename,
        file_path=str(save_path),
        prediction=result["prediction"],
        violence_prob=result["violence_prob"],
        nonviolence_prob=result["nonviolence_prob"],
        confidence=result["confidence"],
        processing_time=result["processing_time"],
        model_used=result["model_used"],
        is_demo=result["is_demo"],
        ground_truth=ground_truth,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)

    return record


@router.post("/annotate", response_model=AnnotateResponse)
async def annotate_prediction(
    req: AnnotateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Set ground-truth label and/or activity category for an existing prediction."""

    result = await db.execute(select(Prediction).where(Prediction.id == req.prediction_id))
    pred = result.scalar_one_or_none()
    if pred is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")

    pred.ground_truth = req.ground_truth
    pred.activity_category = req.activity_category

    # Also save to annotations table
    ann = CategoryAnnotation(
        prediction_id=req.prediction_id,
        filename=pred.filename,
        activity_category=req.activity_category,
        ground_truth=req.ground_truth,
        notes=req.notes,
    )
    db.add(ann)
    await db.commit()

    return AnnotateResponse(
        success=True,
        message=f"Annotation saved for prediction #{req.prediction_id}",
    )
