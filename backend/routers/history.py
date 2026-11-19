"""
VIGIL AI – History Router
GET    /api/history         – Paginated prediction history
DELETE /api/history/{id}    – Delete a prediction record
GET    /api/history/export  – Export history as CSV
"""
import csv
import io
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func
from typing import List, Optional

from database import get_db, Prediction
from schemas.schemas import PredictionResponse

router = APIRouter(prefix="/api/history", tags=["History"])


@router.get("", response_model=List[PredictionResponse])
async def get_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    prediction_filter: Optional[str] = Query(None, description="Filter: 'Violent' or 'Non-Violent'"),
    db: AsyncSession = Depends(get_db),
):
    """Return prediction history, newest first."""
    query = select(Prediction).order_by(Prediction.created_at.desc()).offset(skip).limit(limit)
    if prediction_filter:
        query = query.where(Prediction.prediction == prediction_filter)
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/count")
async def get_history_count(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(func.count()).select_from(Prediction))
    return {"count": result.scalar()}


@router.delete("/{prediction_id}")
async def delete_prediction(prediction_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a single prediction record."""
    result = await db.execute(select(Prediction).where(Prediction.id == prediction_id))
    pred = result.scalar_one_or_none()
    if pred is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    await db.execute(delete(Prediction).where(Prediction.id == prediction_id))
    await db.commit()
    return {"success": True, "message": f"Prediction #{prediction_id} deleted."}


@router.get("/export")
async def export_history_csv(db: AsyncSession = Depends(get_db)):
    """Export all prediction history as a CSV download."""
    result = await db.execute(
        select(Prediction).order_by(Prediction.created_at.desc())
    )
    predictions = result.scalars().all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id", "filename", "prediction", "violence_prob", "nonviolence_prob",
        "confidence", "processing_time_sec", "model_used", "is_demo",
        "ground_truth", "activity_category", "created_at",
    ])
    for p in predictions:
        writer.writerow([
            p.id, p.filename, p.prediction, p.violence_prob, p.nonviolence_prob,
            p.confidence, p.processing_time, p.model_used, p.is_demo,
            p.ground_truth or "", p.activity_category or "",
            p.created_at.isoformat() if p.created_at else "",
        ])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=vigil_prediction_history.csv"},
    )
