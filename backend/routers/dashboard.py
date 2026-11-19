"""
VIGIL AI – Dashboard Router
GET /api/dashboard/stats – Aggregated statistics for the main dashboard
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from database import get_db, Prediction
from schemas.schemas import DashboardStats, PredictionResponse

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


@router.get("/stats", response_model=DashboardStats)
async def get_dashboard_stats(db: AsyncSession = Depends(get_db)):
    """
    Return aggregate statistics for the main dashboard.
    All numbers come from the actual database – no hardcoded values.
    """
    # Total counts
    total_res = await db.execute(select(func.count()).select_from(Prediction))
    total_videos = total_res.scalar() or 0

    violent_res = await db.execute(
        select(func.count()).select_from(Prediction).where(Prediction.prediction == "Violent")
    )
    violent_count = violent_res.scalar() or 0
    nonviolent_count = total_videos - violent_count

    # False positives (requires ground-truth labels)
    fp_res = await db.execute(
        select(func.count()).select_from(Prediction).where(
            Prediction.ground_truth.in_(["Non-Violent", "non-violent", "NonViolent"]),
            Prediction.prediction == "Violent",
        )
    )
    false_positive_count = fp_res.scalar() or 0

    # Average confidence
    avg_conf_res = await db.execute(select(func.avg(Prediction.confidence)).select_from(Prediction))
    avg_confidence = round(float(avg_conf_res.scalar() or 0.0), 4)

    # Recent predictions (last 10)
    recent_res = await db.execute(
        select(Prediction).order_by(Prediction.created_at.desc()).limit(10)
    )
    recent = recent_res.scalars().all()

    # Confidence distribution buckets (0-0.5, 0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, 0.9-1.0)
    all_preds_res = await db.execute(select(Prediction.confidence))
    all_confs = all_preds_res.scalars().all()
    buckets = [
        {"range": "0–50%", "count": 0},
        {"range": "50–60%", "count": 0},
        {"range": "60–70%", "count": 0},
        {"range": "70–80%", "count": 0},
        {"range": "80–90%", "count": 0},
        {"range": "90–100%", "count": 0},
    ]
    for c in all_confs:
        if c < 0.5:
            buckets[0]["count"] += 1
        elif c < 0.6:
            buckets[1]["count"] += 1
        elif c < 0.7:
            buckets[2]["count"] += 1
        elif c < 0.8:
            buckets[3]["count"] += 1
        elif c < 0.9:
            buckets[4]["count"] += 1
        else:
            buckets[5]["count"] += 1

    prediction_distribution = [
        {"name": "Violent", "value": violent_count},
        {"name": "Non-Violent", "value": nonviolent_count},
    ]

    return DashboardStats(
        total_videos=total_videos,
        violent_count=violent_count,
        nonviolent_count=nonviolent_count,
        false_positive_count=false_positive_count,
        overall_accuracy=None,  # Requires ground-truth labels
        avg_confidence=avg_confidence,
        recent_predictions=[PredictionResponse.model_validate(r) for r in recent],
        confidence_distribution=buckets,
        prediction_distribution=prediction_distribution,
    )
