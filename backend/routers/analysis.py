"""
VIGIL AI – Analysis Router
GET /api/analysis/false-positives – Full false positive analysis report
"""
import json
import io
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from services.false_positive import compute_false_positive_analysis
from schemas.schemas import FalsePositiveReport

router = APIRouter(prefix="/api/analysis", tags=["Analysis"])


@router.get("/false-positives", response_model=FalsePositiveReport)
async def get_false_positive_analysis(db: AsyncSession = Depends(get_db)):
    """
    Return a comprehensive false positive analysis report.
    Requires ground-truth labels to be set on predictions.
    Category breakdown requires activity annotations.
    """
    report = await compute_false_positive_analysis(db)
    return report


@router.get("/false-positives/export")
async def export_fp_report(db: AsyncSession = Depends(get_db)):
    """Download the false positive report as a JSON file."""
    report = await compute_false_positive_analysis(db)
    content = json.dumps(report, indent=2, default=str)
    return StreamingResponse(
        io.BytesIO(content.encode()),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=vigil_fp_analysis.json"},
    )
