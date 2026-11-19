"""
VIGIL AI – Dataset Router
GET  /api/dataset/info  – Return info about configured datasets
POST /api/dataset/load  – Trigger dataset scan
"""
from pathlib import Path
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from database import get_db
from schemas.schemas import DatasetInfo, LoadDatasetRequest, LoadDatasetResponse
from config import settings

router = APIRouter(prefix="/api/dataset", tags=["Dataset"])


def _scan_dataset(name: str, base_path: str) -> DatasetInfo:
    """Scan a dataset directory and return statistics."""
    base = Path(base_path)
    violent = 0
    nonviolent = 0
    available = base.exists()

    if available:
        for label_name in ("Fight", "fight", "Violence", "violence"):
            d = base / label_name
            if d.exists():
                violent += len(list(d.glob("*.avi"))) + len(list(d.glob("*.mp4")))
        for sub in base.iterdir():
            if sub.is_dir():
                for label_name in ("Fight", "fight", "Violence", "violence"):
                    d = sub / label_name
                    if d.exists():
                        violent += len(list(d.glob("*.avi"))) + len(list(d.glob("*.mp4")))
                for label_name in ("NonFight", "nonfight", "no-fight", "nofight", "NonViolence"):
                    d = sub / label_name
                    if d.exists():
                        nonviolent += len(list(d.glob("*.avi"))) + len(list(d.glob("*.mp4")))

        for label_name in ("NonFight", "nonfight", "no-fight", "nofight", "NonViolence"):
            d = base / label_name
            if d.exists():
                nonviolent += len(list(d.glob("*.avi"))) + len(list(d.glob("*.mp4")))

    total = violent + nonviolent
    train_count = int(total * 0.8)
    val_count   = int(total * 0.1)
    test_count  = total - train_count - val_count

    has_annotations = Path(settings.annotations_path).exists()

    return DatasetInfo(
        name=name,
        path=base_path,
        available=available,
        total_videos=total,
        violent_count=violent,
        nonviolent_count=nonviolent,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        has_annotations=has_annotations,
        message=(
            "Dataset found and scanned successfully." if available
            else f"Dataset not found at '{base_path}'. See README for setup instructions."
        ),
    )


@router.get("/info", response_model=List[DatasetInfo])
async def get_dataset_info():
    """Return information about all configured datasets."""
    datasets = [
        _scan_dataset("RWF-2000", settings.rwf2000_path),
        _scan_dataset("Hockey Fight Detection", settings.hockey_path),
    ]
    return datasets


@router.post("/load", response_model=LoadDatasetResponse)
async def load_dataset(req: LoadDatasetRequest):
    """Trigger a fresh scan of the specified dataset."""
    datasets = []
    if req.dataset in ("rwf2000", "both"):
        datasets.append(_scan_dataset("RWF-2000", settings.rwf2000_path))
    if req.dataset in ("hockey", "both"):
        datasets.append(_scan_dataset("Hockey Fight Detection", settings.hockey_path))

    success = any(d.available for d in datasets)
    return LoadDatasetResponse(
        success=success,
        message="Dataset scan complete." if success else "No datasets found. See README for setup instructions.",
        datasets=datasets,
    )
