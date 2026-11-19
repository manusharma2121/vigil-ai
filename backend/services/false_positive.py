"""
VIGIL AI – False Positive Analysis Engine

Analyses non-violent clips that the model predicted as violent (false positives).
Groups results by activity category and computes per-category FP rates.

This module requires ground-truth labels (Prediction.ground_truth) to be set.
Category-level breakdown requires activity category annotations, either from:
  1. The annotations CSV file (datasets/annotations/category_annotations.csv)
  2. Manual annotations stored in the CategoryAnnotation table
  3. activity_category field in the Prediction record

If neither is available, the report will clearly indicate this.
"""
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from database import Prediction, CategoryAnnotation
from config import settings

# Default activity categories
ACTIVITY_CATEGORIES = [
    "Sports Activities",
    "Hugging and Friendly Physical Contact",
    "Crowded Scenes",
    "Dancing and Group Movement",
    "Normal Physical Interactions",
    "Other Non-Violent Activities",
]


async def load_annotations_csv() -> Dict[str, str]:
    """
    Load category annotations from the CSV file.
    Returns dict: {filename -> activity_category}
    """
    path = Path(settings.annotations_path)
    if not path.exists():
        return {}

    annotations = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("filename", "").strip()
                cat = row.get("activity_category", "").strip()
                if fname and cat:
                    annotations[fname] = cat
    except Exception:
        pass

    return annotations


async def compute_false_positive_analysis(db: AsyncSession) -> Dict[str, Any]:
    """
    Compute full false positive analysis report.

    Returns a dict matching FalsePositiveReport schema.
    """
    # Fetch all predictions with ground truth labels
    result = await db.execute(
        select(Prediction).where(Prediction.ground_truth.is_not(None))
    )
    labelled = result.scalars().all()

    if not labelled:
        return {
            "total_false_positives": 0,
            "total_evaluated": 0,
            "overall_fp_rate": 0.0,
            "most_misclassified": None,
            "categories": [],
            "has_annotations": False,
            "message": (
                "No labelled data available. Upload videos with ground-truth labels or "
                "run evaluation on a labelled dataset to generate false positive analysis."
            ),
        }

    # Load annotations from CSV for enrichment
    csv_annotations = await load_annotations_csv()

    # Build lookup: filename → category
    # Priority: DB category annotation > CSV annotation
    category_map: Dict[str, str] = {}
    for p in labelled:
        cat = p.activity_category
        if not cat and p.filename in csv_annotations:
            cat = csv_annotations[p.filename]
        if cat:
            category_map[p.filename] = cat

    has_annotations = len(category_map) > 0

    # Identify false positives: ground_truth = Non-Violent, prediction = Violent
    total_evaluated = len(labelled)
    all_fps = [
        p for p in labelled
        if p.ground_truth.lower().strip() in ("non-violent", "nonviolent")
        and p.prediction.lower().strip() in ("violent",)
    ]
    total_fps = len(all_fps)
    overall_fp_rate = round(total_fps / max(total_evaluated, 1), 4)

    # Category-level breakdown
    if has_annotations:
        # Build per-category stats
        cat_stats: Dict[str, Dict[str, int]] = {}

        for p in labelled:
            cat = category_map.get(p.filename)
            if not cat:
                cat = "Other Non-Violent Activities"

            if cat not in cat_stats:
                cat_stats[cat] = {"total": 0, "fps": 0}

            # Only count non-violent ground-truth clips toward FP analysis
            if p.ground_truth.lower().strip() in ("non-violent", "nonviolent"):
                cat_stats[cat]["total"] += 1
                if p.prediction.lower().strip() == "violent":
                    cat_stats[cat]["fps"] += 1

        categories = []
        for cat, stats in sorted(cat_stats.items()):
            fp_rate = round(stats["fps"] / max(stats["total"], 1), 4)
            contrib_pct = round(stats["fps"] / max(total_fps, 1) * 100, 2) if total_fps > 0 else 0.0
            categories.append({
                "category": cat,
                "total_clips": stats["total"],
                "false_positives": stats["fps"],
                "fp_rate": fp_rate,
                "contribution_pct": contrib_pct,
            })

        most_misclassified = None
        if categories:
            most_misclassified = max(categories, key=lambda x: x["false_positives"])["category"]
    else:
        categories = []
        most_misclassified = None

    return {
        "total_false_positives": total_fps,
        "total_evaluated": total_evaluated,
        "overall_fp_rate": overall_fp_rate,
        "most_misclassified": most_misclassified,
        "categories": categories,
        "has_annotations": has_annotations,
        "message": (
            "Analysis computed from labelled predictions."
            if has_annotations else
            "Ground-truth labels are present but activity category annotations are missing. "
            "Add a category_annotations.csv or manually annotate clips to see category breakdown."
        ),
    }
