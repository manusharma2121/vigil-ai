"""
VIGIL AI – Training Router
POST /api/train           – Start model training
GET  /api/train/status    – Poll training progress
GET  /api/model/info      – Return model metadata
"""
from fastapi import APIRouter, HTTPException
from schemas.schemas import TrainRequest, TrainStatus, ModelInfo
from services.trainer import start_training, get_training_status
from services.inference import get_model_info, reload_model

router = APIRouter(tags=["Training"])


@router.post("/api/train")
async def start_train(req: TrainRequest):
    """
    Start model training in a background thread.
    Training will use available dataset files; returns error if no data found.
    """
    result = start_training(
        model_arch=req.model_arch,
        epochs=req.epochs,
        batch_size=req.batch_size,
        learning_rate=req.learning_rate,
        dataset_name=req.dataset,
    )
    if not result.get("started"):
        raise HTTPException(status_code=409, detail=result.get("message", "Could not start training."))
    return result


@router.get("/api/train/status")
async def get_train_status():
    """Poll current training progress."""
    status = get_training_status()
    # Enrich with frontend-compatible fields
    return {
        **status,
        "state": status.get("status", "idle"),
        "message": status.get("error") or ("Training complete." if status.get("status") == "done" else ""),
        "latest_metrics": {
            "train_loss": status.get("train_loss"),
            "val_loss":   status.get("val_loss"),
            "train_acc":  status.get("train_acc"),
            "val_acc":    status.get("val_acc"),
        } if status.get("current_epoch", 0) > 0 else None,
    }


@router.get("/api/model/info", response_model=ModelInfo)
async def model_info():
    """Return information about the currently loaded model."""
    info = get_model_info()
    return ModelInfo(**info)


@router.post("/api/model/reload")
async def reload():
    """Force model reload from disk (e.g., after training completes)."""
    success = reload_model()
    return {
        "success": True,
        "is_demo": not success,
        "message": "Model reloaded from trained weights." if success else "No trained weights found; running in demo mode.",
    }
