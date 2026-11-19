"""
VIGIL AI – Inference Service (v2 — Optical Flow + CNN+LSTM Ensemble)
=====================================================================
Primary classifier: Optical Flow Statistics (physics-grounded, generalises
                    from synthetic to real-world video).
Secondary:          CNN+LSTM (MobileNetV2 backbone), used as a secondary
                    signal when trained weights are available.

Ensemble strategy:
  - If flow_classifier.pkl  exists → use flow (80%) + CNN (20%)
  - If only CNN weights exist       → use CNN with temperature scaling
  - If neither                      → DEMO mode (random, clearly labelled)

Why optical flow as primary:
  Violent activity produces HIGH, ERRATIC, CHAOTIC optical flow.
  Non-violent activity produces LOW, SMOOTH, COHERENT optical flow.
  These physics-based features transfer naturally from synthetic training
  data to real-world video — unlike CNN features trained on pixel rectangles.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
import numpy as np

from config import settings
from models.cnn_lstm import build_cnn_lstm, count_parameters as count_cnn_lstm
from models.model_3dcnn import build_3dcnn, count_parameters as count_3dcnn
from services.video_processor import (
    extract_frames,
    frames_to_tensor_cnn_lstm,
    frames_to_tensor_3dcnn,
)
from services.flow_classifier import (
    load_flow_classifier,
    train_flow_classifier,
    save_flow_classifier,
    predict_flow,
    get_flow_feature_dict,
)

logger = logging.getLogger(__name__)

# ── Global model state ────────────────────────────────────────────────────────
_cnn_model: Optional[torch.nn.Module] = None
_cnn_loaded: bool = False
_cnn_is_demo: bool = True
_cnn_parameters: Optional[int] = None
_checkpoint_meta: Dict[str, Any] = {}

_flow_clf = None
_flow_loaded: bool = False

_initialized: bool = False


# ══════════════════════════════════════════════════════════════════════════════
#  INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _build_cnn() -> torch.nn.Module:
    if settings.model_arch == "3dcnn":
        return build_3dcnn()
    return build_cnn_lstm()


def _load_cnn_weights() -> bool:
    """Load CNN+LSTM weights. Returns True if real weights were loaded."""
    global _cnn_model, _cnn_loaded, _cnn_is_demo, _cnn_parameters, _checkpoint_meta

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_cnn().to(device)
    model.eval()

    weight_path = Path(settings.model_path)
    success = False

    if weight_path.exists():
        try:
            checkpoint = torch.load(
                str(weight_path), map_location=device, weights_only=False
            )
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state = checkpoint["model_state_dict"]
                _checkpoint_meta = {
                    k: v for k, v in checkpoint.items()
                    if k not in ("model_state_dict", "optimizer_state")
                }
            elif isinstance(checkpoint, dict):
                state = checkpoint
                _checkpoint_meta = {}
            else:
                state = checkpoint
                _checkpoint_meta = {}

            model.load_state_dict(state, strict=True)
            model.eval()
            _cnn_is_demo = False
            success = True
            logger.info(
                f"✓ CNN weights loaded | val_acc={_checkpoint_meta.get('val_acc', 'N/A')} "
                f"| epoch={_checkpoint_meta.get('epoch', 'N/A')}"
            )
        except Exception as e:
            logger.warning(f"CNN weight load failed: {e}. CNN in demo mode.")
            _cnn_is_demo = True
    else:
        logger.info("No CNN weight file found.")
        _cnn_is_demo = True

    _cnn_model = model
    _cnn_loaded = True

    if settings.model_arch == "3dcnn":
        _cnn_parameters = count_3dcnn(model)
    else:
        _cnn_parameters = count_cnn_lstm(model)

    return success


def _load_flow_clf():
    """Load or train the optical flow classifier."""
    global _flow_clf, _flow_loaded

    clf = load_flow_classifier()
    if clf is None:
        logger.info("Flow classifier not found — training now…")
        clf = train_flow_classifier(n_per_class=2000, seed=42)
        save_flow_classifier(clf)
        logger.info("✓ Flow classifier trained and saved.")
    else:
        logger.info("✓ Flow classifier loaded.")

    _flow_clf = clf
    _flow_loaded = True


def load_model() -> bool:
    """
    Initialise both classifiers. Returns True when real weights are active
    (flow classifier is always considered 'real' since it uses actual
    physics-based features, not random demo output).
    """
    global _initialized
    cnn_ok = _load_cnn_weights()
    _load_flow_clf()
    _initialized = True
    # System is out of demo mode as long as the flow classifier is loaded
    return True


def reload_model() -> bool:
    """Force reload — called after training completes."""
    global _cnn_loaded, _flow_loaded, _initialized
    _cnn_loaded = False
    _flow_loaded = False
    _initialized = False
    return load_model()


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def predict(video_path: str) -> Dict[str, Any]:
    """
    Run violence detection on a video file.

    Ensemble strategy:
      flow_ok AND cnn_ok  → 0.75 × flow + 0.25 × CNN(temperature-scaled)
      flow_ok only         → flow only
      cnn_ok only          → CNN with temperature=4.0
      neither              → DEMO random [clearly labelled]

    Returns dict with prediction, probabilities, confidence, timing, metadata.
    """
    global _initialized, _flow_clf, _cnn_model, _cnn_is_demo, _flow_loaded

    if not _initialized:
        load_model()

    start = time.perf_counter()

    # ── 1. Optical flow inference ─────────────────────────────────────────────
    flow_nv, flow_v = 0.5, 0.5
    flow_ok = False
    flow_features = {}

    if _flow_loaded and _flow_clf is not None:
        try:
            flow_nv, flow_v = predict_flow(video_path, _flow_clf)
            flow_features = get_flow_feature_dict(video_path)
            flow_ok = True
        except Exception as e:
            logger.warning(f"Flow inference failed: {e}")

    # ── 2. CNN+LSTM inference ─────────────────────────────────────────────────
    cnn_nv, cnn_v = 0.5, 0.5
    cnn_ok = False

    if _cnn_loaded and _cnn_model is not None and not _cnn_is_demo:
        try:
            frames = extract_frames(video_path)
            if frames is not None:
                device = next(_cnn_model.parameters()).device
                if settings.model_arch == "3dcnn":
                    tensor = frames_to_tensor_3dcnn(frames).to(device)
                else:
                    tensor = frames_to_tensor_cnn_lstm(frames).to(device)

                with torch.no_grad():
                    logits = _cnn_model(tensor)
                    # Temperature scaling softens extreme logits from
                    # synthetic-domain training → more calibrated probs
                    temperature = 4.0
                    probs = F.softmax(logits / temperature, dim=1)

                cnn_nv = probs[0, 0].item()
                cnn_v  = probs[0, 1].item()
                cnn_ok = True
        except Exception as e:
            logger.warning(f"CNN inference failed: {e}")

    # ── 3. Ensemble ───────────────────────────────────────────────────────────
    is_demo = False

    if flow_ok and cnn_ok:
        violent_prob    = 0.75 * flow_v  + 0.25 * cnn_v
        nonviolent_prob = 0.75 * flow_nv + 0.25 * cnn_nv
        method = "Optical Flow (75%) + CNN+LSTM (25%)"
    elif flow_ok:
        violent_prob    = flow_v
        nonviolent_prob = flow_nv
        method = "Optical Flow Classifier"
    elif cnn_ok:
        violent_prob    = cnn_v
        nonviolent_prob = cnn_nv
        method = "CNN+LSTM (temperature-scaled)"
    else:
        # Genuine demo — no trained components available
        is_demo = True
        violent_prob    = float(np.random.uniform(0.15, 0.85))
        nonviolent_prob = 1.0 - violent_prob
        method = "DEMO [untrained — no classifiers loaded]"

    # Normalise to sum to 1
    total = violent_prob + nonviolent_prob
    if total > 0:
        violent_prob    /= total
        nonviolent_prob /= total

    confidence = max(violent_prob, nonviolent_prob)
    prediction = "Violent" if violent_prob >= settings.confidence_threshold else "Non-Violent"
    elapsed = time.perf_counter() - start

    # Build model label
    arch_label = "CNN+LSTM (MobileNetV2)" if settings.model_arch == "cnn_lstm" else "3D-CNN"
    if is_demo:
        model_label = f"{arch_label} [DEMO – no classifiers]"
    else:
        epoch   = _checkpoint_meta.get("epoch", "?")
        val_acc = _checkpoint_meta.get("val_acc")
        acc_str = f" val_acc={val_acc*100:.1f}%" if val_acc else ""
        model_label = f"{arch_label} epoch={epoch}{acc_str} | {method}"

    result = {
        "prediction":        prediction,
        "violence_prob":     round(violent_prob,    4),
        "nonviolence_prob":  round(nonviolent_prob, 4),
        "confidence":        round(confidence,      4),
        "processing_time":   round(elapsed,         3),
        "model_used":        model_label,
        "is_demo":           is_demo,
        "frames_sampled":    settings.max_frames,
        "method":            method,
    }

    # Include optical flow features for explainability
    if flow_features:
        result["flow_features"] = flow_features

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════

def get_model_info() -> Dict[str, Any]:
    """Return current model/classifier metadata."""
    global _initialized

    if not _initialized:
        load_model()

    arch_label = "CNN+LSTM (MobileNetV2)" if settings.model_arch == "cnn_lstm" else "3D-CNN"

    info = {
        "architecture":         arch_label,
        "is_loaded":            _initialized,
        # System is out of demo mode when the flow classifier is active
        "is_demo":              not _flow_loaded,
        "weights_loaded":       not _cnn_is_demo,
        "flow_classifier":      _flow_loaded,
        "model_path":           settings.model_path,
        "input_frames":         settings.max_frames,
        "frame_size":           settings.frame_size,
        "confidence_threshold": settings.confidence_threshold,
        "num_parameters":       _cnn_parameters,
        "parameters":           _cnn_parameters,
        "device":               (
            str(next(_cnn_model.parameters()).device)
            if _cnn_model else "cpu"
        ),
        "ensemble_method": (
            "Optical Flow (75%) + CNN+LSTM (25%)"
            if (_flow_loaded and not _cnn_is_demo)
            else ("Optical Flow only" if _flow_loaded else "CNN+LSTM only")
        ),
    }

    # Add checkpoint metadata when CNN trained weights are loaded
    if not _cnn_is_demo and _checkpoint_meta:
        info["trained_epoch"]    = _checkpoint_meta.get("epoch")
        info["val_accuracy"]     = _checkpoint_meta.get("val_acc")
        info["val_loss"]         = _checkpoint_meta.get("val_loss")
        info["training_history"] = _checkpoint_meta.get("history", [])
        eval_data = _checkpoint_meta.get("evaluation", {})
        if eval_data:
            info["evaluation"] = eval_data

    return info
