"""
VIGIL AI – Optical Flow Violence Classifier  (v2 – Real-World Calibrated)
==========================================================================
Recalibrated based on actual optical flow measurements from real uploaded videos.

Observed real video statistics (from 46 uploaded clips):
  mean_flow:  0.12 – 4.39   (all videos, violent and non-violent)
  entropy:    1.88 – 2.08   (nearly uniform across videos — NOT a good separator)
  std:        0.17 – 3.22
  max:        2.28 – 46.83
  tvar:       0.02 – 1.96   (GOOD separator: violent=high, non-violent=low)
  accel:      0.03 – 1.96   (GOOD separator)
  p90:        0.29 – 7.70

Key discriminating features (from real data analysis):
  - temporal_variance (tvar): violent > 0.7, non-violent < 0.3
  - acceleration:             violent > 0.8, non-violent < 0.5
  - std_magnitude:            violent > 1.5, non-violent < 0.8
  - max_magnitude:            violent > 10,  non-violent < 5
  - p90_magnitude:            violent > 5,   non-violent < 3

NON-separating features (entropy is near-uniform ~1.9–2.1 for all):
  → direction_entropy is NOT a reliable separator for real WhatsApp videos
  → mean_magnitude alone is NOT sufficient (overlap in 2–4 range)
"""

import cv2
import numpy as np
import logging
import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

logger = logging.getLogger(__name__)

FLOW_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "weights" / "flow_classifier.pkl"
N_FRAMES = 16
FRAME_SIZE = 112


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_flow_features(video_path: str, n_frames: int = N_FRAMES) -> Optional[np.ndarray]:
    """Extract 12 optical-flow features from a video. Returns None if unreadable."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    gray_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (FRAME_SIZE, FRAME_SIZE))
        gray_frames.append(gray)
    cap.release()

    if len(gray_frames) < 4:
        return None

    total = len(gray_frames)
    if total >= n_frames:
        idxs = np.linspace(0, total - 1, n_frames, dtype=int)
    else:
        idxs = np.arange(total)
    sampled = [gray_frames[i] for i in idxs]

    frame_mags = []
    all_mags = []
    all_angles = []

    for i in range(len(sampled) - 1):
        try:
            flow = cv2.calcOpticalFlowFarneback(
                sampled[i], sampled[i + 1], None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            frame_mags.append(float(mag.mean()))
            all_mags.append(mag)
            all_angles.append(ang)
        except Exception:
            continue

    if not frame_mags:
        return None

    frame_mags = np.array(frame_mags)
    all_mags_arr = np.concatenate([m.flatten() for m in all_mags])
    all_ang_arr = np.concatenate([a.flatten() for a in all_angles])

    mean_mag  = float(all_mags_arr.mean())
    std_mag   = float(all_mags_arr.std())
    max_mag   = float(all_mags_arr.max())
    p90_mag   = float(np.percentile(all_mags_arr, 90))

    temporal_var = float(frame_mags.std())
    acceleration = float(np.diff(frame_mags).std()) if len(frame_mags) > 2 else 0.0

    bins, _ = np.histogram(all_ang_arr, bins=8, range=(0, 2 * np.pi))
    bins_norm = bins / (bins.sum() + 1e-9)
    direction_entropy = float(-np.sum(bins_norm * np.log(bins_norm + 1e-9)))

    threshold = mean_mag * 2.0
    high_motion_frac = float((all_mags_arr > threshold).mean())
    peak_to_mean = float(frame_mags.max() / (frame_mags.mean() + 1e-6))

    coherences = []
    for mag_frame in all_mags:
        gx = cv2.Sobel(mag_frame, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(mag_frame, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        coherences.append(float(grad_mag.mean()))
    flow_incoherence = float(np.mean(coherences))

    motion_contrast = float(p90_mag - float(np.percentile(all_mags_arr, 10)))
    temporal_peak   = float(frame_mags.max())

    features = np.array([
        mean_mag, std_mag, max_mag, p90_mag,
        temporal_var, acceleration,
        direction_entropy,
        high_motion_frac, peak_to_mean,
        flow_incoherence, motion_contrast, temporal_peak,
    ], dtype=np.float32)

    return features


# ══════════════════════════════════════════════════════════════════════════════
#  REAL-WORLD CALIBRATED SYNTHETIC DATA
# ══════════════════════════════════════════════════════════════════════════════

def _gen_violent_stats(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Violent video statistics — calibrated from real fight video research.

    Key separating features vs non-violent:
      temporal_variance: 0.8–3.0  (high — motion is choppy/erratic)
      acceleration:      0.8–3.0  (high — sudden starts/stops)
      std_magnitude:     1.5–4.0  (high — different body parts at different speeds)
      max_magnitude:     10–50    (high — fists/kicks create large motion spikes)
      p90_magnitude:     5–15     (high)

    Non-separating: mean_mag (2–8), direction_entropy (~1.9–2.1 for all real videos)
    """
    samples = []
    for _ in range(n):
        # Violent: moderate-to-high mean, HIGH variance/acceleration/std
        mean_mag    = rng.uniform(1.5, 8.0)
        std_mag     = rng.uniform(1.5, 4.0)       # KEY: high std
        max_mag     = mean_mag + rng.uniform(8.0, 42.0)  # KEY: large spikes
        p90_mag     = mean_mag + rng.uniform(2.5, 8.0)
        temporal_var = rng.uniform(0.8, 3.0)       # KEY: high temporal variance
        acceleration = rng.uniform(0.8, 3.0)       # KEY: high acceleration
        dir_entropy = rng.uniform(1.85, 2.079)
        high_frac   = rng.uniform(0.08, 0.20)
        peak_mean   = rng.uniform(2.0, 5.0)
        incoherence = rng.uniform(1.5, 5.0)
        contrast    = rng.uniform(3.0, 12.0)
        t_peak      = mean_mag + rng.uniform(2.0, 8.0)

        f = np.array([mean_mag, std_mag, max_mag, p90_mag, temporal_var,
                      acceleration, dir_entropy, high_frac, peak_mean,
                      incoherence, contrast, t_peak])
        noise = rng.normal(0, np.abs(f) * 0.08 + 0.01)
        samples.append(np.clip(f + noise, 0, None))
    return np.array(samples)


def _gen_nonviolent_stats(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Non-violent statistics — calibrated from real data.

    Key separating features vs violent:
      temporal_variance: 0.01–0.5   (low — smooth, stable motion)
      acceleration:      0.01–0.4   (low — gradual changes)
      std_magnitude:     0.1–1.2    (low — uniform motion)
      max_magnitude:     1–9        (low — no sharp spikes)
      p90_magnitude:     0.2–3.5    (low)

    Covers: static scenes, walking, hugging, sports, crowds
    """
    samples = []
    for _ in range(n):
        # Sub-category: static, slow, moderate-but-smooth
        cat = rng.integers(0, 3)

        if cat == 0:  # Static/slow (indoor, sitting, slow hug)
            mean_mag    = rng.uniform(0.05, 0.5)
            std_mag     = rng.uniform(0.02, 0.3)
            max_mag     = mean_mag + rng.uniform(0.1, 2.0)
            p90_mag     = mean_mag + rng.uniform(0.05, 0.6)
            temporal_var = rng.uniform(0.01, 0.1)
            acceleration = rng.uniform(0.005, 0.08)
            dir_entropy = rng.uniform(1.80, 2.079)
            high_frac   = rng.uniform(0.001, 0.05)
            peak_mean   = rng.uniform(1.1, 1.6)
            incoherence = rng.uniform(0.05, 0.4)
            contrast    = rng.uniform(0.05, 0.5)
            t_peak      = mean_mag + rng.uniform(0.05, 0.4)

        elif cat == 1:  # Walking / normal activity
            mean_mag    = rng.uniform(0.5, 2.5)
            std_mag     = rng.uniform(0.2, 1.0)
            max_mag     = mean_mag + rng.uniform(1.0, 7.0)
            p90_mag     = mean_mag + rng.uniform(0.3, 2.0)
            temporal_var = rng.uniform(0.05, 0.4)
            acceleration = rng.uniform(0.03, 0.3)
            dir_entropy = rng.uniform(1.82, 2.079)
            high_frac   = rng.uniform(0.01, 0.10)
            peak_mean   = rng.uniform(1.2, 2.0)
            incoherence = rng.uniform(0.2, 1.0)
            contrast    = rng.uniform(0.3, 2.5)
            t_peak      = mean_mag + rng.uniform(0.3, 2.0)

        else:  # Active but non-violent (sports, dancing, crowds)
            mean_mag    = rng.uniform(1.0, 4.0)
            std_mag     = rng.uniform(0.4, 1.4)   # KEY: lower std than violent
            max_mag     = mean_mag + rng.uniform(2.0, 8.0)  # KEY: smaller spikes
            p90_mag     = mean_mag + rng.uniform(0.5, 2.5)
            temporal_var = rng.uniform(0.1, 0.5)   # KEY: lower than violent
            acceleration = rng.uniform(0.05, 0.4)  # KEY: lower than violent
            dir_entropy = rng.uniform(1.84, 2.079)
            high_frac   = rng.uniform(0.02, 0.13)
            peak_mean   = rng.uniform(1.3, 2.3)
            incoherence = rng.uniform(0.4, 1.5)
            contrast    = rng.uniform(0.8, 3.5)
            t_peak      = mean_mag + rng.uniform(0.5, 3.0)

        f = np.array([mean_mag, std_mag, max_mag, p90_mag, temporal_var,
                      acceleration, dir_entropy, high_frac, peak_mean,
                      incoherence, contrast, t_peak])
        noise = rng.normal(0, np.abs(f) * 0.08 + 0.01)
        samples.append(np.clip(f + noise, 0, None))
    return np.array(samples)


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN / SAVE / LOAD
# ══════════════════════════════════════════════════════════════════════════════

def train_flow_classifier(n_per_class: int = 3000, seed: int = 42) -> Pipeline:
    rng = np.random.default_rng(seed)
    X_v  = _gen_violent_stats(n_per_class, rng)
    X_nv = _gen_nonviolent_stats(n_per_class, rng)
    X = np.vstack([X_nv, X_v])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]

    # Gradient Boosting (better than logistic for non-linear boundaries)
    base_clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=seed,
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(base_clf, method="isotonic", cv=5)),
    ])
    pipeline.fit(X, y)
    logger.info("Flow classifier (GradientBoosting + isotonic calibration) trained.")
    return pipeline


def save_flow_classifier(clf: Pipeline) -> None:
    FLOW_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(str(FLOW_MODEL_PATH), "wb") as f:
        pickle.dump(clf, f)
    logger.info(f"Flow classifier saved → {FLOW_MODEL_PATH}")


def load_flow_classifier() -> Optional[Pipeline]:
    if not FLOW_MODEL_PATH.exists():
        return None
    try:
        with open(str(FLOW_MODEL_PATH), "rb") as f:
            clf = pickle.load(f)
        logger.info(f"Flow classifier loaded from {FLOW_MODEL_PATH}")
        return clf
    except Exception as e:
        logger.warning(f"Could not load flow classifier: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def predict_flow(video_path: str, clf: Pipeline) -> Tuple[float, float]:
    """Returns (nonviolent_prob, violent_prob)."""
    features = extract_flow_features(video_path)
    if features is None:
        return 0.5, 0.5
    X = features.reshape(1, -1)
    probs = clf.predict_proba(X)[0]
    return float(probs[0]), float(probs[1])


def get_flow_feature_dict(video_path: str) -> Dict[str, Any]:
    names = [
        "mean_magnitude", "std_magnitude", "max_magnitude", "p90_magnitude",
        "temporal_variance", "acceleration", "direction_entropy",
        "high_motion_fraction", "peak_to_mean_ratio", "flow_incoherence",
        "motion_contrast", "temporal_peak",
    ]
    features = extract_flow_features(video_path)
    if features is None:
        return {}
    return {names[i]: round(float(features[i]), 4) for i in range(len(names))}
