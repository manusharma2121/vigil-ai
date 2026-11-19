"""
VIGIL AI – Recalibrate flow classifier and test on uploaded videos
"""
import sys
import numpy as np
import pickle
sys.path.insert(0, 'backend')

from services.flow_classifier import (
    train_flow_classifier, save_flow_classifier,
    predict_flow, extract_flow_features
)
from pathlib import Path

# ── 1. Retrain ─────────────────────────────────────────────────────────────
print("Step 1: Retraining with real-world calibrated distributions...")
clf = train_flow_classifier(n_per_class=3000, seed=42)
save_flow_classifier(clf)
print("Classifier saved.\n")

# ── 2. Test on uploaded videos ──────────────────────────────────────────────
upload_dir = Path("backend/uploads")
videos = sorted(upload_dir.glob("*.mp4"))
print(f"Step 2: Testing on {len(list(videos))} uploaded videos...\n")

header = f"{'Video':36s}  {'Pred':12s}  v_prob  tvar   accel  std    max"
print(header)
print("-" * 90)

results = []
for vp in videos:
    f = extract_flow_features(str(vp))
    if f is None:
        continue
    nv, v = predict_flow(str(vp), clf)
    pred = "VIOLENT" if v >= 0.45 else "Non-Violent"
    results.append((vp.name, pred, v, nv))
    row = f"{vp.name[:36]:36s}  {pred:12s}  {v:.3f}   {f[4]:.2f}   {f[5]:.2f}   {f[1]:.2f}   {f[2]:.1f}"
    print(row)

v_count  = sum(1 for r in results if r[1] == "VIOLENT")
nv_count = len(results) - v_count
print()
print(f"==> Violent: {v_count}  Non-Violent: {nv_count}  Total: {len(results)}")
print("DONE")
