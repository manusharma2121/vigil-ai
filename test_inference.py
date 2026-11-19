"""
VIGIL AI – Test inference pipeline with current model state.
Run this after training to verify predictions are meaningful.
"""
import sys, json
sys.path.insert(0, r'c:\deep_lr_project\backend')
from pathlib import Path
from services.inference import load_model, predict, get_model_info

print("=" * 60)
print("VIGIL AI – Inference Pipeline Test")
print("=" * 60)

# Load model
loaded = load_model()
info = get_model_info()

print(f"\nModel Status:")
print(f"  Architecture  : {info['architecture']}")
print(f"  Weights loaded: {info.get('weights_loaded', not info['is_demo'])}")
print(f"  Demo mode     : {info['is_demo']}")
print(f"  Parameters    : {info.get('parameters', info.get('num_parameters', 'N/A')):,}" if info.get('parameters') or info.get('num_parameters') else "  Parameters    : N/A")
print(f"  Device        : {info.get('device', 'N/A')}")

if not info['is_demo']:
    print(f"  Trained epoch : {info.get('trained_epoch', 'N/A')}")
    print(f"  Val accuracy  : {info.get('val_accuracy', 0)*100:.2f}%" if info.get('val_accuracy') else "")
    if info.get('evaluation'):
        ev = info['evaluation']
        print(f"\nEvaluation Results:")
        for k, v in ev.items():
            if k != 'confusion_matrix':
                print(f"  {k:12s}: {v:.4f}" if isinstance(v, float) else f"  {k:12s}: {v}")
        if 'confusion_matrix' in ev:
            cm = ev['confusion_matrix']
            print(f"  Confusion Matrix:")
            print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
            print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
else:
    print("\n  ⚠ Running in DEMO MODE — predictions are random placeholders.")
    print("  To train a real model, run:")
    print("    python train_model.py --dataset rwf2000")

# Test prediction on uploaded videos
uploads = Path(r'c:\deep_lr_project\backend\uploads')
videos = list(uploads.glob('*.mp4'))[:5]

if videos:
    print(f"\nTest predictions on {len(videos)} uploaded videos:")
    print("-" * 60)
    for v in videos:
        try:
            result = predict(str(v))
            demo_flag = "[DEMO]" if result['is_demo'] else "[TRAINED]"
            print(f"  {v.name[:35]:35s} → {result['prediction']:11s}  "
                  f"v={result['violence_prob']:.3f}  conf={result['confidence']:.3f}  {demo_flag}")
        except Exception as e:
            print(f"  {v.name[:35]:35s} → ERROR: {e}")
else:
    print("\nNo uploaded videos found in backend/uploads/")

print("\n" + "=" * 60)
