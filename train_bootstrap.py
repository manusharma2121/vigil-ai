"""
VIGIL AI – Bootstrap Training using Uploaded Videos + Pre-trained Features
===========================================================================
Since RWF-2000 is not available, this script:
1. Uses uploaded user videos as a labeled dataset (requires user to label them)
2. Applies heavy data augmentation to expand the dataset
3. Uses MobileNetV2 transfer learning (ImageNet weights) with frozen backbone
4. Trains only the LSTM + classifier head
5. Saves real trained weights to models/weights/vigil_model.pth

This produces a REAL trained model (not demo), even with limited data.
"""

import sys, os, random, time, json, logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
import cv2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

ROOT    = Path(r"c:\deep_lr_project")
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from models.cnn_lstm import build_cnn_lstm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(ROOT / "training_bootstrap.log"), mode="w"),
    ]
)
log = logging.getLogger("vigil.bootstrap")

MEAN       = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD        = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NUM_FRAMES = 16
FRAME_SIZE = 112


# ══════════════════════════════════════════════════════════════════════════════
# Frame extraction with strong augmentation
# ══════════════════════════════════════════════════════════════════════════════

def extract_frames(video_path: str, n_frames: int = NUM_FRAMES,
                   augment: bool = False) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return np.zeros((n_frames, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.float32)

    # Random temporal crop for augmentation
    if augment and total > n_frames * 2:
        start = random.randint(0, total - n_frames * 2)
        end   = start + n_frames * 2
    else:
        start, end = 0, total - 1

    indices = np.linspace(start, end, n_frames, dtype=int)
    indices = np.clip(indices, 0, total - 1)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        frames.append(frame.astype(np.float32))
    cap.release()

    frames = np.stack(frames)  # (T, H, W, 3)

    if augment:
        # Horizontal flip
        if random.random() < 0.5:
            frames = frames[:, :, ::-1, :]
        # Brightness
        alpha = random.uniform(0.8, 1.2)
        frames = np.clip(frames * alpha, 0, 255)
        # Gaussian noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 5, frames.shape).astype(np.float32)
            frames = np.clip(frames + noise, 0, 255)
        # Reverse temporal order (mimics backward motion)
        if random.random() < 0.2:
            frames = frames[::-1]

    # Normalize
    frames = frames / 255.0
    frames = (frames - MEAN) / STD
    return frames.astype(np.float32)


class VideoDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = extract_frames(str(path), augment=self.augment)  # (T, H, W, 3)
        tensor = torch.from_numpy(np.transpose(frames, (0, 3, 1, 2)))  # (T, 3, H, W)
        return tensor, int(label)


# ══════════════════════════════════════════════════════════════════════════════
# Data collection
# ══════════════════════════════════════════════════════════════════════════════

def collect_samples_from_folder(root: Path):
    """Collect from Fight/NonFight or train/Fight/NonFight structure."""
    exts = {".avi", ".mp4", ".mov", ".mkv", ".webm"}
    samples = []

    def _scan(folder, label):
        for f in folder.rglob("*"):
            if f.suffix.lower() in exts:
                samples.append((f, label))

    # Try train/val structure first
    if (root / "train").exists():
        for sub in (root / "train").iterdir():
            if sub.is_dir():
                lname = sub.name.lower()
                lbl = 1 if ("fight" in lname and "non" not in lname) else 0
                _scan(sub, lbl)
    else:
        # Flat Fight/NonFight
        for sub in root.iterdir():
            if sub.is_dir():
                lname = sub.name.lower()
                lbl = 1 if ("fight" in lname and "non" not in lname) else 0
                _scan(sub, lbl)
    return samples


def build_samples(dataset_names=("rwf2000", "hockey")):
    all_samples = []
    for name in dataset_names:
        root = ROOT / "datasets" / name
        if not root.exists():
            continue
        s = collect_samples_from_folder(root)
        if s:
            log.info(f"  {name}: {len(s)} clips — "
                     f"Violent={sum(l for _,l in s)}  "
                     f"Non-Violent={sum(1-l for _,l in s)}")
            all_samples.extend(s)
    return all_samples


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    samples,
    epochs=25,
    batch_size=4,
    lr=5e-5,
    output_path=ROOT / "models" / "weights" / "vigil_model.pth",
    patience=7,
    freeze_backbone=True,
):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  Samples: {len(samples)}  |  Epochs: {epochs}")

    if len(samples) < 4:
        log.error(f"Only {len(samples)} samples — need at least 4 to train.")
        return False

    # Train/val/test split
    random.shuffle(samples)
    n = len(samples)
    n_test  = max(1, int(0.1 * n))
    n_val   = max(1, int(0.1 * n))
    test_s  = samples[:n_test]
    val_s   = samples[n_test:n_test + n_val]
    train_s = samples[n_test + n_val:]
    log.info(f"Split — Train: {len(train_s)}  Val: {len(val_s)}  Test: {len(test_s)}")

    # Class balance
    labels = [l for _, l in train_s]
    n_v  = sum(labels)
    n_nv = len(labels) - n_v
    log.info(f"Train — Violent: {n_v}  Non-Violent: {n_nv}")

    # Weighted sampler
    weights = [1.0/n_nv if l == 0 else 1.0/n_v for l in labels] if n_v > 0 and n_nv > 0 else [1.0]*len(labels)
    sampler = WeightedRandomSampler(weights, len(train_s), replacement=True)

    # Loss weights
    total = n_v + n_nv
    loss_w = torch.tensor(
        [total/(2*max(n_nv,1)), total/(2*max(n_v,1))], dtype=torch.float32
    ).to(device) if n_v > 0 and n_nv > 0 else None

    train_ds = VideoDataset(train_s, augment=True)
    val_ds   = VideoDataset(val_s,   augment=False)
    test_ds  = VideoDataset(test_s,  augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,   num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,   num_workers=0)

    # Model: freeze backbone, train LSTM + head only
    model = build_cnn_lstm(freeze_cnn=freeze_backbone).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(weight=loss_w)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    no_improve = 0
    history = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device).long()
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item() * len(by)
            tc += (out.argmax(1) == by).sum().item()
            tt += len(by)
        scheduler.step()
        train_loss, train_acc = tl/tt, tc/tt

        # Val
        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device).long()
                out = model(bx)
                vl += criterion(out, by).item() * len(by)
                vc += (out.argmax(1) == by).sum().item()
                vt += len(by)
        val_loss, val_acc = vl/vt, vc/vt

        history.append({"epoch": epoch, "train_loss": round(train_loss,4),
                         "val_loss": round(val_loss,4), "train_acc": round(train_acc,4),
                         "val_acc": round(val_acc,4)})

        log.info(f"Epoch {epoch:3d}/{epochs} | "
                 f"TrainLoss={train_loss:.4f} TrainAcc={train_acc*100:.2f}% | "
                 f"ValLoss={val_loss:.4f} ValAcc={val_acc*100:.2f}%")

        improved = val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss)
        if improved:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            no_improve    = 0
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_loss":         val_loss,
                "val_acc":          val_acc,
                "train_loss":       train_loss,
                "train_acc":        train_acc,
                "architecture":     "cnn_lstm",
                "num_frames":       NUM_FRAMES,
                "frame_size":       FRAME_SIZE,
                "history":          history,
            }, str(output_path))
            log.info(f"  ✓ Best model saved  val_acc={val_acc*100:.2f}%")
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"Early stopping at epoch {epoch}"); break

    # Load best and evaluate on test
    ckpt = torch.load(str(output_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            out = model(bx)
            probs = torch.softmax(out, dim=1)[:, 1]
            preds = out.argmax(1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(by.tolist())
            all_probs.extend(probs.cpu().tolist())

    if len(set(all_labels)) >= 2:
        acc  = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec  = recall_score(all_labels, all_preds, zero_division=0)
        f1   = f1_score(all_labels, all_preds, zero_division=0)
        cm   = confusion_matrix(all_labels, all_preds).tolist()
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            auc = 0.0
        metrics = {"accuracy": acc, "precision": prec, "recall": rec,
                   "f1": f1, "roc_auc": auc, "confusion_matrix": cm}
        log.info(f"\n{'='*55}\nTest Results")
        for k, v in metrics.items():
            if k != "confusion_matrix":
                log.info(f"  {k:12s}: {v:.4f}")
        log.info(f"  Confusion Matrix: {cm}")
    else:
        metrics = {"note": "Single class in test set — accuracy not meaningful"}
        log.info("Test set has only one class. Using val metrics.")
        metrics["accuracy"] = ckpt["val_acc"]

    ckpt["evaluation"] = metrics
    torch.save(ckpt, str(output_path))
    log.info(f"\n✓ Training complete. Weights saved to: {output_path}")
    log.info(f"  Best Val Acc: {best_val_acc*100:.2f}%")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",  type=int,   default=25)
    parser.add_argument("--batch",   type=int,   default=4)
    parser.add_argument("--lr",      type=float, default=5e-5)
    parser.add_argument("--unfreeze", action="store_true",
                        help="Unfreeze backbone (slower but better with large dataset)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("VIGIL AI – Training Pipeline")
    log.info("=" * 60)
    log.info("Scanning datasets...\n")

    samples = build_samples(("rwf2000", "hockey"))

    if not samples:
        log.error(
            "\nNo dataset videos found!\n"
            "Please place your dataset in one of these locations:\n"
            "  datasets/rwf2000/train/Fight/     (and NonFight/)\n"
            "  datasets/rwf2000/train/NonFight/\n"
            "  datasets/hockey/Fight/\n"
            "  datasets/hockey/NonFight/\n"
            "\nThen re-run:  python train_bootstrap.py\n"
        )
        sys.exit(1)

    success = train_model(
        samples,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        freeze_backbone=not args.unfreeze,
    )
    sys.exit(0 if success else 1)
