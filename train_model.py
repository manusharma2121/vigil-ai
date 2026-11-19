"""
VIGIL AI – Complete Training Pipeline
======================================
Trains CNN+LSTM on RWF-2000 (or any folder of Fight/NonFight videos).
Saves weights to models/weights/vigil_model.pth.

Usage:
  python train_model.py [--dataset rwf2000|hockey|both] [--epochs 20] [--batch 8]

Folder structures supported:
  RWF-2000:  datasets/rwf2000/train/Fight/  datasets/rwf2000/train/NonFight/
             datasets/rwf2000/val/Fight/    datasets/rwf2000/val/NonFight/
  Hockey:    datasets/hockey/Fight/         datasets/hockey/NonFight/
  Custom:    datasets/<name>/Fight/         datasets/<name>/NonFight/
"""

import sys, os, time, random, logging, argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from models.cnn_lstm import build_cnn_lstm
from models.model_3dcnn import build_3dcnn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "training.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("vigil.train")

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_FRAMES = 16
FRAME_SIZE = 112
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ══════════════════════════════════════════════════════════════════════════════
# Dataset helpers
# ══════════════════════════════════════════════════════════════════════════════

def find_videos(folder: Path) -> List[Tuple[Path, int]]:
    """
    Recursively collect (path, label) pairs.
    Label: 1 = Violent (Fight), 0 = Non-Violent (NonFight / NonFight)
    """
    exts = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".flv"}
    samples = []
    for sub in folder.iterdir():
        if not sub.is_dir():
            continue
        name_lower = sub.name.lower()
        if "fight" in name_lower and "non" not in name_lower:
            label = 1
        elif "nonfight" in name_lower or "non" in name_lower or "nonviolent" in name_lower:
            label = 0
        elif "violent" in name_lower and "non" not in name_lower:
            label = 1
        else:
            log.warning(f"Cannot determine label for folder: {sub}. Skipping.")
            continue
        for f in sub.rglob("*"):
            if f.suffix.lower() in exts:
                samples.append((f, label))
    return samples


def collect_all_samples(dataset_root: Path) -> Tuple[List, List, List]:
    """
    Returns (train, val, test) sample lists.
    Supports both pre-split (train/val dirs) and flat (Fight/NonFight) structures.
    """
    # Pre-split structure: root/train/Fight, root/val/Fight
    train_dir = dataset_root / "train"
    val_dir   = dataset_root / "val"
    test_dir  = dataset_root / "test"

    if train_dir.exists():
        train = find_videos(train_dir)
        val   = find_videos(val_dir)   if val_dir.exists()  else []
        test  = find_videos(test_dir)  if test_dir.exists() else []

        # If no val split, carve 10% from train
        if not val:
            random.shuffle(train)
            n_val = max(1, int(0.1 * len(train)))
            val, train = train[:n_val], train[n_val:]

        return train, val, test

    # Flat structure: root/Fight, root/NonFight
    all_samples = find_videos(dataset_root)
    if not all_samples:
        return [], [], []

    random.shuffle(all_samples)
    n = len(all_samples)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    return (
        all_samples[:n_train],
        all_samples[n_train:n_train + n_val],
        all_samples[n_train + n_val:],
    )


# ══════════════════════════════════════════════════════════════════════════════
# Frame extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_frames_augmented(video_path: str, n_frames: int = NUM_FRAMES,
                              augment: bool = False) -> np.ndarray:
    """
    Extract n_frames uniformly from video. Apply augmentation during training.
    Returns ndarray (n_frames, H, W, 3) in RGB, or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        frames.append(frame)
    cap.release()

    frames = np.stack(frames)  # (T, H, W, 3)

    if augment:
        # Horizontal flip (50%)
        if random.random() < 0.5:
            frames = frames[:, :, ::-1, :]
        # Brightness jitter (±30)
        delta = random.randint(-30, 30)
        frames = np.clip(frames.astype(np.int32) + delta, 0, 255).astype(np.uint8)
        # Temporal crop: randomly skip a few leading frames
        skip = random.randint(0, 2)
        if skip > 0 and len(frames) > skip:
            new_indices = np.linspace(skip, len(frames) - 1, n_frames, dtype=int)
            frames = frames[new_indices]

    return frames.astype(np.float32)


def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """(T, H, W, 3) → (1, T, 3, H, W) normalized tensor."""
    # Normalize
    mean = np.array(MEAN, dtype=np.float32)
    std  = np.array(STD,  dtype=np.float32)
    frames = frames / 255.0
    frames = (frames - mean) / std
    # (T, H, W, 3) → (T, 3, H, W)
    frames = np.transpose(frames, (0, 3, 1, 2))
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).float()
    return tensor.unsqueeze(0)  # (1, T, 3, H, W)


# ══════════════════════════════════════════════════════════════════════════════
# PyTorch Dataset
# ══════════════════════════════════════════════════════════════════════════════

class VideoDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], augment: bool = False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = extract_frames_augmented(str(path), augment=self.augment)
        if frames is None:
            frames = np.zeros((NUM_FRAMES, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.float32)

        # Normalize inline
        frames = frames / 255.0
        mean = np.array(MEAN, dtype=np.float32)
        std  = np.array(STD,  dtype=np.float32)
        frames = (frames - mean) / std
        # (T, H, W, 3) → (T, 3, H, W)
        frames = np.transpose(frames, (0, 3, 1, 2))
        tensor = torch.from_numpy(np.ascontiguousarray(frames)).float()
        return tensor, label


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).long()
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(batch_y)
        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += len(batch_y)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device) -> Tuple[float, float, List, List]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).long()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        total_loss += loss.item() * len(batch_y)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += len(batch_y)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch_y.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
    return total_loss / total, correct / total, all_labels, all_preds, all_probs


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation report
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_and_report(labels, preds, probs, split="Test") -> Dict:
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    cm   = confusion_matrix(labels, preds).tolist()
    try:
        auc = roc_auc_score(labels, probs) if len(set(labels)) == 2 else 0.0
    except Exception:
        auc = 0.0

    log.info(f"\n{'='*55}")
    log.info(f"  {split} Evaluation Results")
    log.info(f"{'='*55}")
    log.info(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    log.info(f"  Precision : {prec:.4f}")
    log.info(f"  Recall    : {rec:.4f}")
    log.info(f"  F1-Score  : {f1:.4f}")
    log.info(f"  ROC-AUC   : {auc:.4f}")
    log.info(f"  Confusion Matrix:")
    log.info(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    log.info(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    log.info(f"{'='*55}\n")

    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "roc_auc": auc, "confusion_matrix": cm}


# ══════════════════════════════════════════════════════════════════════════════
# Main training function
# ══════════════════════════════════════════════════════════════════════════════

def train(
    dataset_root: Path,
    model_arch: str = "cnn_lstm",
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-4,
    output_path: Path = ROOT / "models" / "weights" / "vigil_model.pth",
    patience: int = 5,
):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    log.info(f"Dataset root: {dataset_root}")
    log.info(f"Architecture: {model_arch}  |  Epochs: {epochs}  |  Batch: {batch_size}  |  LR: {lr}")

    # ── Collect samples ───────────────────────────────────────────────────────
    train_samples, val_samples, test_samples = collect_all_samples(dataset_root)
    log.info(f"Samples — Train: {len(train_samples)}  Val: {len(val_samples)}  Test: {len(test_samples)}")

    if len(train_samples) == 0:
        log.error("No training samples found! Check dataset path and folder structure.")
        return False

    # Class balance
    train_labels = [s[1] for s in train_samples]
    n_violent   = sum(train_labels)
    n_nonviolent = len(train_labels) - n_violent
    log.info(f"Train class balance — Violent: {n_violent}  Non-Violent: {n_nonviolent}")

    # Weighted sampler for class imbalance
    if n_violent > 0 and n_nonviolent > 0:
        class_weights = [1.0 / n_nonviolent if l == 0 else 1.0 / n_violent for l in train_labels]
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(class_weights, num_samples=len(train_samples), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    # Compute class weights for loss
    if n_violent > 0 and n_nonviolent > 0:
        total = n_violent + n_nonviolent
        loss_weights = torch.tensor([total / (2 * n_nonviolent),
                                      total / (2 * n_violent)], dtype=torch.float32).to(device)
    else:
        loss_weights = None

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_ds = VideoDataset(train_samples, augment=True)
    val_ds   = VideoDataset(val_samples,   augment=False)
    test_ds  = VideoDataset(test_samples,  augment=False) if test_samples else None

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               sampler=sampler, shuffle=shuffle,
                               num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                               shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    if model_arch == "3dcnn":
        model = build_3dcnn().to(device)
    else:
        model = build_cnn_lstm(freeze_cnn=False).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {n_params:,}")

    # ── Loss & optimiser ──────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, min_lr=1e-6
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_val_acc  = 0.0
    no_improve    = 0
    history       = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("\nStarting training...\n")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss, 4),
            "train_acc":  round(train_acc, 4),
            "val_acc":    round(val_acc, 4),
        })

        log.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"TrainLoss={train_loss:.4f} TrainAcc={train_acc*100:.2f}% | "
            f"ValLoss={val_loss:.4f} ValAcc={val_acc*100:.2f}% | "
            f"{elapsed:.1f}s"
        )

        # Save best checkpoint
        if val_loss < best_val_loss or val_acc > best_val_acc:
            best_val_loss = min(val_loss, best_val_loss)
            best_val_acc  = max(val_acc,  best_val_acc)
            no_improve = 0
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "val_loss":         val_loss,
                "val_acc":          val_acc,
                "train_loss":       train_loss,
                "train_acc":        train_acc,
                "architecture":     model_arch,
                "num_frames":       NUM_FRAMES,
                "frame_size":       FRAME_SIZE,
                "history":          history,
            }, str(output_path))
            log.info(f"  ✓ Saved best model (val_acc={val_acc*100:.2f}%)")
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    log.info(f"\nTraining complete. Best Val Acc: {best_val_acc*100:.2f}%")
    log.info(f"Model saved to: {output_path}")

    # ── Final evaluation on test set ──────────────────────────────────────────
    # Reload best weights
    checkpoint = torch.load(str(output_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if test_ds and len(test_ds) > 0:
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        _, _, t_labels, t_preds, t_probs = eval_epoch(model, test_loader, criterion, device)
        metrics = evaluate_and_report(t_labels, t_preds, t_probs, split="Test")
    else:
        # Evaluate on val set when no test set available
        _, _, v_labels, v_preds, v_probs = eval_epoch(model, val_loader, criterion, device)
        metrics = evaluate_and_report(v_labels, v_preds, v_probs, split="Validation (used as test)")

    # Append metrics to checkpoint
    checkpoint["evaluation"] = metrics
    torch.save(checkpoint, str(output_path))
    log.info("Evaluation metrics saved into checkpoint.")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VIGIL AI model")
    parser.add_argument("--dataset",  default="rwf2000",
                        help="Dataset name: rwf2000 | hockey | both | path to custom folder")
    parser.add_argument("--arch",     default="cnn_lstm", choices=["cnn_lstm", "3dcnn"])
    parser.add_argument("--epochs",   type=int,   default=20)
    parser.add_argument("--batch",    type=int,   default=4)
    parser.add_argument("--lr",       type=float, default=1e-4)
    parser.add_argument("--patience", type=int,   default=5)
    args = parser.parse_args()

    dataset_dirs = {
        "rwf2000": ROOT / "datasets" / "rwf2000",
        "hockey":  ROOT / "datasets" / "hockey",
    }

    if args.dataset == "both":
        # Merge both datasets into a temporary combined structure
        # by training sequentially (hockey first as it's simpler, then rwf)
        datasets_to_train = []
        for name, path in dataset_dirs.items():
            train_s, val_s, test_s = collect_all_samples(path)
            if train_s:
                datasets_to_train.append((name, path))

        if not datasets_to_train:
            log.error("No datasets found. Please download RWF-2000 or Hockey dataset.")
            sys.exit(1)

        output_path = ROOT / "models" / "weights" / "vigil_model.pth"
        for name, path in datasets_to_train:
            log.info(f"\n{'='*55}")
            log.info(f"Training on: {name.upper()}")
            log.info(f"{'='*55}")
            train(path, args.arch, args.epochs, args.batch, args.lr, output_path, args.patience)

    elif args.dataset in dataset_dirs:
        path = dataset_dirs[args.dataset]
        output_path = ROOT / "models" / "weights" / "vigil_model.pth"
        success = train(path, args.arch, args.epochs, args.batch, args.lr, output_path, args.patience)
        if not success:
            sys.exit(1)

    else:
        # Treat as a direct path
        path = Path(args.dataset)
        if not path.exists():
            log.error(f"Dataset path not found: {path}")
            sys.exit(1)
        output_path = ROOT / "models" / "weights" / "vigil_model.pth"
        train(path, args.arch, args.epochs, args.batch, args.lr, output_path, args.patience)
