"""
VIGIL AI – Synthetic Dataset Generator + Full Training Pipeline
==============================================================
Generates realistic synthetic fight/non-fight video clips using OpenCV,
then trains the CNN+LSTM model on them and saves production-ready weights.

Violent clips features:
  - Rapid, erratic motion with sudden direction changes
  - High optical-flow magnitude
  - Object collisions (bounding boxes overlapping with high velocity)
  - Red/warm colour hue flashes (simulating blood/adrenaline visual cues)
  - Multiple agents moving chaotically

Non-violent clips features:
  - Smooth, predictable trajectories
  - Low motion variance
  - Objects maintaining safe distance
  - Cooler colour tones
  - Gradual, coordinated movement

Usage:
    python generate_and_train.py --videos 300 --epochs 25 --batch 4
"""

import os
import sys
import math
import random
import time
import json
import logging
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)

# ── Project path setup ────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "backend"))

from models.cnn_lstm import build_cnn_lstm
from services.video_processor import extract_frames, frames_to_tensor_cnn_lstm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vigil.train")

WEIGHT_PATH = ROOT / "models" / "weights" / "vigil_model.pth"
SYNTH_DIR   = ROOT / "datasets" / "synthetic"
FRAME_SIZE  = 112
NUM_FRAMES  = 16
SEED        = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC VIDEO GENERATION
# ══════════════════════════════════════════════════════════════════════════════

class Agent:
    """A moving rectangular agent (person-like object) in a synthetic scene."""

    def __init__(self, scene_w, scene_h, violent: bool):
        self.w = random.randint(20, 45)
        self.h = random.randint(40, 90)
        self.x = float(random.randint(0, scene_w - self.w))
        self.y = float(random.randint(0, scene_h - self.h))

        speed = random.uniform(6, 18) if violent else random.uniform(1, 5)
        angle = random.uniform(0, 2 * math.pi)
        self.vx = speed * math.cos(angle)
        self.vy = speed * math.sin(angle)

        # Colour: warmer for violent, cooler for non-violent
        if violent:
            r = random.randint(160, 255)
            g = random.randint(50, 150)
            b = random.randint(20, 80)
        else:
            r = random.randint(30, 100)
            g = random.randint(80, 160)
            b = random.randint(140, 240)

        self.color = (b, g, r)  # OpenCV BGR

    def update(self, scene_w, scene_h, violent: bool):
        self.x += self.vx
        self.y += self.vy

        # Violent: random direction perturbations (erratic)
        if violent:
            self.vx += random.uniform(-4, 4)
            self.vy += random.uniform(-4, 4)
            speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
            if speed > 25:
                self.vx = self.vx / speed * 25
        else:
            # Non-violent: smooth drift, mild curve
            self.vx += random.uniform(-0.3, 0.3)
            self.vy += random.uniform(-0.3, 0.3)
            speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
            if speed > 6:
                self.vx = self.vx / speed * 6

        # Bounce off walls
        if self.x < 0 or self.x + self.w > scene_w:
            self.vx = -self.vx
            self.x = max(0.0, min(self.x, scene_w - self.w - 1))
        if self.y < 0 or self.y + self.h > scene_h:
            self.vy = -self.vy
            self.y = max(0.0, min(self.y, scene_h - self.h - 1))

    def rect(self):
        return (int(self.x), int(self.y), int(self.x + self.w), int(self.y + self.h))

    def overlaps(self, other) -> bool:
        ax1, ay1, ax2, ay2 = self.rect()
        bx1, by1, bx2, by2 = other.rect()
        return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def _generate_video_frames(violent: bool, num_frames: int, size: int) -> list:
    """Generate a list of numpy frames (H, W, 3) for one synthetic clip."""
    W, H = size * 2, size * 2  # Render at 2× then resize for quality
    n_agents = random.randint(2, 5) if violent else random.randint(1, 3)
    agents = [Agent(W, H, violent) for _ in range(n_agents)]

    # Background: dark for violent, lighter for non-violent
    if violent:
        bg_base = np.array([random.randint(10, 40)] * 3, dtype=np.uint8)
    else:
        bg_base = np.array([random.randint(60, 120)] * 3, dtype=np.uint8)

    frames = []
    for f_idx in range(num_frames):
        canvas = np.full((H, W, 3), bg_base, dtype=np.uint8)

        # Violent: add noise grain + motion blur hint
        if violent:
            noise = np.random.randint(-20, 20, (H, W, 3), dtype=np.int16)
            canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            # Occasional red flash (impact)
            if random.random() < 0.25:
                flash = np.zeros((H, W, 3), dtype=np.uint8)
                flash[:, :, 2] = random.randint(30, 80)  # Red channel
                canvas = cv2.addWeighted(canvas, 0.8, flash, 0.2, 0)
        else:
            # Soft gradient background
            grad = np.linspace(0, 20, W, dtype=np.uint8)
            canvas[:, :, 0] = np.minimum(canvas[:, :, 0] + grad[np.newaxis, :], 255)

        # Update and draw agents
        for agent in agents:
            agent.update(W, H, violent)

        # Check collisions for violent → draw impact effect
        if violent:
            for i, a in enumerate(agents):
                for b in agents[i + 1:]:
                    if a.overlaps(b):
                        cx = (a.rect()[0] + b.rect()[0]) // 2
                        cy = (a.rect()[1] + b.rect()[1]) // 2
                        cv2.circle(canvas, (cx, cy), random.randint(10, 30),
                                   (0, 100, 255), -1)

        # Draw agents
        for agent in agents:
            x1, y1, x2, y2 = agent.rect()
            cv2.rectangle(canvas, (x1, y1), (x2, y2), agent.color, -1)
            # Simple "head" circle
            cx = (x1 + x2) // 2
            cv2.circle(canvas, (cx, y1 - 8), 10, agent.color, -1)

        # Motion lines for violent (speed trails)
        if violent:
            for agent in agents:
                x1, y1, x2, y2 = agent.rect()
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                trail_x = int(cx - agent.vx * 2)
                trail_y = int(cy - agent.vy * 2)
                cv2.line(canvas, (trail_x, trail_y), (cx, cy), agent.color, 2)

        # Resize down
        frame = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(frame)

    return frames


def generate_synthetic_dataset(
    out_dir: Path,
    n_per_class: int = 200,
    num_frames: int = NUM_FRAMES,
    frame_size: int = FRAME_SIZE,
):
    """Generate synthetic dataset of .avi video clips."""
    for label, violent in [("Fight", True), ("NonFight", False)]:
        for split in ["train", "val", "test"]:
            (out_dir / split / label).mkdir(parents=True, exist_ok=True)

    # Split: 70% train, 15% val, 15% test
    n_train = int(n_per_class * 0.70)
    n_val   = int(n_per_class * 0.15)
    n_test  = n_per_class - n_train - n_val

    total = (n_train + n_val + n_test) * 2
    done  = 0

    for label, violent in [("Fight", True), ("NonFight", False)]:
        counts = {"train": n_train, "val": n_val, "test": n_test}
        for split, count in counts.items():
            for i in range(count):
                vpath = out_dir / split / label / f"clip_{i:04d}.avi"
                if vpath.exists():
                    done += 1
                    continue
                frames = _generate_video_frames(violent, num_frames * 2, frame_size)
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(str(vpath), fourcc, 10.0,
                                         (frame_size, frame_size))
                for fr in frames:
                    writer.write(fr)
                writer.release()
                done += 1
                if done % 50 == 0:
                    log.info(f"  Generated {done}/{total} clips…")

    log.info(f"✓ Synthetic dataset ready: {total} clips in {out_dir}")
    return {
        "train": {"Fight": n_train, "NonFight": n_train},
        "val":   {"Fight": n_val,   "NonFight": n_val},
        "test":  {"Fight": n_test,  "NonFight": n_test},
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET  &  DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

LABEL_MAP = {"Fight": 1, "NonFight": 0}

TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class VideoDataset(Dataset):
    def __init__(self, root: Path, split: str, transform=None, num_frames=NUM_FRAMES):
        self.transform   = transform
        self.num_frames  = num_frames
        self.samples     = []

        split_dir = root / split
        for label_name, label_id in LABEL_MAP.items():
            folder = split_dir / label_name
            if not folder.exists():
                continue
            for vf in sorted(folder.iterdir()):
                if vf.suffix.lower() in (".avi", ".mp4", ".mov"):
                    self.samples.append((str(vf), label_id))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, path: str) -> torch.Tensor:
        cap = cv2.VideoCapture(path)
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(frame)
        cap.release()

        if not all_frames:
            # Return blank tensor if unreadable
            return torch.zeros(self.num_frames, 3, FRAME_SIZE, FRAME_SIZE)

        # Sample evenly
        total = len(all_frames)
        if total >= self.num_frames:
            idxs = np.linspace(0, total - 1, self.num_frames, dtype=int)
            selected = [all_frames[i] for i in idxs]
        else:
            # Repeat last frame
            selected = all_frames + [all_frames[-1]] * (self.num_frames - total)

        tensors = []
        for fr in selected:
            fr_resized = cv2.resize(fr, (FRAME_SIZE, FRAME_SIZE))
            if self.transform:
                t = self.transform(fr_resized)
            else:
                t = transforms.ToTensor()(fr_resized)
                t = transforms.Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])(t)
            tensors.append(t)

        return torch.stack(tensors)   # (T, C, H, W)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = self._load_frames(path)
        return frames, label


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(frames)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        n += len(labels)

    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
    for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device)

        logits = model(frames)
        loss   = criterion(logits, labels)
        probs  = torch.softmax(logits, dim=1)[:, 1]  # violent probability

        total_loss += loss.item() * len(labels)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    n = len(all_labels)
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    cm   = confusion_matrix(all_labels, all_preds).tolist()
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0

    return {
        "loss":      total_loss / n,
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "auc":       auc,
        "confusion_matrix": cm,
    }


def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── 1. Generate synthetic dataset ─────────────────────────────────────────
    log.info(f"Generating synthetic dataset ({args.videos} clips/class)…")
    t0 = time.time()
    split_counts = generate_synthetic_dataset(
        SYNTH_DIR, n_per_class=args.videos,
        num_frames=NUM_FRAMES, frame_size=FRAME_SIZE,
    )
    log.info(f"Dataset generated in {time.time() - t0:.1f}s")

    # ── 2. Build data loaders ─────────────────────────────────────────────────
    train_ds = VideoDataset(SYNTH_DIR, "train", transform=TRAIN_TRANSFORM)
    val_ds   = VideoDataset(SYNTH_DIR, "val",   transform=EVAL_TRANSFORM)
    test_ds  = VideoDataset(SYNTH_DIR, "test",  transform=EVAL_TRANSFORM)

    log.info(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    # Weighted sampler for class balance
    labels  = [s[1] for s in train_ds.samples]
    counts  = np.bincount(labels)
    weights = [1.0 / counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False,
                              num_workers=0)

    # ── 3. Build model ────────────────────────────────────────────────────────
    model = build_cnn_lstm(freeze_cnn=False).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: CNN+LSTM (MobileNetV2)  |  Parameters: {n_params:,}")

    # Class-weighted loss
    class_w = torch.tensor(
        [1.0 / counts[0], 1.0 / counts[1]], dtype=torch.float32
    ).to(device)
    class_w = class_w / class_w.sum()
    criterion = nn.CrossEntropyLoss(weight=class_w * 2)

    # Optimizer + scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── 4. Training loop ──────────────────────────────────────────────────────
    best_val_acc  = 0.0
    best_val_loss = float("inf")
    patience      = args.patience
    no_improve    = 0
    history       = []

    WEIGHT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_ckpt = WEIGHT_PATH

    log.info("=" * 60)
    log.info(f"Training: {args.epochs} epochs, batch={args.batch}, lr={args.lr}")
    log.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics   = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        t_elapsed = time.time() - t_start
        train_loss, train_acc = train_metrics
        val_loss    = val_metrics["loss"]
        val_acc     = val_metrics["accuracy"]
        val_f1      = val_metrics["f1"]

        ep_record = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc,  4),
            "val_loss":   round(val_loss,   4),
            "val_acc":    round(val_acc,    4),
            "val_f1":     round(val_f1,     4),
            "lr":         round(scheduler.get_last_lr()[0], 7),
            "time_s":     round(t_elapsed,  1),
        }
        history.append(ep_record)

        improved = val_acc > best_val_acc or (
            val_acc == best_val_acc and val_loss < best_val_loss
        )
        mark = " ★ BEST" if improved else ""

        log.info(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_f1={val_f1:.3f}  "
            f"[{t_elapsed:.1f}s]{mark}"
        )

        if improved:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            no_improve    = 0

            torch.save({
                "epoch":           epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss":        val_loss,
                "val_acc":         val_acc,
                "train_loss":      train_loss,
                "train_acc":       train_acc,
                "architecture":    "cnn_lstm",
                "num_frames":      NUM_FRAMES,
                "frame_size":      FRAME_SIZE,
                "history":         history,
            }, str(best_ckpt))
            log.info(f"  → Saved checkpoint: val_acc={val_acc:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # ── 5. Load best checkpoint for final evaluation ──────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Final evaluation on test set…")
    ckpt = torch.load(str(best_ckpt), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = eval_epoch(model, test_loader, criterion, device)
    log.info(f"Test accuracy : {test_metrics['accuracy']:.4f}")
    log.info(f"Test F1       : {test_metrics['f1']:.4f}")
    log.info(f"Test AUC      : {test_metrics['auc']:.4f}")
    log.info(f"Confusion mat : {test_metrics['confusion_matrix']}")

    # ── 6. Update checkpoint with evaluation results ──────────────────────────
    ckpt["evaluation"] = {
        "accuracy":         round(test_metrics["accuracy"],  4),
        "precision":        round(test_metrics["precision"], 4),
        "recall":           round(test_metrics["recall"],    4),
        "f1_score":         round(test_metrics["f1"],        4),
        "auc_roc":          round(test_metrics["auc"],       4),
        "confusion_matrix": test_metrics["confusion_matrix"],
        "test_samples":     len(test_ds),
        "dataset":          "synthetic_vigil",
        "note":             (
            "Trained on computer-vision synthetic dataset (motion-pattern learning). "
            "Replace with RWF-2000 or Hockey dataset for real-world performance."
        ),
    }
    ckpt["history"] = history
    torch.save(ckpt, str(best_ckpt))
    log.info(f"✓ Final checkpoint saved → {best_ckpt}")
    log.info(f"  val_acc={ckpt['val_acc']:.4f}  epoch={ckpt['epoch']}")

    # ── 7. Print summary JSON ─────────────────────────────────────────────────
    summary = {
        "status":        "done",
        "best_val_acc":  round(best_val_acc, 4),
        "best_val_loss": round(best_val_loss, 4),
        "total_epochs":  len(history),
        "test_metrics":  {k: round(v, 4) if isinstance(v, float) else v
                          for k, v in test_metrics.items()},
        "checkpoint":    str(best_ckpt),
    }
    print("\nTRAINING_SUMMARY=" + json.dumps(summary))
    return summary


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VIGIL AI – Generate & Train")
    parser.add_argument("--videos",   type=int,   default=250,
                        help="Number of synthetic clips per class (default: 250)")
    parser.add_argument("--epochs",   type=int,   default=25,
                        help="Max training epochs (default: 25)")
    parser.add_argument("--batch",    type=int,   default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("--lr",       type=float, default=1e-4,
                        help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--patience", type=int,   default=6,
                        help="Early stopping patience (default: 6)")
    parser.add_argument("--clean",    action="store_true",
                        help="Delete existing synthetic dataset before generating")
    args = parser.parse_args()

    if args.clean and SYNTH_DIR.exists():
        log.info(f"Cleaning {SYNTH_DIR}…")
        shutil.rmtree(SYNTH_DIR)

    summary = run_training(args)
    sys.exit(0 if summary["test_metrics"]["accuracy"] > 0.0 else 1)
