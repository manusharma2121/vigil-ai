"""
VIGIL AI – Training Service
Runs the full training loop in a background thread.
Emits progress via a shared state dictionary polled by the /api/train/status endpoint.
"""
import threading
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import logging

from config import settings
from models.cnn_lstm import build_cnn_lstm
from models.model_3dcnn import build_3dcnn
from services.video_processor import extract_frames, frames_to_tensor_cnn_lstm, frames_to_tensor_3dcnn

logger = logging.getLogger(__name__)

# Shared training state – accessed from main thread for status polling
training_state: Dict[str, Any] = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "train_loss": 0.0,
    "val_loss": 0.0,
    "train_acc": 0.0,
    "val_acc": 0.0,
    "status": "idle",
    "history": [],
    "error": None,
}

_training_thread: Optional[threading.Thread] = None


class VideoDataset(Dataset):
    """
    Generic dataset that reads video files from a list of (path, label) tuples.
    Label: 0 = Non-Violent, 1 = Violent
    """
    def __init__(self, samples: List[tuple], model_arch: str = "cnn_lstm"):
        self.samples = samples
        self.model_arch = model_arch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = extract_frames(str(video_path))
        if frames is None:
            # Return zeros tensor on failure
            if self.model_arch == "3dcnn":
                tensor = torch.zeros(3, settings.max_frames, settings.frame_size, settings.frame_size)
            else:
                tensor = torch.zeros(settings.max_frames, 3, settings.frame_size, settings.frame_size)
        else:
            if self.model_arch == "3dcnn":
                tensor = frames_to_tensor_3dcnn(frames).squeeze(0)
            else:
                tensor = frames_to_tensor_cnn_lstm(frames).squeeze(0)
        return tensor, label


def _collect_dataset_samples(dataset_name: str) -> List[tuple]:
    """
    Scan dataset directories and collect (video_path, label) tuples.
    Supports RWF-2000 folder structure: Fight/ and NonFight/
    Supports Hockey dataset structure: fight/ and no-fight/
    """
    samples = []

    if dataset_name in ("rwf2000", "both"):
        base = Path(settings.rwf2000_path)
        for split in ("train", "val", "test", ""):
            split_dir = base / split if split else base
            for label_name, label in [("Fight", 1), ("NonFight", 0), ("fight", 1), ("nonfight", 0)]:
                label_dir = split_dir / label_name
                if label_dir.exists():
                    for f in label_dir.glob("*.avi"):
                        samples.append((f, label))
                    for f in label_dir.glob("*.mp4"):
                        samples.append((f, label))

    if dataset_name in ("hockey", "both"):
        base = Path(settings.hockey_path)
        for label_name, label in [("fight", 1), ("no-fight", 0), ("nofight", 0), ("non-fight", 0)]:
            label_dir = base / label_name
            if label_dir.exists():
                for f in label_dir.glob("*.avi"):
                    samples.append((f, label))
                for f in label_dir.glob("*.mp4"):
                    samples.append((f, label))

    return samples


def _training_loop(
    model_arch: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    dataset_name: str,
):
    """Background training loop – runs in a separate thread."""
    global training_state

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    try:
        training_state.update({
            "status": "collecting_data",
            "error": None,
        })

        samples = _collect_dataset_samples(dataset_name)
        if len(samples) < 4:
            raise ValueError(
                f"Not enough video samples found (found {len(samples)}). "
                "Please ensure dataset directories are set up correctly per the README."
            )

        # Shuffle and split 80/10/10
        np.random.shuffle(samples)
        n = len(samples)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        train_samples = samples[:train_end]
        val_samples = samples[train_end:val_end]

        train_ds = VideoDataset(train_samples, model_arch)
        val_ds = VideoDataset(val_samples, model_arch)

        # Weighted sampler for class balance
        train_labels = [s[1] for s in train_samples]
        n_v  = sum(train_labels)
        n_nv = len(train_labels) - n_v
        if n_v > 0 and n_nv > 0:
            w = [1.0/n_nv if l == 0 else 1.0/n_v for l in train_labels]
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(w, len(train_samples), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
            total = n_v + n_nv
            loss_weights = torch.tensor(
                [total/(2*n_nv), total/(2*n_v)], dtype=torch.float32
            ).to(device)
        else:
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
            loss_weights = None

        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # Build model
        if model_arch == "3dcnn":
            model = build_3dcnn().to(device)
        else:
            model = build_cnn_lstm(freeze_cnn=True).to(device)

        criterion = nn.CrossEntropyLoss(weight=loss_weights if 'loss_weights' in dir() else None)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        training_state.update({
            "status": "training",
            "total_epochs": epochs,
            "history": [],
        })

        for epoch in range(1, epochs + 1):
            # ── Train ────────────────────────────────────────────────────────
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += inputs.size(0)

            # ── Validate ─────────────────────────────────────────────────────
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += inputs.size(0)

            scheduler.step()

            avg_train_loss = round(train_loss / max(train_total, 1), 4)
            avg_val_loss   = round(val_loss   / max(val_total,   1), 4)
            avg_train_acc  = round(train_correct / max(train_total, 1), 4)
            avg_val_acc    = round(val_correct   / max(val_total,   1), 4)

            epoch_data = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss":   avg_val_loss,
                "train_acc":  avg_train_acc,
                "val_acc":    avg_val_acc,
            }

            training_state["current_epoch"] = epoch
            training_state["train_loss"] = avg_train_loss
            training_state["val_loss"]   = avg_val_loss
            training_state["train_acc"]  = avg_train_acc
            training_state["val_acc"]    = avg_val_acc
            training_state["history"].append(epoch_data)

            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.4f}"
            )

        # ── Save checkpoint (full format) ─────────────────────────────────────
        weights_dir = Path("../models/weights")
        weights_dir.mkdir(parents=True, exist_ok=True)
        save_path = weights_dir / "vigil_model.pth"

        final_epoch   = training_state.get("current_epoch", epochs)
        final_val_acc = training_state.get("val_acc", 0.0)
        final_val_loss = training_state.get("val_loss", 0.0)

        torch.save({
            "epoch":            final_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "val_loss":         final_val_loss,
            "val_acc":          final_val_acc,
            "train_loss":       training_state.get("train_loss", 0.0),
            "train_acc":        training_state.get("train_acc", 0.0),
            "architecture":     model_arch,
            "num_frames":       settings.max_frames,
            "frame_size":       settings.frame_size,
            "history":          training_state.get("history", []),
        }, str(save_path))
        logger.info(f"Model checkpoint saved to {save_path}")

        # Reload inference engine with trained weights
        try:
            from services.inference import reload_model
            reloaded = reload_model()
            logger.info(f"Inference engine reloaded. Demo mode: {not reloaded}")
            training_state["demo_mode_active"] = not reloaded
        except Exception as re:
            logger.warning(f"Could not auto-reload inference model: {re}")

        training_state["status"] = "done"

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        training_state["status"] = "error"
        training_state["error"] = str(e)
    finally:
        training_state["is_training"] = False


def start_training(
    model_arch: str = "cnn_lstm",
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    dataset_name: str = "both",
) -> Dict[str, Any]:
    """
    Launch training in a background thread.
    Returns immediately with current training state.
    """
    global _training_thread, training_state

    if training_state["is_training"]:
        return {"started": False, "message": "Training already in progress."}

    # Reset state
    training_state = {
        "is_training": True,
        "current_epoch": 0,
        "total_epochs": epochs,
        "train_loss": 0.0,
        "val_loss": 0.0,
        "train_acc": 0.0,
        "val_acc": 0.0,
        "status": "starting",
        "history": [],
        "error": None,
    }

    _training_thread = threading.Thread(
        target=_training_loop,
        args=(model_arch, epochs, batch_size, learning_rate, dataset_name),
        daemon=True,
    )
    _training_thread.start()

    return {"started": True, "message": "Training started in background."}


def get_training_status() -> Dict[str, Any]:
    return dict(training_state)
