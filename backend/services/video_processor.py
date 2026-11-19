"""
VIGIL AI – Video Processor
Extracts, resizes, and normalises frames from uploaded video files using OpenCV.
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional
from config import settings


# ImageNet normalisation constants
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def extract_frames(
    video_path: str,
    num_frames: int = None,
    frame_size: int = None,
) -> Optional[np.ndarray]:
    """
    Read a video file and return uniformly sampled, resized, normalised frames.

    Returns:
        np.ndarray of shape (num_frames, H, W, C) with float32 values in [0,1]
        normalised with ImageNet stats, or None on failure.
    """
    num_frames = num_frames or settings.max_frames
    frame_size = frame_size or settings.frame_size

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    # Uniformly sample frame indices
    indices = np.linspace(0, total - 1, num=num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            # Pad with last good frame if read fails
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((frame_size, frame_size, 3), dtype=np.float32))
            continue

        # BGR → RGB, resize, normalise
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_size, frame_size))
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - MEAN) / STD
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None

    return np.stack(frames, axis=0)  # (T, H, W, C)


def frames_to_tensor_cnn_lstm(frames: np.ndarray) -> torch.Tensor:
    """
    Convert frames array to CNN+LSTM input tensor.
    Input : (T, H, W, C)
    Output: (1, T, C, H, W)  – batched
    """
    # (T, H, W, C) → (T, C, H, W)
    t = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    return t.unsqueeze(0)  # (1, T, C, H, W)


def frames_to_tensor_3dcnn(frames: np.ndarray) -> torch.Tensor:
    """
    Convert frames array to 3D-CNN input tensor.
    Input : (T, H, W, C)
    Output: (1, C, T, H, W)  – batched
    """
    # (T, H, W, C) → (C, T, H, W)
    t = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
    return t.unsqueeze(0)  # (1, C, T, H, W)


def get_video_metadata(video_path: str) -> dict:
    """Return basic video metadata without reading all frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    meta = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_sec": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(cap.get(cv2.CAP_PROP_FPS), 1),
    }
    cap.release()
    return meta
