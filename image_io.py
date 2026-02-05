"""
Centralized image I/O with mandatory timestamping.

All saves include timestamps to prevent accidental overwrites.
"""
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image


def _get_timestamp() -> str:
    """Get current timestamp string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _inject_timestamp(path: Path) -> Path:
    """Inject timestamp into filename: foo.png -> foo_20260204_041500.png"""
    ts = _get_timestamp()
    return path.parent / f"{path.stem}_{ts}{path.suffix}"


def save_image(
    data: Union[torch.Tensor, np.ndarray],
    path: Union[str, Path],
    timestamp: bool = True
) -> Path:
    """
    Save image with mandatory timestamp injection.

    Args:
        data: Image data as torch.Tensor (H,W,3) or numpy array
        path: Output path (timestamp will be injected before extension)
        timestamp: If True (default), inject timestamp. Only False for explicit overwrites.

    Returns:
        Actual path where file was saved (with timestamp if enabled)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if timestamp:
        path = _inject_timestamp(path)

    # Convert to numpy if torch
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    else:
        arr = data

    # Normalize to uint8
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    Image.fromarray(arr).save(path)
    print(f"Saved: {path}")
    return path


def load_image(path: Union[str, Path], device: torch.device = None) -> torch.Tensor:
    """
    Load image as RGB float32 [0,1] torch tensor.

    Args:
        path: Image file path
        device: Target device (default: CUDA if available, else CPU)

    Returns:
        (H, W, 3) torch tensor in [0, 1] range
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img = Image.open(path).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).to(device)


def save_comparison_grid(
    images: list,
    path: Union[str, Path],
    labels: list = None,
    cols: int = None
) -> Path:
    """
    Save multiple images as a comparison grid with timestamps.

    Args:
        images: List of (H,W,3) tensors or arrays
        path: Output path (timestamp injected)
        labels: Optional list of labels for each image
        cols: Number of columns (default: sqrt of count)

    Returns:
        Actual saved path
    """
    import math

    n = len(images)
    if cols is None:
        cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    # Convert all to numpy uint8
    imgs = []
    for img in images:
        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu().numpy()
        else:
            arr = img
        if arr.dtype in (np.float32, np.float64):
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        imgs.append(arr)

    # Get max dimensions
    max_h = max(img.shape[0] for img in imgs)
    max_w = max(img.shape[1] for img in imgs)

    # Create grid
    grid = np.zeros((rows * max_h, cols * max_w, 3), dtype=np.uint8)

    for i, img in enumerate(imgs):
        r, c = i // cols, i % cols
        h, w = img.shape[:2]
        grid[r*max_h:r*max_h+h, c*max_w:c*max_w+w] = img

    return save_image(grid, path, timestamp=True)
