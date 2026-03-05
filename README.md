# tourPlanner

# src/utils/device.py

from __future__ import annotations
import torch


def get_device() -> torch.device:
    """
    Returns the device used for training/evaluation.

    Priority:
    1) CUDA GPU
    2) CPU
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[device] Using CPU")

    return device


def to_device(batch, device: torch.device):
    """
    Moves tensors in a batch to the selected device.
    Useful when dataloaders return tuples.
    """

    if isinstance(batch, (list, tuple)):
        return [b.to(device) if hasattr(b, "to") else b for b in batch]

    if hasattr(batch, "to"):
        return batch.to(device)

    return batch

--

# src/utils/io.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: Path) -> None:
    """
    Create directory if it does not exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    """
    Load JSON file and return dictionary.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    """
    Save dictionary to JSON file.
    """
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration.
    """
    import yaml

    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

--

# src/utils/seed.py

from __future__ import annotations

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set seeds for reproducibility.
    """

    print(f"[seed] Setting global seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    # deterministic behaviour (slightly slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

xxx