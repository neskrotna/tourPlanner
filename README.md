# tourPlanner

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch.nn as nn

try:
    import timm
except ImportError as e:
    raise ImportError(
        "timm is required for model creation. Install it via: pip install timm"
    ) from e


# Optional: keep a small whitelist so typos fail fast.
# These are the names you're using in YAML tabs (resnet18, efficientnet_b0, convnext_small).
SUPPORTED_TIMM_MODELS = {
    "resnet18",
    "efficientnet_b0",
    "convnext_small",
}


@dataclass(frozen=True)
class ModelConfig:
    name: str
    pretrained: bool = True
    num_classes: int = 2
    in_chans: int = 3  # RGB crops


def _parse_model_config(cfg: Dict[str, Any]) -> ModelConfig:
    """
    Expected YAML structure:
      model:
        name: resnet18
        pretrained: true
        num_classes: 2
    """
    if "model" not in cfg or not isinstance(cfg["model"], dict):
        raise ValueError("Config must contain a 'model' dict (e.g. cfg['model']['name']).")

    m = cfg["model"]
    name = str(m.get("name", "")).strip()
    if not name:
        raise ValueError("cfg['model']['name'] is missing or empty.")

    pretrained = bool(m.get("pretrained", True))
    num_classes = int(m.get("num_classes", 2))
    in_chans = int(m.get("in_chans", 3))

    return ModelConfig(
        name=name,
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=in_chans,
    )


def create_model(cfg: Dict[str, Any], *, strict_name_check: bool = True) -> nn.Module:
    """
    Create a classification model based on the YAML config.

    - Uses timm models (pretrained backbones supported).
    - Sets the final classifier head to num_classes automatically.

    Args:
        cfg: loaded YAML as a dict
        strict_name_check: if True, rejects model names outside SUPPORTED_TIMM_MODELS

    Returns:
        torch.nn.Module
    """
    mc = _parse_model_config(cfg)

    if strict_name_check and mc.name not in SUPPORTED_TIMM_MODELS:
        raise ValueError(
            f"Unsupported model name '{mc.name}'. "
            f"Supported: {sorted(SUPPORTED_TIMM_MODELS)}. "
            f"If you intentionally want another timm model, call create_model(..., strict_name_check=False)."
        )

    # timm will replace the classifier head when num_classes is given.
    model = timm.create_model(
        mc.name,
        pretrained=mc.pretrained,
        num_classes=mc.num_classes,
        in_chans=mc.in_chans,
    )

    # Safety check: ensure output shape matches expected classes (best-effort).
    # We can't run a forward pass here without knowing image size, so we just keep it lightweight.
    if not isinstance(model, nn.Module):
        raise TypeError("timm.create_model did not return a torch.nn.Module.")

    return model


def get_num_params(model: nn.Module, *, trainable_only: bool = False) -> int:
    """
    Small helper: count parameters, useful for printing in train.py.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

xxx

python -c "import yaml; from src.models.factory import create_model, get_num_params; cfg=yaml.safe_load(open('configs/binary/resnet18.yaml','r',encoding='utf-8')); m=create_model(cfg); print(type(m).__name__, get_num_params(m))"