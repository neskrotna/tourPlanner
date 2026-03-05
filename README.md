# tourPlanner

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from src.data.packcrop_dataset import PackCropDataset
from src.models.factory import create_model
from src.utils.device import get_device
from src.utils.seed import set_seed


class Trainer:
    def __init__(self, cfg):

        self.cfg = cfg

        set_seed(cfg["train"]["seed"])
        self.device = get_device()

        self.train_ds = PackCropDataset(
            Path(cfg["dataset"]["manifest_csv"]),
            Path(cfg["dataset"]["classes_json"]),
            Path(cfg["dataset"]["split_json"]),
            split="train",
            transform=self._build_transforms()
        )

        self.val_ds = PackCropDataset(
            Path(cfg["dataset"]["manifest_csv"]),
            Path(cfg["dataset"]["classes_json"]),
            Path(cfg["dataset"]["split_json"]),
            split="val",
            transform=self._build_transforms()
        )

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg["train"]["batch_size"],
            shuffle=True,
            num_workers=cfg["train"]["num_workers"]
        )

        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=cfg["train"]["num_workers"]
        )

        self.model = create_model(cfg).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"]
        )

        self.criterion = torch.nn.CrossEntropyLoss()

        self.exp_dir = Path(cfg["output"]["exp_root"])
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def _build_transforms(self):

        size = self.cfg["train"]["image_size"]

        return T.Compose([
            T.ToPILImage(),
            T.Resize((size, size)),
            T.ToTensor()
        ])

    def train(self):

        epochs = self.cfg["train"]["epochs"]

        for epoch in range(epochs):

            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self._validate_epoch()

            print(
                f"Epoch {epoch+1}/{epochs} "
                f"train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.4f}"
            )

            self._save_checkpoint(epoch)

    def _train_epoch(self):

        self.model.train()

        total_loss = 0
        correct = 0
        total = 0

        for x, y, _ in self.train_loader:

            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(x)
            loss = self.criterion(logits, y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / total
        acc = correct / total

        return avg_loss, acc

    def _validate_epoch(self):

        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():

            for x, y, _ in self.val_loader:

                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss = self.criterion(logits, y)

                total_loss += loss.item() * x.size(0)

                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        avg_loss = total_loss / total
        acc = correct / total

        return avg_loss, acc

    def _save_checkpoint(self, epoch):

        path = self.exp_dir / f"model_epoch_{epoch+1}.pt"

        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            },
            path
        )

--

# src/models/factory.py

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch.nn as nn
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    convnext_small,
    ConvNeXt_Small_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights,
)


def _get_model_cfg(cfg: Dict[str, Any]) -> Tuple[str, bool, int]:
    if "model" not in cfg:
        raise ValueError("Config is missing the 'model' section.")

    mcfg = cfg["model"]
    name = mcfg.get("name")
    if not name:
        raise ValueError("Config is missing model.name (e.g. 'resnet18').")

    pretrained = bool(mcfg.get("pretrained", False))
    num_classes = int(mcfg.get("num_classes", 2))

    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}.")

    return str(name).lower(), pretrained, num_classes


def create_model(cfg: Dict[str, Any]) -> nn.Module:
    """
    Torchvision-only model factory (no Hugging Face).

    Supported model names:
      - "resnet18"
      - "convnext_small"   (also accepts "convnext-small")
      - "efficientnet_b0"  (also accepts "efficientnet-b0")

    Expected cfg:
      cfg["model"]["name"]
      cfg["model"]["pretrained"]  (bool)
      cfg["model"]["num_classes"] (int)
    """
    name, pretrained, num_classes = _get_model_cfg(cfg)

    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if name in {"convnext_small", "convnext-small"}:
        weights = ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        model = convnext_small(weights=weights)

        # Torchvision ConvNeXt classifier is: Sequential(LayerNorm2d, Flatten, Linear)
        # Replace the last Linear.
        if not isinstance(model.classifier, nn.Sequential) or len(model.classifier) < 1:
            raise RuntimeError("Unexpected ConvNeXt classifier structure.")

        last = model.classifier[-1]
        if not isinstance(last, nn.Linear):
            raise RuntimeError("Unexpected ConvNeXt last classifier layer (expected nn.Linear).")

        in_features = last.in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if name in {"efficientnet_b0", "efficientnet-b0"}:
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)

        # Torchvision EfficientNet classifier is: Sequential(Dropout, Linear)
        if not isinstance(model.classifier, nn.Sequential) or len(model.classifier) < 1:
            raise RuntimeError("Unexpected EfficientNet classifier structure.")

        last = model.classifier[-1]
        if not isinstance(last, nn.Linear):
            raise RuntimeError("Unexpected EfficientNet last classifier layer (expected nn.Linear).")

        in_features = last.in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(
        f"Unsupported model name '{name}'. Supported: "
        "['resnet18', 'convnext_small', 'efficientnet_b0']."
    )


def get_num_params(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count parameters.
      - trainable_only=False: counts all parameters
      - trainable_only=True: counts only parameters with requires_grad=True
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

xxx