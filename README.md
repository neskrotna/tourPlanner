# tourPlanner

# src/cli/train.py

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import yaml

from src.data.packcrop_dataset import PackCropDataset
from src.models.factory import create_model, get_num_params


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Keep it simple/stable for PoC. (Deterministic training can be slower.)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def build_transforms(image_size: int, is_train: bool) -> T.Compose:
    # You can extend this later (color jitter, rotation augmentation etc.)
    if is_train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
        ])
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y, _meta in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * y.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += int((preds == y).sum().item())
        total += int(y.size(0))

    avg_loss = total_loss / max(1, total)
    avg_acc = total_correct / max(1, total)
    return avg_loss, avg_acc


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y, _meta in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * y.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += int((preds == y).sum().item())
        total += int(y.size(0))

    avg_loss = total_loss / max(1, total)
    avg_acc = total_correct / max(1, total)
    return avg_loss, avg_acc


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, cfg: Dict[str, Any]) -> None:
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg,
    }
    torch.save(ckpt, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg: Dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # ---- read cfg ----
    seed = int(cfg["train"].get("seed", 42))
    image_size = int(cfg["train"]["image_size"])
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])
    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"].get("weight_decay", 0.0))
    num_workers = int(cfg["train"].get("num_workers", 0))

    manifest_csv = Path(cfg["dataset"]["manifest_csv"])
    classes_json = Path(cfg["dataset"]["classes_json"])
    split_json = Path(cfg["dataset"]["split_json"])

    exp_root = Path(cfg["output"]["exp_root"])
    ensure_dir(exp_root)

    # ---- setup ----
    set_seed(seed)
    device = get_device()
    print(f"[train] device={device}")

    # ---- datasets ----
    train_tf = build_transforms(image_size=image_size, is_train=True)
    val_tf = build_transforms(image_size=image_size, is_train=False)

    train_ds = PackCropDataset(
        manifest_csv=manifest_csv,
        classes_json=classes_json,
        split_json=split_json,
        split="train",
        transform=train_tf,
        strict=True,
    )
    val_ds = PackCropDataset(
        manifest_csv=manifest_csv,
        classes_json=classes_json,
        split_json=split_json,
        split="val",
        transform=val_tf,
        strict=True,
    )

    print(f"[train] train_samples={len(train_ds)} counts={train_ds.class_counts()}")
    print(f"[train] val_samples={len(val_ds)} counts={val_ds.class_counts()}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ---- model ----
    model = create_model(cfg).to(device)
    print(f"[train] model={type(model).__name__} params={get_num_params(model):,}")

    # ---- loss + optimizer ----
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- training loop ----
    best_val_acc = -1.0
    best_path = exp_root / "best.pt"

    history = {
        "config": str(cfg_path),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"[epoch {epoch:02d}/{epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        history["epochs"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        # Save last checkpoint each epoch (simple + reliable)
        save_checkpoint(exp_root / "last.pt", model, optimizer, epoch, cfg)

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(best_path, model, optimizer, epoch, cfg)
            print(f"[train] new best val_acc={best_val_acc:.4f} -> {best_path}")

    history["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    history["best_val_acc"] = best_val_acc

    (exp_root / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[train] done. best_val_acc={best_val_acc:.4f} history={exp_root/'history.json'}")


if __name__ == "__main__":
    main()

xxx