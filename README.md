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

xxx