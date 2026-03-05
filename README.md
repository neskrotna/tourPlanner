# tourPlanner

# src/eval/metrics.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class EvalMetrics:
    accuracy: float
    precision_pos: float
    recall_pos: float
    f1_pos: float
    confusion: List[List[int]]  # [[tn, fp],[fn, tp]]
    support_neg: int
    support_pos: int


def confusion_2x2(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> Tuple[int, int, int, int]:
    """
    Returns (tn, fp, fn, tp) for a binary problem.
    Assumes labels are integers (e.g. 0/1).
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    neg = 0 if pos_label == 1 else 1
    tn = int(np.sum((y_true == neg) & (y_pred == neg)))
    fp = int(np.sum((y_true == neg) & (y_pred == pos_label)))
    fn = int(np.sum((y_true == pos_label) & (y_pred == neg)))
    tp = int(np.sum((y_true == pos_label) & (y_pred == pos_label)))
    return tn, fp, fn, tp


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 else 0.0


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> EvalMetrics:
    tn, fp, fn, tp = confusion_2x2(y_true, y_pred, pos_label=pos_label)

    acc = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    # support
    support_pos = int(np.sum(y_true == pos_label))
    support_neg = int(np.sum(y_true != pos_label))

    return EvalMetrics(
        accuracy=acc,
        precision_pos=precision,
        recall_pos=recall,
        f1_pos=f1,
        confusion=[[tn, fp], [fn, tp]],
        support_neg=support_neg,
        support_pos=support_pos,
    )


def metrics_to_dict(m: EvalMetrics) -> Dict:
    return {
        "accuracy": m.accuracy,
        "precision_pos": m.precision_pos,
        "recall_pos": m.recall_pos,
        "f1_pos": m.f1_pos,
        "confusion": m.confusion,
        "support_neg": m.support_neg,
        "support_pos": m.support_pos,
    }

--

# src/eval/evaluator.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

from src.data.packcrop_dataset import PackCropDataset
from src.eval.metrics import EvalMetrics, compute_binary_metrics


@dataclass(frozen=True)
class EvalResult:
    metrics: EvalMetrics
    y_true: List[int]
    y_pred: List[int]
    y_prob_pos: List[float]  # probability for class 1 (rotated)
    paths: List[str]         # crop paths aligned with y_true/y_pred


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def build_eval_transforms(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])


@torch.no_grad()
def run_eval(
    cfg: Dict[str, Any],
    checkpoint_path: Path,
    split: str = "test",
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> EvalResult:
    """
    Loads model from checkpoint and evaluates on the given split.

    Assumes binary classification with class 1 = "rotated".
    """
    device = get_device()

    image_size = int(cfg["train"]["image_size"])
    bs = int(batch_size if batch_size is not None else cfg["train"]["batch_size"])
    nw = int(num_workers if num_workers is not None else cfg["train"].get("num_workers", 0))

    manifest_csv = Path(cfg["dataset"]["manifest_csv"])
    classes_json = Path(cfg["dataset"]["classes_json"])
    split_json = Path(cfg["dataset"]["split_json"])

    ds = PackCropDataset(
        manifest_csv=manifest_csv,
        classes_json=classes_json,
        split_json=split_json,
        split=split,
        transform=build_eval_transforms(image_size),
        strict=True,
    )

    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
    )

    # model
    from src.models.factory import create_model
    model = create_model(cfg).to(device)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt)  # tolerate raw state_dict saves
    model.load_state_dict(state, strict=True)
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob_pos: List[float] = []
    paths: List[str] = []

    softmax = nn.Softmax(dim=1)

    for x, y, meta in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        probs = softmax(logits)

        pred = torch.argmax(probs, dim=1)

        # prob for positive class (id=1)
        prob_pos = probs[:, 1]

        y_true.extend([int(v) for v in y.detach().cpu().numpy().tolist()])
        y_pred.extend([int(v) for v in pred.detach().cpu().numpy().tolist()])
        y_prob_pos.extend([float(v) for v in prob_pos.detach().cpu().numpy().tolist()])

        # meta is a dict of lists because DataLoader collates it
        if isinstance(meta, dict) and "crop_path" in meta:
            paths.extend([str(p) for p in meta["crop_path"]])
        else:
            paths.extend([""] * y.size(0))

    y_true_np = np.array(y_true, dtype=int)
    y_pred_np = np.array(y_pred, dtype=int)

    metrics = compute_binary_metrics(y_true_np, y_pred_np, pos_label=1)

    return EvalResult(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        y_prob_pos=y_prob_pos,
        paths=paths,
    )

--

# src/cli/eval.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from src.eval.evaluator import run_eval
from src.eval.metrics import metrics_to_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (default: <exp_root>/best.pt)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg: Dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    exp_root = Path(cfg["output"]["exp_root"])
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (exp_root / "best.pt")

    result = run_eval(cfg, checkpoint_path=ckpt_path, split=args.split)

    m = result.metrics
    print(f"[eval] split={args.split} checkpoint={ckpt_path}")
    print(f"[eval] accuracy={m.accuracy:.4f} precision(rotated)={m.precision_pos:.4f} recall(rotated)={m.recall_pos:.4f} f1(rotated)={m.f1_pos:.4f}")
    print(f"[eval] confusion=[[tn,fp],[fn,tp]] = {m.confusion}  support_neg={m.support_neg} support_pos={m.support_pos}")

    out = {
        "config": str(cfg_path),
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "metrics": metrics_to_dict(m),
    }

    out_path = exp_root / f"eval_{args.split}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()

xxx