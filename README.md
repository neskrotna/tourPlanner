# tourPlanner

python - <<'PY'
import yaml
import torch
from pathlib import Path

from src.data.packcrop_dataset import PackCropDataset
from src.models.factory import create_model

cfg = yaml.safe_load(open("configs/binary/resnet18.yaml"))

manifest = cfg["dataset"]["manifest_csv"]
classes = cfg["dataset"]["classes_json"]
split = cfg["dataset"]["split_json"]

ds = PackCropDataset(manifest, classes, split, split="train")

print("Dataset size:", len(ds))
x, y, meta = ds[0]
print("Sample shape:", x.shape, "label:", y)

model = create_model(cfg)
model.eval()

x = x.unsqueeze(0)   # add batch dimension
logits = model(x)

print("Forward pass OK")
print("Output shape:", logits.shape)
PY

--

python - <<'PY'
from src.data.packcrop_dataset import PackCropDataset
import yaml

cfg = yaml.safe_load(open("configs/binary/resnet18.yaml"))

ds = PackCropDataset(
    cfg["dataset"]["manifest_csv"],
    cfg["dataset"]["classes_json"],
    cfg["dataset"]["split_json"],
    split="train"
)

print(ds.class_counts())
PY

xxx