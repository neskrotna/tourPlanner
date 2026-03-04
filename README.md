# tourPlanner

# src/data/packcrop_dataset.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    crop_path: Path
    label_id: int
    label_name: str
    source_image: str
    json_file: str
    ann_index: int
    ann_id: str
    bbox: Tuple[int, int, int, int]
    img_size: Tuple[int, int]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_classes(classes_json: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Expected format:
    {
      "0": "normal",
      "1": "rotated"
    }
    Returns:
      name_to_id: {"normal": 0, "rotated": 1}
      id_to_name: {0: "normal", 1: "rotated"}
    """
    data = _read_json(classes_json)
    id_to_name: Dict[int, str] = {}
    name_to_id: Dict[str, int] = {}
    for k, v in data.items():
        cid = int(k)
        cname = str(v)
        id_to_name[cid] = cname
        name_to_id[cname] = cid
    return name_to_id, id_to_name


def _load_split_list(split_json: Path, split: str) -> List[str]:
    """
    Expects your split file structure:
    {
      "train": ["data/crops/normal/....png", ...],
      "val":   [...],
      "test":  [...]
      ...
    }
    It also tolerates a wrapper, but your current one looks like above.
    """
    data = _read_json(split_json)

    if split in data and isinstance(data[split], list):
        return [str(x) for x in data[split]]

    # Fallback: some people store under "splits": {"train": [...]}
    maybe_splits = data.get("splits")
    if isinstance(maybe_splits, dict) and split in maybe_splits and isinstance(maybe_splits[split], list):
        return [str(x) for x in maybe_splits[split]]

    raise ValueError(f"Split '{split}' not found in {split_json}. Keys: {list(data.keys())}")


def _as_posix_rel(p: str) -> str:
    # normalize Windows backslashes in CSV/split files
    return Path(p).as_posix()


class PackCropDataset(Dataset):
    """
    Dataset for binary (or multiclass) classification on already-cropped images.

    Uses:
      - manifest_csv: rows with crop_path + label_id/label_name + debug columns
      - split_json: list of crop_path strings per split (train/val/test)

    Key design choice:
      - We trust split_json as the "source of truth" for which samples belong to a split.
      - We then look those paths up in manifest_csv.
    """

    def __init__(
        self,
        manifest_csv: Path,
        classes_json: Path,
        split_json: Path,
        split: str,
        transform=None,
        strict: bool = True,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.classes_json = Path(classes_json)
        self.split_json = Path(split_json)
        self.split = split
        self.transform = transform
        self.strict = strict

        self.name_to_id, self.id_to_name = _load_classes(self.classes_json)

        df = pd.read_csv(self.manifest_csv)

        # Required columns for training
        required = {"crop_path", "label_id", "label_name"}
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise ValueError(f"Manifest {self.manifest_csv} missing columns: {sorted(missing_cols)}")

        # Normalize crop_path formatting for reliable matching with split_json
        df["crop_path_norm"] = df["crop_path"].astype(str).apply(_as_posix_rel)

        # Load split list and normalize
        split_list = [_as_posix_rel(p) for p in _load_split_list(self.split_json, self.split)]
        split_set = set(split_list)

        # Keep only rows that are in the split list
        df_split = df[df["crop_path_norm"].isin(split_set)].copy()

        if strict:
            # If strict, error when split contains paths not found in manifest (common gotcha)
            manifest_set = set(df["crop_path_norm"].tolist())
            not_in_manifest = [p for p in split_list if p not in manifest_set]
            if not_in_manifest:
                preview = "\n".join(not_in_manifest[:20])
                raise ValueError(
                    f"{len(not_in_manifest)} crop paths from {self.split_json} split '{self.split}' "
                    f"are not present in manifest {self.manifest_csv}.\n"
                    f"First examples:\n{preview}"
                )

        # Sort deterministically (nice for reproducibility)
        df_split = df_split.sort_values("crop_path_norm").reset_index(drop=True)

        self._df = df_split
        self._samples: List[Sample] = []

        root = self.manifest_csv.parent.parent  # data/meta -> data
        # But crop_path is already like "data/crops/..." (repo-relative), so we resolve from repo root.
        # We'll resolve from CWD at runtime, but also support absolute.
        for _, r in df_split.iterrows():
            crop_rel = Path(str(r["crop_path_norm"]))
            crop_path = crop_rel if crop_rel.is_absolute() else Path.cwd() / crop_rel

            label_id = int(r["label_id"])
            label_name = str(r["label_name"])

            # Optional debug columns (fallback safe)
            source_image = str(r["source_image"]) if "source_image" in df_split.columns else ""
            json_file = str(r["json_file"]) if "json_file" in df_split.columns else ""
            ann_index = int(r["ann_index"]) if "ann_index" in df_split.columns else -1
            ann_id = str(r["ann_id"]) if "ann_id" in df_split.columns else ""

            x1 = int(r["bbox_x1"]) if "bbox_x1" in df_split.columns else -1
            y1 = int(r["bbox_y1"]) if "bbox_y1" in df_split.columns else -1
            x2 = int(r["bbox_x2"]) if "bbox_x2" in df_split.columns else -1
            y2 = int(r["bbox_y2"]) if "bbox_y2" in df_split.columns else -1

            w = int(r["img_w"]) if "img_w" in df_split.columns else -1
            h = int(r["img_h"]) if "img_h" in df_split.columns else -1

            self._samples.append(
                Sample(
                    crop_path=crop_path,
                    label_id=label_id,
                    label_name=label_name,
                    source_image=source_image,
                    json_file=json_file,
                    ann_index=ann_index,
                    ann_id=ann_id,
                    bbox=(x1, y1, x2, y2),
                    img_size=(w, h),
                )
            )

        if len(self._samples) == 0:
            raise RuntimeError(
                f"No samples for split '{self.split}'. "
                f"Check split_json={self.split_json} and manifest_csv={self.manifest_csv}."
            )

        # Quick class sanity (helps catch label mismatches after merging)
        bad = []
        for s in self._samples:
            expected_name = self.id_to_name.get(s.label_id)
            if expected_name is None:
                bad.append(f"{s.crop_path} has label_id={s.label_id} not in {self.classes_json}")
            elif expected_name != s.label_name:
                bad.append(
                    f"{s.crop_path} label mismatch: id {s.label_id} -> '{expected_name}', "
                    f"but manifest label_name is '{s.label_name}'"
                )
        if bad:
            preview = "\n".join(bad[:20])
            raise ValueError(
                f"Found {len(bad)} label inconsistencies between manifest and classes_json.\n"
                f"First examples:\n{preview}"
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        s = self._samples[idx]

        # Load image
        if not s.crop_path.exists():
            raise FileNotFoundError(f"Crop image not found: {s.crop_path}")

        with Image.open(s.crop_path) as im:
            im = im.convert("RGB")
            if self.transform is not None:
                x = self.transform(im)
            else:
                # Minimal default: convert to tensor [0,1], CHW
                x = torch.from_numpy(
                    (torch.ByteTensor(torch.ByteStorage.from_buffer(im.tobytes()))
                     .view(im.size[1], im.size[0], 3)
                     .numpy())
                )
                x = x.permute(2, 0, 1).float() / 255.0

        y = torch.tensor(s.label_id, dtype=torch.long)

        # Returning meta is super useful for debugging misclassifications later
        meta = {
            "crop_path": str(s.crop_path),
            "label_name": s.label_name,
            "source_image": s.source_image,
            "json_file": s.json_file,
            "ann_index": s.ann_index,
            "ann_id": s.ann_id,
        }
        return x, y, meta

    @property
    def samples(self) -> Sequence[Sample]:
        return self._samples

    def class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for s in self._samples:
            counts[s.label_name] = counts.get(s.label_name, 0) + 1
        return counts

--
python -c "from pathlib import Path; from src.data.packcrop_dataset import PackCropDataset; ds=PackCropDataset(Path('data/meta/crops_manifest_binary_merged.csv'), Path('data/meta/classes_binary.json'), Path('data/splits/split_binary_v1.json'), split='train'); print(len(ds), ds.class_counts()); x,y,meta=ds[0]; print(x.shape, y.item(), meta['crop_path'])"

xxx