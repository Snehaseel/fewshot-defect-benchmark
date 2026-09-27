"""Dataset loading for MVTec AD and VisA, and reproducible k-shot subset sampling.

Images are loaded once per category, resized (whole image, no crop) to CACHE_SIZE,
and kept as uint8 arrays. Each method then resizes to its own input size. Ground-truth
masks are resized to EVAL_SIZE, the common grid on which all anomaly maps are scored.
"""
from __future__ import annotations

import csv
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

CACHE_SIZE = 512
EVAL_SIZE = 256

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]
MVTEC_TEXTURES = {"carpet", "grid", "leather", "tile", "wood"}
VISA_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2",
    "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]
# Plain-language names for WinCLIP's text prompts.
PROMPT_NAMES = {
    "metal_nut": "metal nut", "chewinggum": "chewing gum", "macaroni1": "macaroni",
    "macaroni2": "macaroni", "pcb1": "printed circuit board", "pcb2": "printed circuit board",
    "pcb3": "printed circuit board", "pcb4": "printed circuit board", "pipe_fryum": "fryum",
    "capsules": "capsules",
}
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".JPG", ".PNG")


@dataclass
class CategoryData:
    dataset: str
    category: str
    train_paths: list[str]
    train_images: np.ndarray            # (N, CACHE, CACHE, 3) uint8
    test_paths: list[str]
    test_images: np.ndarray             # (M, CACHE, CACHE, 3) uint8
    test_labels: np.ndarray             # (M,) 0 = good, 1 = defective
    test_masks: np.ndarray              # (M, EVAL, EVAL) bool
    test_defect_types: list[str] = field(default_factory=list)

    @property
    def is_texture(self) -> bool:
        return self.dataset == "mvtec" and self.category in MVTEC_TEXTURES

    @property
    def prompt_name(self) -> str:
        return PROMPT_NAMES.get(self.category, self.category.replace("_", " "))


def _load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((CACHE_SIZE, CACHE_SIZE), Image.BICUBIC)
        return np.asarray(im, dtype=np.uint8)


def _load_mask(path: str | None) -> np.ndarray:
    if path is None or not os.path.exists(path):
        return np.zeros((EVAL_SIZE, EVAL_SIZE), dtype=bool)
    with Image.open(path) as im:
        im = im.convert("L").resize((EVAL_SIZE, EVAL_SIZE), Image.NEAREST)
        return np.asarray(im) > 0


def _files(folder: Path) -> list[str]:
    return sorted(str(p) for p in folder.iterdir() if p.suffix in IMG_EXT)


def load_mvtec(root: str, category: str) -> CategoryData:
    base = Path(root) / category
    train_paths = _files(base / "train" / "good")
    test_paths, labels, masks, types = [], [], [], []
    for defect_dir in sorted((base / "test").iterdir()):
        if not defect_dir.is_dir():
            continue
        for p in _files(defect_dir):
            test_paths.append(p)
            good = defect_dir.name == "good"
            labels.append(0 if good else 1)
            types.append(defect_dir.name)
            mask_path = None if good else str(
                base / "ground_truth" / defect_dir.name / (Path(p).stem + "_mask.png"))
            masks.append(_load_mask(mask_path))
    return CategoryData(
        "mvtec", category, train_paths, np.stack([_load_rgb(p) for p in train_paths]),
        test_paths, np.stack([_load_rgb(p) for p in test_paths]), np.array(labels),
        np.stack(masks), types)


def load_visa(root: str, category: str) -> CategoryData:
    """Uses the official one-class split, VisA_20220922/split_csv/1cls.csv."""
    root_p = Path(root)
    split_csv = root_p / "split_csv" / "1cls.csv"
    train_paths, test_paths, labels, masks, types = [], [], [], [], []
    with open(split_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["object"] != category:
                continue
            path = str(root_p / row["image"])
            anomalous = row["label"].strip().lower() == "anomaly"
            if row["split"] == "train":
                if not anomalous:
                    train_paths.append(path)
                continue
            test_paths.append(path)
            labels.append(int(anomalous))
            types.append("anomaly" if anomalous else "good")
            mask_rel = row.get("mask") or ""
            masks.append(_load_mask(str(root_p / mask_rel) if (anomalous and mask_rel) else None))
    train_paths.sort()
    return CategoryData(
        "visa", category, train_paths, np.stack([_load_rgb(p) for p in train_paths]),
        test_paths, np.stack([_load_rgb(p) for p in test_paths]), np.array(labels),
        np.stack(masks), types)


def load_category(dataset: str, root: str, category: str) -> CategoryData:
    if dataset == "mvtec":
        return load_mvtec(root, category)
    if dataset == "visa":
        return load_visa(root, category)
    raise ValueError(f"Unknown dataset {dataset!r}")


def categories(dataset: str) -> list[str]:
    return {"mvtec": MVTEC_CATEGORIES, "visa": VISA_CATEGORIES}[dataset]


def subset_seed(dataset: str, category: str, k: int | str, subset: int) -> int:
    """Deterministic seed that depends only on (dataset, category, k, subset), never on method,
    so every method sees exactly the same training images."""
    key = f"{dataset}|{category}|{k}|{subset}".encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def sample_subset(n_train: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_train, size=k, replace=False))


def make_synthetic_mvtec(root: str, n_categories: int = 2, n_train: int = 20,
                         n_test_good: int = 8, n_test_bad: int = 8, size: int = 128) -> list[str]:
    """Writes a tiny MVTec-format dataset (textured images; defects are bright blobs)
    so that the whole pipeline can be smoke-tested in minutes."""
    rng = np.random.default_rng(0)
    names = [f"synth{i}" for i in range(n_categories)]
    yy, xx = np.mgrid[0:size, 0:size]
    for c, name in enumerate(names):
        base = Path(root) / name

        def good_image():
            freq = 0.15 + 0.05 * c
            img = 0.5 + 0.25 * np.sin(freq * xx + rng.normal(0, 0.3)) * np.cos(freq * yy)
            img = img[..., None] * np.array([0.9, 0.6 + 0.1 * c, 0.4])
            img += rng.normal(0, 0.03, img.shape)
            return np.clip(img, 0, 1)

        def save(arr, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((arr * 255).astype(np.uint8)).save(path)

        for i in range(n_train):
            save(good_image(), base / "train" / "good" / f"{i:03d}.png")
        for i in range(n_test_good):
            save(good_image(), base / "test" / "good" / f"{i:03d}.png")
        for i in range(n_test_bad):
            img = good_image()
            cy, cx = rng.integers(20, size - 20, 2)
            blob = (yy - cy) ** 2 + (xx - cx) ** 2 < rng.integers(36, 144)
            img[blob] = [1.0, 1.0, 1.0]
            save(img, base / "test" / "blob" / f"{i:03d}.png")
            mpath = base / "ground_truth" / "blob" / f"{i:03d}_mask.png"
            mpath.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((blob * 255).astype(np.uint8)).save(mpath)
    return names
