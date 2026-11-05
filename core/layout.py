"""
Single data-folder layout used by every step of the app.

    <data>/
        images/            flat after download, images/{train,val,test}/ after split
        labels/            flat after convert,  labels/{train,val,test}/ after split
        annotations/instances_default.json
        data.yaml
        runs/              training results (runs/detect, runs/segment) and runs/test

Folder names are fixed: Ultralytics finds each label by replacing /images/ with /labels/ in the
image path, so `labels` must be spelled exactly like this.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class DatasetLayout:
    def __init__(self, root):
        self.root = Path(root)

    @property
    def images(self) -> Path:
        return self.root / "images"

    @property
    def labels(self) -> Path:
        return self.root / "labels"

    @property
    def annotations(self) -> Path:
        return self.root / "annotations"

    @property
    def coco_json(self) -> Path:
        return self.annotations / "instances_default.json"

    @property
    def data_yaml(self) -> Path:
        return self.root / "data.yaml"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def test_runs(self) -> Path:
        return self.runs / "test"

    # ---- discovery -------------------------------------------------------------------------
    def image_locations(self) -> dict[str, Path]:
        """file name -> path, for images in images/ and images/{train,val,test}/."""
        out: dict[str, Path] = {}
        for d in [self.images] + [self.images / s for s in SPLITS]:
            if not d.is_dir():
                continue
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    out.setdefault(p.name, p)
        return out

    def label_files(self) -> list[Path]:
        out = []
        for d in [self.labels] + [self.labels / s for s in SPLITS]:
            if d.is_dir():
                out.extend(p for p in d.glob("*.txt") if p.is_file())
        return out

    def split_of(self, image_path: Path) -> str:
        """'train'/'val'/'test' when the image sits in a split folder, '' when flat."""
        parent = Path(image_path).parent
        return parent.name if parent.parent == self.images and parent.name in SPLITS else ""

    def label_for(self, image_path: Path) -> Path:
        sub = self.split_of(image_path)
        base = self.labels / sub if sub else self.labels
        return base / f"{Path(image_path).stem}.txt"

    def is_split(self) -> bool:
        return any((self.images / s).is_dir() and any((self.images / s).iterdir()) for s in SPLITS)

    def counts(self) -> dict:
        locs = self.image_locations()
        per = {s: 0 for s in SPLITS}
        flat = 0
        for p in locs.values():
            s = self.split_of(p)
            if s:
                per[s] += 1
            else:
                flat += 1
        labelled = sum(1 for p in locs.values() if self.label_for(p).exists())
        return {"images": len(locs), "flat": flat, "labelled": labelled, **per}

    def class_names(self) -> list[str]:
        """Class names in the order the converter assigns ids (COCO categories sorted by id)."""
        if not self.coco_json.exists():
            return []
        with open(self.coco_json, "r", encoding="utf-8") as f:
            cats = json.load(f).get("categories", [])
        return [c["name"] for c in sorted(cats, key=lambda c: c["id"])]

    # ---- maintenance -----------------------------------------------------------------------
    def clear_labels(self) -> int:
        """Remove every YOLO label (and Ultralytics *.cache) so a new conversion starts clean."""
        n = 0
        for p in self.label_files():
            p.unlink()
            n += 1
        self.clear_caches()
        return n

    def clear_caches(self):
        for d in [self.labels] + [self.labels / s for s in SPLITS]:
            if d.is_dir():
                for c in d.glob("*.cache"):
                    c.unlink(missing_ok=True)
        if self.labels.is_dir():
            for c in self.labels.parent.glob("labels*.cache"):
                c.unlink(missing_ok=True)

    def place_labels_next_to_images(self) -> int:
        """
        After converting into flat labels/, move each label into labels/<split>/ when its image
        has already been split. Returns how many labels were moved.
        """
        moved = 0
        for name, img in self.image_locations().items():
            sub = self.split_of(img)
            if not sub:
                continue
            src = self.labels / f"{img.stem}.txt"
            if src.exists():
                dst = self.labels / sub / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    dst.unlink()
                shutil.move(str(src), str(dst))
                moved += 1
        return moved

    def write_data_yaml(self, names: list[str], has_test: bool) -> Path:
        abs_root = self.root.resolve().as_posix()
        lines = [
            f"path: '{abs_root}'",
            "train: images/train",
            "val: images/val",
            "test: images/test" if has_test else "test: images/val",
            f"nc: {len(names)}",
            "names: [" + ", ".join("'" + n.replace("'", "''") + "'" for n in names) + "]",
        ]
        self.data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return self.data_yaml