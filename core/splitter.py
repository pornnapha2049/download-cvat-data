"""
Dataset Splitter module for YOLO datasets.
Splits images and labels into train/val/test partitions.
CRITICAL: Includes ALL images (both labeled images and unannotated background images).
Generates the Ultralytics data.yaml configuration file.
"""

import json
import random
import shutil
import typing
from pathlib import Path


class DatasetSplitter:
    """Splits dataset into train/val/test splits and generates data.yaml."""

    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    def split_dataset(
        self,
        images_dir: Path,
        labels_dir: Path,
        output_dataset_dir: Path,
        coco_json_path: typing.Optional[Path] = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.20,
        test_ratio: float = 0.10,
        class_names: typing.Optional[list[str]] = None,
        copy_files: bool = True,
        seed: int = 42,
        progress_callback: typing.Optional[typing.Callable[[int, int, str], None]] = None,
    ) -> dict:
        """
        Split dataset into train, val, and test splits.
        
        Args:
            images_dir: Directory containing input images.
            labels_dir: Directory containing YOLO .txt label files.
            output_dataset_dir: Destination folder for the split dataset.
            coco_json_path: Path to COCO JSON for extracting class names (if not directly passed).
            train_ratio: Ratio for training set (e.g. 0.70).
            val_ratio: Ratio for validation set (e.g. 0.20).
            test_ratio: Ratio for testing set (e.g. 0.10).
            class_names: Explicit list of class names in index order.
            copy_files: If True, copies files; if False, moves files.
            seed: Random seed for reproducible shuffling.
            progress_callback: Optional progress reporter.
            
        Returns:
            Dictionary with split counts and path to generated data.yaml.
        """
        # Normalize ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if total_ratio <= 0:
            train_ratio, val_ratio, test_ratio = 0.70, 0.20, 0.10
            total_ratio = 1.0

        r_train = train_ratio / total_ratio
        r_val = val_ratio / total_ratio
        r_test = test_ratio / total_ratio

        splits = ["train", "val", "test"]
        for s in splits:
            (output_dataset_dir / "images" / s).mkdir(parents=True, exist_ok=True)
            (output_dataset_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

        # Collect ALL images, regardless of whether a label exists
        all_images = [
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTS
        ]

        if not all_images:
            raise FileNotFoundError(f"No valid images found in: {images_dir}")

        random.seed(seed)
        shuffled_images = list(all_images)
        random.shuffle(shuffled_images)

        n = len(shuffled_images)
        n_train = int(n * r_train)
        n_val = int(n * r_val)
        n_test = n - n_train - n_val

        # Ensure validation has at least 1 image if n >= 2 and val_ratio > 0
        if r_val > 0 and n_val == 0 and n >= 2:
            n_val = 1
            if n_train > 1:
                n_train -= 1
            elif n_test > 0:
                n_test -= 1

        split_assignment = {}
        for i, img_path in enumerate(shuffled_images):
            if i < n_train:
                s = "train"
            elif i < n_train + n_val:
                s = "val"
            else:
                s = "test"
            split_assignment[img_path] = s

        transfer_func = shutil.copy2 if copy_files else shutil.move
        labeled_count = 0
        background_count = 0

        for idx, (img_path, s) in enumerate(split_assignment.items()):
            if progress_callback and (idx % 25 == 0 or idx == n - 1):
                progress_callback(idx + 1, n, f"Splitting images into {s} ({idx+1}/{n})...")

            dest_img = output_dataset_dir / "images" / s / img_path.name
            transfer_func(str(img_path), str(dest_img))

            # Handle label
            src_lbl = labels_dir / f"{img_path.stem}.txt"
            dest_lbl = output_dataset_dir / "labels" / s / f"{img_path.stem}.txt"

            if src_lbl.exists() and src_lbl.stat().st_size > 0:
                transfer_func(str(src_lbl), str(dest_lbl))
                labeled_count += 1
            else:
                # Background image: create empty label file so YOLO handles negative samples cleanly
                dest_lbl.write_text("", encoding="utf-8")
                background_count += 1

        # Determine class names
        names = []
        if class_names:
            names = list(class_names)
        elif coco_json_path and coco_json_path.exists():
            try:
                with open(coco_json_path, "r", encoding="utf-8") as f:
                    coco_data = json.load(f)
                cats = sorted(coco_data.get("categories", []), key=lambda c: c["id"])
                names = [c["name"] for c in cats]
            except Exception:
                pass

        if not names:
            # Fallback default class names
            names = ["object"]

        # Generate data.yaml
        abs_root = output_dataset_dir.resolve().as_posix()
        yaml_lines = [
            f"path: '{abs_root}'",
            "train: images/train",
            "val: images/val",
        ]
        if n_test > 0:
            yaml_lines.append("test: images/test")
        else:
            # Point test to val if no test images
            yaml_lines.append("test: images/val")

        yaml_lines.append(f"nc: {len(names)}")
        names_formatted = ", ".join(f"'{name}'" for name in names)
        yaml_lines.append(f"names: [{names_formatted}]")

        data_yaml_path = output_dataset_dir / "data.yaml"
        data_yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

        return {
            "total_images": n,
            "train_count": n_train,
            "val_count": n_val,
            "test_count": n_test,
            "labeled_images": labeled_count,
            "background_images": background_count,
            "classes": names,
            "num_classes": len(names),
            "data_yaml_path": data_yaml_path,
            "dataset_dir": output_dataset_dir,
        }
