"""
Dataset Merger module for combining multiple CVAT COCO 1.0 export folders.
Resolves duplicate category names, unique image filenames, and remaps annotation/image IDs.
"""

import json
import shutil
import hashlib
import typing
from pathlib import Path


class CocoDatasetMerger:
    """Merges multiple COCO 1.0 datasets into a single unified COCO dataset."""

    def __init__(self):
        self.cat_name_to_newid: dict[str, int] = {}
        self.new_categories: list[dict] = []
        self.next_cat_id: int = 1
        self.seen_file_names: set[str] = set()

    def _reset(self):
        self.cat_name_to_newid.clear()
        self.new_categories.clear()
        self.next_cat_id = 1
        self.seen_file_names.clear()

    def get_or_create_cat_id(self, cat: dict) -> int:
        name = cat["name"]
        if name not in self.cat_name_to_newid:
            self.cat_name_to_newid[name] = self.next_cat_id
            self.new_categories.append({
                "id": self.next_cat_id,
                "name": name,
                "supercategory": cat.get("supercategory", ""),
            })
            self.next_cat_id += 1
        return self.cat_name_to_newid[name]

    def unique_name(self, src_path: Path, original_name: str) -> str:
        """Generate a unique filename to prevent collisions between different tasks."""
        if original_name not in self.seen_file_names:
            self.seen_file_names.add(original_name)
            return original_name

        stem = Path(original_name).stem
        suffix = Path(original_name).suffix
        try:
            st = src_path.stat()
            key = f"{src_path.as_posix()}|{st.st_size}|{int(st.st_mtime)}"
        except Exception:
            key = src_path.as_posix()

        h = hashlib.md5(key.encode("utf-8")).hexdigest()[:8]
        new_name = f"{stem}_{h}{suffix}"
        if new_name in self.seen_file_names:
            i = 2
            while True:
                alt = f"{stem}_{h}_{i}{suffix}"
                if alt not in self.seen_file_names:
                    new_name = alt
                    break
                i += 1

        self.seen_file_names.add(new_name)
        return new_name

    @staticmethod
    def guess_image_dir(coco_json_path: Path) -> Path:
        ann_dir = coco_json_path.parent
        root = ann_dir.parent
        img_dir = root / "images"
        if not img_dir.exists():
            # Check fallback if images/default still exists
            default_dir = root / "images" / "default"
            if default_dir.exists():
                return default_dir
            raise FileNotFoundError(f"Image directory not found for {coco_json_path}\nExpected: {img_dir}")
        return img_dir

    def merge_datasets(
        self,
        source_dirs_or_base: typing.Union[Path, list[Path]],
        output_dir: Path,
        progress_callback: typing.Optional[typing.Callable[[int, int, str], None]] = None,
    ) -> dict:
        """
        Merge multiple COCO dataset directories into output_dir.
        
        Args:
            source_dirs_or_base: Either a single base Path to search with rglob,
                                 or a list of specific task directory Paths.
            output_dir: Target directory where merged images and annotations will be saved.
            progress_callback: Optional callback(current, total, message).
        
        Returns:
            Dictionary with merge statistics and output paths.
        """
        self._reset()

        coco_jsons: list[Path] = []
        if isinstance(source_dirs_or_base, list):
            for s_dir in source_dirs_or_base:
                p = Path(s_dir)
                if p.is_file() and p.name.endswith(".json"):
                    coco_jsons.append(p)
                elif p.is_dir():
                    found = list(p.rglob("instances_default.json"))
                    if not found:
                        found = list(p.glob("*.json"))
                    coco_jsons.extend(found)
        else:
            base = Path(source_dirs_or_base)
            coco_jsons = sorted(base.rglob("annotations/instances_default.json"))
            if not coco_jsons:
                coco_jsons = sorted(base.rglob("*.json"))

        # Deduplicate JSON paths, keeping the given order: the first dataset decides the class order,
        # so adding new tasks to an existing dataset never re-numbers its classes.
        coco_jsons = list(dict.fromkeys(coco_jsons))

        if not coco_jsons:
            raise FileNotFoundError("No COCO json files found to merge.")

        out_img_dir = output_dir / "images"
        out_ann_dir = output_dir / "annotations"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_ann_dir.mkdir(parents=True, exist_ok=True)
        out_ann_path = out_ann_dir / "instances_default.json"

        merged_images = []
        merged_annotations = []
        next_image_id = 1
        next_ann_id = 1
        sources_info = []

        total_files = len(coco_jsons)

        for idx, coco_json in enumerate(coco_jsons):
            if progress_callback:
                progress_callback(idx, total_files, f"Merging ({idx+1}/{total_files}): {coco_json.parent.parent.name}")

            img_dir = self.guess_image_dir(coco_json)

            with open(coco_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            old_to_new_img_id = {}
            old_catid_to_new = {}

            # Map categories
            for cat in data.get("categories", []):
                new_id = self.get_or_create_cat_id(cat)
                old_catid_to_new[cat["id"]] = new_id

            # Process images
            for img in data.get("images", []):
                raw_file_name = img["file_name"]
                # Resolve possible subpath like default/xxx.jpg
                clean_name = Path(raw_file_name).name
                src_path = img_dir / raw_file_name
                if not src_path.exists():
                    src_path = img_dir / clean_name
                if not src_path.exists():
                    raise FileNotFoundError(f"Missing image file: {src_path}")

                target_name = self.unique_name(src_path, clean_name)
                dst_path = out_img_dir / target_name

                if not dst_path.exists():
                    shutil.copy2(src_path, dst_path)

                merged_images.append({
                    "id": next_image_id,
                    "file_name": target_name,
                    "width": img.get("width"),
                    "height": img.get("height"),
                })
                old_to_new_img_id[img["id"]] = next_image_id
                next_image_id += 1

                sources_info.append({
                    "out_file_name": target_name,
                    "src": str(src_path),
                    "coco_json": str(coco_json),
                })

            # Process annotations
            for ann in data.get("annotations", []):
                old_img_id = ann.get("image_id")
                old_cat_id = ann.get("category_id")

                if old_img_id not in old_to_new_img_id:
                    continue
                if old_cat_id not in old_catid_to_new:
                    continue

                merged_annotations.append({
                    "id": next_ann_id,
                    "image_id": old_to_new_img_id[old_img_id],
                    "category_id": old_catid_to_new[old_cat_id],
                    "bbox": ann.get("bbox"),
                    "area": ann.get("area"),
                    "iscrowd": ann.get("iscrowd", 0),
                    "segmentation": ann.get("segmentation"),
                })
                next_ann_id += 1

        merged_coco = {
            "images": merged_images,
            "annotations": merged_annotations,
            "categories": self.new_categories,
        }

        with open(out_ann_path, "w", encoding="utf-8") as f:
            json.dump(merged_coco, f, ensure_ascii=False, indent=2)

        # Also write instances_merged.json for convenience
        alt_ann_path = out_ann_dir / "instances_merged.json"
        with open(alt_ann_path, "w", encoding="utf-8") as f:
            json.dump(merged_coco, f, ensure_ascii=False, indent=2)

        if progress_callback:
            progress_callback(total_files, total_files, "Merge complete!")

        return {
            "jsons_count": len(coco_jsons),
            "images_count": len(merged_images),
            "annotations_count": len(merged_annotations),
            "categories": self.new_categories,
            "output_dir": output_dir,
            "images_dir": out_img_dir,
            "annotations_file": out_ann_path,
        }