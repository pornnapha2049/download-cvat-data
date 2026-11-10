"""
Auto-Labeler Engine for Pre-annotating Images using Local YOLO Models.
Generates standard COCO 1.0 JSON annotations and integrates with CVAT.
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
import typing
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def scan_local_models(app_dir: Path) -> list[dict]:
    """
    Scans project directory for available YOLO .pt weights.
    Looks in root, models/ directory, and runs/ experiment directories.
    """
    found = []
    seen_paths = set()

    def add_weight(p: Path, category: str):
        p_resolved = p.resolve()
        if p_resolved in seen_paths or not p.is_file() or p.suffix.lower() != ".pt":
            return
        seen_paths.add(p_resolved)
        try:
            sz_mb = round(p.stat().st_size / (1024 * 1024), 1)
        except OSError:
            sz_mb = 0.0

        try:
            rel = str(p.relative_to(app_dir))
        except ValueError:
            rel = p.name

        found.append({
            "path": p_resolved,
            "name": p.name,
            "category": category,
            "display": f"{p.name} [{category}] ({sz_mb} MB)",
            "size_mb": sz_mb,
            "rel_path": rel,
        })

    # 1. Project root
    for item in app_dir.glob("*.pt"):
        add_weight(item, "Root")

    # 2. models/ directory
    models_dir = app_dir / "models"
    if models_dir.exists():
        for item in models_dir.glob("*.pt"):
            add_weight(item, "Models")

    # 3. runs/ weights
    runs_dir = app_dir / "runs"
    if runs_dir.exists():
        for item in runs_dir.rglob("*.pt"):
            add_weight(item, "Runs")

    return found


def inspect_model_metadata(model_path: Path) -> dict:
    """Loads a YOLO model and returns metadata including task type and class list."""
    try:
        model = YOLO(str(model_path))
        task = getattr(model, "task", "detect")
        names = getattr(model, "names", {})
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        elif not isinstance(names, dict):
            names = {}

        return {
            "status": "success",
            "task": task,
            "names": names,
            "num_classes": len(names),
            "classes_list": [names[k] for k in sorted(names.keys())],
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "task": "unknown",
            "names": {},
            "num_classes": 0,
            "classes_list": [],
        }


class AutoLabelEngine:
    """
    Performs batch inference on images using local YOLO models
    and creates standardized COCO 1.0 dataset JSON files.
    """

    def __init__(self):
        self.stop_requested: bool = False
        self._cached_model = None
        self._cached_model_path = None

    def stop(self):
        """Signals active inference loop to abort gracefully."""
        self.stop_requested = True

    def get_model(self, model_path: Path | str) -> YOLO:
        p_str = str(model_path)
        if self._cached_model is not None and self._cached_model_path == p_str:
            return self._cached_model
        model = YOLO(p_str)
        self._cached_model = model
        self._cached_model_path = p_str
        return model

    def run_inference_and_export_coco(
        self,
        model_path: Path,
        image_paths: list[Path],
        output_json_path: Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "auto",
        progress_callback: typing.Optional[typing.Callable[[int, int, str], None]] = None,
        preview_callback: typing.Optional[typing.Callable[[np.ndarray, dict], None]] = None,
        selected_classes: typing.Optional[list[str]] = None,
    ) -> dict:
        """
        Runs inference on given image files, formats detections into COCO 1.0 JSON,
        and saves instances_default.json.
        """
        self.stop_requested = False
        t_start = time.time()

        if not image_paths:
            raise ValueError("No images provided for auto-labeling.")

        if progress_callback:
            progress_callback(1, 100, f"Loading model: {model_path.name}...")

        model = self.get_model(model_path)
        names = getattr(model, "names", {})
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}

        # Build COCO categories
        # category_id = class_idx + 1 (1-indexed for standard COCO)
        categories = []
        name_to_cat_id = {}
        for cls_idx, cls_name in sorted(names.items(), key=lambda x: int(x[0])):
            cat_id = int(cls_idx) + 1
            name_to_cat_id[cls_name] = cat_id
            categories.append({
                "id": cat_id,
                "name": str(cls_name),
                "supercategory": "",
            })

        # Selected classes filter set
        allowed_classes = set(selected_classes) if selected_classes else None

        coco_images = []
        coco_annotations = []
        ann_id_counter = 1
        class_detection_counts: dict[str, int] = {cat["name"]: 0 for cat in categories}

        total_images = len(image_paths)

        # Prepare device parameter for Ultralytics
        dev_arg = None
        if device == "cpu":
            dev_arg = "cpu"
        elif device == "cuda" or device == "gpu":
            dev_arg = 0

        for idx, img_path in enumerate(image_paths):
            if self.stop_requested:
                raise InterruptedError("Auto-labeling inference cancelled by user.")

            img_id = idx + 1
            filename = img_path.name

            # Read image dimensions
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception:
                # Fallback to OpenCV if PIL fails
                cv_img = cv2.imread(str(img_path))
                if cv_img is None:
                    continue
                h, w = cv_img.shape[:2]

            coco_images.append({
                "id": img_id,
                "file_name": filename,
                "width": int(w),
                "height": int(h),
            })

            # Run prediction
            predict_kwargs = {
                "source": str(img_path),
                "conf": conf,
                "iou": iou,
                "imgsz": imgsz,
                "verbose": False,
            }
            if dev_arg is not None:
                predict_kwargs["device"] = dev_arg

            results = model.predict(**predict_kwargs)
            res = results[0]

            image_ann_count = 0
            has_masks = res.masks is not None and len(res.masks) > 0

            if res.boxes is not None and len(res.boxes) > 0:
                xyxy_arr = res.boxes.xyxy.cpu().numpy()
                confs_arr = res.boxes.conf.cpu().numpy()
                cls_arr = res.boxes.cls.cpu().numpy().astype(int)

                masks_xy = res.masks.xy if has_masks else None

                for b_idx, (box, box_conf, cls_idx) in enumerate(zip(xyxy_arr, confs_arr, cls_arr)):
                    cls_name = names.get(cls_idx, str(cls_idx))
                    if allowed_classes is not None and cls_name not in allowed_classes:
                        continue

                    cat_id = name_to_cat_id.get(cls_name, int(cls_idx) + 1)
                    x1, y1, x2, y2 = box
                    x1 = max(0.0, min(float(x1), float(w)))
                    y1 = max(0.0, min(float(y1), float(h)))
                    x2 = max(0.0, min(float(x2), float(w)))
                    y2 = max(0.0, min(float(y2), float(h)))
                    bw = max(0.0, x2 - x1)
                    bh = max(0.0, y2 - y1)
                    area = float(bw * bh)

                    # Segmentation polygon
                    segmentation = []
                    if has_masks and masks_xy and b_idx < len(masks_xy):
                        poly = masks_xy[b_idx]
                        if len(poly) >= 3:
                            segmentation = [poly.flatten().tolist()]

                    # Default 4-point polygon if no segmentation mask
                    if not segmentation:
                        segmentation = [[
                            float(x1), float(y1),
                            float(x1 + bw), float(y1),
                            float(x1 + bw), float(y1 + bh),
                            float(x1), float(y1 + bh),
                        ]]

                    coco_annotations.append({
                        "id": ann_id_counter,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "segmentation": segmentation,
                        "area": round(area, 2),
                        "bbox": [round(float(x1), 2), round(float(y1), 2), round(bw, 2), round(bh, 2)],
                        "iscrowd": 0,
                        "score": round(float(box_conf), 4),
                    })

                    ann_id_counter += 1
                    image_ann_count += 1
                    class_detection_counts[cls_name] = class_detection_counts.get(cls_name, 0) + 1

            # Live preview callback
            if preview_callback and (idx % max(1, total_images // 50) == 0 or idx == total_images - 1):
                try:
                    annotated_bgr = res.plot(line_width=2, labels=True, conf=True)
                    stats = {
                        "image_index": idx + 1,
                        "total_images": total_images,
                        "image_name": filename,
                        "detections": image_ann_count,
                        "total_annotations": len(coco_annotations),
                    }
                    preview_callback(annotated_bgr, stats)
                except Exception:
                    pass

            if progress_callback:
                pct = int(((idx + 1) / total_images) * 100)
                progress_callback(
                    idx + 1,
                    total_images,
                    f"Annotating image {idx + 1}/{total_images}: {filename} ({image_ann_count} objects)"
                )

        # Assemble COCO 1.0 JSON structure
        coco_dataset = {
            "info": {
                "description": f"Auto-labeled with {model_path.name} via CVAT to YOLO Studio",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [],
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": categories,
        }

        # Write to output_json_path
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(coco_dataset, f, ensure_ascii=False, indent=2)

        elapsed = round(time.time() - t_start, 2)

        return {
            "status": "success",
            "json_path": output_json_path,
            "total_images": len(coco_images),
            "total_annotations": len(coco_annotations),
            "categories": categories,
            "class_counts": {k: v for k, v in class_detection_counts.items() if v > 0},
            "elapsed_seconds": elapsed,
        }
