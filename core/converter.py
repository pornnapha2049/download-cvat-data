"""
COCO 1.0 to YOLO Converter module.
Converts COCO JSON annotations to YOLO format for Detection and Segmentation tasks.
- Detection: converts polygons to bounding boxes (cls xc yc w h).
- Segmentation: converts polygons to normalized points (cls x1 y1 x2 y2 ...).
"""

import json
import typing
from pathlib import Path


class CocoToYoloConverter:
    """Converts COCO 1.0 formatted annotations to YOLO txt format."""

    @staticmethod
    def _clamp(val: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, val))

    def convert(
        self,
        coco_json_path: Path,
        output_labels_dir: Path,
        mode: str = "detect",  # "detect" or "segment"
        progress_callback: typing.Optional[typing.Callable[[int, int, str], None]] = None,
    ) -> dict:
        """
        Convert COCO annotations into YOLO .txt label files.
        
        Args:
            coco_json_path: Path to the COCO JSON file.
            output_labels_dir: Directory where YOLO .txt label files will be saved.
            mode: 'detect' for bounding boxes, 'segment' for polygon segmentation.
            progress_callback: Optional progress reporter callback(current, total, message).
            
        Returns:
            Dictionary with conversion statistics and category mappings.
        """
        output_labels_dir.mkdir(parents=True, exist_ok=True)

        with open(coco_json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # Build category map: category_id -> 0-indexed YOLO class_id
        cats = sorted(coco.get("categories", []), key=lambda c: c["id"])
        cat_map = {c["id"]: idx for idx, c in enumerate(cats)}
        class_names = [c["name"] for c in cats]

        img_map = {im["id"]: im for im in coco.get("images", [])}
        ann_map: dict[int, list[dict]] = {}
        for a in coco.get("annotations", []):
            ann_map.setdefault(a["image_id"], []).append(a)

        total_images = len(img_map)
        converted_count = 0
        total_annotations_written = 0

        for idx, (img_id, img) in enumerate(img_map.items()):
            if progress_callback and (idx % 20 == 0 or idx == total_images - 1):
                progress_callback(idx + 1, total_images, f"Converting labels ({idx+1}/{total_images})...")

            w = img.get("width")
            h = img.get("height")
            if not w or not h:
                continue

            lines = []
            for ann in ann_map.get(img_id, []):
                cat_id = ann.get("category_id")
                if cat_id not in cat_map:
                    continue
                cls_idx = cat_map[cat_id]

                seg = ann.get("segmentation")
                bbox = ann.get("bbox")

                if mode == "detect":
                    # Detection Mode: convert segmentation polygon to bbox if present, otherwise use bbox
                    norm_box = None
                    if seg and isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], list) and len(seg[0]) >= 6:
                        # Extract bbox from polygon points
                        all_x = []
                        all_y = []
                        for poly in seg:
                            if isinstance(poly, list):
                                for i in range(0, len(poly) - 1, 2):
                                    all_x.append(poly[i])
                                    all_y.append(poly[i + 1])
                        if all_x and all_y:
                            min_x, max_x = min(all_x), max(all_x)
                            min_y, max_y = min(all_y), max(all_y)
                            bw = max(0.0, max_x - min_x)
                            bh = max(0.0, max_y - min_y)
                            xc = min_x + bw / 2.0
                            yc = min_y + bh / 2.0
                            norm_box = (
                                self._clamp(xc / w),
                                self._clamp(yc / h),
                                self._clamp(bw / w),
                                self._clamp(bh / h),
                            )

                    if not norm_box and bbox and len(bbox) >= 4:
                        bx, by, bw, bh = bbox[:4]
                        xc = bx + bw / 2.0
                        yc = by + bh / 2.0
                        norm_box = (
                            self._clamp(xc / w),
                            self._clamp(yc / h),
                            self._clamp(bw / w),
                            self._clamp(bh / h),
                        )

                    if norm_box:
                        n_xc, n_yc, n_bw, n_bh = norm_box
                        lines.append(f"{cls_idx} {n_xc:.6f} {n_yc:.6f} {n_bw:.6f} {n_bh:.6f}")
                        total_annotations_written += 1

                elif mode == "segment":
                    # Segmentation Mode: normalized polygon points
                    wrote_poly = False
                    if seg and isinstance(seg, list):
                        for poly in seg:
                            if isinstance(poly, list) and len(poly) >= 6:
                                norm_pts = []
                                for i, v in enumerate(poly):
                                    norm_val = self._clamp((v / w) if i % 2 == 0 else (v / h))
                                    norm_pts.append(f"{norm_val:.6f}")
                                lines.append(f"{cls_idx} " + " ".join(norm_pts))
                                total_annotations_written += 1
                                wrote_poly = True

                    # Fallback if no polygon but bbox is available: make 4-corner polygon
                    if not wrote_poly and bbox and len(bbox) >= 4:
                        bx, by, bw, bh = bbox[:4]
                        corners = [
                            self._clamp(bx / w), self._clamp(by / h),
                            self._clamp((bx + bw) / w), self._clamp(by / h),
                            self._clamp((bx + bw) / w), self._clamp((by + bh) / h),
                            self._clamp(bx / w), self._clamp((by + bh) / h),
                        ]
                        lines.append(f"{cls_idx} " + " ".join(f"{c:.6f}" for c in corners))
                        total_annotations_written += 1

            # Write label file for this image
            file_name = img.get("file_name", "")
            stem = Path(file_name).stem
            txt_path = output_labels_dir / f"{stem}.txt"

            if lines:
                txt_path.write_text("\n".join(lines), encoding="utf-8")
                converted_count += 1
            else:
                # Write empty file if desired for background representation
                txt_path.write_text("", encoding="utf-8")

        return {
            "mode": mode,
            "total_images": total_images,
            "images_with_labels": converted_count,
            "total_annotations_written": total_annotations_written,
            "classes": class_names,
            "num_classes": len(class_names),
            "output_dir": output_labels_dir,
        }
