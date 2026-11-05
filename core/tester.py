"""
Model Testing & Inference Engine for YOLO (Images, Folders, and Videos).
Supports detection and segmentation inference with configurable confidence,
IoU threshold, visual preview, and optional saving of results.
"""

from pathlib import Path
import typing
import cv2
import numpy as np
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv"}


class _IdDict(dict):
    """Dictionary mapping keys to their string representation, falling back to str(key)."""
    def __missing__(self, key):
        return str(key)
    def get(self, key, default=None):
        return str(key)


class YoloTestEngine:
    """Manages model prediction for single images, folders, or video files."""

    def __init__(self):
        self.stop_requested: bool = False
        self._loaded_model = None
        self._loaded_path = None
        self.label_mode: str = "name"   # "name" (class name), "id" (numeric class id), "none" (no label text)
        self.font_size: int = 14
        self.show_conf: bool = True

    def set_visual_settings(
        self,
        label_mode: str | None = None,
        font_size: int | None = None,
        show_conf: bool | None = None,
    ):
        """Updates real-time visual annotation settings dynamically during or before inference."""
        if label_mode is not None:
            self.label_mode = label_mode
        if font_size is not None:
            self.font_size = max(6, int(font_size))
        if show_conf is not None:
            self.show_conf = bool(show_conf)

    def get_visual_settings(self) -> tuple[str, int, bool]:
        """Returns current (label_mode, font_size, show_conf)."""
        return self.label_mode, self.font_size, self.show_conf

    def plot_result(
        self,
        res,
        label_mode: str | None = None,
        font_size: int | None = None,
        show_conf: bool | None = None,
    ) -> np.ndarray:
        """
        Plots annotations on a YOLO Results object with custom label mode and font size.
        - label_mode: 'name' (class name), 'id' (numeric class id), 'none' (bounding box/mask only)
        - font_size: font size in pixels (rendered via PIL truetype)
        - show_conf: whether confidence score is displayed
        """
        mode = label_mode if label_mode is not None else self.label_mode
        fs = font_size if font_size is not None else self.font_size
        conf = show_conf if show_conf is not None else self.show_conf

        orig_names = res.names
        try:
            if mode == "none":
                annotated = res.plot(pil=True, font_size=fs, labels=False, conf=conf)
            elif mode == "id":
                res.names = _IdDict({k: str(k) for k in orig_names})
                annotated = res.plot(pil=True, font_size=fs, labels=True, conf=conf)
            else:  # "name" (default)
                annotated = res.plot(pil=True, font_size=fs, labels=True, conf=conf)
        finally:
            res.names = orig_names

        return np.ascontiguousarray(np.asarray(annotated))

    def stop(self):
        """Signals active inference loop to abort gracefully."""
        self.stop_requested = True

    def get_model(self, model_path: Path | str) -> YOLO:
        """Caches and returns loaded YOLO model instance."""
        p_str = str(model_path)
        if self._loaded_model is not None and self._loaded_path == p_str:
            return self._loaded_model
        model = YOLO(p_str)
        self._loaded_model = model
        self._loaded_path = p_str
        return model

    def predict_image(
        self,
        model_path: Path | str,
        image_path: Path | str,
        conf: float = 0.25,
        iou: float = 0.45,
        save: bool = False,
        save_dir: Path | None = None,
        font_size: int | None = None,
        label_mode: str | None = None,
    ) -> dict:
        """
        Runs prediction on a single image.
        Returns dictionary containing annotated image (BGR), detections list, speed, etc.
        """
        self.stop_requested = False
        img_p = Path(image_path)
        if not img_p.exists():
            raise FileNotFoundError(f"Image not found: {img_p}")

        model = self.get_model(model_path)
        results = model.predict(
            source=str(img_p),
            conf=conf,
            iou=iou,
            save=False,
            verbose=False
        )

        res = results[0]
        mode = label_mode if label_mode is not None else self.label_mode
        fs = font_size if font_size is not None else self.font_size
        annotated_bgr = self.plot_result(res, label_mode=mode, font_size=fs, show_conf=self.show_conf)

        detections = []
        if res.boxes is not None and len(res.boxes) > 0:
            for i, box in enumerate(res.boxes):
                cls_id = int(box.cls[0].item())
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                conf_val = float(box.conf[0].item())
                xyxy = [float(x) for x in box.xyxy[0].tolist()]
                detections.append({
                    "id": i,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "conf": conf_val,
                    "box": xyxy,
                })

        saved_file = None
        if save and save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            saved_file = save_dir / f"pred_{img_p.name}"
            cv2.imwrite(str(saved_file), annotated_bgr)

        # Classes summary count
        cls_summary = {}
        for d in detections:
            cname = d["class_name"]
            cls_summary[cname] = cls_summary.get(cname, 0) + 1

        return {
            "source": str(img_p),
            "file_name": img_p.name,
            "annotated_bgr": annotated_bgr,
            "raw_result": res,
            "detections": detections,
            "count": len(detections),
            "classes_summary": cls_summary,
            "saved_path": saved_file,
            "speed": getattr(res, "speed", {}),
        }

    def predict_folder(
        self,
        model_path: Path | str,
        folder_path: Path | str,
        conf: float = 0.25,
        iou: float = 0.45,
        save: bool = False,
        save_dir: Path | None = None,
        progress_callback: typing.Optional[typing.Callable[[int, int, str], None]] = None,
        frame_callback: typing.Optional[typing.Callable[[np.ndarray, int, int, dict], None]] = None,
        font_size: int | None = None,
        label_mode: str | None = None,
    ) -> dict:
        """
        Runs prediction sequentially across all images in a folder.
        """
        self.stop_requested = False
        f_path = Path(folder_path)
        if not f_path.is_dir():
            raise NotADirectoryError(f"Folder not found: {f_path}")

        image_files = sorted([
            f for f in f_path.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        ])

        if not image_files:
            raise ValueError(f"No valid images found in directory: {f_path}")

        total = len(image_files)
        results = []
        overall_classes = {}

        for idx, img_p in enumerate(image_files):
            if self.stop_requested:
                break

            if progress_callback:
                progress_callback(idx + 1, total, f"Testing image {idx + 1}/{total}: {img_p.name}")

            res = self.predict_image(
                model_path=model_path,
                image_path=img_p,
                conf=conf,
                iou=iou,
                save=save,
                save_dir=save_dir,
                font_size=font_size,
                label_mode=label_mode,
            )
            results.append(res)

            for cname, count in res["classes_summary"].items():
                overall_classes[cname] = overall_classes.get(cname, 0) + count

            if frame_callback:
                frame_callback(res["annotated_bgr"], idx + 1, total, res["classes_summary"])

        return {
            "folder": str(f_path),
            "total_images": len(image_files),
            "processed": len(results),
            "results": results,
            "classes_summary": overall_classes,
            "saved_dir": str(save_dir) if (save and save_dir) else None,
        }

    def predict_video(
        self,
        model_path: Path | str,
        video_path: Path | str,
        conf: float = 0.25,
        iou: float = 0.45,
        save: bool = False,
        save_dir: Path | None = None,
        progress_callback: typing.Optional[typing.Callable[[int, int, str], None]] = None,
        frame_callback: typing.Optional[typing.Callable[[np.ndarray, int, int, dict], None]] = None,
        font_size: int | None = None,
        label_mode: str | None = None,
    ) -> dict:
        """
        Runs prediction on a video file with frame-by-frame preview and optional output video recording.
        """
        self.stop_requested = False
        vid_p = Path(video_path)
        if not vid_p.exists():
            raise FileNotFoundError(f"Video file not found: {vid_p}")

        model = self.get_model(model_path)

        cap = cv2.VideoCapture(str(vid_p))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {vid_p}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        saved_video_path = None
        if save and save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            saved_video_path = save_dir / f"pred_{vid_p.stem}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(saved_video_path), fourcc, fps, (width, height))

        frame_idx = 0
        overall_classes = {}
        last_annotated = None
        last_res = None

        try:
            while cap.isOpened():
                if self.stop_requested:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # Predict on frame
                results = model.predict(
                    source=frame,
                    conf=conf,
                    iou=iou,
                    save=False,
                    verbose=False
                )
                res = results[0]
                cur_mode = label_mode if label_mode is not None else self.label_mode
                cur_fs = font_size if font_size is not None else self.font_size
                annotated = self.plot_result(res, label_mode=cur_mode, font_size=cur_fs, show_conf=self.show_conf)
                last_annotated = annotated
                last_res = res

                # Collect detected class counts
                frame_summary = {}
                if res.boxes is not None and len(res.boxes) > 0:
                    for box in res.boxes:
                        cid = int(box.cls[0].item())
                        cname = model.names.get(cid, f"class_{cid}")
                        frame_summary[cname] = frame_summary.get(cname, 0) + 1
                        overall_classes[cname] = overall_classes.get(cname, 0) + 1

                if writer is not None:
                    writer.write(annotated)

                if frame_callback and (frame_idx % 2 == 0 or frame_idx == total_frames):
                    frame_callback(annotated, frame_idx, total_frames, frame_summary)

                if progress_callback and (frame_idx % 5 == 0 or frame_idx == total_frames):
                    progress_callback(frame_idx, total_frames, f"Processing Frame {frame_idx}/{total_frames}")

        finally:
            cap.release()
            if writer is not None:
                writer.release()

        return {
            "video_path": str(vid_p),
            "total_frames": total_frames,
            "processed_frames": frame_idx,
            "fps": fps,
            "last_annotated": last_annotated,
            "last_result": last_res,
            "saved_path": saved_video_path,
            "classes_summary": overall_classes,
        }
