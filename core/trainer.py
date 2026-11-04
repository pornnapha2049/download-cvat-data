"""
YOLO Training Engine module using Ultralytics.
Supports Detection and Segmentation tasks, auto-hyperparameter tuning,
early stopping (patience), auto-versioned folder naming (_v1, _v2, ...),
and custom naming of best.pt weights.
"""

import os
import re
import sys
import shutil
import time
import typing
from pathlib import Path
import torch
from ultralytics import YOLO


def auto_tune_hyperparameters(
    num_images: int,
    num_annotations: int,
    num_classes: int,
    is_gpu: bool = False,
) -> dict:
    """
    Intelligently recommends training hyperparameters tailored to dataset size and hardware.
    User can freely modify all returned parameters in the GUI.
    """
    params = {
        "imgsz": 640,
        "workers": 2 if sys.platform == "win32" else 4,
        "device": "0" if is_gpu else "cpu",
        "save": True,
        "plots": True,
        "verbose": True,
    }

    if num_images <= 60:
        # Very small dataset: use transfer learning with AdamW, moderate epochs, smaller batch
        params.update({
            "epochs": 150,
            "batch": 4 if not is_gpu else 8,
            "patience": 25,  # Give enough room before early stop
            "lr0": 0.005,
            "lrf": 0.01,
            "warmup_epochs": 4,
            "optimizer": "AdamW",
            "weight_decay": 0.0005,
        })
    elif num_images <= 300:
        # Small to medium dataset
        params.update({
            "epochs": 120,
            "batch": 8 if not is_gpu else 16,
            "patience": 20,
            "lr0": 0.01,
            "lrf": 0.01,
            "warmup_epochs": 3,
            "optimizer": "AdamW",
            "weight_decay": 0.0005,
        })
    elif num_images <= 1200:
        # Medium dataset
        params.update({
            "epochs": 100,
            "batch": 16 if not is_gpu else 32,
            "patience": 15,
            "lr0": 0.01,
            "lrf": 0.01,
            "warmup_epochs": 3,
            "optimizer": "auto",
            "weight_decay": 0.0005,
        })
    else:
        # Large dataset
        params.update({
            "epochs": 80,
            "batch": 16 if not is_gpu else 32,
            "patience": 12,
            "lr0": 0.01,
            "lrf": 0.01,
            "warmup_epochs": 3,
            "optimizer": "auto",
            "weight_decay": 0.0005,
        })

    return params


def get_next_run_version(base_runs_dir: Path, custom_name: str) -> tuple[str, Path]:
    """
    Determines next available run version: {custom_name}_v1, {custom_name}_v2, etc.
    Returns (versioned_name, full_run_path).
    """
    base_runs_dir.mkdir(parents=True, exist_ok=True)
    clean_name = re.sub(r'[\\/*?:"<>|]', "_", custom_name).strip() or "yolo_model"

    version = 1
    while True:
        candidate_name = f"{clean_name}_v{version}"
        candidate_path = base_runs_dir / candidate_name
        if not candidate_path.exists():
            return candidate_name, candidate_path
        version += 1


class YoloTrainerEngine:
    """Manages YOLO model training, progress tracking, and weights naming."""

    def __init__(self):
        self.is_training: bool = False
        self.stop_requested: bool = False
        self.active_trainer = None

    def stop_training(self):
        """Signals active training process to stop cleanly."""
        self.stop_requested = True
        if self.active_trainer:
            try:
                self.active_trainer.stop = True
            except Exception:
                pass

    def train_model(
        self,
        task: str,  # "detect" or "segment"
        model_name_or_path: str,
        data_yaml_path: Path,
        custom_run_name: str,
        base_runs_dir: Path,
        hyperparams: dict,
        log_callback: typing.Optional[typing.Callable[[str], None]] = None,
        epoch_callback: typing.Optional[typing.Callable[[int, int, dict], None]] = None,
    ) -> dict:
        """
        Executes YOLO training session.
        
        Args:
            task: "detect" or "segment".
            model_name_or_path: Model checkpoint or preset (e.g. "yolo11n.pt", "yolo11n-seg.pt").
            data_yaml_path: Path to dataset data.yaml.
            custom_run_name: User specified experiment name.
            base_runs_dir: Base directory (e.g. runs/detect or runs/segment).
            hyperparams: Dictionary of training hyperparameters.
            log_callback: Callback receiving stdout lines.
            epoch_callback: Callback(epoch, total_epochs, metrics_dict).
            
        Returns:
            Dictionary containing run details, best weights paths, and final metrics.
        """
        self.is_training = True
        self.stop_requested = False

        task_subfolder = "segment" if task == "segment" else "detect"
        project_dir = base_runs_dir / task_subfolder

        versioned_run_name, target_run_dir = get_next_run_version(project_dir, custom_run_name)

        if log_callback:
            log_callback(f"🚀 Initializing YOLO Training for [{task.upper()}]...")
            log_callback(f"📂 Output directory: {target_run_dir}")
            log_callback(f"📦 Model: {model_name_or_path}")

        # Resolve model name: if user picked custom or YOLO26 preset that doesn't exist yet,
        # provide fallback or notify
        model_resolved = model_name_or_path
        if not Path(model_resolved).exists() and ("26" in model_resolved):
            # YOLO26 placeholder alias fallback to YOLO11 if weights not locally present
            fallback = "yolo11n-seg.pt" if task == "segment" else "yolo11n.pt"
            if log_callback:
                log_callback(f"ℹ️ '{model_name_or_path}' not found locally. Using base '{fallback}' architecture.")
            model_resolved = fallback

        model = YOLO(model_resolved)

        # Callbacks for live UI updates
        def on_train_epoch_end(trainer):
            if self.stop_requested:
                trainer.stop = True
                return

            self.active_trainer = trainer
            epoch = trainer.epoch + 1
            total_epochs = trainer.epochs

            def _num(v, default=0.0):
                """Ultralytics fields can be None (before 1st val) or tensors."""
                try:
                    return float(v.item() if hasattr(v, "item") else v)
                except (TypeError, ValueError):
                    return default

            loss = getattr(trainer, "loss", None)
            loss_val = _num(loss.sum() if loss is not None and hasattr(loss, "sum") else loss)

            metrics = getattr(trainer, "metrics", {}) or {}
            fitness = getattr(trainer, "fitness", 0.0)

            # Build readable summary
            summary = {
                "epoch": epoch,
                "total_epochs": total_epochs,
                "loss": loss_val,
                "fitness": _num(fitness),
                "best_fitness": _num(getattr(trainer, "best_fitness", None)),
            }

            for k, v in metrics.items():
                num = _num(v, None)
                if num is not None:
                    summary[k] = num

            if epoch_callback:
                epoch_callback(epoch, total_epochs, summary)

            if log_callback and (epoch % 1 == 0 or epoch == total_epochs):
                fit_str = f", Fitness={summary['fitness']:.4f}" if summary['fitness'] else ""
                log_callback(f"📊 Epoch {epoch}/{total_epochs} - Loss: {loss_val:.4f}{fit_str}")

        # Register callback
        model.add_callback("on_train_epoch_end", on_train_epoch_end)

        # Prepare training arguments
        train_args = {
            "data": str(data_yaml_path),
            "project": str(project_dir),
            "name": versioned_run_name,
            "exist_ok": True,
            "verbose": True,
        }
        train_args.update(hyperparams)

        if log_callback:
            log_callback(f"⚙️ Hyperparameters: Epochs={train_args.get('epochs')}, Batch={train_args.get('batch')}, "
                         f"ImgSz={train_args.get('imgsz')}, Patience={train_args.get('patience')}, "
                         f"LR0={train_args.get('lr0')}, Device={train_args.get('device')}")

        t_start = time.time()
        try:
            results = model.train(**train_args)
        finally:
            self.is_training = False
            self.active_trainer = None

        weights_dir = target_run_dir / "weights"
        best_pt = weights_dir / "best.pt"
        last_pt = weights_dir / "last.pt"

        # Directly rename best.pt to user's custom named versioned weight:
        # e.g. runs/detect/{custom_name}_v{run_num}/weights/{versioned_run_name}.pt
        target_custom_pt = weights_dir / f"{versioned_run_name}.pt"

        if best_pt.exists():
            if target_custom_pt.exists():
                target_custom_pt.unlink()
            best_pt.rename(target_custom_pt)
            if log_callback:
                log_callback(f"✅ Renamed best.pt -> {target_custom_pt.name} inside {weights_dir}")

        if log_callback:
            log_callback(f"🎉 Training session finished! Run saved to: {target_run_dir}")

        return {
            "run_dir": target_run_dir,
            "run_name": versioned_run_name,
            "task": task,
            "model": str(model_resolved),
            "data_yaml": str(data_yaml_path),
            "hyperparams": dict(train_args),
            "seconds": time.time() - t_start,
            "stopped_by_user": self.stop_requested,
            "best_pt": target_custom_pt if target_custom_pt.exists() else None,
            "custom_pt": target_custom_pt if target_custom_pt.exists() else None,
            "last_pt": last_pt if last_pt.exists() else None,
            "results": results,
        }