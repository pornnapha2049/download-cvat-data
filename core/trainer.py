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

try:
    import psutil
except Exception:          # psutil ships with ultralytics; stay safe anyway
    psutil = None

_MEMORY_ERROR_HINTS = (
    "insufficient memory", "outofmemoryerror", "out of memory", "cannot allocate memory",
    "paging file is too small", "winerror 1455", "unable to allocate",
    "dataloader worker", "killed by signal", "memoryerror", "bad allocation",
)


def is_memory_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return isinstance(exc, MemoryError) or any(h in text for h in _MEMORY_ERROR_HINTS)


def available_ram_gb() -> float:
    try:
        return psutil.virtual_memory().available / 1024 ** 3 if psutil else 8.0
    except Exception:
        return 8.0


def safe_workers(requested: typing.Optional[int] = None) -> int:
    """
    DataLoader worker processes. Each worker on Windows is a full Python process (torch + a copy of
    the dataset index + decoded images), so the Ultralytics default of 8 easily runs a PC out of RAM
    with large images. Pick by free RAM; never more than the CPU count.
    """
    if requested is not None:
        return max(0, int(requested))
    free = available_ram_gb()
    if free < 6:
        w = 0
    elif free < 12:
        w = 2
    elif free < 24:
        w = 4
    else:
        w = 6
    if sys.platform == "win32":
        w = min(w, 4)
    return min(w, max(1, (os.cpu_count() or 2) - 1))


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
        "workers": safe_workers(),
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


def resolve_model_path(model_name_or_path: str, models_dir: typing.Optional[Path] = None) -> str:
    """
    Existing file path -> used as is (custom .pt).
    Preset name (e.g. 'yolo26x.pt') -> <models_dir>/<name>. Ultralytics downloads the official
    weights straight into that path the first time, and re-uses the file afterwards.
    """
    p = Path(model_name_or_path)
    if p.exists() or models_dir is None or p.parent != Path("."):
        return str(p)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    return str(models_dir / p.name)


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
        models_dir: typing.Optional[Path] = None,
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
        model_resolved = resolve_model_path(model_name_or_path, models_dir)
        if log_callback:
            if Path(model_resolved).exists():
                log_callback(f"📁 Base weights: {model_resolved}")
            else:
                log_callback(f"⬇️ Downloading '{Path(model_resolved).name}' into {Path(model_resolved).parent} (first use only)...")

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

        # Prepare training arguments
        train_args = {
            "data": str(data_yaml_path),
            "project": str(project_dir),
            "name": versioned_run_name,
            "exist_ok": True,
            "verbose": True,
        }
        train_args.update(hyperparams)
        train_args["workers"] = safe_workers(hyperparams.get("workers"))
        train_args.setdefault("cache", False)        # never keep every decoded image in RAM

        if log_callback:
            log_callback(f"⚙️ Hyperparameters: Epochs={train_args.get('epochs')}, Batch={train_args.get('batch')}, "
                         f"ImgSz={train_args.get('imgsz')}, Patience={train_args.get('patience')}, "
                         f"LR0={train_args.get('lr0')}, Device={train_args.get('device')}, "
                         f"Workers={train_args['workers']} (free RAM {available_ram_gb():.1f} GB)")

        # Out of memory -> retry automatically with lighter settings instead of failing:
        #   1st retry: workers=0 (no extra worker processes)   2nd/3rd: halve the batch
        t_start = time.time()
        results = None
        attempt = 0
        try:
            while True:
                model = YOLO(model_resolved)
                model.add_callback("on_train_epoch_end", on_train_epoch_end)
                try:
                    results = model.train(**train_args)
                    break
                except Exception as e:
                    if self.stop_requested or not is_memory_error(e) or attempt >= 3:
                        raise
                    attempt += 1
                    old_w, old_b = train_args.get("workers"), train_args.get("batch", 16)
                    if train_args.get("workers", 0) > 0:
                        train_args["workers"] = 0
                    elif isinstance(old_b, (int, float)) and old_b > 1:
                        train_args["batch"] = max(1, int(old_b) // 2)
                    else:
                        raise
                    if log_callback:
                        log_callback(f"⚠️ Out of memory ({str(e).strip().splitlines()[-1][:160]})")
                        log_callback(f"🔁 Retrying from epoch 1 with lighter settings: workers {old_w}→{train_args['workers']}, "
                                     f"batch {old_b}→{train_args['batch']} (retry {attempt}/3)")
                    del model
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
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