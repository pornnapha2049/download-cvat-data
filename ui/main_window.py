"""
Main Window for the CVAT-to-YOLO GUI application.
Featuring:
- Dedicated First Login Screen (view_login) with centered card and wiggling chubby cat mascot
- Full Studio Pipeline (view_studio) with cohesive top stepper bar (Steps 1 to 6)
- Connected workflow (not isolated standalone): users can switch tabs freely or follow Next/Prev flow
- Single unified 'data/' directory architecture
- Blank CVAT server URL by default
- In-place weights renaming (runs/.../weights/[custom_name]_v[num].pt) with NO extra outside weights folder
- Model Testing & Inference page (Single image, entire folder, video, with save option and visual preview)
"""

import os
import sys
import json
import re
import shutil
import subprocess
from pathlib import Path
import numpy as np
import cv2
import torch

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QStackedWidget, QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox,
    QDoubleSpinBox, QSlider, QCheckBox, QRadioButton, QButtonGroup,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox,
    QProgressBar, QPlainTextEdit, QFrame, QGroupBox, QScrollArea, QSplitter,
    QApplication, QGraphicsDropShadowEffect, QCompleter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QSettings
from PyQt5.QtGui import QImage, QPixmap, QColor
from datetime import datetime

from core.cvat_client import CvatClientWrapper, sanitize_filename
from core.merger import CocoDatasetMerger
from core.converter import CocoToYoloConverter
from core.splitter import DatasetSplitter
from core.layout import DatasetLayout
from core.trainer import YoloTrainerEngine, auto_tune_hyperparameters, get_next_run_version, resolve_model_path
from core.tester import YoloTestEngine, IMAGE_EXTS, VIDEO_EXTS
from core.evaluator import (
    dataset_stats, read_training_results, validate_on_split, evaluate_external,
    has_separate_test_split, resolve_split_images, write_report_files,
)
from ui.dancing_svg import DancingSvgMascot, AnimatedWavePill, get_cat_pixmap
from ui.theme import DARK_STYLESHEET, PALETTE
from ui.svg_icons import get_svg_icon, get_svg_pixmap, icon_name_for
from ui.report_dialog import ReportDialog, StatCard, pct, fmt_secs
from ui.notifications import NotificationCenter

from ui import secure_store   # keyring, or Windows DPAPI when keyring is not installed


def bgr_to_qpixmap(bgr_array: np.ndarray) -> QPixmap:
    """Converts OpenCV BGR numpy array to a QPixmap."""
    if bgr_array is None or bgr_array.size == 0:
        return QPixmap()
    h, w, ch = bgr_array.shape
    bytes_per_line = ch * w
    rgb = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


# =====================================================================
# Background Worker Threads
# =====================================================================

class CvatAuthWorker(QThread):
    finished_signal = pyqtSignal(dict, list, list)
    error_signal = pyqtSignal(str)

    def __init__(self, client: CvatClientWrapper, url: str, username: str, password: str):
        super().__init__()
        self.client = client
        self.url = url
        self.username = username
        self.password = password

    def run(self):
        try:
            info = self.client.connect(self.url, self.username, self.password)
            projects = self.client.get_projects()
            tasks = self.client.get_tasks()
            self.finished_signal.emit(info, projects, tasks)
        except Exception as e:
            self.error_signal.emit(str(e))


def find_leftover_task_dirs(dataset_dir: Path) -> list[Path]:
    """Per-task CVAT export folders (task_<id>_<name> with annotations/*.json) left in the dataset folder."""
    out = []
    try:
        for d in Path(dataset_dir).iterdir():
            if d.is_dir() and re.match(r"task_\d+_", d.name) and (d / "annotations").is_dir():
                out.append(d)
    except OSError:
        pass
    return sorted(out)


class CvatDownloadWorker(QThread):
    """
    Downloads the selected CVAT tasks (COCO 1.0), merges them into ONE dataset in `dataset_dir`
    (images/ + annotations/instances_default.json) and deletes the per-task folders.

    append=True keeps the dataset already in `dataset_dir` and adds the new tasks to it.
    The existing dataset is only replaced after download + merge both succeed.
    """
    progress_signal = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    DERIVED = ("images", "annotations", "labels")   # rebuilt from the merged dataset

    def __init__(self, client: CvatClientWrapper, merger: CocoDatasetMerger, tasks: list[dict],
                 dataset_dir: Path, append: bool = False):
        super().__init__()
        self.client = client
        self.merger = merger
        self.tasks = tasks
        self.dataset_dir = Path(dataset_dir)
        self.append = append

    def run(self):
        dl_root = self.dataset_dir / "_downloads"
        tmp_out = self.dataset_dir / "_merge_tmp"
        try:
            for d in (dl_root, tmp_out):
                shutil.rmtree(d, ignore_errors=True)
            dl_root.mkdir(parents=True, exist_ok=True)

            # 1) download every task (0-80%)
            total_tasks = len(self.tasks)
            jsons = []
            for idx, task in enumerate(self.tasks):
                task_id, task_name = task["id"], task["name"]

                def cb(cur, tot, desc, idx=idx, task_id=task_id, task_name=task_name):
                    pct = int(((idx + (cur / max(tot, 1))) / total_tasks) * 80)
                    self.progress_signal.emit(pct, 100, f"Downloading [{task_id}] {task_name}: {desc}")

                out_dir = self.client.download_and_extract_task(
                    task_id=task_id, task_name=task_name, output_dir=dl_root, progress_callback=cb)
                js = Path(out_dir) / "annotations" / "instances_default.json"
                if not js.exists():
                    found = list(Path(out_dir).rglob("instances_default.json"))
                    if not found:
                        raise FileNotFoundError(f"Task {task_id} ({task_name}) has no COCO annotations file.")
                    js = found[0]
                jsons.append(js)

            # 2) merge (80-97%)
            existing_json = self.dataset_dir / "annotations" / "instances_default.json"
            if self.append and existing_json.exists():
                jsons.insert(0, existing_json)

            def mcb(cur, tot, desc):
                self.progress_signal.emit(80 + int(17 * cur / max(tot, 1)), 100, desc)

            stats = self.merger.merge_datasets(jsons, tmp_out, progress_callback=mcb)

            # 3) swap in the merged dataset, then remove per-task folders (97-100%)
            self.progress_signal.emit(97, 100, "Replacing dataset with the merged result...")
            for name in self.DERIVED:
                shutil.rmtree(self.dataset_dir / name, ignore_errors=True)
            old_yaml = self.dataset_dir / "data.yaml"
            if old_yaml.exists():
                old_yaml.unlink()
            shutil.move(str(tmp_out / "images"), str(self.dataset_dir / "images"))
            shutil.move(str(tmp_out / "annotations"), str(self.dataset_dir / "annotations"))
            for d in (tmp_out, dl_root, self.dataset_dir / "raw_tasks"):   # raw_tasks = old per-task layout
                shutil.rmtree(d, ignore_errors=True)
            for d in find_leftover_task_dirs(self.dataset_dir):          # task_<id>_* from older versions
                shutil.rmtree(d, ignore_errors=True)

            stats.update({
                "tasks": [{"id": t["id"], "name": t["name"]} for t in self.tasks],
                "appended": self.append and len(jsons) > len(self.tasks),
                "dataset_dir": str(self.dataset_dir),
                "images_dir": str(self.dataset_dir / "images"),
                "annotations_file": str(self.dataset_dir / "annotations" / "instances_default.json"),
            })
            self.progress_signal.emit(100, 100, "Download & merge complete!")
            self.finished_signal.emit(stats)
        except Exception as e:
            # existing dataset is untouched on failure; just drop the temporary folders
            for d in (tmp_out, dl_root):
                shutil.rmtree(d, ignore_errors=True)
            self.error_signal.emit(str(e))


class ConvertWorker(QThread):
    """COCO -> YOLO into <data>/labels/. Old labels are removed first; if the images were
    already split, each label is moved to labels/<split>/ next to its image."""
    progress_signal = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, converter: CocoToYoloConverter, dataset_dir: Path, mode: str):
        super().__init__()
        self.converter = converter
        self.layout = DatasetLayout(dataset_dir)
        self.mode = mode

    def run(self):
        try:
            lay = self.layout
            if not lay.coco_json.exists():
                raise FileNotFoundError(f"COCO annotations not found:\n{lay.coco_json}\nDownload data first (Step 1).")
            removed = lay.clear_labels()

            def cb(cur, tot, desc):
                self.progress_signal.emit(cur, tot, desc)

            stats = self.converter.convert(lay.coco_json, lay.labels, mode=self.mode, progress_callback=cb)
            stats["old_labels_removed"] = removed
            stats["moved_into_splits"] = lay.place_labels_next_to_images()
            stats["already_split"] = lay.is_split()
            self.finished_signal.emit(stats)
        except Exception as e:
            self.error_signal.emit(str(e))


class SplitWorker(QThread):
    """Moves images/labels inside <data> into images|labels/{train,val,test} and writes data.yaml."""
    progress_signal = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, splitter: DatasetSplitter, dataset_dir: Path, ratios: tuple[float, float, float],
                 class_names: list[str] | None = None):
        super().__init__()
        self.splitter = splitter
        self.dataset_dir = Path(dataset_dir)
        self.ratios = ratios
        self.class_names = class_names

    def run(self):
        try:
            def cb(cur, tot, desc):
                self.progress_signal.emit(cur, tot, desc)

            r_train, r_val, r_test = self.ratios
            stats = self.splitter.split_in_place(
                self.dataset_dir, train_ratio=r_train, val_ratio=r_val, test_ratio=r_test,
                class_names=self.class_names, progress_callback=cb)
            self.finished_signal.emit(stats)
        except Exception as e:
            self.error_signal.emit(str(e))


class TrainWorker(QThread):
    log_signal = pyqtSignal(str)
    epoch_signal = pyqtSignal(int, int, dict)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, engine: YoloTrainerEngine, task: str, model_name: str, data_yaml: Path, run_name: str, base_dir: Path,
                 hyperparams: dict, models_dir: Path | None = None):
        super().__init__()
        self.models_dir = models_dir
        self.engine = engine
        self.task = task
        self.model_name = model_name
        self.data_yaml = data_yaml
        self.run_name = run_name
        self.base_dir = base_dir
        self.hyperparams = hyperparams

    def run(self):
        try:
            def log_cb(msg):
                self.log_signal.emit(msg)

            def epoch_cb(ep, tot, metrics):
                self.epoch_signal.emit(ep, tot, metrics)

            res = self.engine.train_model(
                task=self.task,
                model_name_or_path=self.model_name,
                data_yaml_path=self.data_yaml,
                custom_run_name=self.run_name,
                base_runs_dir=self.base_dir,
                hyperparams=self.hyperparams,
                log_callback=log_cb,
                epoch_callback=epoch_cb,
                models_dir=self.models_dir,
            )
            res["report_data"] = self._collect_report(res, log_cb)
            self.finished_signal.emit(res)
        except Exception as e:
            self.error_signal.emit(str(e))


    def _collect_report(self, res: dict, log) -> dict:
        """Dataset counts + results.csv + accuracy on the test split. Never fails the training."""
        data = {"warnings": []}
        try:
            data["dataset"] = dataset_stats(self.data_yaml)
        except Exception as e:
            data["warnings"].append(f"Could not read dataset stats: {e}")
        try:
            data["training"] = read_training_results(res["run_dir"])
        except Exception as e:
            data["warnings"].append(f"Could not read results.csv: {e}")
        best = res.get("best_pt") or res.get("last_pt")
        if best:
            try:
                if has_separate_test_split(self.data_yaml):
                    log("🧪 Evaluating best weights on the TEST split...")
                    data["test_metrics"] = validate_on_split(best, self.data_yaml, Path(res["run_dir"]), "test",
                                                            imgsz=self.hyperparams.get("imgsz"), log=log)
                else:
                    data["warnings"].append("No separate test split in data.yaml — accuracy shown is on the validation split.")
                log("📏 Re-checking best weights on the VAL split (per-class + confusion matrix)...")
                data["val_metrics"] = validate_on_split(best, self.data_yaml, Path(res["run_dir"]), "val",
                                                       imgsz=self.hyperparams.get("imgsz"), log=log)
            except Exception as e:
                data["warnings"].append(f"Evaluation failed: {e}")
        return data


class TestInferenceWorker(QThread):
    progress_signal = pyqtSignal(int, int, str)
    frame_signal = pyqtSignal(object, int, int, dict)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, engine: YoloTestEngine, mode: str, model_path: Path, src_path: Path, conf: float, iou: float,
                 save: bool, save_dir: Path, evaluate: bool = True, eval_dir: Path | None = None):
        super().__init__()
        self.evaluate = evaluate
        self.eval_dir = eval_dir or save_dir
        self.engine = engine
        self.mode = mode
        self.model_path = model_path
        self.src_path = src_path
        self.conf = conf
        self.iou = iou
        self.save = save
        self.save_dir = save_dir

    def run(self):
        try:
            if self.mode == "image":
                res = self.engine.predict_image(
                    model_path=self.model_path,
                    image_path=self.src_path,
                    conf=self.conf,
                    iou=self.iou,
                    save=self.save,
                    save_dir=self.save_dir,
                )
                self.finished_signal.emit({"mode": "image", "data": res, "eval": self._evaluate(self.src_path)})
            elif self.mode == "folder":
                def cb(cur, tot, desc):
                    self.progress_signal.emit(cur, tot, desc)
                def fcb(frame, cur, tot, summary):
                    self.frame_signal.emit(frame, cur, tot, summary)
                res = self.engine.predict_folder(
                    model_path=self.model_path,
                    folder_path=self.src_path,
                    conf=self.conf,
                    iou=self.iou,
                    save=self.save,
                    save_dir=self.save_dir,
                    progress_callback=cb,
                    frame_callback=fcb,
                )
                self.finished_signal.emit({"mode": "folder", "data": res, "eval": self._evaluate(self.src_path)})
            elif self.mode == "split":
                # src_path = data.yaml ; preview predictions on the test (or val) images, then measure accuracy
                split = "test" if has_separate_test_split(self.src_path) else "val"
                imgs = resolve_split_images(self.src_path, split)
                if not imgs:
                    raise FileNotFoundError(f"No images found in the '{split}' split of {self.src_path}")
                def cb(cur, tot, desc):
                    self.progress_signal.emit(cur, tot, f"[{split}] {desc}")
                def fcb(frame, cur, tot, summary):
                    self.frame_signal.emit(frame, cur, tot, summary)
                res = self.engine.predict_folder(
                    model_path=self.model_path,
                    folder_path=imgs[0].parent,
                    conf=self.conf,
                    iou=self.iou,
                    save=self.save,
                    save_dir=self.save_dir,
                    progress_callback=cb,
                    frame_callback=fcb,
                )
                ev = None
                if not self.engine.stop_requested:
                    self.progress_signal.emit(0, 0, f"Measuring accuracy on the {split} split...")
                    try:
                        ev = validate_on_split(self.model_path, self.src_path, self.eval_dir, split, cm_conf=self.conf)
                    except Exception as e:
                        ev = {"error": str(e)}
                self.finished_signal.emit({"mode": "split", "data": res, "eval": ev})
            elif self.mode == "video":
                def pcb(cur, tot, desc):
                    self.progress_signal.emit(cur, tot, desc)
                def fcb(frame, cur, tot, summary):
                    self.frame_signal.emit(frame, cur, tot, summary)
                res = self.engine.predict_video(
                    model_path=self.model_path,
                    video_path=self.src_path,
                    conf=self.conf,
                    iou=self.iou,
                    save=self.save,
                    save_dir=self.save_dir,
                    progress_callback=pcb,
                    frame_callback=fcb,
                )
                self.finished_signal.emit({"mode": "video", "data": res})
        except Exception as e:
            self.error_signal.emit(str(e))


    def _evaluate(self, source: Path):
        """Accuracy when the image/folder has labels; None for unlabeled data."""
        if not self.evaluate or self.engine.stop_requested:
            return None
        self.progress_signal.emit(0, 0, "Looking for labels to measure accuracy...")
        try:
            return evaluate_external(self.model_path, source, self.eval_dir, cm_conf=self.conf)
        except Exception as e:
            return {"error": str(e)}


# =====================================================================
# Main GUI Window (Login View + Studio Connected Workflow)
# =====================================================================

class MainWindow(QMainWindow):
    STEP_TITLES = [
        ("1. Download & Merge", "CVAT Tasks → One Dataset"),
        ("2. COCO to YOLO", "Detect or Segment"),
        ("3. Dataset Splitter", "Train/Val/Test All Imgs"),
        ("4. YOLO Auto-Trainer", "Auto-Tune & Patience"),
        ("5. Test Model", "Image / Folder / Video"),
    ]

    STEP_DOWNLOAD, STEP_CONVERT, STEP_SPLIT, STEP_TRAIN, STEP_TEST = range(5)

    STEP_GUIDES = [
        "Select tasks from CVAT — they are downloaded, merged into one dataset in data/ "
        "(duplicate file names resolved, class ids unified) and the per-task folders are removed.",
        "Convert COCO annotations into YOLO .txt format for Detection (BBox) or Segmentation (Polygon).",
        "Partition dataset with 100% images guarantee (including unannotated background images).",
        "Intelligently auto-tune hyperparameters, train YOLO model, and inspect outputs in runs/.",
        "Evaluate trained YOLO weights on single images, folders, or video files with visual preview."
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CVAT to YOLO Studio")
        self.setMinimumSize(980, 640)
        self._fit_to_screen()

        # Single Unified Data Folder Architecture
        self.app_dir = Path("C:/Articulus/cvat").resolve()
        self.data_dir = self.app_dir / "data"
        self.models_dir = self.app_dir / "models"   # base weights (yolo11n.pt, yolo26x.pt …) are kept here

        for d in [self.data_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # State Engines
        self.cvat_client = CvatClientWrapper()
        self.merger_engine = CocoDatasetMerger()
        self.converter_engine = CocoToYoloConverter()
        self.splitter_engine = DatasetSplitter()
        self.trainer_engine = YoloTrainerEngine()
        self.tester_engine = YoloTestEngine()

        self.all_tasks_data: list[dict] = []
        self.active_worker = None

        # Test state
        self.folder_test_results: list[dict] = []
        self.current_folder_test_idx: int = 0
        self.single_test_result: dict | None = None
        self.last_video_result: dict | None = None
        self.last_video_raw_res = None
        self.last_trained_weight: Path | None = None
        self.last_class_names: list[str] | None = None
        self.last_train_report: dict | None = None
        self.last_eval_report: dict | None = None
        self.settings = QSettings("Articulus", "CVAT to YOLO Studio")
        self.notify = NotificationCenter(self, self.settings)

        self._init_ui()
        self.setStyleSheet(DARK_STYLESHEET)
        self._apply_inverse_icons()
        self._load_saved_login()
        saved_ds = self.settings.value("paths/dataset_dir", "", type=str)
        if saved_ds and Path(saved_ds).is_dir():
            self.txt_dl_folder.setText(saved_ds)
        self._sync_dataset_paths(self._ds().root, save=False)

    # -----------------------------------------------------------------
    # Desktop sizing & styling helpers
    # -----------------------------------------------------------------
    def _fit_to_screen(self):
        """Open at a comfortable desktop size: ~88% of the available screen,
        capped so it doesn't look stretched on big monitors, and centred."""
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1280, 820)
            return
        avail = screen.availableGeometry()
        w = max(self.minimumWidth(), min(int(avail.width() * 0.88), 1480))
        h = max(self.minimumHeight(), min(int(avail.height() * 0.90), 960))
        w, h = min(w, avail.width()), min(h, avail.height())
        self.resize(w, h)
        self.move(avail.x() + (avail.width() - w) // 2,
                  avail.y() + (avail.height() - h) // 2)

    DARK_BUTTON_CLASSES = ("PrimaryButton", "StepNavButton", "LoginButton", "DangerButton")

    def _apply_inverse_icons(self):
        """Dark (navy / red) buttons get the white version of their icon."""
        for btn in self.findChildren(QPushButton):
            if (btn.property("class") or "") not in self.DARK_BUTTON_CLASSES or btn.icon().isNull():
                continue
            found = icon_name_for(btn.icon())
            if found:
                name, size = found
                btn.setIcon(get_svg_icon(name, size, inverse=True))

    @staticmethod
    def _repolish(widget: QWidget):
        """Re-apply stylesheet to a widget and all its children (after a class change)."""
        for w in [widget] + widget.findChildren(QWidget):
            w.style().unpolish(w)
            w.style().polish(w)
            w.update()

    @staticmethod
    def _soft_shadow(widget: QWidget, blur: int = 36, dy: int = 10, alpha: int = 28):
        """Soft floating shadow like the cards in the reference."""
        eff = QGraphicsDropShadowEffect(widget)
        eff.setBlurRadius(blur)
        eff.setOffset(0, dy)
        eff.setColor(QColor(14, 75, 90, alpha))
        widget.setGraphicsEffect(eff)

    @staticmethod
    def _time_greeting() -> str:
        hour = datetime.now().hour
        if hour < 12:
            return "Good Morning"
        if hour < 17:
            return "Good Afternoon"
        return "Good Evening"

    def _make_choice_card(self, variant: str, title: str, subtitle: str, on_click) -> QFrame:
        """Square-ish choice card like "Hang out" / "Laugh": cat badge, title, round → badge."""
        card = QFrame()
        card.setProperty("class", "ChoiceCardPink" if variant == "pink" else "ChoiceCardNavy")
        card.setCursor(Qt.PointingHandCursor)
        card.setMinimumHeight(172)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(18, 16, 14, 14)
        lay.setSpacing(4)

        face = QLabel()
        face.setPixmap(get_cat_pixmap(64, "teal" if variant == "pink" else "pink"))
        lay.addWidget(face, alignment=Qt.AlignLeft)
        lay.addStretch(1)

        bottom = QHBoxLayout()
        bottom.setSpacing(6)
        txt = QVBoxLayout()
        txt.setSpacing(1)
        lbl_t = QLabel(title)
        lbl_t.setProperty("class", "ChoiceTitlePink" if variant == "pink" else "ChoiceTitleNavy")
        lbl_s = QLabel(subtitle)
        lbl_s.setProperty("class", "ChoiceSubPink" if variant == "pink" else "ChoiceSubNavy")
        lbl_s.setWordWrap(True)
        txt.addWidget(lbl_t)
        txt.addWidget(lbl_s)
        bottom.addLayout(txt, 1)

        arrow = QLabel("→")
        arrow.setAlignment(Qt.AlignCenter)
        arrow.setProperty("class", "ArrowBadgePink" if variant == "pink" else "ArrowBadgeNavy")
        bottom.addWidget(arrow, 0, Qt.AlignBottom)
        lay.addLayout(bottom)

        card.mousePressEvent = lambda e: on_click()
        return card

    def _make_todo_row(self, icon: str, tile_bg: str, title: str, subtitle: str) -> QFrame:
        """Row like "Mind Detox / Gratitude Notes": icon tile, two lines of text, play circle."""
        row = QFrame()
        row.setProperty("class", "SoftCard")
        lay = QHBoxLayout(row)
        lay.setContentsMargins(10, 10, 14, 10)
        lay.setSpacing(12)

        tile = QLabel()
        tile.setFixedSize(46, 46)
        tile.setAlignment(Qt.AlignCenter)
        tile.setPixmap(get_svg_pixmap(icon, 28, 28))
        tile.setStyleSheet(f"background-color: {tile_bg}; border-radius: 14px;")
        lay.addWidget(tile)

        txt = QVBoxLayout()
        txt.setSpacing(0)
        t = QLabel(title)
        t.setStyleSheet(f"font-weight: 700; font-size: 13px; color: {PALETTE['ink']};")
        st = QLabel(subtitle)
        st.setStyleSheet(f"font-size: 11.5px; color: {PALETTE['muted']};")
        txt.addWidget(t)
        txt.addWidget(st)
        lay.addLayout(txt, 1)

        play = QLabel("▶")
        play.setAlignment(Qt.AlignCenter)
        play.setFixedSize(30, 30)
        play.setStyleSheet(f"border: 2px solid {PALETTE['ink']}; border-radius: 15px; "
                           f"color: {PALETTE['ink']}; font-size: 10px; padding-left: 2px;")
        lay.addWidget(play)
        return row

    def _wrap_page_scroll(self, widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(widget)
        return scroll

    def _init_ui(self):
        central_widget = QWidget(self)
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Root Stack: 0 -> First Dedicated Login Screen, 1 -> Connected Studio Workflow
        self.root_stack = QStackedWidget()
        self.view_login = self._build_login_view()
        self.view_studio = self._build_studio_view()

        self.root_stack.addWidget(self._wrap_page_scroll(self.view_login))
        self.root_stack.addWidget(self.view_studio)
        root_layout.addWidget(self.root_stack)

        self.root_stack.setCurrentIndex(0)

    # =================================================================
    # VIEW 0: DEDICATED FIRST LOGIN SCREEN (The Original Design)
    # =================================================================
    def _build_login_view(self) -> QWidget:
        view = QWidget()
        outer = QVBoxLayout(view)
        outer.setContentsMargins(36, 24, 36, 18)
        outer.setSpacing(0)
        outer.addStretch(1)

        # Centred content column (keeps a calm, phone-like rhythm on wide monitors)
        content = QWidget()
        content.setMaximumWidth(1080)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(22)

        center_row = QHBoxLayout()
        center_row.addStretch(1)
        center_row.addWidget(content, 100)
        center_row.addStretch(1)

        # ---------- Greeting header + bell ----------
        header = QHBoxLayout()
        header.setSpacing(12)
        head_txt = QVBoxLayout()
        head_txt.setSpacing(6)

        lbl_greeting = QLabel(f"{self._time_greeting()}, Developer")
        lbl_greeting.setObjectName("greetingTitle")
        head_txt.addWidget(lbl_greeting)

        lbl_sub = QLabel("Select what you want more for now")
        lbl_sub.setObjectName("greetingSubtitle")
        head_txt.addWidget(lbl_sub)
        header.addLayout(head_txt, 1)

        bell = QPushButton()
        bell.setObjectName("circleNavButton")
        bell.setIcon(get_svg_icon("bell", 24))
        bell.setIconSize(QSize(24, 24))
        self.notify.attach_button(bell)
        header.addWidget(bell, 0, Qt.AlignTop)
        layout.addLayout(header)

        # ---------- Two columns ----------
        cols = QHBoxLayout()
        cols.setSpacing(24)

        # LEFT: choice cards + quote/mascot card
        left_col = QVBoxLayout()
        left_col.setSpacing(18)

        choices = QHBoxLayout()
        choices.setSpacing(16)
        choices.addWidget(self._make_choice_card(
            "pink", "Work Offline", "Use local COCO / YOLO data", self._on_skip_login), 1)
        choices.addWidget(self._make_choice_card(
            "navy", "Auto-Trainer", "Train YOLO detect & segment",
            lambda: (self._on_skip_login(), self._go_to_step(self.STEP_TRAIN))), 1)
        left_col.addLayout(choices)

        quote_card = QFrame()
        quote_card.setProperty("class", "LoginCard")
        qc_lay = QVBoxLayout(quote_card)
        qc_lay.setContentsMargins(26, 20, 22, 22)
        qc_lay.setSpacing(10)

        q_top = QHBoxLayout()
        q_top.setSpacing(10)
        q_txt = QVBoxLayout()
        q_txt.setSpacing(8)
        tag_row = QHBoxLayout()
        pill = QLabel("Day advice")
        pill.setProperty("class", "TagPink")
        tag_row.addWidget(pill)
        tag_row.addStretch()
        q_txt.addLayout(tag_row)

        lbl_quote = QLabel("\u201cPatience is the companion of wisdom.\u201d")
        lbl_quote.setObjectName("heroQuoteText")
        lbl_quote.setWordWrap(True)
        q_txt.addWidget(lbl_quote)

        lbl_author = QLabel("— Saint Augustine")
        lbl_author.setObjectName("heroQuoteAuthor")
        q_txt.addWidget(lbl_author)
        q_txt.addStretch(1)
        q_top.addLayout(q_txt, 1)

        self.login_mascot = DancingSvgMascot(size=140)
        q_top.addWidget(self.login_mascot, 0, Qt.AlignVCenter)
        qc_lay.addLayout(q_top)

        lbl_things = QLabel("Things you can do")
        lbl_things.setProperty("class", "SectionTitle")
        qc_lay.addWidget(lbl_things)

        for icon, tile_bg, title, sub in (
            ("step_cvat", PALETTE["lilac"], "Fetch CVAT tasks", "Download COCO 1.0 && flatten images"),
            ("step_train", PALETTE["sky"], "Train && test YOLO", "Auto-tune, early stop, preview results"),
        ):
            qc_lay.addWidget(self._make_todo_row(icon, tile_bg, title.replace("&&", "&"), sub.replace("&&", "&")))
        qc_lay.addStretch(1)
        left_col.addWidget(quote_card, 1)

        cols.addLayout(left_col, 11)

        # RIGHT: Sign-in card
        right_card = QFrame()
        right_card.setProperty("class", "LoginCard")
        right_card.setMinimumWidth(400)
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(34, 28, 34, 28)
        right_layout.setSpacing(10)

        form_tag_row = QHBoxLayout()
        for text, cls in (("CVAT", "TagYellow"), ("Server", "TagGreen"), ("Portal", "TagPurple")):
            t = QLabel(text)
            t.setProperty("class", cls)
            form_tag_row.addWidget(t)
        form_tag_row.addStretch()
        right_layout.addLayout(form_tag_row)
        right_layout.addSpacing(4)

        lbl_form_title = QLabel("Sign in to CVAT")
        lbl_form_title.setObjectName("loginTitle")
        right_layout.addWidget(lbl_form_title)

        lbl_form_sub = QLabel("Enter your CVAT server credentials to access tasks, images & annotations.")
        lbl_form_sub.setObjectName("loginSubtitle")
        lbl_form_sub.setWordWrap(True)
        right_layout.addWidget(lbl_form_sub)
        right_layout.addSpacing(4)

        # Alert Banner (Hidden by default)
        self.lbl_login_alert = QLabel()
        self.lbl_login_alert.setProperty("class", "AlertError")
        self.lbl_login_alert.setWordWrap(True)
        self.lbl_login_alert.hide()
        right_layout.addWidget(self.lbl_login_alert)

        # Inputs
        lbl_url = QLabel("Server URL")
        lbl_url.setProperty("class", "FieldLabel")
        right_layout.addWidget(lbl_url)
        self.txt_login_url = QLineEdit("")
        self.txt_login_url.setPlaceholderText("http://<your-cvat-server>:8080")
        self.txt_login_url.setProperty("class", "LoginInput")
        right_layout.addWidget(self.txt_login_url)

        lbl_user = QLabel("Username")
        lbl_user.setProperty("class", "FieldLabel")
        right_layout.addWidget(lbl_user)
        self.txt_login_user = QLineEdit("")
        self.txt_login_user.setPlaceholderText("Enter your CVAT username")
        self.txt_login_user.setProperty("class", "LoginInput")
        right_layout.addWidget(self.txt_login_user)

        lbl_pass = QLabel("Password")
        lbl_pass.setProperty("class", "FieldLabel")
        right_layout.addWidget(lbl_pass)

        pass_row = QHBoxLayout()
        pass_row.setSpacing(8)
        pass_row.setContentsMargins(0, 0, 0, 0)
        self.txt_login_pass = QLineEdit("")
        self.txt_login_pass.setEchoMode(QLineEdit.Password)
        self.txt_login_pass.setPlaceholderText("Enter your CVAT password")
        self.txt_login_pass.setProperty("class", "LoginInput")
        self.txt_login_pass.returnPressed.connect(self._on_do_login)

        self.btn_toggle_pass = QPushButton()
        self.btn_toggle_pass.setObjectName("btnTogglePassword")
        self.btn_toggle_pass.setIcon(get_svg_icon("eye_off", 20))
        self.btn_toggle_pass.setToolTip("Show / Hide Password (แสดงหรือซ่อนรหัสผ่าน)")
        self.btn_toggle_pass.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_pass.setFixedSize(42, 42)
        self.btn_toggle_pass.clicked.connect(self._toggle_password_visibility)

        pass_row.addWidget(self.txt_login_pass, 1)
        pass_row.addWidget(self.btn_toggle_pass)
        right_layout.addLayout(pass_row)

        # Remember me (URL + username in QSettings, password in the OS credential store)
        remember_row = QHBoxLayout()
        remember_row.setSpacing(14)
        self.chk_remember = QCheckBox("Remember URL && username")
        self.chk_remember.setChecked(True)
        self.chk_remember_pwd = QCheckBox("Remember password")
        self.chk_remember_pwd.setToolTip(secure_store.describe())
        # disabled checkboxes look almost the same in this theme -> make the state obvious
        self.chk_remember_pwd.setStyleSheet(f"QCheckBox:disabled {{ color: {PALETTE['faint']}; }}")
        self.chk_remember_pwd.setEnabled(secure_store.available())
        self.chk_remember.toggled.connect(self._on_remember_toggled)
        remember_row.addWidget(self.chk_remember)
        remember_row.addWidget(self.chk_remember_pwd)
        remember_row.addStretch()
        self.btn_forget_login = QPushButton("Forget saved login")
        self.btn_forget_login.setProperty("class", "SkipLoginButton")
        self.btn_forget_login.setCursor(Qt.PointingHandCursor)
        self.btn_forget_login.clicked.connect(self._forget_saved_login)
        remember_row.addWidget(self.btn_forget_login)
        right_layout.addLayout(remember_row)
        if not secure_store.available():
            hint = QLabel("Remember password needs the 'keyring' package on this system: pip install keyring")
            hint.setWordWrap(True)
            hint.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 11px;")
            right_layout.addWidget(hint)

        self.lbl_login_status = QLabel("")
        self.lbl_login_status.setAlignment(Qt.AlignCenter)
        self.lbl_login_status.setStyleSheet(f"color: {PALETTE['teal']}; font-weight: 600;")
        right_layout.addWidget(self.lbl_login_status)
        right_layout.addStretch(1)

        self.btn_do_login = QPushButton("Sign In && Launch Studio")
        self.btn_do_login.setProperty("class", "LoginButton")
        self.btn_do_login.setCursor(Qt.PointingHandCursor)
        self.btn_do_login.clicked.connect(self._on_do_login)
        right_layout.addWidget(self.btn_do_login)

        cols.addWidget(right_card, 9)
        layout.addLayout(cols)

        for w in (quote_card, right_card):
            self._soft_shadow(w)

        outer.addLayout(center_row)
        outer.addStretch(1)

        lbl_foot = QLabel("CVAT to YOLO Studio • Navy Enterprise Edition v3.0")
        lbl_foot.setStyleSheet(f"color: {PALETTE['faint']}; font-size: 11px;")
        lbl_foot.setAlignment(Qt.AlignCenter)
        outer.addWidget(lbl_foot)

        return view

    # =================================================================
    # VIEW 1: CONNECTED STUDIO WORKFLOW (6 STEPS)
    # =================================================================
    def _build_studio_view(self) -> QWidget:
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(22, 14, 22, 14)
        layout.setSpacing(12)

        # 1. Studio Header with Cute Chubby Cat Mascot & User Info
        layout.addWidget(self._build_studio_header())

        # 2. Interactive Stepper Bar (Step 1 -> 5, Cohesive Pipeline)
        layout.addWidget(self._build_stepper_bar())

        # 3. Dynamic Guide Box
        layout.addWidget(self._build_guide_box())

        # 4. Connected Multi-Page Container (QStackedWidget)
        self.pages = QStackedWidget()
        self.pages.addWidget(self._wrap_page_scroll(self._build_page_cvat()))       # Step 1: Download & auto-merge
        self.pages.addWidget(self._wrap_page_scroll(self._build_page_convert()))    # Step 2: Convert COCO to YOLO
        self.pages.addWidget(self._wrap_page_scroll(self._build_page_split()))      # Step 3: Dataset Splitter
        self.pages.addWidget(self._wrap_page_scroll(self._build_page_train()))      # Step 4: YOLO Auto-Trainer
        self.pages.addWidget(self._wrap_page_scroll(self._build_page_test()))       # Step 5: Test Model
        layout.addWidget(self.pages, stretch=1)

        # 5. Bottom Stepper Footer Navigation (< Previous | Step Pills | Next >)
        layout.addWidget(self._build_stepper_footer())

        return view

    # -----------------------------------------------------------------
    # Studio Header
    # -----------------------------------------------------------------
    def _build_studio_header(self) -> QFrame:
        """Top app bar sitting on the background (like the reference phones): cat, greeting, actions."""
        card = QFrame()
        card.setObjectName("topAppBar")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setSpacing(12)

        self.studio_mascot = DancingSvgMascot(size=76)
        layout.addWidget(self.studio_mascot)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)

        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        title_lbl = QLabel("CVAT to YOLO Studio")
        title_lbl.setObjectName("appTitleLabel")
        title_row.addWidget(title_lbl)
        self.wave_indicator = AnimatedWavePill(44, 20)
        title_row.addWidget(self.wave_indicator)
        title_row.addStretch()
        text_layout.addLayout(title_row)

        self.lbl_user_badge = QLabel("Local Mode • Ready to process datasets")
        self.lbl_user_badge.setObjectName("userBadge")
        text_layout.addWidget(self.lbl_user_badge)
        layout.addLayout(text_layout, stretch=1)

        btn_data = QPushButton("data/ Folder")
        btn_data.setProperty("class", "SecondaryNavButton")
        btn_data.setIcon(get_svg_icon("folder", 16))
        btn_data.setIconSize(QSize(16, 16))
        btn_data.setCursor(Qt.PointingHandCursor)
        btn_data.clicked.connect(lambda: self._open_dir(self._ds().root))
        layout.addWidget(btn_data)

        btn_logout = QPushButton("Logout")
        btn_logout.setProperty("class", "LogoutButton")
        btn_logout.setIcon(get_svg_icon("logout", 14))
        btn_logout.setIconSize(QSize(14, 14))
        btn_logout.setCursor(Qt.PointingHandCursor)
        btn_logout.clicked.connect(self._on_logout)
        layout.addWidget(btn_logout)

        bell = QPushButton()
        bell.setObjectName("circleNavButton")
        bell.setIcon(get_svg_icon("bell", 22))
        bell.setIconSize(QSize(22, 22))
        self.notify.attach_button(bell)
        layout.addWidget(bell)

        return card

    # -----------------------------------------------------------------
    # Stepper — segmented pills ("Day | Week | Month" style)
    # -----------------------------------------------------------------
    def _build_stepper_bar(self) -> QFrame:
        bar = QFrame()
        bar.setProperty("class", "StepperContainer")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.step_buttons: list[QPushButton] = []
        for idx, (title, sub) in enumerate(self.STEP_TITLES):
            btn = QPushButton()
            btn.setProperty("class", "StepButtonActive" if idx == 0 else "StepButton")
            btn.setText(f"{title}\n{sub}".replace("&", "&&"))
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, step_idx=idx: self._go_to_step(step_idx))
            layout.addWidget(btn, stretch=1)
            self.step_buttons.append(btn)

        return bar

    # -----------------------------------------------------------------
    # Guide Box — pink card like "Yes or No Quiz"
    # -----------------------------------------------------------------
    def _build_guide_box(self) -> QFrame:
        box = QFrame()
        box.setProperty("class", "GuideBox")
        layout = QHBoxLayout(box)
        layout.setContentsMargins(16, 10, 18, 10)
        layout.setSpacing(12)

        icon_lbl = QLabel()
        icon_lbl.setPixmap(get_cat_pixmap(34, "teal"))
        layout.addWidget(icon_lbl)

        tag = QLabel("Tip")
        tag.setProperty("class", "TagNavy")
        layout.addWidget(tag)

        self.lbl_guide = QLabel(self.STEP_GUIDES[0])
        self.lbl_guide.setObjectName("guideLabel")
        self.lbl_guide.setWordWrap(True)
        layout.addWidget(self.lbl_guide, stretch=1)

        return box

    # -----------------------------------------------------------------
    # Bottom Footer — date-strip style step circles
    # -----------------------------------------------------------------
    def _build_stepper_footer(self) -> QFrame:
        footer = QFrame()
        footer.setObjectName("bottomNavFooter")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        self.btn_footer_prev = QPushButton("Previous")
        self.btn_footer_prev.setProperty("class", "SecondaryNavButton")
        self.btn_footer_prev.setIcon(get_svg_icon("arrow_left", 16))
        self.btn_footer_prev.setIconSize(QSize(16, 16))
        self.btn_footer_prev.setCursor(Qt.PointingHandCursor)
        self.btn_footer_prev.clicked.connect(self._on_prev_step)
        layout.addWidget(self.btn_footer_prev)

        layout.addStretch()

        self.step_pills: list[QPushButton] = []
        for idx in range(len(self.STEP_TITLES)):
            pill = QPushButton(str(idx + 1))
            pill.setProperty("class", "StepPillActive" if idx == 0 else "StepPill")
            pill.setCursor(Qt.PointingHandCursor)
            pill.setToolTip(self.STEP_TITLES[idx][0])
            pill.clicked.connect(lambda checked, step_idx=idx: self._go_to_step(step_idx))
            layout.addWidget(pill)
            self.step_pills.append(pill)

        layout.addSpacing(8)
        self.lbl_step_indicator = QLabel(f"Step 1 of {len(self.STEP_TITLES)}")
        self.lbl_step_indicator.setObjectName("stepIndicator")
        layout.addWidget(self.lbl_step_indicator)

        layout.addStretch()

        self.btn_footer_next = QPushButton("Next Step")
        self.btn_footer_next.setProperty("class", "StepNavButton")
        self.btn_footer_next.setLayoutDirection(Qt.RightToLeft)
        self.btn_footer_next.setIcon(get_svg_icon("arrow_right", 16))
        self.btn_footer_next.setIconSize(QSize(16, 16))
        self.btn_footer_next.setCursor(Qt.PointingHandCursor)
        self.btn_footer_next.clicked.connect(self._on_next_step)
        layout.addWidget(self.btn_footer_next)

        return footer

    # -----------------------------------------------------------------
    # STEP 1: CVAT Downloader
    # -----------------------------------------------------------------
    def _build_page_cvat(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(12)

        # Filter Card
        filter_card = QFrame()
        filter_card.setProperty("class", "CardFrame")
        filter_layout = QHBoxLayout(filter_card)
        filter_layout.setContentsMargins(14, 10, 14, 10)
        filter_layout.setSpacing(10)

        filter_layout.addWidget(QLabel("Project:"))
        self.cmb_proj_filter = QComboBox()
        self.cmb_proj_filter.addItem("All Projects", None)
        self.cmb_proj_filter.currentIndexChanged.connect(self._apply_task_filters)
        filter_layout.addWidget(self.cmb_proj_filter)

        filter_layout.addWidget(QLabel("Stage:"))
        self.cmb_stage_filter = QComboBox()
        self.cmb_stage_filter.addItems(["All", "acceptance", "annotation", "validation"])
        self.cmb_stage_filter.currentIndexChanged.connect(self._apply_task_filters)
        filter_layout.addWidget(self.cmb_stage_filter)

        filter_layout.addWidget(QLabel("State:"))
        self.cmb_state_filter = QComboBox()
        self.cmb_state_filter.addItems(["All", "completed", "in progress", "new", "rejected"])
        self.cmb_state_filter.currentIndexChanged.connect(self._apply_task_filters)
        filter_layout.addWidget(self.cmb_state_filter)

        filter_layout.addWidget(QLabel("Search:"))
        self.txt_task_search = QLineEdit()
        self.txt_task_search.setPlaceholderText("Filter task name...")
        self.txt_task_search.textChanged.connect(self._apply_task_filters)
        filter_layout.addWidget(self.txt_task_search)

        btn_select_all = QPushButton("Select All")
        btn_select_all.setIcon(get_svg_icon("check", 14))
        btn_select_all.setIconSize(QSize(14, 14))
        btn_select_all.clicked.connect(lambda: self._set_all_tasks_checked(True))
        filter_layout.addWidget(btn_select_all)

        btn_clear_sel = QPushButton("Deselect All")
        btn_clear_sel.setIcon(get_svg_icon("filter", 14))
        btn_clear_sel.setIconSize(QSize(14, 14))
        btn_clear_sel.clicked.connect(lambda: self._set_all_tasks_checked(False))
        filter_layout.addWidget(btn_clear_sel)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.setIcon(get_svg_icon("refresh", 14))
        btn_refresh.setIconSize(QSize(14, 14))
        btn_refresh.clicked.connect(self._fetch_cvat_tasks)
        filter_layout.addWidget(btn_refresh)

        layout.addWidget(filter_card)

        # Tasks Table
        self.tbl_tasks = QTableWidget()
        self.tbl_tasks.setColumnCount(8)
        self.tbl_tasks.setHorizontalHeaderLabels([
            "Select", "Task ID", "Task Name", "Project", "Stage", "State", "Status", "Total Frames"
        ])
        self.tbl_tasks.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.tbl_tasks.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.tbl_tasks.verticalHeader().setDefaultSectionSize(36)
        self.tbl_tasks.verticalHeader().setVisible(False)
        self.tbl_tasks.setMinimumHeight(280)
        self.tbl_tasks.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.tbl_tasks, stretch=1)

        # Download Card -> merged dataset folder (per-task folders are temporary)
        dl_card = QFrame()
        dl_card.setProperty("class", "CardFrame")
        dl_layout = QHBoxLayout(dl_card)
        dl_layout.setContentsMargins(16, 12, 16, 12)

        dl_layout.addWidget(QLabel("Dataset Folder:"))
        self.txt_dl_folder = QLineEdit(str(self.data_dir))
        self.txt_dl_folder.setToolTip("Merged dataset goes here: images/ + annotations/instances_default.json")
        dl_layout.addWidget(self.txt_dl_folder, stretch=1)

        btn_browse_dl = QPushButton("Browse...")
        btn_browse_dl.setIcon(get_svg_icon("folder", 16))
        btn_browse_dl.setIconSize(QSize(16, 16))
        btn_browse_dl.clicked.connect(lambda: (self._browse_folder(self.txt_dl_folder), self._on_dataset_folder_edited()))
        dl_layout.addWidget(btn_browse_dl)

        self.btn_download_tasks = QPushButton("Download && Merge Selected (COCO 1.0)")
        self.btn_download_tasks.setProperty("class", "PrimaryButton")
        self.btn_download_tasks.setIcon(get_svg_icon("download", 18))
        self.btn_download_tasks.setIconSize(QSize(18, 18))
        self.btn_download_tasks.clicked.connect(self._on_download_selected_tasks)
        dl_layout.addWidget(self.btn_download_tasks)

        layout.addWidget(dl_card)

        # Leftover per-task folders (e.g. from an older version) -> one-click clean up
        self.leftover_row = QFrame()
        self.leftover_row.setProperty("class", "CardFrame")
        lo = QHBoxLayout(self.leftover_row)
        lo.setContentsMargins(16, 8, 16, 8)
        self.lbl_leftover = QLabel("")
        self.lbl_leftover.setWordWrap(True)
        lo.addWidget(self.lbl_leftover, 1)
        self.btn_clean_leftover = QPushButton("Clean Up")
        self.btn_clean_leftover.setProperty("class", "DangerButton")
        self.btn_clean_leftover.clicked.connect(self._on_clean_leftover_tasks)
        lo.addWidget(self.btn_clean_leftover)
        self.leftover_row.hide()
        layout.insertWidget(layout.indexOf(self.tbl_tasks), self.leftover_row)   # above the table, always visible
        self.txt_dl_folder.editingFinished.connect(self._on_dataset_folder_edited)

        # Progress & Status
        self.prog_dl = QProgressBar()
        self.prog_dl.setValue(0)
        self.prog_dl.setTextVisible(True)
        layout.addWidget(self.prog_dl)

        status_row = QHBoxLayout()
        self.lbl_cvat_status = QLabel("Select tasks above and click Download.")
        self.lbl_cvat_status.setStyleSheet("color: #6b7f88; font-weight: 600;")
        status_row.addWidget(self.lbl_cvat_status, stretch=1)


        layout.addLayout(status_row)
        return page

    # -----------------------------------------------------------------
    # STEP 3: COCO to YOLO Converter
    # -----------------------------------------------------------------
    def _build_page_convert(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(14)

        # 2 Large Choice Cards side-by-side
        choice_row = QHBoxLayout()
        choice_row.setSpacing(14)

        # Detection Card
        self.card_det = QFrame()
        self.card_det.setProperty("class", "ChoiceCardNavy")
        card_det_layout = QVBoxLayout(self.card_det)
        card_det_layout.setContentsMargins(18, 16, 18, 16)

        c_det_top = QHBoxLayout()
        icon_box_lbl = QLabel()
        icon_box_lbl.setPixmap(get_svg_pixmap("box", 28, 28))
        c_det_top.addWidget(icon_box_lbl)
        c_det_top.addStretch()
        self.lbl_det_badge = QLabel()
        c_det_top.addWidget(self.lbl_det_badge)
        card_det_layout.addLayout(c_det_top)

        lbl_det_title = QLabel("YOLO Detection")
        lbl_det_title.setProperty("class", "ChoiceTitleNavy")
        card_det_layout.addWidget(lbl_det_title)

        lbl_det_sub = QLabel("Converts polygons to normalized bounding boxes (xc, yc, w, h).")
        lbl_det_sub.setProperty("class", "ChoiceSubNavy")
        lbl_det_sub.setWordWrap(True)
        card_det_layout.addWidget(lbl_det_sub)

        self.rb_mode_det = QRadioButton("Detection")
        card_det_layout.addWidget(self.rb_mode_det)

        choice_row.addWidget(self.card_det, stretch=1)

        # Segmentation Card
        self.card_seg = QFrame()
        self.card_seg.setProperty("class", "ChoiceCardPink")
        card_seg_layout = QVBoxLayout(self.card_seg)
        card_seg_layout.setContentsMargins(18, 16, 18, 16)

        c_seg_top = QHBoxLayout()
        icon_poly_lbl = QLabel()
        icon_poly_lbl.setPixmap(get_svg_pixmap("polygon", 28, 28))
        c_seg_top.addWidget(icon_poly_lbl)
        c_seg_top.addStretch()
        self.lbl_seg_badge = QLabel()
        c_seg_top.addWidget(self.lbl_seg_badge)
        card_seg_layout.addLayout(c_seg_top)

        lbl_seg_title = QLabel("YOLO Segmentation")
        lbl_seg_title.setProperty("class", "ChoiceTitlePink")
        card_seg_layout.addWidget(lbl_seg_title)

        lbl_seg_sub = QLabel("Preserves exact polygon vertex contours (cls x1 y1 x2 y2...).")
        lbl_seg_sub.setProperty("class", "ChoiceSubPink")
        lbl_seg_sub.setWordWrap(True)
        card_seg_layout.addWidget(lbl_seg_sub)

        self.rb_mode_seg = QRadioButton("Segmentation")
        card_seg_layout.addWidget(self.rb_mode_seg)

        choice_row.addWidget(self.card_seg, stretch=1)
        layout.addLayout(choice_row)

        # The two radios live in different cards, so Qt does NOT make them exclusive on its own
        # (that let both be checked and the convert always ran as "detect"). Group them explicitly.
        self.grp_conv_mode = QButtonGroup(self)
        self.grp_conv_mode.setExclusive(True)
        self.grp_conv_mode.addButton(self.rb_mode_det, 0)
        self.grp_conv_mode.addButton(self.rb_mode_seg, 1)
        self.rb_mode_det.setChecked(True)
        self.grp_conv_mode.buttonToggled.connect(lambda *_: self._on_task_mode_changed())
        for card, rb in ((self.card_det, self.rb_mode_det), (self.card_seg, self.rb_mode_seg)):
            card.setCursor(Qt.PointingHandCursor)
            card.mousePressEvent = lambda e, r=rb: r.setChecked(True)   # click anywhere on the card
        self._on_task_mode_changed()

        # Conversion Paths Card
        conv_paths = QFrame()
        conv_paths.setProperty("class", "CardFrame")
        cp_layout = QVBoxLayout(conv_paths)
        cp_layout.setContentsMargins(18, 16, 18, 16)
        cp_layout.setSpacing(12)

        cp_layout.addWidget(self._make_data_info("convert"))

        self.btn_run_convert = QPushButton("Convert COCO to YOLO (.txt)")
        self.btn_run_convert.setProperty("class", "SuccessButton")
        self.btn_run_convert.setIcon(get_svg_icon("step_convert", 18))
        self.btn_run_convert.setIconSize(QSize(18, 18))
        self.btn_run_convert.clicked.connect(self._on_run_convert)
        cp_layout.addWidget(self.btn_run_convert)

        self.prog_convert = QProgressBar()
        self.prog_convert.setValue(0)
        cp_layout.addWidget(self.prog_convert)

        self.lbl_conv_stats = QLabel("No conversion performed yet.")
        self.lbl_conv_stats.setStyleSheet("color: #10232b; font-weight: 600; min-height: 24px;")
        cp_layout.addWidget(self.lbl_conv_stats)

        layout.addWidget(conv_paths)


        layout.addStretch()
        return page

    # -----------------------------------------------------------------
    # STEP 4: Dataset Splitter
    # -----------------------------------------------------------------
    def _build_page_split(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(14)

        layout.addWidget(self._make_data_info("split", card=True))

        # Ratios Card
        ratios_card = QFrame()
        ratios_card.setProperty("class", "CardFrame")
        box_ratios_layout = QGridLayout(ratios_card)
        box_ratios_layout.setContentsMargins(18, 16, 18, 16)
        box_ratios_layout.setHorizontalSpacing(14)
        box_ratios_layout.setVerticalSpacing(10)

        box_ratios_layout.addWidget(QLabel("Train Ratio:"), 0, 0)
        self.slider_train = QSlider(Qt.Horizontal)
        self.slider_train.setRange(10, 95)
        self.slider_train.setValue(70)
        box_ratios_layout.addWidget(self.slider_train, 0, 1)
        self.spin_train = QSpinBox()
        self.spin_train.setRange(10, 95)
        self.spin_train.setValue(70)
        self.spin_train.setSuffix("%")
        box_ratios_layout.addWidget(self.spin_train, 0, 2)

        box_ratios_layout.addWidget(QLabel("Validation Ratio:"), 1, 0)
        self.slider_val = QSlider(Qt.Horizontal)
        self.slider_val.setRange(5, 50)
        self.slider_val.setValue(20)
        box_ratios_layout.addWidget(self.slider_val, 1, 1)
        self.spin_val = QSpinBox()
        self.spin_val.setRange(5, 50)
        self.spin_val.setValue(20)
        self.spin_val.setSuffix("%")
        box_ratios_layout.addWidget(self.spin_val, 1, 2)

        box_ratios_layout.addWidget(QLabel("Test Ratio:"), 2, 0)
        self.slider_test = QSlider(Qt.Horizontal)
        self.slider_test.setRange(0, 40)
        self.slider_test.setValue(10)
        box_ratios_layout.addWidget(self.slider_test, 2, 1)
        self.spin_test = QSpinBox()
        self.spin_test.setRange(0, 40)
        self.spin_test.setValue(10)
        self.spin_test.setSuffix("%")
        box_ratios_layout.addWidget(self.spin_test, 2, 2)

        # Dragging a slider or typing a number re-balances the other two so the total is always 100%
        self._ratio_widgets = {
            "train": (self.slider_train, self.spin_train),
            "val": (self.slider_val, self.spin_val),
            "test": (self.slider_test, self.spin_test),
        }
        self._ratio_busy = False
        self._split_img_count_cache: tuple[str, int] | None = None
        for key, (slider, spin) in self._ratio_widgets.items():
            slider.valueChanged.connect(lambda v, k=key: self._on_ratio_changed(k, v))
            spin.valueChanged.connect(lambda v, k=key: self._on_ratio_changed(k, v))

        self.lbl_ratio_total = QLabel("")
        self.lbl_ratio_total.setProperty("class", "StatText")
        box_ratios_layout.addWidget(self.lbl_ratio_total, 3, 0, 1, 3)

        note = QLabel("All images are used: images without objects get an empty label (background). "
                      "Files are moved inside the data folder — running Split again re-shuffles with the new ratios.")
        note.setWordWrap(True)
        note.setProperty("class", "MutedText")
        box_ratios_layout.addWidget(note, 4, 0, 1, 3)
        self._update_ratio_summary()

        self.btn_run_split = QPushButton("Split Dataset && Generate data.yaml")
        self.btn_run_split.setProperty("class", "PrimaryButton")
        self.btn_run_split.setIcon(get_svg_icon("step_split", 18))
        self.btn_run_split.setIconSize(QSize(18, 18))
        self.btn_run_split.clicked.connect(self._on_run_split)
        box_ratios_layout.addWidget(self.btn_run_split, 5, 0, 1, 3)

        self.prog_split = QProgressBar()
        self.prog_split.setValue(0)
        box_ratios_layout.addWidget(self.prog_split, 6, 0, 1, 3)

        self.lbl_split_stats = QLabel("No dataset split yet.")
        self.lbl_split_stats.setStyleSheet("color: #10232b; font-weight: 600; min-height: 24px;")
        box_ratios_layout.addWidget(self.lbl_split_stats, 7, 0, 1, 3)

        layout.addWidget(ratios_card)


        layout.addStretch()
        return page

    # -----------------------------------------------------------------
    # STEP 5: YOLO Auto-Trainer
    # -----------------------------------------------------------------
    def _build_page_train(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(12)

        # Model & Hyperparameters Card
        train_card = QFrame()
        train_card.setProperty("class", "CardFrame")
        h_layout = QGridLayout(train_card)
        h_layout.setContentsMargins(18, 16, 18, 16)
        h_layout.setHorizontalSpacing(12)
        h_layout.setVerticalSpacing(8)

        h_layout.addWidget(QLabel("Task:"), 0, 0)
        self.cmb_train_task = QComboBox()
        self.cmb_train_task.addItems(["detect", "segment"])
        self.cmb_train_task.currentIndexChanged.connect(self._update_model_presets)
        h_layout.addWidget(self.cmb_train_task, 0, 1)

        h_layout.addWidget(QLabel("Model Preset:"), 0, 2)
        self.cmb_model_preset = QComboBox()
        h_layout.addWidget(self.cmb_model_preset, 0, 3)

        h_layout.addWidget(QLabel("Or Custom .pt:"), 0, 4)
        self.txt_custom_model = QLineEdit()
        self.txt_custom_model.setPlaceholderText("Optional path to custom .pt weights")
        h_layout.addWidget(self.txt_custom_model, 0, 5)
        btn_browse_pt = QPushButton("Browse...")
        btn_browse_pt.setIcon(get_svg_icon("folder", 16))
        btn_browse_pt.setIconSize(QSize(16, 16))
        btn_browse_pt.clicked.connect(self._browse_model_pt)
        h_layout.addWidget(btn_browse_pt, 0, 6)

        self.cmb_model_preset.currentIndexChanged.connect(lambda _: self._update_model_file_label())
        self.txt_custom_model.textChanged.connect(lambda _: self._update_model_file_label())

        model_row = QHBoxLayout()
        self.lbl_model_file = QLabel("")
        self.lbl_model_file.setProperty("class", "MutedText")
        self.lbl_model_file.setTextInteractionFlags(Qt.TextSelectableByMouse)
        model_row.addWidget(self.lbl_model_file, 1)
        btn_models_dir = QPushButton("models/ Folder")
        btn_models_dir.setIcon(get_svg_icon("folder", 16))
        btn_models_dir.setIconSize(QSize(16, 16))
        btn_models_dir.clicked.connect(lambda: self._open_dir(self.models_dir))
        model_row.addWidget(btn_models_dir)
        h_layout.addLayout(model_row, 7, 0, 1, 7)

        h_layout.addWidget(QLabel("data.yaml Path:"), 1, 0)
        self.txt_data_yaml = QLineEdit(str(self.data_dir / "data.yaml"))
        self.txt_data_yaml.setReadOnly(True)
        self.txt_data_yaml.setToolTip("Fixed: <data folder>/data.yaml (written by Dataset Splitter). Change the data folder in Step 1.")
        h_layout.addWidget(self.txt_data_yaml, 1, 1, 1, 5)
        btn_open_data = QPushButton("Open")
        btn_open_data.setIcon(get_svg_icon("folder", 16))
        btn_open_data.setIconSize(QSize(16, 16))
        btn_open_data.clicked.connect(lambda: self._open_dir(self._ds().root))
        h_layout.addWidget(btn_open_data, 1, 6)

        self.btn_auto_tune = QPushButton("Auto-Tune Hyperparameters for Dataset")
        self.btn_auto_tune.setProperty("class", "SuccessButton")
        self.btn_auto_tune.setIcon(get_svg_icon("brain", 18))
        self.btn_auto_tune.setIconSize(QSize(18, 18))
        self.btn_auto_tune.clicked.connect(self._on_auto_tune_clicked)
        h_layout.addWidget(self.btn_auto_tune, 2, 0, 1, 7)

        h_layout.addWidget(QLabel("Epochs:"), 3, 0)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(100)
        h_layout.addWidget(self.spin_epochs, 3, 1)

        h_layout.addWidget(QLabel("Batch Size:"), 3, 2)
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(-1, 256)
        self.spin_batch.setValue(8 if not torch.cuda.is_available() else 16)
        self.spin_batch.setSpecialValueText("Auto (-1)")
        h_layout.addWidget(self.spin_batch, 3, 3)

        h_layout.addWidget(QLabel("Image Size:"), 3, 4)
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(160, 2560)
        self.spin_imgsz.setSingleStep(32)
        self.spin_imgsz.setValue(640)
        h_layout.addWidget(self.spin_imgsz, 3, 5)

        h_layout.addWidget(QLabel("Early Stop (Patience):"), 4, 0)
        self.spin_patience = QSpinBox()
        self.spin_patience.setRange(1, 100)
        self.spin_patience.setValue(15)
        h_layout.addWidget(self.spin_patience, 4, 1)

        h_layout.addWidget(QLabel("Initial LR (lr0):"), 4, 2)
        self.spin_lr0 = QDoubleSpinBox()
        self.spin_lr0.setRange(0.00001, 0.1)
        self.spin_lr0.setDecimals(5)
        self.spin_lr0.setValue(0.01)
        h_layout.addWidget(self.spin_lr0, 4, 3)

        h_layout.addWidget(QLabel("Optimizer:"), 4, 4)
        self.cmb_optimizer = QComboBox()
        self.cmb_optimizer.addItems(["auto", "AdamW", "Adam", "SGD", "RMSProp"])
        h_layout.addWidget(self.cmb_optimizer, 4, 5)

        h_layout.addWidget(QLabel("Custom Run Name:"), 5, 0)
        self.txt_custom_run_name = QLineEdit("my_yolo_model")
        self.txt_custom_run_name.textChanged.connect(self._update_run_preview)
        h_layout.addWidget(self.txt_custom_run_name, 5, 1, 1, 2)

        h_layout.addWidget(QLabel("Runs Folder:"), 5, 3)
        self.txt_runs_base = QLineEdit(str(self.data_dir / "runs"))
        self.txt_runs_base.setReadOnly(True)
        self.txt_runs_base.setToolTip("Fixed: <data folder>/runs")
        h_layout.addWidget(self.txt_runs_base, 5, 4, 1, 2)
        btn_runs_base = QPushButton("Open")
        btn_runs_base.setIcon(get_svg_icon("folder", 16))
        btn_runs_base.setIconSize(QSize(16, 16))
        btn_runs_base.clicked.connect(lambda: self._open_dir(self._ds().runs))
        h_layout.addWidget(btn_runs_base, 5, 6)

        # Output preview notice (In-place weights renaming inside runs/)
        self.lbl_run_preview = QLabel("Output Preview: runs/detect/my_yolo_model_v1/weights/my_yolo_model_v1.pt")
        self.lbl_run_preview.setProperty("class", "RunPreview")
        h_layout.addWidget(self.lbl_run_preview, 6, 0, 1, 7)

        layout.addWidget(train_card)

        # Training Execution Card
        exec_card = QFrame()
        exec_card.setProperty("class", "CardFrame")
        exec_layout = QVBoxLayout(exec_card)
        exec_layout.setContentsMargins(16, 14, 16, 14)
        exec_layout.setSpacing(10)

        btn_row = QHBoxLayout()
        self.btn_start_train = QPushButton("Start YOLO Model Training")
        self.btn_start_train.setProperty("class", "PrimaryButton")
        self.btn_start_train.setIcon(get_svg_icon("play", 18))
        self.btn_start_train.setIconSize(QSize(18, 18))
        self.btn_start_train.clicked.connect(self._on_start_training)
        btn_row.addWidget(self.btn_start_train, stretch=3)

        self.btn_stop_train = QPushButton("Stop Training")
        self.btn_stop_train.setProperty("class", "DangerButton")
        self.btn_stop_train.setIcon(get_svg_icon("stop", 16))
        self.btn_stop_train.setIconSize(QSize(16, 16))
        self.btn_stop_train.setEnabled(False)
        self.btn_stop_train.clicked.connect(self._on_stop_training)
        btn_row.addWidget(self.btn_stop_train, stretch=1)

        exec_layout.addLayout(btn_row)

        self.prog_train = QProgressBar()
        self.prog_train.setValue(0)
        exec_layout.addWidget(self.prog_train)

        self.lbl_train_metrics = QLabel("Waiting to begin training session...")
        self.lbl_train_metrics.setStyleSheet("color: #10232b; font-weight: 600;")
        exec_layout.addWidget(self.lbl_train_metrics)

        # Splitter between log and yaml preview
        split_area = QSplitter(Qt.Horizontal)

        # Training log
        log_box = QGroupBox("Training Output Console")
        l_layout = QVBoxLayout(log_box)
        self.txt_train_log = QPlainTextEdit()
        self.txt_train_log.setReadOnly(True)
        self.txt_train_log.setMinimumHeight(180)
        l_layout.addWidget(self.txt_train_log)
        split_area.addWidget(log_box)

        # data.yaml preview
        yaml_box = QGroupBox("data.yaml Preview")
        y_layout = QVBoxLayout(yaml_box)
        self.txt_yaml_preview = QPlainTextEdit()
        self.txt_yaml_preview.setReadOnly(True)
        self.txt_yaml_preview.setMinimumHeight(180)
        y_layout.addWidget(self.txt_yaml_preview)
        split_area.addWidget(yaml_box)

        exec_layout.addWidget(split_area)
        layout.addWidget(exec_card)

        # Latest training result (filled in when training finishes)
        self.train_result_card = QFrame()
        self.train_result_card.setProperty("class", "CardFrame")
        rc_layout = QVBoxLayout(self.train_result_card)
        rc_layout.setContentsMargins(16, 14, 16, 14)
        rc_layout.setSpacing(10)
        rc_head = QHBoxLayout()
        self.lbl_train_result_title = QLabel("Latest Training Result")
        self.lbl_train_result_title.setProperty("class", "SectionTitle")
        rc_head.addWidget(self.lbl_train_result_title)
        rc_head.addStretch()
        self.btn_train_report = QPushButton("View Full Report (graphs && confusion matrix)")
        self.btn_train_report.setProperty("class", "PrimaryButton")
        self.btn_train_report.setIcon(get_svg_icon("chart", 18))
        self.btn_train_report.setIconSize(QSize(18, 18))
        self.btn_train_report.clicked.connect(lambda: self._show_report(self.last_train_report))
        rc_head.addWidget(self.btn_train_report)
        rc_layout.addLayout(rc_head)
        self.train_result_grid = QGridLayout()
        self.train_result_grid.setSpacing(10)
        rc_layout.addLayout(self.train_result_grid)
        self.train_result_card.hide()
        layout.addWidget(self.train_result_card)


        self._update_model_presets()
        return page

    # -----------------------------------------------------------------
    # STEP 6: Test Model / Inference
    # -----------------------------------------------------------------
    def _build_page_test(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(12)

        # Configuration Card
        cfg_card = QFrame()
        cfg_card.setProperty("class", "CardFrame")
        cg_layout = QGridLayout(cfg_card)
        cg_layout.setContentsMargins(18, 16, 18, 16)
        cg_layout.setHorizontalSpacing(14)
        cg_layout.setVerticalSpacing(10)

        # 1. Model Weights
        cg_layout.addWidget(QLabel("Model Weights (.pt):"), 0, 0)
        self.txt_test_model = QLineEdit()
        self.txt_test_model.setPlaceholderText("Path to trained model .pt (e.g. runs/detect/.../weights/my_model_v1.pt)")
        cg_layout.addWidget(self.txt_test_model, 0, 1, 1, 4)

        btn_browse_test_pt = QPushButton("Browse Model...")
        btn_browse_test_pt.setIcon(get_svg_icon("folder", 16))
        btn_browse_test_pt.setIconSize(QSize(16, 16))
        btn_browse_test_pt.clicked.connect(self._browse_test_model_pt)
        cg_layout.addWidget(btn_browse_test_pt, 0, 5)

        # 2. Source Type Selector (Single Image, Entire Folder, Video)
        cg_layout.addWidget(QLabel("Test Input Source:"), 1, 0)
        src_mode_box = QHBoxLayout()
        self.rb_src_image = QRadioButton("Single Image")
        self.rb_src_folder = QRadioButton("Entire Images Folder")
        self.rb_src_video = QRadioButton("Video File")
        self.rb_src_split = QRadioButton("Test Split (data.yaml)")
        self.rb_src_split.setToolTip("Use the test split created in Step 3 (falls back to val if there is no test split).")
        self.rb_src_image.setChecked(True)

        self.btn_grp_source = QButtonGroup(self)
        self.btn_grp_source.addButton(self.rb_src_image, 0)
        self.btn_grp_source.addButton(self.rb_src_folder, 1)
        self.btn_grp_source.addButton(self.rb_src_video, 2)
        self.btn_grp_source.addButton(self.rb_src_split, 3)
        self.btn_grp_source.buttonToggled.connect(self._on_test_source_mode_changed)

        src_mode_box.addWidget(self.rb_src_image)
        src_mode_box.addWidget(self.rb_src_folder)
        src_mode_box.addWidget(self.rb_src_video)
        src_mode_box.addWidget(self.rb_src_split)
        src_mode_box.addStretch()
        self.chk_eval_labels = QCheckBox("Measure accuracy if labels exist")
        self.chk_eval_labels.setChecked(True)
        self.chk_eval_labels.setToolTip(
            "Images/folders: looks for YOLO .txt labels (next to the images or in a labels/ folder) "
            "or a COCO json (e.g. CVAT export). No labels → prediction only.")
        src_mode_box.addWidget(self.chk_eval_labels)
        cg_layout.addLayout(src_mode_box, 1, 1, 1, 5)

        # 3. Source Path
        self.lbl_test_src_path = QLabel("Image File Path:")
        cg_layout.addWidget(self.lbl_test_src_path, 2, 0)
        self.txt_test_source = QLineEdit()
        self.txt_test_source.setPlaceholderText("Select image to evaluate...")
        cg_layout.addWidget(self.txt_test_source, 2, 1, 1, 4)

        self.btn_browse_test_src = QPushButton("Browse...")
        self.btn_browse_test_src.setIcon(get_svg_icon("folder", 16))
        self.btn_browse_test_src.setIconSize(QSize(16, 16))
        self.btn_browse_test_src.clicked.connect(self._browse_test_source)
        cg_layout.addWidget(self.btn_browse_test_src, 2, 5)

        # 4. Confidence & IoU
        cg_layout.addWidget(QLabel("Confidence (conf):"), 3, 0)
        self.slider_test_conf = QSlider(Qt.Horizontal)
        self.slider_test_conf.setRange(5, 100)
        self.slider_test_conf.setValue(25)
        cg_layout.addWidget(self.slider_test_conf, 3, 1)
        self.spin_test_conf = QDoubleSpinBox()
        self.spin_test_conf.setRange(0.05, 1.0)
        self.spin_test_conf.setSingleStep(0.05)
        self.spin_test_conf.setValue(0.25)
        self.slider_test_conf.valueChanged.connect(lambda v: self.spin_test_conf.setValue(v / 100.0))
        self.spin_test_conf.valueChanged.connect(lambda v: self.slider_test_conf.setValue(int(v * 100)))
        cg_layout.addWidget(self.spin_test_conf, 3, 2)

        cg_layout.addWidget(QLabel("IoU Threshold:"), 3, 3)
        self.slider_test_iou = QSlider(Qt.Horizontal)
        self.slider_test_iou.setRange(10, 100)
        self.slider_test_iou.setValue(45)
        cg_layout.addWidget(self.slider_test_iou, 3, 4)
        self.spin_test_iou = QDoubleSpinBox()
        self.spin_test_iou.setRange(0.1, 1.0)
        self.spin_test_iou.setSingleStep(0.05)
        self.spin_test_iou.setValue(0.45)
        self.slider_test_iou.valueChanged.connect(lambda v: self.spin_test_iou.setValue(v / 100.0))
        self.spin_test_iou.valueChanged.connect(lambda v: self.slider_test_iou.setValue(int(v * 100)))
        cg_layout.addWidget(self.spin_test_iou, 3, 5)

        # 5. Label Display & Font Size (can be adjusted in real-time during predict)
        cg_layout.addWidget(QLabel("Label Display:"), 4, 0)
        lbl_mode_box = QHBoxLayout()
        self.rb_label_name = QRadioButton("Class Name (ชื่อ)")
        self.rb_label_id = QRadioButton("Class ID (เลข)")
        self.rb_label_none = QRadioButton("No Label (ไม่ใส่)")
        self.rb_label_name.setChecked(True)
        self.rb_label_name.setToolTip("Show class name and confidence score (e.g. 'person 0.85')")
        self.rb_label_id.setToolTip("Show class numeric ID and confidence score (e.g. '0 0.85')")
        self.rb_label_none.setToolTip("Hide label text, show bounding boxes / masks only")

        self.btn_grp_label = QButtonGroup(self)
        self.btn_grp_label.addButton(self.rb_label_name, 0)
        self.btn_grp_label.addButton(self.rb_label_id, 1)
        self.btn_grp_label.addButton(self.rb_label_none, 2)
        self.btn_grp_label.buttonToggled.connect(self._on_test_label_mode_changed)

        lbl_mode_box.addWidget(self.rb_label_name)
        lbl_mode_box.addWidget(self.rb_label_id)
        lbl_mode_box.addWidget(self.rb_label_none)
        lbl_mode_box.addStretch()
        cg_layout.addLayout(lbl_mode_box, 4, 1, 1, 2)

        cg_layout.addWidget(QLabel("Font Size:"), 4, 3)
        font_box = QHBoxLayout()
        self.slider_test_font = QSlider(Qt.Horizontal)
        self.slider_test_font.setRange(8, 48)
        self.slider_test_font.setValue(14)
        self.slider_test_font.setToolTip("Adjust annotation font size (8-48px). Can be adjusted live during prediction!")
        self.spin_test_font = QSpinBox()
        self.spin_test_font.setRange(8, 48)
        self.spin_test_font.setValue(14)
        self.spin_test_font.setSuffix(" px")
        self.slider_test_font.valueChanged.connect(self.spin_test_font.setValue)
        self.spin_test_font.valueChanged.connect(self.slider_test_font.setValue)
        self.slider_test_font.valueChanged.connect(self._on_test_font_size_changed)

        font_box.addWidget(self.slider_test_font)
        font_box.addWidget(self.spin_test_font)
        cg_layout.addLayout(font_box, 4, 4, 1, 2)

        # 6. Save results checkbox & destination folder
        self.chk_save_test = QCheckBox("Save Test Prediction Results")
        self.chk_save_test.setChecked(True)
        self.chk_save_test.setStyleSheet("font-weight: 700; color: #0e4b5a;")
        cg_layout.addWidget(self.chk_save_test, 5, 0)

        self.txt_test_save_dir = QLineEdit(str(self.data_dir / "runs" / "test"))
        self.txt_test_save_dir.setReadOnly(True)
        self.txt_test_save_dir.setToolTip("Fixed: <data folder>/runs/test")
        cg_layout.addWidget(self.txt_test_save_dir, 5, 1, 1, 5)

        # Action Buttons
        act_row = QHBoxLayout()
        self.btn_run_test = QPushButton("Run Model Test / Predict")
        self.btn_run_test.setProperty("class", "PrimaryButton")
        self.btn_run_test.setIcon(get_svg_icon("play", 18))
        self.btn_run_test.setIconSize(QSize(18, 18))
        self.btn_run_test.clicked.connect(self._on_run_test)
        act_row.addWidget(self.btn_run_test, stretch=3)

        self.btn_stop_test = QPushButton("Stop Test")
        self.btn_stop_test.setProperty("class", "DangerButton")
        self.btn_stop_test.setIcon(get_svg_icon("stop", 16))
        self.btn_stop_test.setIconSize(QSize(16, 16))
        self.btn_stop_test.setEnabled(False)
        self.btn_stop_test.clicked.connect(self._on_stop_test)
        act_row.addWidget(self.btn_stop_test, stretch=1)

        btn_open_results = QPushButton("Open Results Folder")
        btn_open_results.setProperty("class", "SecondaryNavButton")
        btn_open_results.setIcon(get_svg_icon("folder", 16))
        btn_open_results.setIconSize(QSize(16, 16))
        btn_open_results.clicked.connect(lambda: self._open_dir(Path(self.txt_test_save_dir.text().strip())))
        act_row.addWidget(btn_open_results, stretch=1)

        cg_layout.addLayout(act_row, 6, 0, 1, 6)

        self.prog_test = QProgressBar()
        self.prog_test.setValue(0)
        cg_layout.addWidget(self.prog_test, 7, 0, 1, 6)

        self.lbl_test_status = QLabel("Select model weights and input source above, then click 'Run Model Test / Predict'.")
        self.lbl_test_status.setStyleSheet("color: #6b7f88; font-weight: 600;")
        cg_layout.addWidget(self.lbl_test_status, 8, 0, 1, 6)

        layout.addWidget(cfg_card)

        # Accuracy result (shown when the tested data had labels)
        self.test_eval_card = QFrame()
        self.test_eval_card.setProperty("class", "CardFrame")
        ec_layout = QVBoxLayout(self.test_eval_card)
        ec_layout.setContentsMargins(16, 14, 16, 14)
        ec_layout.setSpacing(10)
        ec_head = QHBoxLayout()
        self.lbl_test_eval_title = QLabel("Accuracy")
        self.lbl_test_eval_title.setProperty("class", "SectionTitle")
        ec_head.addWidget(self.lbl_test_eval_title)
        ec_head.addStretch()
        self.btn_test_report = QPushButton("View Accuracy Report (confusion matrix && curves)")
        self.btn_test_report.setProperty("class", "PrimaryButton")
        self.btn_test_report.setIcon(get_svg_icon("chart", 18))
        self.btn_test_report.setIconSize(QSize(18, 18))
        self.btn_test_report.clicked.connect(lambda: self._show_report(self.last_eval_report))
        ec_head.addWidget(self.btn_test_report)
        ec_layout.addLayout(ec_head)
        self.test_eval_grid = QGridLayout()
        self.test_eval_grid.setSpacing(10)
        ec_layout.addLayout(self.test_eval_grid)
        self.lbl_test_eval_note = QLabel("")
        self.lbl_test_eval_note.setWordWrap(True)
        self.lbl_test_eval_note.setProperty("class", "MutedText")
        ec_layout.addWidget(self.lbl_test_eval_note)
        self.test_eval_card.hide()
        layout.addWidget(self.test_eval_card)

        # Visual Result & Detections Inspector Area
        inspect_card = QFrame()
        inspect_card.setProperty("class", "CardFrame")
        ic_layout = QHBoxLayout(inspect_card)
        ic_layout.setContentsMargins(14, 14, 14, 14)
        ic_layout.setSpacing(14)

        # Left: Image / Frame Preview Canvas
        canvas_col = QVBoxLayout()
        canvas_top = QHBoxLayout()
        lbl_canvas_title = QLabel("Prediction Visual Preview")
        lbl_canvas_title.setProperty("class", "SectionTitle")
        canvas_top.addWidget(lbl_canvas_title)

        # Quick Font Zoom Buttons right beside Carousel
        lbl_qfont = QLabel("Font:")
        lbl_qfont.setStyleSheet("color: #6b7f88; font-weight: 600; margin-left: 10px;")
        canvas_top.addWidget(lbl_qfont)

        btn_font_minus = QPushButton("A-")
        btn_font_minus.setToolTip("Decrease Font Size (-2 px)")
        btn_font_minus.setFixedWidth(36)
        btn_font_minus.clicked.connect(lambda: self.slider_test_font.setValue(max(8, self.slider_test_font.value() - 2)))
        canvas_top.addWidget(btn_font_minus)

        btn_font_plus = QPushButton("A+")
        btn_font_plus.setToolTip("Increase Font Size (+2 px)")
        btn_font_plus.setFixedWidth(36)
        btn_font_plus.clicked.connect(lambda: self.slider_test_font.setValue(min(48, self.slider_test_font.value() + 2)))
        canvas_top.addWidget(btn_font_plus)

        canvas_top.addStretch()

        # Folder Carousel Navigator
        self.btn_prev_test_img = QPushButton("< Prev")
        self.btn_prev_test_img.setEnabled(False)
        self.btn_prev_test_img.clicked.connect(self._on_prev_test_img)
        canvas_top.addWidget(self.btn_prev_test_img)

        self.lbl_test_img_counter = QLabel("0 / 0")
        self.lbl_test_img_counter.setStyleSheet("font-weight: 700; color: #24363d;")
        canvas_top.addWidget(self.lbl_test_img_counter)

        self.btn_next_test_img = QPushButton("Next >")
        self.btn_next_test_img.setEnabled(False)
        self.btn_next_test_img.clicked.connect(self._on_next_test_img)
        canvas_top.addWidget(self.btn_next_test_img)

        canvas_col.addLayout(canvas_top)

        # Canvas Frame
        canvas_frame = QFrame()
        canvas_frame.setObjectName("testCanvas")
        cf_layout = QVBoxLayout(canvas_frame)
        cf_layout.setContentsMargins(4, 4, 4, 4)

        self.lbl_test_preview = QLabel("Visual evaluation result will appear here")
        self.lbl_test_preview.setAlignment(Qt.AlignCenter)
        self.lbl_test_preview.setObjectName("testPreview")
        self.lbl_test_preview.setMinimumSize(480, 300)
        cf_layout.addWidget(self.lbl_test_preview)
        canvas_col.addWidget(canvas_frame, stretch=1)

        ic_layout.addLayout(canvas_col, stretch=3)

        # Right: Detection Summary & Table
        table_col = QVBoxLayout()
        lbl_det_table_title = QLabel("Detected Objects & Metrics")
        lbl_det_table_title.setProperty("class", "SectionTitle")
        table_col.addWidget(lbl_det_table_title)

        self.lbl_test_classes_badge = QLabel("No detections yet")
        self.lbl_test_classes_badge.setStyleSheet("color: #6b7f88; font-weight: 600; padding: 4px;")
        self.lbl_test_classes_badge.setWordWrap(True)
        table_col.addWidget(self.lbl_test_classes_badge)

        self.tbl_test_detections = QTableWidget()
        self.tbl_test_detections.setColumnCount(4)
        self.tbl_test_detections.setHorizontalHeaderLabels(["Class", "Confidence", "Coordinates (xyxy)", "ID"])
        self.tbl_test_detections.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.tbl_test_detections.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tbl_test_detections.verticalHeader().setVisible(False)
        table_col.addWidget(self.tbl_test_detections, stretch=1)

        ic_layout.addLayout(table_col, stretch=2)
        layout.addWidget(inspect_card, stretch=1)

        return page

    # =================================================================
    # Studio Navigation & Pipeline Slots
    # =================================================================
    def _open_dir(self, dir_path: Path):
        dir_path.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(dir_path))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(dir_path)])
        else:
            subprocess.Popen(["xdg-open", str(dir_path)])

    def _go_to_step(self, step_idx: int):
        if step_idx < 0 or step_idx >= self.pages.count():
            return

        self.pages.setCurrentIndex(step_idx)
        if step_idx != self.STEP_DOWNLOAD:
            self._sync_dataset_paths(self._ds().root, save=False)   # files may have changed outside the app
            self._refresh_yaml_preview()

        # Update Stepper Buttons active style
        for idx, btn in enumerate(self.step_buttons):
            if idx == step_idx:
                btn.setProperty("class", "StepButtonActive")
            else:
                btn.setProperty("class", "StepButton")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        for idx, pill in enumerate(getattr(self, "step_pills", [])):
            pill.setProperty("class", "StepPillActive" if idx == step_idx else "StepPill")
            pill.style().unpolish(pill)
            pill.style().polish(pill)

        # Update Guide Box
        self.lbl_guide.setText(self.STEP_GUIDES[step_idx])

        # Update Footer Navigation
        self.lbl_step_indicator.setText(f"Step {step_idx + 1} of {self.pages.count()}")
        self.btn_footer_prev.setEnabled(step_idx > 0)
        self.btn_footer_next.setEnabled(step_idx < self.pages.count() - 1)
        if step_idx < len(self.STEP_TITLES) - 1:
            nxt = self.STEP_TITLES[step_idx + 1][0].split(". ", 1)[-1]
            self.btn_footer_next.setText(f"Next: {nxt}".replace("&", "&&"))
        else:
            self.btn_footer_next.setText("Next Step")
        if step_idx > 0:
            prv = self.STEP_TITLES[step_idx - 1][0].split(". ", 1)[-1]
            self.btn_footer_prev.setText(f"Back: {prv}".replace("&", "&&"))
        else:
            self.btn_footer_prev.setText("Previous")

        if step_idx == self.STEP_DOWNLOAD:
            self._refresh_leftover_tasks()

        # Refresh preview if going to Train
        if step_idx == self.STEP_TRAIN:
            self._refresh_yaml_preview()
            self._update_run_preview()

        # Pre-fill model path if going to Test (Step 6 / index 5)
        if step_idx == self.STEP_TEST and self.last_trained_weight and self.last_trained_weight.exists():
            if not self.txt_test_model.text().strip():
                self.txt_test_model.setText(str(self.last_trained_weight))

    def _on_prev_step(self):
        cur = self.pages.currentIndex()
        if cur > 0:
            self._go_to_step(cur - 1)

    def _on_next_step(self):
        cur = self.pages.currentIndex()
        if cur < self.pages.count() - 1:
            self._go_to_step(cur + 1)

    # -----------------------------------------------------------------
    # Authentication & Login Flow Slots
    # -----------------------------------------------------------------
    def _toggle_password_visibility(self):
        """Toggle password visibility between masked and clear text."""
        if self.txt_login_pass.echoMode() == QLineEdit.Password:
            self.txt_login_pass.setEchoMode(QLineEdit.Normal)
            self.btn_toggle_pass.setIcon(get_svg_icon("eye", 20))
            self.btn_toggle_pass.setToolTip("Hide Password (ซ่อนรหัสผ่าน)")
        else:
            self.txt_login_pass.setEchoMode(QLineEdit.Password)
            self.btn_toggle_pass.setIcon(get_svg_icon("eye_off", 20))
            self.btn_toggle_pass.setToolTip("Show Password (แสดงรหัสผ่าน)")

    def _on_do_login(self):
        url = self.txt_login_url.text().strip()
        user = self.txt_login_user.text().strip()
        pwd = self.txt_login_pass.text()

        self.lbl_login_alert.hide()

        if not url:
            self.lbl_login_alert.setText("Please enter your CVAT Server URL (e.g. http://<server-ip>:8080).")
            self.lbl_login_alert.show()
            return

        self.btn_do_login.setEnabled(False)
        self.lbl_login_status.setText("Connecting to CVAT server...")
        self.login_mascot.set_state(DancingSvgMascot.STATE_DOWNLOADING)

        self.active_worker = CvatAuthWorker(self.cvat_client, url, user, pwd)
        self.active_worker.finished_signal.connect(self._on_auth_success)
        self.active_worker.error_signal.connect(self._on_auth_error)
        self.active_worker.start()

    def _on_auth_success(self, login_info: dict, projects: list[dict], tasks: list[dict]):
        self.btn_do_login.setEnabled(True)
        self._save_login()
        self.login_mascot.set_state(DancingSvgMascot.STATE_SUCCESS)
        username = login_info.get("username") or self.txt_login_user.text().strip() or "User"
        server_url = login_info.get("server_url") or self.txt_login_url.text().strip()

        # Update Studio Header Profile Badge
        self.lbl_user_badge.setText(f"{username} @ {server_url} • {len(tasks)} tasks available")

        # Populate projects dropdown in Step 1
        self.cmb_proj_filter.blockSignals(True)
        self.cmb_proj_filter.clear()
        self.cmb_proj_filter.addItem("All Projects", None)
        for p in projects:
            self.cmb_proj_filter.addItem(f"[{p['id']}] {p['name']}", p['id'])
        self.cmb_proj_filter.blockSignals(False)

        self.all_tasks_data = tasks
        self._populate_tasks_table(self.all_tasks_data)
        self.lbl_cvat_status.setText(f"Loaded {len(tasks)} tasks across {len(projects)} projects.")

        # Switch smoothly to Studio view!
        self.root_stack.setCurrentIndex(1)
        self.studio_mascot.set_state(DancingSvgMascot.STATE_IDLE)
        self._go_to_step(0)

    def _on_auth_error(self, err_msg: str):
        self.btn_do_login.setEnabled(True)
        self.login_mascot.set_state(DancingSvgMascot.STATE_ERROR)
        self.lbl_login_status.setText("")
        self.lbl_login_alert.setText(f"Connection Failed: {err_msg}")
        self.lbl_login_alert.show()

    # -----------------------------------------------------------------
    # Remember me
    # -----------------------------------------------------------------
    def _on_remember_toggled(self, on: bool):
        self.chk_remember_pwd.setEnabled(on and secure_store.available())
        if not on:
            self.chk_remember_pwd.setChecked(False)

    def _load_saved_login(self):
        urls = self.settings.value("login/url_history", [], type=list) or []
        users = self.settings.value("login/user_history", [], type=list) or []
        for edit, items in ((self.txt_login_url, urls), (self.txt_login_user, users)):
            comp = QCompleter(items, self)
            comp.setCaseSensitivity(Qt.CaseInsensitive)
            comp.setFilterMode(Qt.MatchContains)
            edit.setCompleter(comp)

        remember = self.settings.value("login/remember", True, type=bool)
        self.chk_remember.setChecked(remember)
        self.btn_forget_login.setVisible(bool(urls or users))
        if not remember:
            return
        url = self.settings.value("login/url", "", type=str)
        user = self.settings.value("login/username", "", type=str)
        self.txt_login_url.setText(url)
        self.txt_login_user.setText(user)
        remember_pwd = self.settings.value("login/remember_password", False, type=bool) and secure_store.available()
        self.chk_remember_pwd.setChecked(remember_pwd)
        if remember_pwd and url and user:
            pwd = secure_store.load_password(self.settings, f"{user}@{url}")
            if pwd:
                self.txt_login_pass.setText(pwd)
        (self.txt_login_pass if user else self.txt_login_user).setFocus()

    def _save_login(self):
        url = self.txt_login_url.text().strip()
        user = self.txt_login_user.text().strip()
        remember = self.chk_remember.isChecked()
        self.settings.setValue("login/remember", remember)
        if not remember:
            self._forget_saved_login(silent=True)
            return

        def push(key, value):
            items = [v for v in (self.settings.value(key, [], type=list) or []) if v != value]
            self.settings.setValue(key, ([value] + items)[:8])

        self.settings.setValue("login/url", url)
        self.settings.setValue("login/username", user)
        push("login/url_history", url)
        if user:
            push("login/user_history", user)

        remember_pwd = self.chk_remember_pwd.isChecked() and secure_store.available()
        account = f"{user}@{url}"
        if user and remember_pwd:
            remember_pwd = secure_store.save_password(self.settings, account, self.txt_login_pass.text())
        elif user:
            secure_store.delete_password(self.settings, account)
        self.settings.setValue("login/remember_password", remember_pwd)
        self.settings.sync()
        self.btn_forget_login.setVisible(True)

    def _forget_saved_login(self, silent: bool = False):
        url = self.settings.value("login/url", "", type=str)
        user = self.settings.value("login/username", "", type=str)
        if user:
            secure_store.delete_password(self.settings, f"{user}@{url}")
        secure_store.delete_all(self.settings)
        for k in ("login/url", "login/username", "login/url_history", "login/user_history", "login/remember_password"):
            self.settings.remove(k)
        self.settings.sync()
        self.btn_forget_login.setVisible(False)
        if not silent:
            self.txt_login_pass.clear()
            self.chk_remember_pwd.setChecked(False)
            for edit in (self.txt_login_url, self.txt_login_user):
                edit.setCompleter(QCompleter([], self))
            self.lbl_login_status.setText("Saved login removed.")

    def _on_skip_login(self):
        """Allows direct entry to studio for local dataset processing."""
        self.lbl_user_badge.setText("Local Mode (Offline Dataset Processing)")
        self.root_stack.setCurrentIndex(1)
        self.studio_mascot.set_state(DancingSvgMascot.STATE_IDLE)
        self._go_to_step(0)

    def _on_logout(self):
        """Switches back to dedicated login screen."""
        self.root_stack.setCurrentIndex(0)
        self.login_mascot.set_state(DancingSvgMascot.STATE_IDLE)
        self.lbl_login_status.setText("")
        self.lbl_login_alert.hide()

    # -----------------------------------------------------------------
    # STEP 1: Downloader Slots
    # -----------------------------------------------------------------
    def _fetch_cvat_tasks(self):
        if not self.cvat_client.is_authenticated():
            self._on_logout()
            return
        try:
            self.all_tasks_data = self.cvat_client.get_tasks()
            self._populate_tasks_table(self.all_tasks_data)
            self.lbl_cvat_status.setText(f"Refreshed: {len(self.all_tasks_data)} tasks loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Refresh Error", f"Error fetching tasks:\n{e}")

    def _populate_tasks_table(self, tasks: list[dict]):
        self.tbl_tasks.setRowCount(0)
        self.tbl_tasks.setRowCount(len(tasks))

        for row, t in enumerate(tasks):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setCheckState(Qt.Unchecked)
            self.tbl_tasks.setItem(row, 0, chk)

            id_item = QTableWidgetItem(str(t["id"]))
            id_item.setTextAlignment(Qt.AlignCenter)
            self.tbl_tasks.setItem(row, 1, id_item)

            name_item = QTableWidgetItem(t["name"])
            self.tbl_tasks.setItem(row, 2, name_item)

            proj_item = QTableWidgetItem(str(t["project_name"] or "-"))
            self.tbl_tasks.setItem(row, 3, proj_item)

            stage_item = QTableWidgetItem(str(t["stage"]))
            stage_item.setTextAlignment(Qt.AlignCenter)
            self.tbl_tasks.setItem(row, 4, stage_item)

            state_item = QTableWidgetItem(str(t["state"]))
            state_item.setTextAlignment(Qt.AlignCenter)
            self.tbl_tasks.setItem(row, 5, state_item)

            status_item = QTableWidgetItem(str(t["status"]))
            status_item.setTextAlignment(Qt.AlignCenter)
            self.tbl_tasks.setItem(row, 6, status_item)

            frames_item = QTableWidgetItem(str(t["size"]))
            frames_item.setTextAlignment(Qt.AlignCenter)
            self.tbl_tasks.setItem(row, 7, frames_item)

    def _apply_task_filters(self):
        sel_proj_id = self.cmb_proj_filter.currentData()
        sel_stage = self.cmb_stage_filter.currentText()
        sel_state = self.cmb_state_filter.currentText()
        search_query = self.txt_task_search.text().strip().lower()

        filtered = []
        for t in self.all_tasks_data:
            if sel_proj_id is not None and t["project_id"] != sel_proj_id:
                continue
            if sel_stage != "All" and str(t["stage"]).lower() != sel_stage.lower():
                continue
            if sel_state != "All" and str(t["state"]).lower() != sel_state.lower():
                continue
            if search_query and search_query not in t["name"].lower():
                continue
            filtered.append(t)

        self._populate_tasks_table(filtered)
        self.lbl_cvat_status.setText(f"Showing {len(filtered)} of {len(self.all_tasks_data)} tasks.")

    def _set_all_tasks_checked(self, checked: bool):
        state = Qt.Checked if checked else Qt.Unchecked
        for r in range(self.tbl_tasks.rowCount()):
            item = self.tbl_tasks.item(r, 0)
            if item:
                item.setCheckState(state)

    def _on_download_selected_tasks(self):
        if not self.cvat_client.is_authenticated():
            QMessageBox.warning(self, "Not Connected", "Please sign in to CVAT first.")
            self._on_logout()
            return

        selected_tasks = []
        for r in range(self.tbl_tasks.rowCount()):
            chk = self.tbl_tasks.item(r, 0)
            if chk and chk.checkState() == Qt.Checked:
                tid = int(self.tbl_tasks.item(r, 1).text())
                task_dict = next((t for t in self.all_tasks_data if t["id"] == tid), None)
                if task_dict:
                    selected_tasks.append(task_dict)

        if not selected_tasks:
            QMessageBox.warning(self, "No Tasks Selected", "Please select at least one task using the checkboxes.")
            return

        dest_folder = Path(self.txt_dl_folder.text().strip())
        append = False
        existing = dest_folder / "annotations" / "instances_default.json"
        if existing.exists():
            try:
                with open(existing, "r", encoding="utf-8") as f:
                    n_old = len(json.load(f).get("images", []))
            except Exception:
                n_old = "?"
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Question)
            box.setWindowTitle("Dataset Already Exists")
            box.setText(f"{dest_folder} already has a merged dataset ({n_old} images).")
            box.setInformativeText(
                "Add: keep it and add the new tasks.\n"
                "Replace: start a new dataset from the selected tasks only.\n\n"
                "Either way images/, annotations/, labels/ and data.yaml are rebuilt — "
                "run COCO → YOLO and Split again afterwards.")
            b_add = box.addButton("Add to Existing", QMessageBox.AcceptRole)
            b_rep = box.addButton("Replace", QMessageBox.DestructiveRole)
            box.addButton(QMessageBox.Cancel)
            box.setDefaultButton(b_add)
            box.exec_()
            if box.clickedButton() not in (b_add, b_rep):
                return
            append = box.clickedButton() is b_add

        self.btn_download_tasks.setEnabled(False)
        self.prog_dl.setValue(0)
        self.studio_mascot.set_state(DancingSvgMascot.STATE_DOWNLOADING)

        self.active_worker = CvatDownloadWorker(self.cvat_client, self.merger_engine, selected_tasks, dest_folder, append)
        self.active_worker.progress_signal.connect(self._on_download_progress)
        self.active_worker.finished_signal.connect(self._on_download_finished)
        self.active_worker.error_signal.connect(self._on_download_error)
        self.active_worker.start()

    def _sync_dataset_paths(self, ds: Path, save: bool = True):
        """Point Convert / Split / Train at <dataset>/annotations, images, labels and data.yaml."""
        lay = DatasetLayout(ds)
        self.txt_data_yaml.setText(str(lay.data_yaml))
        self.txt_runs_base.setText(str(lay.runs))
        self.txt_test_save_dir.setText(str(lay.test_runs))
        if save:
            self.settings.setValue("paths/dataset_dir", str(lay.root))
        self._refresh_data_info()
        if hasattr(self, "lbl_run_preview"):
            self._update_run_preview()

    def _on_dataset_folder_edited(self):
        ds = Path(self.txt_dl_folder.text().strip())
        if ds.is_dir():
            self._sync_dataset_paths(ds)
        self._refresh_leftover_tasks()

    def _stray_label_files(self, ds: Path) -> list[Path]:
        """YOLO .txt files sitting in the dataset root whose names match images/ (from a wrong Labels Output Folder)."""
        if not (ds / "images").is_dir():
            return []
        try:
            stems = {Path(n).stem for n in DatasetLayout(ds).image_locations()}
            return sorted(p for p in ds.iterdir() if p.is_file() and p.suffix.lower() == ".txt" and p.stem in stems)
        except OSError:
            return []

    def _refresh_leftover_tasks(self):
        ds = Path(self.txt_dl_folder.text().strip())
        left = find_leftover_task_dirs(ds)
        stray = self._stray_label_files(ds)
        has_merged = (ds / "annotations" / "instances_default.json").exists()
        parts = []
        if left and has_merged:
            parts.append(f"{len(left)} per-task folders (task_{left[0].name.split('_')[1]}_…)")
        if stray:
            parts.append(f"{len(stray)} label .txt files in the dataset root (they belong in labels/)")
        if parts:
            self.lbl_leftover.setText(f"⚠  Found in {ds}: " + " and ".join(parts) +
                                      ". images/ and annotations/ are kept.")
            self.btn_clean_leftover.setText("Clean Up")
            self.leftover_row.show()
        else:
            self.leftover_row.hide()

    def _on_clean_leftover_tasks(self):
        ds = Path(self.txt_dl_folder.text().strip())
        left = find_leftover_task_dirs(ds) if (ds / "annotations" / "instances_default.json").exists() else []
        stray = self._stray_label_files(ds)
        if not left and not stray:
            self.leftover_row.hide()
            return
        lines = []
        if left:
            lines.append(f"• {len(left)} per-task folders: " + ", ".join(d.name for d in left[:4]) + (" …" if len(left) > 4 else ""))
        if stray:
            lines.append(f"• {len(stray)} label .txt files in the dataset root (e.g. {stray[0].name}) — "
                         f"run COCO → YOLO again to recreate them in labels/")
        ans = QMessageBox.question(
            self, "Clean Up Dataset Folder",
            f"Delete from {ds}:\n\n" + "\n".join(lines) + "\n\nimages/ and annotations/ (the merged dataset) are kept.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ans != QMessageBox.Yes:
            return
        failed = []
        for d in left:
            try:
                shutil.rmtree(d)
            except Exception as e:
                failed.append(f"{d.name}: {e}")
        for f in stray:
            try:
                f.unlink()
            except Exception as e:
                failed.append(f"{f.name}: {e}")
        self._refresh_leftover_tasks()
        if failed:
            QMessageBox.warning(self, "Some Folders Not Removed", "\n".join(failed[:10]))
        else:
            self.notify.add("success", "Dataset folder cleaned",
                            f"{len(left)} task folders and {len(stray)} stray label files deleted from {ds}",
                            {"type": "folder", "path": str(ds)}, toast=False)

    def _on_download_progress(self, cur: int, tot: int, desc: str):
        self.prog_dl.setMaximum(tot)
        self.prog_dl.setValue(cur)
        self.lbl_cvat_status.setText(desc)

    def _on_download_finished(self, stats: dict):
        self.btn_download_tasks.setEnabled(True)
        self.studio_mascot.set_state(DancingSvgMascot.STATE_SUCCESS)
        self.prog_dl.setValue(100)
        cats = [c.get("name", "") for c in stats.get("categories", [])]
        n_tasks = len(stats.get("tasks", []))
        summary = (f"{'Added' if stats.get('appended') else 'Merged'} {n_tasks} task(s) → "
                   f"{stats.get('images_count', 0)} images, {stats.get('annotations_count', 0)} annotations, "
                   f"{len(cats)} classes ({', '.join(cats)})")
        self.lbl_cvat_status.setText(summary)

        # next steps read the merged dataset
        ds = Path(stats.get("dataset_dir", self.txt_dl_folder.text().strip()))
        self._sync_dataset_paths(ds)
        self.last_class_names = None   # converter will report the new class list

        self._refresh_leftover_tasks()
        self.notify.add("success", "Download & merge finished", summary, {"type": "folder", "path": str(ds)})
        QMessageBox.information(
            self, "Download & Merge Finished",
            f"{summary}\n\nData folder: {ds}\n  images/  +  annotations/instances_default.json\n\n"
            f"Per-task folders were removed. labels/ and data.yaml were reset (runs/ is kept).\n"
            f"Next: COCO → YOLO, then Split.")

    def _on_download_error(self, err_msg: str):
        self.notify.add("error", "Download / merge failed", err_msg[:300])
        self.btn_download_tasks.setEnabled(True)
        self.studio_mascot.set_state(DancingSvgMascot.STATE_ERROR)
        self.lbl_cvat_status.setText(f"Error: {err_msg}")
        QMessageBox.critical(self, "Download Error",
                             f"Download / merge failed — the existing dataset was not changed.\n\n{err_msg}")

    # -----------------------------------------------------------------
    # STEP 3: Convert Slots
    # -----------------------------------------------------------------
    def _on_task_mode_changed(self):
        """Selected card = navy with a 'SELECTED' badge; the other card is light. Keeps Step 4 task in sync."""
        is_det = self.rb_mode_det.isChecked()
        for card, badge, selected in ((self.card_det, self.lbl_det_badge, is_det),
                                      (self.card_seg, self.lbl_seg_badge, not is_det)):
            card.setProperty("class", "ChoiceCardNavy" if selected else "ChoiceCardPink")
            badge.setText("✓  SELECTED" if selected else "Click to select")
            badge.setStyleSheet(
                "background: #ffffff; color: %s; border-radius: 4px; padding: 3px 10px; font-weight: 700; font-size: 11px;"
                % PALETTE["teal"] if selected else
                f"color: {PALETTE['muted']}; font-size: 11px; background: transparent;")
            self._repolish(card)
        mode = "detect" if is_det else "segment"
        if hasattr(self, "cmb_train_task") and self.cmb_train_task.currentText() != mode:
            self.cmb_train_task.setCurrentText(mode)   # training task must match the label format

    def _ds(self) -> DatasetLayout:
        """The one data folder chosen in Step 1; every step reads/writes inside it."""
        return DatasetLayout(Path(self.txt_dl_folder.text().strip() or str(self.data_dir)))

    def _make_data_info(self, step: str, card: bool = False) -> QWidget:
        """Read-only box showing where this step reads and writes (all inside the Step 1 folder)."""
        box = QFrame()
        if card:
            box.setProperty("class", "CardFrame")
        lay = QHBoxLayout(box)
        lay.setContentsMargins(18, 14, 18, 14) if card else lay.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel()
        lbl.setTextFormat(Qt.RichText)
        lbl.setWordWrap(True)
        lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(lbl, 1)
        btn = QPushButton("Open Data Folder")
        btn.setIcon(get_svg_icon("folder", 16))
        btn.setIconSize(QSize(16, 16))
        btn.clicked.connect(lambda: self._open_dir(self._ds().root))
        lay.addWidget(btn, 0, Qt.AlignTop)
        if not hasattr(self, "_data_info_labels"):
            self._data_info_labels = {}
        self._data_info_labels[step] = lbl
        return box

    def _refresh_data_info(self):
        labels = getattr(self, "_data_info_labels", {})
        if not labels:
            return
        lay = self._ds()
        try:
            c = lay.counts()
            n_lbl = len(lay.label_files())
        except OSError:
            c, n_lbl = {"images": 0, "flat": 0, "labelled": 0, "train": 0, "val": 0, "test": 0}, 0
        ok, bad = "#2e7d4f", "#b3261e"

        def row(name, path, state, good):
            col = ok if good else bad
            return (f"<tr><td style='padding-right:14px'><b>{name}</b></td>"
                    f"<td style='padding-right:14px'>{path}</td><td style='color:{col}'>{state}</td></tr>")

        root = lay.root.as_posix()
        split_txt = (f"split: train {c['train']} · val {c['val']} · test {c['test']}"
                     if lay.is_split() else "not split yet")
        img_row = row("images/", f"{root}/images", f"{c['images']} images, {split_txt}", c["images"] > 0)
        json_row = row("annotations/", f"{root}/annotations/instances_default.json",
                       "found" if lay.coco_json.exists() else "missing — download in Step 1", lay.coco_json.exists())
        lbl_row = row("labels/", f"{root}/labels",
                      f"{n_lbl} label files" if n_lbl else "empty — run COCO to YOLO", n_lbl > 0)
        yaml_row = row("data.yaml", f"{root}/data.yaml",
                       "found" if lay.data_yaml.exists() else "not created yet — run Split", lay.data_yaml.exists())
        head = f"<div style='margin-bottom:4px'>Data folder (chosen in Step 1): <b>{root}</b></div>"
        tables = {
            "convert": [json_row, img_row, lbl_row],
            "split": [img_row, lbl_row, yaml_row],
        }
        for step, lbl in labels.items():
            rows = "".join(tables.get(step, []))
            lbl.setText(head + f"<table cellspacing='0' cellpadding='2'>{rows}</table>")
        if hasattr(self, "_split_img_count_cache"):
            self._split_img_count_cache = None
            self._update_ratio_summary()

    def _on_run_convert(self):
        lay = self._ds()
        mode = "detect" if self.rb_mode_det.isChecked() else "segment"
        if not lay.coco_json.exists():
            QMessageBox.warning(self, "Annotations Not Found",
                                f"COCO annotations not found:\n{lay.coco_json}\n\nDownload data in Step 1 first.")
            return
        n_old = len(lay.label_files())
        if n_old:
            ans = QMessageBox.question(
                self, "Replace Labels",
                f"labels/ already has {n_old} label files.\nThey will be deleted and created again in "
                f"[{mode.upper()}] mode.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Yes)
            if ans != QMessageBox.Yes:
                return

        self.btn_run_convert.setEnabled(False)
        self.prog_convert.setMaximum(100)
        self.prog_convert.setValue(0)
        self.lbl_conv_stats.setText(f"Converting in [{mode.upper()}] mode...")
        self.studio_mascot.set_state(DancingSvgMascot.STATE_DOWNLOADING)

        self.active_worker = ConvertWorker(self.converter_engine, lay.root, mode)
        self.active_worker.progress_signal.connect(lambda c, t, d: (self.prog_convert.setMaximum(max(t, 1)), self.prog_convert.setValue(c), self.lbl_conv_stats.setText(d)))
        self.active_worker.finished_signal.connect(self._on_convert_finished)
        self.active_worker.error_signal.connect(self._on_convert_error)
        self.active_worker.start()

    def _on_convert_finished(self, stats: dict):
        self.btn_run_convert.setEnabled(True)
        self.studio_mascot.set_state(DancingSvgMascot.STATE_SUCCESS)
        self.prog_convert.setMaximum(100)   # max was the image count during conversion
        self.prog_convert.setValue(100)

        self.last_class_names = list(stats.get("classes") or []) or None
        summary = (
            f"Images: {stats.get('total_images', 0)} | "
            f"With labels: {stats.get('images_with_labels', 0)} | "
            f"Annotations: {stats.get('total_annotations_written', 0)} | "
            f"Classes ({stats.get('num_classes', 0)}): {', '.join(stats.get('classes', []))} | "
            f"Mode: {stats.get('mode', 'unknown')}"
        )
        if stats.get("already_split"):
            summary += f" | Labels placed in labels/train|val|test ({stats.get('moved_into_splits', 0)})"
        self.lbl_conv_stats.setText(summary)
        out = self._ds().labels
        self.notify.add("success", "COCO → YOLO finished", summary, {"type": "folder", "path": str(out)})
        self._refresh_data_info()
        QMessageBox.information(self, "Conversion Complete", f"COCO to YOLO Conversion Completed!\n\n{summary}\nOutput: {out}")

    def _on_convert_error(self, err_msg: str):
        self.notify.add("error", "Conversion failed", err_msg[:300])
        self.btn_run_convert.setEnabled(True)
        self.studio_mascot.set_state(DancingSvgMascot.STATE_ERROR)
        self.lbl_conv_stats.setText(f"Conversion Error: {err_msg}")
        QMessageBox.critical(self, "Conversion Error", f"Conversion failed:\n{err_msg}")

    # -----------------------------------------------------------------
    # STEP 4: Split Slots
    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    # Split ratios: always sum to 100%
    # -----------------------------------------------------------------
    def _set_ratio(self, key: str, value: int):
        slider, spin = self._ratio_widgets[key]
        for w in (slider, spin):
            w.blockSignals(True)
            w.setValue(int(value))
            w.blockSignals(False)

    def _on_ratio_changed(self, changed: str, value: int):
        """Re-balances the other two ratios so train + val + test = 100.
        train changed -> val & test share the rest (keeping their proportion)
        val changed   -> test absorbs the difference, then train if test hits its limit
        test changed  -> val absorbs the difference, then train if val hits its limit
        """
        if self._ratio_busy:
            return
        self._ratio_busy = True
        try:
            self._set_ratio(changed, value)  # keep slider & spin box in sync
            t, v, te = self.spin_train.value(), self.spin_val.value(), self.spin_test.value()
            v_lo, v_hi = self.spin_val.minimum(), self.spin_val.maximum()
            te_lo, te_hi = self.spin_test.minimum(), self.spin_test.maximum()

            if changed == "train":
                rest = 100 - t
                share = v / (v + te) if (v + te) > 0 else 1.0
                v = round(rest * share)
                v = max(v_lo, rest - te_hi, min(v, v_hi, rest - te_lo))
                te = rest - v
            elif changed == "val":
                te = 100 - t - v
                if te < te_lo:
                    te = te_lo
                elif te > te_hi:
                    te = te_hi
                t = 100 - v - te
            else:  # test
                v = 100 - t - te
                if v < v_lo:
                    v = v_lo
                elif v > v_hi:
                    v = v_hi
                t = 100 - v - te

            self._set_ratio("train", t)
            self._set_ratio("val", v)
            self._set_ratio("test", te)
        finally:
            self._ratio_busy = False
        self._update_ratio_summary()

    def _update_ratio_summary(self):
        """Shows the total and how many images each split will get (same rounding as the splitter)."""
        t, v, te = self.spin_train.value(), self.spin_val.value(), self.spin_test.value()
        total = t + v + te
        src = str(self._ds().root) if hasattr(self, "txt_dl_folder") else ""
        if not self._split_img_count_cache or self._split_img_count_cache[0] != src:
            try:
                n_imgs = len(self._ds().image_locations()) if src else 0
            except Exception:
                n_imgs = 0
            self._split_img_count_cache = (src, n_imgs)
        n = self._split_img_count_cache[1]
        text = f"Total: {total}%"
        if n:
            n_train, n_val = int(n * t / 100), int(n * v / 100)
            if v > 0 and n_val == 0 and n >= 2:
                n_val = 1
                n_train = max(0, n_train - 1)
            n_test = n - n_train - n_val
            text += f"   →   Train {n_train} · Val {n_val} · Test {n_test}   (of {n} images)"
        self.lbl_ratio_total.setText(text)

    def _on_run_split(self):
        lay = self._ds()
        if not lay.image_locations():
            QMessageBox.warning(self, "Images Not Found", f"No images in:\n{lay.images}\n\nDownload data in Step 1 first.")
            return
        if not lay.label_files():
            QMessageBox.warning(self, "Labels Not Found", f"No YOLO labels in:\n{lay.labels}\n\nRun COCO to YOLO first.")
            return

        r_train = self.spin_train.value() / 100.0
        r_val = self.spin_val.value() / 100.0
        r_test = self.spin_test.value() / 100.0
        total_r = r_train + r_val + r_test
        if abs(total_r - 1.0) > 0.01:
            QMessageBox.warning(self, "Invalid Ratios", f"Train + Val + Test ratios must sum to 100% (currently {int(total_r*100)}%).")
            return
        if lay.is_split():
            ans = QMessageBox.question(
                self, "Split Again",
                "This data folder is already split.\nAll images will be re-shuffled into train/val/test with the "
                f"new ratios ({self.spin_train.value()}/{self.spin_val.value()}/{self.spin_test.value()}).\n\nContinue?",
                QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Yes)
            if ans != QMessageBox.Yes:
                return

        self.btn_run_split.setEnabled(False)
        self.prog_split.setMaximum(100)
        self.prog_split.setValue(0)
        self.lbl_split_stats.setText("Splitting dataset in place (all images included)...")
        self.studio_mascot.set_state(DancingSvgMascot.STATE_DOWNLOADING)

        # class names always come from the COCO file -> same ids the converter wrote
        self.active_worker = SplitWorker(self.splitter_engine, lay.root, (r_train, r_val, r_test))
        self.active_worker.progress_signal.connect(lambda c, t, d: (self.prog_split.setMaximum(max(t, 1)), self.prog_split.setValue(c), self.lbl_split_stats.setText(d)))
        self.active_worker.finished_signal.connect(self._on_split_finished)
        self.active_worker.error_signal.connect(self._on_split_error)
        self.active_worker.start()

    def _on_split_finished(self, stats: dict):
        self.btn_run_split.setEnabled(True)
        self.studio_mascot.set_state(DancingSvgMascot.STATE_SUCCESS)
        self.prog_split.setMaximum(100)
        self.prog_split.setValue(100)

        summary = (
            f"Train: {stats.get('train_count', 0)} | "
            f"Val: {stats.get('val_count', 0)} | "
            f"Test: {stats.get('test_count', 0)} | "
            f"Total: {stats.get('total_images', 0)} images | "
            f"data.yaml generated!"
        )
        self.lbl_split_stats.setText(summary)
        self.notify.add("success", "Dataset split finished", summary,
                        {"type": "file", "path": str(stats.get("data_yaml_path", "")),
                         "fallback": str(self._ds().root)})
        self.txt_data_yaml.setText(str(self._ds().data_yaml))
        self._refresh_yaml_preview()
        self._refresh_data_info()
        QMessageBox.information(self, "Split Complete", f"Dataset Partition Completed!\n\n{summary}\nOutput: {self._ds().root}")

    def _on_split_error(self, err_msg: str):
        self.notify.add("error", "Split failed", err_msg[:300])
        self.btn_run_split.setEnabled(True)
        self.studio_mascot.set_state(DancingSvgMascot.STATE_ERROR)
        self.lbl_split_stats.setText(f"Split Error: {err_msg}")
        QMessageBox.critical(self, "Split Error", f"Split failed:\n{err_msg}")

    # -----------------------------------------------------------------
    # STEP 5: YOLO Auto-Trainer Slots
    # -----------------------------------------------------------------
    def _update_model_presets(self):
        task = self.cmb_train_task.currentText()
        self.cmb_model_preset.clear()
        if task == "detect":
            presets = [
                "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt",
                "yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
                "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"
            ]
        else:
            presets = [
                "yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt",
                "yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt",
                "yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt", "yolo26l-seg.pt", "yolo26x-seg.pt"
            ]
        self.cmb_model_preset.addItems(presets)
        self._update_run_preview()
        self._update_model_file_label()

    def _selected_base_model(self) -> str:
        custom = self.txt_custom_model.text().strip()
        return custom if custom else self.cmb_model_preset.currentText()

    def _update_model_file_label(self):
        if not hasattr(self, "lbl_model_file"):
            return
        custom = self.txt_custom_model.text().strip()
        if custom:
            ok = Path(custom).exists()
            self.lbl_model_file.setText(f"Base model (custom): {custom}" + ("" if ok else "   ⚠ file not found"))
            return
        name = self.cmb_model_preset.currentText()
        if not name:
            return
        path = Path(resolve_model_path(name, self.models_dir))
        if path.exists():
            self.lbl_model_file.setText(f"Base model: {path}   ✓ downloaded ({path.stat().st_size / 1e6:.0f} MB)")
        else:
            self.lbl_model_file.setText(f"Base model: {path}   — will be downloaded on first training")

    def _update_run_preview(self):
        task = self.cmb_train_task.currentText()
        custom_name = self.txt_custom_run_name.text().strip() or "my_yolo_model"
        base_runs = Path(self.txt_runs_base.text().strip()) / task
        next_version_name, _ = get_next_run_version(base_runs, custom_name)

        # In-place weights renaming inside runs/detect/{custom_name}_v{num}/weights/
        self.lbl_run_preview.setText(
            f"Weights Destination: runs/{task}/{next_version_name}/weights/{next_version_name}.pt (and last.pt)"
        )

    def _on_auto_tune_clicked(self):
        yaml_path = self._ds().data_yaml
        if not yaml_path.exists():
            QMessageBox.warning(self, "File Not Found", f"data.yaml not found at:\n{yaml_path}")
            return

        try:
            st = dataset_stats(yaml_path)
        except Exception as e:
            QMessageBox.warning(self, "Auto-Tune", f"Could not read the dataset from:\n{yaml_path}\n\n{e}")
            return
        n_train = (st.get("splits", {}).get("train") or {}).get("images", 0)
        params = auto_tune_hyperparameters(
            num_images=n_train or st.get("total_images", 0),
            num_annotations=st.get("total_instances", 0),
            num_classes=st.get("num_classes", 0),
            is_gpu=torch.cuda.is_available(),
        )
        if params.get("optimizer") and self.cmb_optimizer.findText(params["optimizer"]) >= 0:
            self.cmb_optimizer.setCurrentText(params["optimizer"])
        self.spin_epochs.setValue(params["epochs"])
        self.spin_batch.setValue(params["batch"])
        self.spin_imgsz.setValue(params["imgsz"])
        self.spin_patience.setValue(params["patience"])
        self.spin_lr0.setValue(params["lr0"])

        QMessageBox.information(
            self,
            "Auto-Tune Completed",
            f"Recommended Hyperparameters applied based on dataset size "
            f"({n_train or st.get('total_images', 0)} train images, {st.get('total_instances', 0)} objects, "
            f"{st.get('num_classes', 0)} classes, {'GPU' if torch.cuda.is_available() else 'CPU'}):\n\n"
            f"• Epochs: {params['epochs']}\n"
            f"• Batch Size: {params['batch']}\n"
            f"• Image Size: {params['imgsz']}\n"
            f"• Early Stopping (Patience): {params['patience']}\n"
            f"• Learning Rate (lr0): {params['lr0']}\n"
            f"• Optimizer: {params.get('optimizer', 'auto')}"
        )

    def _on_start_training(self):
        yaml_path = self._ds().data_yaml
        self.txt_data_yaml.setText(str(yaml_path))
        if not yaml_path.exists():
            QMessageBox.warning(self, "Missing data.yaml",
                                f"Cannot find dataset configuration at:\n{yaml_path}\n\nRun Dataset Splitter (Step 3) first.")
            return

        task = self.cmb_train_task.currentText()
        custom_pt = self.txt_custom_model.text().strip()
        if custom_pt and not Path(custom_pt).exists():
            QMessageBox.warning(self, "Custom Model Not Found",
                                f"The custom weights file does not exist:\n{custom_pt}\n\n"
                                "Fix the path, or clear the field to use the selected preset.")
            return
        model_name = custom_pt if custom_pt else self.cmb_model_preset.currentText()
        run_name = self.txt_custom_run_name.text().strip() or "yolo_model"
        base_runs_dir = self._ds().runs
        base_runs_dir.mkdir(parents=True, exist_ok=True)

        hyperparams = {
            "epochs": self.spin_epochs.value(),
            "batch": self.spin_batch.value(),
            "imgsz": self.spin_imgsz.value(),
            "patience": self.spin_patience.value(),
            "lr0": self.spin_lr0.value(),
            "optimizer": self.cmb_optimizer.currentText(),
            "device": "0" if torch.cuda.is_available() else "cpu",
        }

        self.btn_start_train.setEnabled(False)
        self.btn_stop_train.setEnabled(True)
        self.prog_train.setMaximum(self.spin_epochs.value())
        self.prog_train.setValue(0)
        self.txt_train_log.clear()
        self.studio_mascot.set_state(DancingSvgMascot.STATE_TRAINING)

        self.active_worker = TrainWorker(
            self.trainer_engine, task, model_name, yaml_path, run_name, base_runs_dir, hyperparams,
            models_dir=self.models_dir,
        )
        self.active_worker.log_signal.connect(self._append_train_log)
        self.active_worker.epoch_signal.connect(self._on_train_epoch)
        self.active_worker.finished_signal.connect(self._on_train_finished)
        self.active_worker.error_signal.connect(self._on_train_error)
        self.active_worker.start()

    def _on_stop_training(self):
        if self.trainer_engine:
            self.trainer_engine.stop_training()
            self.btn_stop_train.setEnabled(False)
            self.txt_train_log.appendPlainText("\n⚠️ Stopping training after current batch...")

    def _append_train_log(self, msg: str):
        self.txt_train_log.appendPlainText(msg)
        self.txt_train_log.verticalScrollBar().setValue(self.txt_train_log.verticalScrollBar().maximum())

    def _on_train_epoch(self, epoch: int, total: int, metrics: dict):
        self.prog_train.setValue(epoch)
        loss = metrics.get("loss", 0.0)
        fit = metrics.get("fitness", 0.0)
        fit_str = f" | Fitness: {fit:.4f}" if fit else ""
        self.lbl_train_metrics.setText(f"Epoch {epoch}/{total} | Loss: {loss:.4f}{fit_str}")

    def _on_train_finished(self, res: dict):
        self.btn_start_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
        self.studio_mascot.set_state(DancingSvgMascot.STATE_SUCCESS)
        self.prog_train.setValue(self.prog_train.maximum())
        self._update_run_preview()
        self._update_model_file_label()

        run_dir = res.get("run_dir")
        best_pt = res.get("best_pt")
        last_pt = res.get("last_pt")

        if best_pt:
            self.last_trained_weight = Path(best_pt)
            self.txt_test_model.setText(str(best_pt))

        report = self._build_train_report(res)
        self.last_train_report = report
        self._fill_stat_grid(self.train_result_grid, report["cards"])
        self.lbl_train_result_title.setText(f"Latest Training Result — {res.get('run_name', '')}")
        self.train_result_card.show()
        QApplication.processEvents()
        scroll = self.pages.currentWidget()
        if isinstance(scroll, QScrollArea):
            scroll.ensureWidgetVisible(self.train_result_card, 0, 40)
        self._append_train_log(f"📄 Report saved: {report.get('report_md', '-')}")

        main = report.get("test_metrics") or report.get("val_metrics") or {}
        mbox = main.get("box") or {}
        which = "Test" if report.get("test_metrics") else "Val"
        stopped = res.get("stopped_by_user")
        self.notify.add(
            "warning" if stopped else "success",
            f"Training {'stopped' if stopped else 'finished'} — {res.get('run_name', '')}",
            f"{which} mAP50 {pct(mbox.get('map50'))} · Precision {pct(mbox.get('precision'))} · "
            f"Recall {pct(mbox.get('recall'))} · time {fmt_secs(res.get('seconds'))}",
            {"type": "report", "path": report.get("report_md"), "fallback": str(run_dir)},
            payload=report,
        )
        self._show_report(report)

    # -----------------------------------------------------------------
    # Reports (training + evaluation)
    # -----------------------------------------------------------------
    def _show_report(self, report: dict | None):
        if report:
            ReportDialog(report, self).exec_()

    @staticmethod
    def _fill_stat_grid(grid: QGridLayout, cards: list, per_row: int = 5):
        while grid.count():
            item = grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, c in enumerate(cards):
            grid.addWidget(StatCard(c[0], c[1], c[2] if len(c) > 2 else "", accent=(len(c) > 3 and c[3])),
                           i // per_row, i % per_row)

    @staticmethod
    def _acc_row(name: str, m: dict | None) -> list | None:
        if not m:
            return None
        box = m.get("box") or {}
        return [name, m.get("images", "-"), m.get("instances", "-"), pct(box.get("precision")), pct(box.get("recall")),
                pct(box.get("f1")), pct(box.get("map50")), pct(box.get("map50_95")), pct(m.get("accuracy"))]

    @staticmethod
    def _per_class_table(m: dict | None, ds: dict | None) -> dict | None:
        if not m or not m.get("per_class"):
            return None
        pc_inst = (ds or {}).get("per_class", {})
        has_ds = bool(pc_inst)
        headers = ["Class"] + (["Train inst.", "Val inst.", "Test inst."] if has_ds else ["Instances"]) + \
                  ["Precision", "Recall", "F1", "mAP50", "mAP50-95"]
        rows = []
        for r in m["per_class"]:
            name = r["class"]
            if has_ds:
                inst = [pc_inst.get(name, {}).get(k, 0) for k in ("train", "val", "test")]
            else:
                inst = [m.get("per_class_instances", {}).get(name, 0)]
            rows.append([name] + inst + [pct(r["precision"]), pct(r["recall"]), pct(r.get("f1")),
                                         pct(r["map50"]), pct(r["map50_95"])])
        return {"headers": headers, "rows": rows}

    def _build_train_report(self, res: dict) -> dict:
        rd = res.get("report_data") or {}
        ds, tr = rd.get("dataset") or {}, rd.get("training") or {}
        val, test = rd.get("val_metrics"), rd.get("test_metrics")
        hp = res.get("hyperparams") or {}
        main = test or val or {"box": (tr.get("best") or {}).get("box", {})}
        mbox = main.get("box") or {}
        main_name = "Test" if test else "Val"

        sp = ds.get("splits", {})
        split_txt = " · ".join(f"{k} {v['images']}" + (" (=val)" if v.get("same_as") else "") for k, v in sp.items())
        planned = hp.get("epochs", "-")
        done = tr.get("epochs_completed", 0)
        early = ""
        if res.get("stopped_by_user"):
            early = " · stopped by user"
        elif isinstance(planned, int) and done and done < planned:
            early = f" · early stop (patience {hp.get('patience', '-')})"
        names = ds.get("names", [])

        cards = [
            ("Images", str(ds.get("total_images", "-")), split_txt),
            ("Labels (instances)", str(ds.get("total_instances", "-")),
             f"{ds.get('total_labelled_images', 0)} labelled images"),
            ("Classes", str(len(names)), ", ".join(names)[:70]),
            ("Epochs", f"{done}/{planned}", f"best epoch {tr.get('best_epoch', '-')}{early}"),
            ("Training time", fmt_secs(res.get("seconds")), f"{hp.get('device', '')} · batch {hp.get('batch', '-')} · img {hp.get('imgsz', '-')}"),
            (f"{main_name} mAP50", pct(mbox.get("map50")), f"mAP50-95 {pct(mbox.get('map50_95'))}", True),
            (f"{main_name} Precision", pct(mbox.get("precision"))),
            (f"{main_name} Recall", pct(mbox.get("recall"))),
            (f"{main_name} F1", pct(mbox.get("f1"))),
            (f"{main_name} Accuracy*", pct(main.get("accuracy")), "TP / (TP+FP+FN)"),
        ]

        acc_rows = [r for r in (self._acc_row("Validation split", val), self._acc_row("Test split", test)) if r]
        if not acc_rows and tr.get("best"):
            b = tr["best"].get("box", {})
            acc_rows = [["Validation (results.csv)", "-", "-", pct(b.get("precision")), pct(b.get("recall")),
                         pct(b.get("f1")), pct(b.get("map50")), pct(b.get("map50_95")), "-"]]

        hist = tr.get("history", {})
        loss = {k.replace("/", " "): hist[k] for k in ("train/box_loss", "val/box_loss", "train/cls_loss",
                                                       "val/cls_loss", "train/seg_loss", "val/seg_loss") if k in hist}
        metr = {lbl: hist[k] for lbl, k in (("Precision", "metrics/precision(B)"), ("Recall", "metrics/recall(B)"),
                                            ("mAP50", "metrics/mAP50(B)"), ("mAP50-95", "metrics/mAP50-95(B)"),
                                            ("Mask mAP50", "metrics/mAP50(M)")) if k in hist}
        plots = dict(tr.get("plots", {}))
        for src in (val, test):  # evaluation plots (confusion matrix / curves) override training ones
            if src:
                plots.update({k: v for k, v in (src.get("plots") or {}).items() if k not in ("results", "labels")})

        report = {
            "title": f"Training Report — {res.get('run_name', '')}",
            "subtitle": f"{Path(str(res.get('model', ''))).name} · {res.get('task', '')} · data: {res.get('data_yaml', '')}",
            "cards": cards,
            "accuracy_rows": acc_rows,
            "per_class": self._per_class_table(test or val, ds),
            "curves": {"loss": loss, "metrics": metr, "best_epoch": tr.get("best_epoch")} if hist else None,
            "plots": plots,
            "confusion": (test or val or {}).get("confusion"),
            "warnings": rd.get("warnings", []),
            "folder": str(res.get("run_dir", "")),
            "dataset": ds, "val_metrics": val, "test_metrics": test,
            "best_pt": str(res.get("best_pt") or ""),
        }
        try:
            files = write_report_files({k: v for k, v in report.items() if k != "curves"}, Path(res["run_dir"]))
            report["report_md"] = files["md"]
        except Exception as e:
            report["warnings"].append(f"Could not write report files: {e}")
        return report

    def _build_eval_report(self, ev: dict, model_path: Path) -> dict:
        box = ev.get("box") or {}
        warnings = []
        if ev.get("labelled_images", 0) < ev.get("images", 0):
            warnings.append(f"{ev['images'] - ev['labelled_images']} of {ev['images']} images have no label — "
                            f"they are counted as background (no objects).")
        if ev.get("unknown_classes"):
            warnings.append("Classes in the labels that the model does not know (ignored): "
                            + ", ".join(ev["unknown_classes"]))
        cards = [
            ("Images", str(ev.get("images", "-")), f"{ev.get('labelled_images', 0)} with labels"),
            ("Labels (instances)", str(ev.get("instances", "-")), ev.get("label_source") or ev.get("source", "")),
            ("mAP50", pct(box.get("map50")), f"mAP50-95 {pct(box.get('map50_95'))}", True),
            ("Precision", pct(box.get("precision"))),
            ("Recall", pct(box.get("recall"))),
            ("F1", pct(box.get("f1"))),
            ("Accuracy*", pct(ev.get("accuracy")), f"TP / (TP+FP+FN) @ conf {ev.get('cm_conf', 0.25):.2f}"),
            ("Eval time", fmt_secs(ev.get("seconds"))),
        ]
        name = "Test split" if ev.get("split") == "test" else ("Val split" if ev.get("split") == "val" else "External data")
        report = {
            "title": f"Accuracy Report — {name}",
            "subtitle": f"{model_path.name} · {ev.get('source', '')}",
            "cards": cards,
            "accuracy_rows": [self._acc_row(name, ev)],
            "per_class": self._per_class_table(ev, None),
            "plots": ev.get("plots", {}),
            "confusion": ev.get("confusion"),
            "warnings": warnings,
            "folder": ev.get("save_dir", ""),
            "eval": ev,
        }
        try:
            files = write_report_files(report, Path(ev.get("save_dir") or self._ds().test_runs))
            report["report_md"] = files["md"]
        except Exception:
            pass
        return report

    def _on_train_error(self, err_msg: str):
        self.notify.add("error", "Training failed", err_msg[:300])
        self.btn_start_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
        self.studio_mascot.set_state(DancingSvgMascot.STATE_ERROR)
        self.txt_train_log.appendPlainText(f"\n❌ Training Error: {err_msg}")
        low = err_msg.lower()
        if any(h in low for h in ("insufficient memory", "out of memory", "outofmemory", "paging file", "cannot allocate")):
            QMessageBox.critical(
                self, "Not Enough Memory",
                "Training ran out of memory, even after automatically retrying with workers = 0 and a smaller batch.\n\n"
                "Try:\n"
                "  • Lower Image Size (e.g. 640 → 512 or 416)\n"
                "  • Batch Size 2–4\n"
                "  • A smaller model (n or s instead of l / x)\n"
                "  • Close other programs (browser, other Python processes)\n"
                "  • Windows: increase the page file (Virtual memory)\n\n"
                f"Details (last line):\n{err_msg.strip().splitlines()[-1][:300]}")
        else:
            QMessageBox.critical(self, "Training Error", f"Training encountered an error:\n{err_msg}")

    def _refresh_yaml_preview(self):
        yaml_path = Path(self.txt_data_yaml.text().strip())
        if yaml_path.exists():
            try:
                self.txt_yaml_preview.setPlainText(yaml_path.read_text(encoding="utf-8"))
            except Exception as e:
                self.txt_yaml_preview.setPlainText(f"Error reading {yaml_path}:\n{e}")
        else:
            self.txt_yaml_preview.setPlainText(f"data.yaml not found at:\n{yaml_path}")

    # -----------------------------------------------------------------
    # STEP 6: Model Test / Inference Slots
    # -----------------------------------------------------------------
    def _browse_test_model_pt(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model Weights", "", "PyTorch Weights (*.pt)")
        if f:
            self.txt_test_model.setText(f)

    def _on_test_source_mode_changed(self):
        idx = self.btn_grp_source.checkedId()
        if idx == 0:
            self.lbl_test_src_path.setText("Image File Path:")
            self.txt_test_source.setPlaceholderText("Select image to evaluate (.jpg, .png, ...)")
        elif idx == 1:
            self.lbl_test_src_path.setText("Images Folder Path:")
            self.txt_test_source.setPlaceholderText("Select folder containing test images...")
        elif idx == 2:
            self.lbl_test_src_path.setText("Video File Path:")
            self.txt_test_source.setPlaceholderText("Select video to evaluate (.mp4, .avi, ...)")
        elif idx == 3:
            self.lbl_test_src_path.setText("data.yaml Path:")
            self.txt_test_source.setPlaceholderText("data.yaml created in Step 3 (uses its test split)")
            yaml_guess = str(self._ds().data_yaml)
            if yaml_guess and (not self.txt_test_source.text().strip().endswith((".yaml", ".yml"))):
                self.txt_test_source.setText(yaml_guess)
        self.chk_eval_labels.setEnabled(idx in (0, 1))

    def _get_current_test_label_mode(self) -> str:
        """Returns active label mode: 'name' (class name), 'id' (numeric class id), or 'none' (no label)."""
        idx = self.btn_grp_label.checkedId()
        return {0: "name", 1: "id", 2: "none"}.get(idx, "name")

    def _on_test_label_mode_changed(self):
        """Called whenever user toggles between Class Name, Class ID, and No Label."""
        mode = self._get_current_test_label_mode()
        self.tester_engine.set_visual_settings(label_mode=mode)
        self._refresh_current_test_preview()

    def _on_test_font_size_changed(self, val: int):
        """Called whenever user adjusts font size slider/spinbox (even while predicting)."""
        self.tester_engine.set_visual_settings(font_size=val)
        self._refresh_current_test_preview()

    def _refresh_current_test_preview(self):
        """Immediately re-renders the current visible test image or frame with updated font size / label mode."""
        mode = self._get_current_test_label_mode()
        fs = self.slider_test_font.value()

        # 1. Single image mode
        if getattr(self, "single_test_result", None) and "raw_result" in self.single_test_result:
            res = self.single_test_result["raw_result"]
            new_bgr = self.tester_engine.plot_result(res, label_mode=mode, font_size=fs)
            self.single_test_result["annotated_bgr"] = new_bgr
            pix = bgr_to_qpixmap(new_bgr)
            if not pix.isNull():
                scaled = pix.scaled(self.lbl_test_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.lbl_test_preview.setPixmap(scaled)
            return

        # 2. Folder / Split mode
        if self.folder_test_results and 0 <= self.current_folder_test_idx < len(self.folder_test_results):
            item = self.folder_test_results[self.current_folder_test_idx]
            res = item.get("raw_result")
            if res is not None:
                new_bgr = self.tester_engine.plot_result(res, label_mode=mode, font_size=fs)
                item["annotated_bgr"] = new_bgr
                pix = bgr_to_qpixmap(new_bgr)
                if not pix.isNull():
                    scaled = pix.scaled(self.lbl_test_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.lbl_test_preview.setPixmap(scaled)
            return

        # 3. Video mode (when finished or paused)
        if getattr(self, "last_video_raw_res", None) is not None:
            res = self.last_video_raw_res
            new_bgr = self.tester_engine.plot_result(res, label_mode=mode, font_size=fs)
            pix = bgr_to_qpixmap(new_bgr)
            if not pix.isNull():
                scaled = pix.scaled(self.lbl_test_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.lbl_test_preview.setPixmap(scaled)
            return

    def _browse_test_source(self):
        idx = self.btn_grp_source.checkedId()
        if idx == 0:
            f, _ = QFileDialog.getOpenFileName(self, "Select Test Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp *.webp)")
            if f:
                self.txt_test_source.setText(f)
        elif idx == 1:
            d = QFileDialog.getExistingDirectory(self, "Select Folder of Images")
            if d:
                self.txt_test_source.setText(d)
        elif idx == 2:
            f, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv *.webm)")
            if f:
                self.txt_test_source.setText(f)
        elif idx == 3:
            f, _ = QFileDialog.getOpenFileName(self, "Select data.yaml", str(self._ds().root), "YAML Files (*.yaml *.yml)")
            if f:
                self.txt_test_source.setText(f)

    def _on_run_test(self):
        model_p = Path(self.txt_test_model.text().strip())
        src_p = Path(self.txt_test_source.text().strip())
        save_enabled = self.chk_save_test.isChecked()
        save_dir = self._ds().test_runs
        save_dir.mkdir(parents=True, exist_ok=True)
        self.txt_test_save_dir.setText(str(save_dir))

        if not model_p.exists():
            QMessageBox.warning(self, "Model Not Found", f"Model weights not found:\n{model_p}")
            return

        if not src_p.exists():
            QMessageBox.warning(self, "Source Not Found", f"Input test source not found:\n{src_p}")
            return

        mode_id = self.btn_grp_source.checkedId()
        mode_str = {0: "image", 1: "folder", 2: "video", 3: "split"}.get(mode_id, "image")
        if mode_str == "split" and src_p.suffix.lower() not in (".yaml", ".yml"):
            QMessageBox.warning(self, "data.yaml Required", "Please select the data.yaml created in Step 3.")
            return
        self.test_eval_card.hide()
        self.last_eval_report = None
        self.single_test_result = None
        self.last_video_result = None
        self.last_video_raw_res = None
        self.folder_test_results = []

        conf = self.spin_test_conf.value()
        iou = self.spin_test_iou.value()

        # Sync live visual settings to engine
        self.tester_engine.set_visual_settings(
            label_mode=self._get_current_test_label_mode(),
            font_size=self.slider_test_font.value(),
        )

        self.btn_run_test.setEnabled(False)
        self.btn_stop_test.setEnabled(True)
        self.prog_test.setValue(0)
        self.lbl_test_status.setText(f"Running inference ({mode_str.upper()})...")
        self.studio_mascot.set_state(DancingSvgMascot.STATE_DOWNLOADING)

        evaluate = mode_str == "split" or (mode_str in ("image", "folder") and self.chk_eval_labels.isChecked())
        self.active_worker = TestInferenceWorker(
            self.tester_engine, mode_str, model_p, src_p, conf, iou, save_enabled, save_dir,
            evaluate=evaluate, eval_dir=save_dir,
        )
        self.active_worker.progress_signal.connect(self._on_test_progress)
        self.active_worker.frame_signal.connect(self._on_test_frame)
        self.active_worker.finished_signal.connect(self._on_test_finished)
        self.active_worker.error_signal.connect(self._on_test_error)
        self.active_worker.start()

    def _on_stop_test(self):
        if self.tester_engine:
            self.tester_engine.stop()
            self.btn_stop_test.setEnabled(False)
            self.lbl_test_status.setText("Aborting inference...")

    def _on_test_progress(self, cur: int, tot: int, desc: str):
        self.prog_test.setMaximum(tot)  # tot == 0 -> busy indicator while measuring accuracy
        self.prog_test.setValue(cur)
        self.lbl_test_status.setText(desc)

    def _on_test_frame(self, frame_bgr: np.ndarray, cur: int, tot: int, summary: dict):
        pix = bgr_to_qpixmap(frame_bgr)
        if not pix.isNull():
            scaled = pix.scaled(self.lbl_test_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_test_preview.setPixmap(scaled)

        mode_name = getattr(self.active_worker, "mode", "frame")
        prefix = "Image" if mode_name in ("folder", "split") else "Frame"
        self.lbl_test_img_counter.setText(f"{prefix} {cur} / {tot}")
        badge_text = ", ".join([f"{k}: {v}" for k, v in summary.items()]) or "No objects detected in frame"
        self.lbl_test_classes_badge.setText(f"Active Objects: {badge_text}")

    def _on_test_finished(self, out: dict):
        self.btn_run_test.setEnabled(True)
        self.btn_stop_test.setEnabled(False)
        self.studio_mascot.set_state(DancingSvgMascot.STATE_SUCCESS)
        self.prog_test.setMaximum(100)
        self.prog_test.setValue(100)

        mode = out.get("mode")
        data = out.get("data", {})
        if mode == "split":
            mode = "folder"

        if mode == "image":
            self.single_test_result = data
            annotated_bgr = data.get("annotated_bgr")
            pix = bgr_to_qpixmap(annotated_bgr)
            if not pix.isNull():
                scaled = pix.scaled(self.lbl_test_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.lbl_test_preview.setPixmap(scaled)

            self.lbl_test_img_counter.setText("Image 1 / 1")
            self.btn_prev_test_img.setEnabled(False)
            self.btn_next_test_img.setEnabled(False)

            detections = data.get("detections", [])
            self._populate_detections_table(detections)

            cls_sum = data.get("classes_summary", {})
            badge_str = ", ".join([f"{k} ({v})" for k, v in cls_sum.items()]) or "No objects detected"
            saved_msg = f" | Saved: {data.get('saved_path')}" if data.get('saved_path') else ""
            self.lbl_test_classes_badge.setText(f"Found: {badge_str}{saved_msg}")
            self.lbl_test_status.setText(f"Single image test complete! Found {len(detections)} objects.{saved_msg}")

        elif mode == "folder":
            self.folder_test_results = data.get("results", [])
            self.current_folder_test_idx = 0

            if self.folder_test_results:
                self._show_folder_test_item(0)
                self.btn_prev_test_img.setEnabled(False)
                self.btn_next_test_img.setEnabled(len(self.folder_test_results) > 1)
            else:
                self.lbl_test_preview.setText("No images evaluated.")

            cls_sum = data.get("classes_summary", {})
            badge_str = ", ".join([f"{k}: {v}" for k, v in cls_sum.items()]) or "No objects detected"
            saved_msg = f" | Results in: {data.get('saved_dir')}" if data.get('saved_dir') else ""
            self.lbl_test_classes_badge.setText(f"Folder Summary: {badge_str}")
            self.lbl_test_status.setText(f"Batch test complete! Processed {len(self.folder_test_results)} images.{saved_msg}")

        elif mode == "video":
            self.last_video_result = data
            self.last_video_raw_res = data.get("last_result")
            annotated_bgr = data.get("last_annotated")
            if annotated_bgr is not None:
                pix = bgr_to_qpixmap(annotated_bgr)
                if not pix.isNull():
                    scaled = pix.scaled(self.lbl_test_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.lbl_test_preview.setPixmap(scaled)

            cls_sum = data.get("classes_summary", {})
            badge_str = ", ".join([f"{k}: {v}" for k, v in cls_sum.items()]) or "No objects detected"
            saved_msg = f"\nSaved video to: {data.get('saved_path')}" if data.get('saved_path') else ""
            self.lbl_test_classes_badge.setText(f"Video Summary: {badge_str}")
            self.lbl_test_status.setText(f"Video test complete! Processed {data.get('processed_frames')} frames.{saved_msg}")

        self._show_test_eval(out.get("eval"), Path(self.txt_test_model.text().strip()))

    def _show_test_eval(self, ev: dict | None, model_path: Path):
        """Shows accuracy when the tested data had labels; otherwise explains it was prediction-only."""
        self._notify_test(ev, model_path)
        if ev and ev.get("error"):
            self.test_eval_card.hide()
            QMessageBox.warning(self, "Accuracy Not Measured",
                                f"Predictions finished, but measuring accuracy failed:\n{ev['error']}")
            return
        if not ev:
            self.test_eval_card.hide()
            note = ("No labels were found, so this was prediction only (no accuracy)."
                    if self.chk_eval_labels.isChecked() and self.btn_grp_source.checkedId() in (0, 1) else
                    "Prediction only.")
            self.lbl_test_status.setText(self.lbl_test_status.text() + "  •  " + note)
            QMessageBox.information(self, "Test Complete", f"Model prediction completed.\n{note}")
            return

        report = getattr(self, "_pending_eval_report", None) or self._build_eval_report(ev, model_path)
        self._pending_eval_report = None
        self.last_eval_report = report
        box = ev.get("box") or {}
        self.lbl_test_eval_title.setText(f"Accuracy — {report['accuracy_rows'][0][0]}")
        self._fill_stat_grid(self.test_eval_grid, report["cards"][:7], per_row=7)
        self.lbl_test_eval_note.setText("  ".join(report.get("warnings", [])) or
                                        f"Labels: {ev.get('label_source') or ev.get('source', '')}")
        self.test_eval_card.show()
        QApplication.processEvents()
        scroll = self.pages.currentWidget()
        if isinstance(scroll, QScrollArea):
            scroll.ensureWidgetVisible(self.test_eval_card, 0, 40)
        QMessageBox.information(
            self, "Test Complete",
            f"Accuracy on {ev.get('images', 0)} images ({ev.get('instances', 0)} labelled objects):\n\n"
            f"mAP50: {pct(box.get('map50'))}    mAP50-95: {pct(box.get('map50_95'))}\n"
            f"Precision: {pct(box.get('precision'))}    Recall: {pct(box.get('recall'))}\n"
            f"Accuracy*: {pct(ev.get('accuracy'))}\n\nOpen 'View Accuracy Report' for the confusion matrix and curves.")

    def _show_folder_test_item(self, idx: int):
        if not self.folder_test_results or idx < 0 or idx >= len(self.folder_test_results):
            return

        self.current_folder_test_idx = idx
        item = self.folder_test_results[idx]

        res = item.get("raw_result")
        if res is not None:
            mode = self._get_current_test_label_mode()
            fs = self.slider_test_font.value()
            annotated_bgr = self.tester_engine.plot_result(res, label_mode=mode, font_size=fs)
            item["annotated_bgr"] = annotated_bgr
        else:
            annotated_bgr = item.get("annotated_bgr")

        pix = bgr_to_qpixmap(annotated_bgr)
        if not pix.isNull():
            scaled = pix.scaled(self.lbl_test_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_test_preview.setPixmap(scaled)

        total = len(self.folder_test_results)
        self.lbl_test_img_counter.setText(f"Image {idx + 1} / {total} ({item.get('file_name')})")
        self.btn_prev_test_img.setEnabled(idx > 0)
        self.btn_next_test_img.setEnabled(idx < total - 1)

        detections = item.get("detections", [])
        self._populate_detections_table(detections)

    def _on_prev_test_img(self):
        if self.current_folder_test_idx > 0:
            self._show_folder_test_item(self.current_folder_test_idx - 1)

    def _on_next_test_img(self):
        if self.current_folder_test_idx < len(self.folder_test_results) - 1:
            self._show_folder_test_item(self.current_folder_test_idx + 1)

    def _populate_detections_table(self, detections: list[dict]):
        self.tbl_test_detections.setRowCount(0)
        self.tbl_test_detections.setRowCount(len(detections))

        for row, det in enumerate(detections):
            c_item = QTableWidgetItem(str(det["class_name"]))
            c_item.setTextAlignment(Qt.AlignCenter)
            self.tbl_test_detections.setItem(row, 0, c_item)

            conf_item = QTableWidgetItem(f"{det['conf']:.2%}")
            conf_item.setTextAlignment(Qt.AlignCenter)
            self.tbl_test_detections.setItem(row, 1, conf_item)

            coords = det.get("box", [])
            coord_str = f"[{int(coords[0])}, {int(coords[1])}, {int(coords[2])}, {int(coords[3])}]" if len(coords) >= 4 else "-"
            box_item = QTableWidgetItem(coord_str)
            box_item.setTextAlignment(Qt.AlignCenter)
            self.tbl_test_detections.setItem(row, 2, box_item)

            id_item = QTableWidgetItem(str(det.get("id", row)))
            id_item.setTextAlignment(Qt.AlignCenter)
            self.tbl_test_detections.setItem(row, 3, id_item)

    def _notify_test(self, ev: dict | None, model_path: Path):
        save_dir = self.txt_test_save_dir.text().strip()
        if ev and not ev.get("error"):
            box = ev.get("box") or {}
            report = self._build_eval_report(ev, model_path)
            self._pending_eval_report = report
            self.notify.add("success", f"Test finished — {report['accuracy_rows'][0][0]}",
                            f"{ev.get('images', 0)} images · mAP50 {pct(box.get('map50'))} · "
                            f"Accuracy {pct(ev.get('accuracy'))}",
                            {"type": "report", "path": report.get("report_md"), "fallback": ev.get("save_dir")},
                            payload=report)
        else:
            note = "accuracy not measured: " + ev["error"][:120] if (ev and ev.get("error")) else "prediction only (no labels)"
            self.notify.add("info", "Test finished", f"{model_path.name} · {note}",
                            {"type": "folder", "path": save_dir})

    def _on_test_error(self, err_msg: str):
        self.notify.add("error", "Test failed", err_msg[:300])
        self.btn_run_test.setEnabled(True)
        self.btn_stop_test.setEnabled(False)
        self.studio_mascot.set_state(DancingSvgMascot.STATE_ERROR)
        self.lbl_test_status.setText(f"Test Error: {err_msg}")
        QMessageBox.critical(self, "Inference Error", f"Model testing encountered an error:\n{err_msg}")

    # -----------------------------------------------------------------
    # Helper File & Folder Browsers
    # -----------------------------------------------------------------
    def _browse_folder(self, target_line_edit: QLineEdit):
        folder = QFileDialog.getExistingDirectory(self, "Select Directory", target_line_edit.text())
        if folder:
            target_line_edit.setText(folder)

    def _browse_file(self, target_line_edit: QLineEdit, filter_str: str = "All Files (*)"):
        f, _ = QFileDialog.getOpenFileName(self, "Select File", target_line_edit.text(), filter_str)
        if f:
            target_line_edit.setText(f)

    def _browse_model_pt(self):
        start = str(self.models_dir if self.models_dir.exists() else self._ds().runs)
        f, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model Weights", start, "PyTorch Weights (*.pt)")
        if f:
            self.txt_custom_model.setText(f)

    def _browse_data_yaml(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select data.yaml", self.txt_data_yaml.text(), "YAML Files (*.yaml *.yml)")
        if f:
            self.txt_data_yaml.setText(f)
            self._refresh_yaml_preview()