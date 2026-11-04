"""
Main Window for the CVAT-to-YOLO GUI application.
Provides a comprehensive end-to-end interface:
1. CVAT Authentication & Download (COCO 1.0 + unzipping + cleanup default folder)
2. Task Merging & COCO-to-YOLO Conversion (Detection & Segmentation)
3. Train/Val/Test Splitter (including all background images)
4. YOLO Training with Auto-Tuning, Early Stopping, and Auto-Versioning
5. Outputs Explorer
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
import torch

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QTabWidget, QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox,
    QDoubleSpinBox, QSlider, QCheckBox, QRadioButton, QButtonGroup,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox,
    QProgressBar, QPlainTextEdit, QFrame, QSplitter, QGroupBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QFont, QColor

from core.cvat_client import CvatClientWrapper, sanitize_filename
from core.merger import CocoDatasetMerger
from core.converter import CocoToYoloConverter
from core.splitter import DatasetSplitter
from core.trainer import YoloTrainerEngine, auto_tune_hyperparameters, get_next_run_version
from ui.dancing_svg import DancingSvgMascot, AnimatedWavePill
from ui.theme import DARK_STYLESHEET


# =====================================================================
# Background Worker Threads
# =====================================================================

class CvatFetchWorker(QThread):
    finished_signal = pyqtSignal(list, list)  # projects, tasks
    error_signal = pyqtSignal(str)

    def __init__(self, client: CvatClientWrapper, project_id=None, stage=None, state=None, status=None):
        super().__init__()
        self.client = client
        self.project_id = project_id
        self.stage = stage
        self.state = state
        self.status = status

    def run(self):
        try:
            projects = self.client.refresh_projects()
            tasks = self.client.get_tasks(
                project_id=self.project_id,
                stage_filter=self.stage,
                state_filter=self.state,
                status_filter=self.status,
            )
            self.finished_signal.emit(projects, tasks)
        except Exception as e:
            self.error_signal.emit(str(e))


class CvatDownloadWorker(QThread):
    progress_signal = pyqtSignal(int, int, str)
    task_completed_signal = pyqtSignal(int, str, str)  # task_id, name, folder
    all_completed_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)

    def __init__(self, client: CvatClientWrapper, selected_tasks: list[dict], dest_dir: Path):
        super().__init__()
        self.client = client
        self.selected_tasks = selected_tasks
        self.dest_dir = dest_dir
        self.is_cancelled = False

    def run(self):
        try:
            completed_folders = []
            total = len(self.selected_tasks)

            for idx, task in enumerate(self.selected_tasks):
                if self.is_cancelled:
                    break

                t_id = task["id"]
                t_name = task["name"]

                self.progress_signal.emit(
                    int((idx / total) * 100), 100,
                    f"Processing task {idx+1}/{total}: [{t_id}] {t_name}..."
                )

                def sub_progress(cur, tot, msg):
                    task_pct = (cur / tot) if tot > 0 else 0
                    overall = int(((idx + task_pct) / total) * 100)
                    self.progress_signal.emit(overall, 100, f"[{idx+1}/{total}] {msg}")

                out_folder = self.client.download_and_extract_task(
                    task_id=t_id,
                    output_dir=self.dest_dir,
                    progress_callback=sub_progress,
                    task_name=t_name,
                )
                completed_folders.append(out_folder)
                self.task_completed_signal.emit(t_id, t_name, str(out_folder))

            self.progress_signal.emit(100, 100, f"Successfully downloaded and cleaned {len(completed_folders)} tasks!")
            self.all_completed_signal.emit(completed_folders)
        except Exception as e:
            self.error_signal.emit(str(e))


class MergeWorker(QThread):
    progress_signal = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, source_dirs: list[Path], output_dir: Path):
        super().__init__()
        self.source_dirs = source_dirs
        self.output_dir = output_dir

    def run(self):
        try:
            merger = CocoDatasetMerger()
            res = merger.merge_datasets(
                source_dirs_or_base=self.source_dirs,
                output_dir=self.output_dir,
                progress_callback=lambda c, t, m: self.progress_signal.emit(c, t, m),
            )
            self.finished_signal.emit(res)
        except Exception as e:
            self.error_signal.emit(str(e))


class ConvertWorker(QThread):
    progress_signal = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, coco_json: Path, output_dir: Path, mode: str):
        super().__init__()
        self.coco_json = coco_json
        self.output_dir = output_dir
        self.mode = mode

    def run(self):
        try:
            converter = CocoToYoloConverter()
            res = converter.convert(
                coco_json_path=self.coco_json,
                output_labels_dir=self.output_dir,
                mode=self.mode,
                progress_callback=lambda c, t, m: self.progress_signal.emit(c, t, m),
            )
            self.finished_signal.emit(res)
        except Exception as e:
            self.error_signal.emit(str(e))


class SplitWorker(QThread):
    progress_signal = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, images_dir: Path, labels_dir: Path, out_dir: Path, coco_json: Path,
                 train_r: float, val_r: float, test_r: float, seed: int = 42):
        super().__init__()
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.out_dir = out_dir
        self.coco_json = coco_json
        self.train_r = train_r
        self.val_r = val_r
        self.test_r = test_r
        self.seed = seed

    def run(self):
        try:
            splitter = DatasetSplitter()
            res = splitter.split_dataset(
                images_dir=self.images_dir,
                labels_dir=self.labels_dir,
                output_dataset_dir=self.out_dir,
                coco_json_path=self.coco_json,
                train_ratio=self.train_r,
                val_ratio=self.val_r,
                test_ratio=self.test_r,
                seed=self.seed,
                progress_callback=lambda c, t, m: self.progress_signal.emit(c, t, m),
            )
            self.finished_signal.emit(res)
        except Exception as e:
            self.error_signal.emit(str(e))


class TrainWorker(QThread):
    log_signal = pyqtSignal(str)
    epoch_signal = pyqtSignal(int, int, dict)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, engine: YoloTrainerEngine, task: str, model_name: str,
                 data_yaml: Path, run_name: str, base_dir: Path, hyperparams: dict):
        super().__init__()
        self.engine = engine
        self.task = task
        self.model_name = model_name
        self.data_yaml = data_yaml
        self.run_name = run_name
        self.base_dir = base_dir
        self.hyperparams = hyperparams

    def run(self):
        try:
            res = self.engine.train_model(
                task=self.task,
                model_name_or_path=self.model_name,
                data_yaml_path=self.data_yaml,
                custom_run_name=self.run_name,
                base_runs_dir=self.base_dir,
                hyperparams=self.hyperparams,
                log_callback=lambda m: self.log_signal.emit(m),
                epoch_callback=lambda e, t, d: self.epoch_signal.emit(e, t, d),
            )
            self.finished_signal.emit(res)
        except Exception as e:
            self.error_signal.emit(str(e))


# =====================================================================
# Main Application Window
# =====================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CVAT to YOLO Studio • Enterprise AI Pipeline")
        self.resize(1240, 860)
        self.setMinimumSize(1000, 720)

        # Working Directories
        self.app_dir = Path("C:/Articulus/cvat").resolve()
        self.downloads_dir = self.app_dir / "downloads"
        self.merged_dir = self.app_dir / "merged"
        self.dataset_dir = self.app_dir / "dataset"
        self.runs_dir = self.app_dir / "runs"

        for d in [self.downloads_dir, self.merged_dir, self.dataset_dir, self.runs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # State Engines
        self.cvat_client = CvatClientWrapper()
        self.trainer_engine = YoloTrainerEngine()
        self.all_tasks_data: list[dict] = []

        # Active Workers
        self.active_worker = None

        self._init_ui()
        self.setStyleSheet(DARK_STYLESHEET)

    def _init_ui(self):
        central_widget = QWidget(self)
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 14, 16, 14)
        main_layout.setSpacing(12)

        # 1. Header with Animated Mascot
        main_layout.addWidget(self._build_header())

        # 2. Main Tab Widget
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_tab_cvat(), "🌐 1. CVAT Connect & Download")
        self.tabs.addTab(self._build_tab_merge_convert(), "🔄 2. Merge & Convert (COCO → YOLO)")
        self.tabs.addTab(self._build_tab_split(), "✂️ 3. Dataset Splitter")
        self.tabs.addTab(self._build_tab_train(), "🧠 4. YOLO Auto-Trainer")
        self.tabs.addTab(self._build_tab_explorer(), "📂 5. Runs & Explorer")
        main_layout.addWidget(self.tabs, stretch=1)

        # 3. Bottom Global Status Bar
        main_layout.addWidget(self._build_global_status_bar())

    # -----------------------------------------------------------------
    # UI Header
    # -----------------------------------------------------------------
    def _build_header(self) -> QFrame:
        card = QFrame()
        card.setProperty("class", "HeaderCard")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(18, 12, 18, 12)

        # Title & Subtitle
        text_layout = QVBoxLayout()
        text_layout.setSpacing(4)

        title_row = QHBoxLayout()
        title_lbl = QLabel("🚀 CVAT to YOLO Studio & Auto-Trainer")
        title_lbl.setObjectName("titleLabel")
        title_row.addWidget(title_lbl)

        # Pulsing Activity Indicator
        self.wave_indicator = AnimatedWavePill(40, 20)
        title_row.addWidget(self.wave_indicator)
        title_row.addStretch()

        text_layout.addLayout(title_row)

        subtitle_lbl = QLabel(
            "Seamless CVAT Downloader • Auto-Extract & Cleanup /images/default • "
            "COCO 1.0 Merger • Polygon-to-YOLO Converter (Det/Seg) • Background Images Splitter • YOLO Auto-Tuning Trainer"
        )
        subtitle_lbl.setObjectName("subtitleLabel")
        text_layout.addWidget(subtitle_lbl)

        # Quick hardware pill
        cuda_ok = torch.cuda.is_available()
        hw_text = f"Hardware: {'⚡ NVIDIA GPU (' + torch.cuda.get_device_name(0) + ')' if cuda_ok else '💻 CPU Mode'}"
        self.hw_badge = QLabel(hw_text)
        self.hw_badge.setProperty("class", "BadgeSuccess" if cuda_ok else "BadgeInfo")
        text_layout.addWidget(self.hw_badge)

        layout.addLayout(text_layout, stretch=1)

        # Mascot Options
        mascot_box = QVBoxLayout()
        mascot_box.setAlignment(Qt.AlignCenter)

        # Dancing SVG Mascot ("Byte")
        self.mascot = DancingSvgMascot(size=115)
        mascot_box.addWidget(self.mascot, alignment=Qt.AlignCenter)

        self.chk_mascot = QCheckBox("✨ SVG Mascot")
        self.chk_mascot.setChecked(True)
        self.chk_mascot.toggled.connect(self.mascot.toggle_animation)
        mascot_box.addWidget(self.chk_mascot, alignment=Qt.AlignCenter)

        layout.addLayout(mascot_box)

        return card

    # -----------------------------------------------------------------
    # Tab 1: CVAT Connect & Download
    # -----------------------------------------------------------------
    def _build_tab_cvat(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # CVAT Credentials Box
        auth_card = QFrame()
        auth_card.setProperty("class", "CardFrame")
        auth_layout = QGridLayout(auth_card)
        auth_layout.setContentsMargins(14, 12, 14, 12)
        auth_layout.setHorizontalSpacing(10)
        auth_layout.setVerticalSpacing(8)

        auth_layout.addWidget(QLabel("CVAT URL:"), 0, 0)
        self.txt_cvat_url = QLineEdit("http://1.10.131.222:8080/projects")
        auth_layout.addWidget(self.txt_cvat_url, 0, 1)

        auth_layout.addWidget(QLabel("Username:"), 0, 2)
        self.txt_cvat_user = QLineEdit()
        self.txt_cvat_user.setPlaceholderText("Enter username")
        auth_layout.addWidget(self.txt_cvat_user, 0, 3)

        auth_layout.addWidget(QLabel("Password:"), 0, 4)
        self.txt_cvat_pass = QLineEdit()
        self.txt_cvat_pass.setEchoMode(QLineEdit.Password)
        self.txt_cvat_pass.setPlaceholderText("Enter password")
        auth_layout.addWidget(self.txt_cvat_pass, 0, 5)

        self.btn_cvat_login = QPushButton("🔑 Connect & Fetch Tasks")
        self.btn_cvat_login.setProperty("class", "PrimaryButton")
        self.btn_cvat_login.clicked.connect(self._on_cvat_login)
        auth_layout.addWidget(self.btn_cvat_login, 0, 6)

        layout.addWidget(auth_card)

        # Filter Card (Project, Stage, State, Status, Search)
        filter_card = QFrame()
        filter_card.setProperty("class", "CardFrame")
        filter_layout = QHBoxLayout(filter_card)
        filter_layout.setContentsMargins(14, 8, 14, 8)
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
        self.txt_task_search.setPlaceholderText("Filter by task name...")
        self.txt_task_search.textChanged.connect(self._apply_task_filters)
        filter_layout.addWidget(self.txt_task_search)

        btn_select_all = QPushButton("Select All")
        btn_select_all.clicked.connect(lambda: self._set_all_tasks_checked(True))
        filter_layout.addWidget(btn_select_all)

        btn_clear_sel = QPushButton("Deselect All")
        btn_clear_sel.clicked.connect(lambda: self._set_all_tasks_checked(False))
        filter_layout.addWidget(btn_clear_sel)

        layout.addWidget(filter_card)

        # Tasks Table
        self.tbl_tasks = QTableWidget()
        self.tbl_tasks.setColumnCount(8)
        self.tbl_tasks.setHorizontalHeaderLabels([
            "Select", "Task ID", "Task Name", "Project", "Stage", "State", "Status", "Total Frames"
        ])
        self.tbl_tasks.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.tbl_tasks.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.tbl_tasks.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.tbl_tasks, stretch=1)

        # Download Destination & Action Bar
        dl_card = QFrame()
        dl_card.setProperty("class", "CardFrame")
        dl_layout = QHBoxLayout(dl_card)
        dl_layout.setContentsMargins(14, 10, 14, 10)

        dl_layout.addWidget(QLabel("Download Folder:"))
        self.txt_dl_folder = QLineEdit(str(self.downloads_dir))
        dl_layout.addWidget(self.txt_dl_folder, stretch=1)

        btn_browse_dl = QPushButton("Browse...")
        btn_browse_dl.clicked.connect(lambda: self._browse_folder(self.txt_dl_folder))
        dl_layout.addWidget(btn_browse_dl)

        self.btn_download_tasks = QPushButton("⬇️ Download & Auto-Extract Tasks (COCO 1.0)")
        self.btn_download_tasks.setProperty("class", "PrimaryButton")
        self.btn_download_tasks.clicked.connect(self._on_download_selected_tasks)
        dl_layout.addWidget(self.btn_download_tasks)

        layout.addWidget(dl_card)

        # Download Progress & Status
        self.prog_dl = QProgressBar()
        self.prog_dl.setValue(0)
        self.prog_dl.setTextVisible(True)
        layout.addWidget(self.prog_dl)

        self.lbl_cvat_status = QLabel("Ready to connect to CVAT.")
        layout.addWidget(self.lbl_cvat_status)

        return tab

    # -----------------------------------------------------------------
    # Tab 2: Merge & Convert (COCO → YOLO)
    # -----------------------------------------------------------------
    def _build_tab_merge_convert(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Card 1: Merge Downloaded Tasks
        merge_box = QGroupBox("Step 1: Merge Downloaded CVAT Tasks into Unified Dataset")
        merge_layout = QVBoxLayout(merge_box)
        merge_layout.setSpacing(10)

        row_src = QHBoxLayout()
        row_src.addWidget(QLabel("Tasks Directory:"))
        self.txt_merge_src = QLineEdit(str(self.downloads_dir))
        row_src.addWidget(self.txt_merge_src, stretch=1)
        btn_src = QPushButton("Browse...")
        btn_src.clicked.connect(lambda: self._browse_folder(self.txt_merge_src))
        row_src.addWidget(btn_src)
        merge_layout.addLayout(row_src)

        row_dst = QHBoxLayout()
        row_dst.addWidget(QLabel("Merged Output:"))
        self.txt_merge_dst = QLineEdit(str(self.merged_dir))
        row_dst.addWidget(self.txt_merge_dst, stretch=1)
        btn_dst = QPushButton("Browse...")
        btn_dst.clicked.connect(lambda: self._browse_folder(self.txt_merge_dst))
        row_dst.addWidget(btn_dst)
        merge_layout.addLayout(row_dst)

        merge_action_row = QHBoxLayout()
        self.btn_run_merge = QPushButton("🧩 Merge Datasets (Deduplicate Images & Unify Annotations)")
        self.btn_run_merge.setProperty("class", "PrimaryButton")
        self.btn_run_merge.clicked.connect(self._on_run_merge)
        merge_action_row.addWidget(self.btn_run_merge)
        merge_layout.addLayout(merge_action_row)

        self.lbl_merge_stats = QLabel("No merged dataset yet.")
        merge_layout.addWidget(self.lbl_merge_stats)

        layout.addWidget(merge_box)

        # Card 2: Convert COCO 1.0 to YOLO Format
        conv_box = QGroupBox("Step 2: Convert Unified COCO 1.0 to YOLO Format")
        conv_layout = QVBoxLayout(conv_box)
        conv_layout.setSpacing(10)

        # Task Selection: Detection vs Segmentation
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("YOLO Task Mode:"))
        self.rb_mode_det = QRadioButton("🎯 Detection (Converts Segmentation Polygons to Bounding Boxes)")
        self.rb_mode_seg = QRadioButton("🔷 Segmentation (Keeps Precise Polygons)")
        self.rb_mode_det.setChecked(True)
        self.rb_mode_det.toggled.connect(self._on_task_mode_changed)

        mode_row.addWidget(self.rb_mode_det)
        mode_row.addWidget(self.rb_mode_seg)
        mode_row.addStretch()
        conv_layout.addLayout(mode_row)

        row_lbl_dst = QHBoxLayout()
        row_lbl_dst.addWidget(QLabel("Labels Destination:"))
        self.txt_conv_labels_dst = QLineEdit(str(self.merged_dir / "labels"))
        row_lbl_dst.addWidget(self.txt_conv_labels_dst, stretch=1)
        btn_lbl_dst = QPushButton("Browse...")
        btn_lbl_dst.clicked.connect(lambda: self._browse_folder(self.txt_conv_labels_dst))
        row_lbl_dst.addWidget(btn_lbl_dst)
        conv_layout.addLayout(row_lbl_dst)

        conv_action_row = QHBoxLayout()
        self.btn_run_convert = QPushButton("⚡ Convert COCO to YOLO (.txt)")
        self.btn_run_convert.setProperty("class", "SuccessButton")
        self.btn_run_convert.clicked.connect(self._on_run_convert)
        conv_action_row.addWidget(self.btn_run_convert)
        conv_layout.addLayout(conv_action_row)

        self.lbl_conv_stats = QLabel("No conversion performed yet.")
        conv_layout.addWidget(self.lbl_conv_stats)

        layout.addWidget(conv_box)

        # Progress bar for Tab 2
        self.prog_merge_conv = QProgressBar()
        self.prog_merge_conv.setValue(0)
        layout.addWidget(self.prog_merge_conv)

        layout.addStretch()
        return tab

    # -----------------------------------------------------------------
    # Tab 3: Dataset Splitter
    # -----------------------------------------------------------------
    def _build_tab_split(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        box_paths = QGroupBox("Dataset Sources & Destination")
        box_paths_layout = QGridLayout(box_paths)
        box_paths_layout.setHorizontalSpacing(10)
        box_paths_layout.setVerticalSpacing(8)

        box_paths_layout.addWidget(QLabel("Images Directory:"), 0, 0)
        self.txt_split_img_src = QLineEdit(str(self.merged_dir / "images"))
        box_paths_layout.addWidget(self.txt_split_img_src, 0, 1)
        btn_img_src = QPushButton("Browse...")
        btn_img_src.clicked.connect(lambda: self._browse_folder(self.txt_split_img_src))
        box_paths_layout.addWidget(btn_img_src, 0, 2)

        box_paths_layout.addWidget(QLabel("Labels Directory:"), 1, 0)
        self.txt_split_lbl_src = QLineEdit(str(self.merged_dir / "labels"))
        box_paths_layout.addWidget(self.txt_split_lbl_src, 1, 1)
        btn_lbl_src = QPushButton("Browse...")
        btn_lbl_src.clicked.connect(lambda: self._browse_folder(self.txt_split_lbl_src))
        box_paths_layout.addWidget(btn_lbl_src, 1, 2)

        box_paths_layout.addWidget(QLabel("Split Output Dataset:"), 2, 0)
        self.txt_split_dst = QLineEdit(str(self.dataset_dir))
        box_paths_layout.addWidget(self.txt_split_dst, 2, 1)
        btn_split_dst = QPushButton("Browse...")
        btn_split_dst.clicked.connect(lambda: self._browse_folder(self.txt_split_dst))
        box_paths_layout.addWidget(btn_split_dst, 2, 2)

        layout.addWidget(box_paths)

        # Split Ratio Configuration
        box_ratios = QGroupBox("Split Ratios (Train / Val / Test)")
        box_ratios_layout = QGridLayout(box_ratios)
        box_ratios_layout.setHorizontalSpacing(12)
        box_ratios_layout.setVerticalSpacing(8)

        # Train Ratio
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

        # Val Ratio
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

        # Test Ratio
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

        # Connect sliders to spinboxes
        self.slider_train.valueChanged.connect(self.spin_train.setValue)
        self.spin_train.valueChanged.connect(self.slider_train.setValue)
        self.slider_val.valueChanged.connect(self.spin_val.setValue)
        self.spin_val.valueChanged.connect(self.slider_val.setValue)
        self.slider_test.valueChanged.connect(self.spin_test.setValue)
        self.spin_test.valueChanged.connect(self.slider_test.setValue)

        # Crucial Requirement: Include ALL images (labeled + unannotated background images)
        self.chk_all_images = QCheckBox("☑️ Include ALL images in dataset (background images without annotations will have empty labels)")
        self.chk_all_images.setChecked(True)
        self.chk_all_images.setEnabled(False)  # Enforce requirement
        box_ratios_layout.addWidget(self.chk_all_images, 3, 0, 1, 3)

        layout.addWidget(box_ratios)

        # Action Button & Progress
        self.btn_run_split = QPushButton("📦 Split Dataset & Generate data.yaml")
        self.btn_run_split.setProperty("class", "PrimaryButton")
        self.btn_run_split.clicked.connect(self._on_run_split)
        layout.addWidget(self.btn_run_split)

        self.prog_split = QProgressBar()
        self.prog_split.setValue(0)
        layout.addWidget(self.prog_split)

        self.lbl_split_stats = QLabel("No dataset split yet.")
        layout.addWidget(self.lbl_split_stats)

        layout.addStretch()
        return tab

    # -----------------------------------------------------------------
    # Tab 4: YOLO Auto-Trainer
    # -----------------------------------------------------------------
    def _build_tab_train(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Upper row: Model selection & Dataset yaml
        box_model = QGroupBox("Model & Dataset Configuration")
        m_layout = QGridLayout(box_model)
        m_layout.setHorizontalSpacing(10)
        m_layout.setVerticalSpacing(8)

        m_layout.addWidget(QLabel("Task:"), 0, 0)
        self.cmb_train_task = QComboBox()
        self.cmb_train_task.addItems(["detect", "segment"])
        self.cmb_train_task.currentIndexChanged.connect(self._update_model_presets)
        m_layout.addWidget(self.cmb_train_task, 0, 1)

        m_layout.addWidget(QLabel("Model Preset:"), 0, 2)
        self.cmb_model_preset = QComboBox()
        self._update_model_presets()
        m_layout.addWidget(self.cmb_model_preset, 0, 3)

        m_layout.addWidget(QLabel("Or Custom .pt:"), 0, 4)
        self.txt_custom_model = QLineEdit()
        self.txt_custom_model.setPlaceholderText("Optional path to custom .pt weights")
        m_layout.addWidget(self.txt_custom_model, 0, 5)
        btn_browse_pt = QPushButton("Browse...")
        btn_browse_pt.clicked.connect(self._browse_model_pt)
        m_layout.addWidget(btn_browse_pt, 0, 6)

        m_layout.addWidget(QLabel("data.yaml Path:"), 1, 0)
        self.txt_data_yaml = QLineEdit(str(self.dataset_dir / "data.yaml"))
        m_layout.addWidget(self.txt_data_yaml, 1, 1, 1, 5)
        btn_browse_yaml = QPushButton("Browse...")
        btn_browse_yaml.clicked.connect(self._browse_data_yaml)
        m_layout.addWidget(btn_browse_yaml, 1, 6)

        layout.addWidget(box_model)

        # Hyperparameters Card (Auto-Tune + User Editable)
        box_hparams = QGroupBox("Training Hyperparameters (Intelligently Auto-Tuned & Fully Editable)")
        h_layout = QGridLayout(box_hparams)
        h_layout.setHorizontalSpacing(12)
        h_layout.setVerticalSpacing(8)

        # Auto-Tune Trigger Button
        self.btn_auto_tune = QPushButton("⚡ Auto-Tune Hyperparameters for Dataset")
        self.btn_auto_tune.setProperty("class", "SuccessButton")
        self.btn_auto_tune.clicked.connect(self._on_auto_tune_clicked)
        h_layout.addWidget(self.btn_auto_tune, 0, 0, 1, 6)

        # Fields
        h_layout.addWidget(QLabel("Epochs:"), 1, 0)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(100)
        h_layout.addWidget(self.spin_epochs, 1, 1)

        h_layout.addWidget(QLabel("Batch Size:"), 1, 2)
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(-1, 256)
        self.spin_batch.setValue(8 if not torch.cuda.is_available() else 16)
        self.spin_batch.setSpecialValueText("Auto (-1)")
        h_layout.addWidget(self.spin_batch, 1, 3)

        h_layout.addWidget(QLabel("Image Size:"), 1, 4)
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(160, 2560)
        self.spin_imgsz.setSingleStep(32)
        self.spin_imgsz.setValue(640)
        h_layout.addWidget(self.spin_imgsz, 1, 5)

        # Early Stopping / Patience requirement:
        # "หยุด train เมื่อค่าเท่าเดิมหรือไม่ดีขึ้นด้วย ขอแบบโหดๆ"
        h_layout.addWidget(QLabel("Early Stop (Patience):"), 2, 0)
        self.spin_patience = QSpinBox()
        self.spin_patience.setRange(1, 100)
        self.spin_patience.setValue(15)
        self.spin_patience.setToolTip("Stops training automatically if validation fitness does not improve for this many epochs")
        h_layout.addWidget(self.spin_patience, 2, 1)

        h_layout.addWidget(QLabel("Initial LR (lr0):"), 2, 2)
        self.spin_lr0 = QDoubleSpinBox()
        self.spin_lr0.setRange(0.00001, 0.1)
        self.spin_lr0.setDecimals(5)
        self.spin_lr0.setValue(0.01)
        h_layout.addWidget(self.spin_lr0, 2, 3)

        h_layout.addWidget(QLabel("Optimizer:"), 2, 4)
        self.cmb_optimizer = QComboBox()
        self.cmb_optimizer.addItems(["auto", "AdamW", "Adam", "SGD", "RMSProp"])
        h_layout.addWidget(self.cmb_optimizer, 2, 5)

        h_layout.addWidget(QLabel("Device:"), 3, 0)
        self.txt_device = QLineEdit("0" if torch.cuda.is_available() else "cpu")
        h_layout.addWidget(self.txt_device, 3, 1)

        h_layout.addWidget(QLabel("Workers:"), 3, 2)
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(0, 16)
        self.spin_workers.setValue(2)
        h_layout.addWidget(self.spin_workers, 3, 3)

        layout.addWidget(box_hparams)

        # Folder & Weights Naming Requirement:
        # "มีให้ตั้งชื่อ folder ด้วยว่าจะเก็บไว้ไหนเช่น runs/segment/(ตรงนี้ตั้งชื่อได้และมี _v(1 ถ้ารันชื่อเดียวกันอีกรอบให้เป็น v_2, v_3, _v...))/weights/(ตรง best.pt ให้ใช้ชื่อตาม folderที่ user ตั้ง)"
        box_naming = QGroupBox("Custom Run & Weights Naming")
        naming_layout = QGridLayout(box_naming)
        naming_layout.setHorizontalSpacing(10)
        naming_layout.setVerticalSpacing(8)

        naming_layout.addWidget(QLabel("Custom Run Name:"), 0, 0)
        self.txt_custom_run_name = QLineEdit("my_yolo_model")
        self.txt_custom_run_name.textChanged.connect(self._update_run_preview)
        naming_layout.addWidget(self.txt_custom_run_name, 0, 1)

        naming_layout.addWidget(QLabel("Base Runs Folder:"), 0, 2)
        self.txt_runs_base = QLineEdit(str(self.runs_dir))
        naming_layout.addWidget(self.txt_runs_base, 0, 3)
        btn_runs_base = QPushButton("Browse...")
        btn_runs_base.clicked.connect(lambda: self._browse_folder(self.txt_runs_base))
        naming_layout.addWidget(btn_runs_base, 0, 4)

        self.lbl_run_preview = QLabel("Output Preview: runs/detect/my_yolo_model_v1/weights/my_yolo_model.pt")
        self.lbl_run_preview.setStyleSheet("color: #38bdf8; font-weight: 600;")
        naming_layout.addWidget(self.lbl_run_preview, 1, 0, 1, 5)
        self._update_run_preview()

        layout.addWidget(box_naming)

        # Training Action Buttons & Live Metrics Row
        action_row = QHBoxLayout()
        self.btn_start_train = QPushButton("🚀 Start YOLO Training")
        self.btn_start_train.setProperty("class", "PrimaryButton")
        self.btn_start_train.clicked.connect(self._on_start_training)
        action_row.addWidget(self.btn_start_train, stretch=2)

        self.btn_stop_train = QPushButton("⏹️ Stop Training")
        self.btn_stop_train.setProperty("class", "DangerButton")
        self.btn_stop_train.setEnabled(False)
        self.btn_stop_train.clicked.connect(self._on_stop_training)
        action_row.addWidget(self.btn_stop_train, stretch=1)

        layout.addLayout(action_row)

        # Real-time Metrics Card
        metrics_card = QFrame()
        metrics_card.setProperty("class", "CardFrame")
        metrics_layout = QHBoxLayout(metrics_card)
        metrics_layout.setContentsMargins(12, 8, 12, 8)

        self.lbl_metric_epoch = QLabel("Epoch: --/--")
        self.lbl_metric_loss = QLabel("Loss: --")
        self.lbl_metric_fitness = QLabel("Best Fitness / mAP: --")
        self.lbl_metric_early_stop = QLabel("Early Stopping: Active")

        for m_lbl in [self.lbl_metric_epoch, self.lbl_metric_loss, self.lbl_metric_fitness, self.lbl_metric_early_stop]:
            m_lbl.setStyleSheet("font-weight: 700; font-size: 13px; color: #f8fafc;")
            metrics_layout.addWidget(m_lbl)
            metrics_layout.addStretch()

        layout.addWidget(metrics_card)

        # Live Epoch Progress Bar
        self.prog_train = QProgressBar()
        self.prog_train.setValue(0)
        layout.addWidget(self.prog_train)

        # Live Terminal Logs
        self.txt_train_log = QPlainTextEdit()
        self.txt_train_log.setReadOnly(True)
        layout.addWidget(self.txt_train_log, stretch=1)

        return tab

    # -----------------------------------------------------------------
    # Tab 5: Runs & Explorer
    # -----------------------------------------------------------------
    def _build_tab_explorer(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        box_quick = QGroupBox("Quick Open Workspaces in File Explorer")
        quick_layout = QGridLayout(box_quick)
        quick_layout.setHorizontalSpacing(12)
        quick_layout.setVerticalSpacing(10)

        btn_open_dl = QPushButton("📁 Open Downloads Folder")
        btn_open_dl.clicked.connect(lambda: self._open_dir(self.downloads_dir))
        quick_layout.addWidget(btn_open_dl, 0, 0)

        btn_open_mrg = QPushButton("📁 Open Merged Folder")
        btn_open_mrg.clicked.connect(lambda: self._open_dir(self.merged_dir))
        quick_layout.addWidget(btn_open_mrg, 0, 1)

        btn_open_ds = QPushButton("📁 Open Split Dataset Folder")
        btn_open_ds.clicked.connect(lambda: self._open_dir(self.dataset_dir))
        quick_layout.addWidget(btn_open_ds, 1, 0)

        btn_open_runs = QPushButton("📁 Open Runs / Weights Folder")
        btn_open_runs.clicked.connect(lambda: self._open_dir(self.runs_dir))
        quick_layout.addWidget(btn_open_runs, 1, 1)

        layout.addWidget(box_quick)

        # Dataset Preview
        box_yaml = QGroupBox("Active data.yaml Configuration Preview")
        yaml_layout = QVBoxLayout(box_yaml)
        self.txt_yaml_preview = QPlainTextEdit()
        self.txt_yaml_preview.setReadOnly(True)
        yaml_layout.addWidget(self.txt_yaml_preview)

        btn_refresh_yaml = QPushButton("🔄 Refresh data.yaml Preview")
        btn_refresh_yaml.clicked.connect(self._refresh_yaml_preview)
        yaml_layout.addWidget(btn_refresh_yaml)

        layout.addWidget(box_yaml, stretch=1)

        return tab

    # -----------------------------------------------------------------
    # Global Status Bar
    # -----------------------------------------------------------------
    def _build_global_status_bar(self) -> QWidget:
        bar = QFrame()
        bar.setProperty("class", "CardFrame")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 6, 12, 6)

        self.lbl_global_status = QLabel("🟢 Ready")
        layout.addWidget(self.lbl_global_status, stretch=1)

        lbl_workspace = QLabel(f"Workspace: {self.app_dir}")
        lbl_workspace.setStyleSheet("color: #64748b; font-size: 11px;")
        layout.addWidget(lbl_workspace)

        return bar

    # =================================================================
    # Controller / Slot Implementations
    # =================================================================

    def _browse_folder(self, line_edit: QLineEdit):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", line_edit.text() or str(self.app_dir))
        if folder:
            line_edit.setText(folder)

    def _browse_model_pt(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select YOLO Weights (.pt)", str(self.app_dir), "PyTorch Weights (*.pt)")
        if file_path:
            self.txt_custom_model.setText(file_path)

    def _browse_data_yaml(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select data.yaml", str(self.dataset_dir), "YAML Files (*.yaml *.yml)")
        if file_path:
            self.txt_data_yaml.setText(file_path)

    def _open_dir(self, directory: Path):
        try:
            if sys.platform == "win32":
                os.startfile(str(directory))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(directory)])
            else:
                subprocess.Popen(["xdg-open", str(directory)])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open directory: {e}")

    # -----------------------------------------------------------------
    # CVAT Slots
    # -----------------------------------------------------------------
    def _on_cvat_login(self):
        url = self.txt_cvat_url.text().strip()
        user = self.txt_cvat_user.text().strip()
        pwd = self.txt_cvat_pass.text()

        if not url:
            QMessageBox.warning(self, "Input Error", "Please provide a valid CVAT URL.")
            return

        self.btn_cvat_login.setEnabled(False)
        self.lbl_cvat_status.setText("Connecting and logging in to CVAT...")
        self.mascot.set_state(DancingSvgMascot.STATE_DOWNLOADING)

        try:
            res = self.cvat_client.login(url, user, pwd)
            self.lbl_cvat_status.setText(f"Connected! Server Version: {res['server_version']}")
            self._fetch_cvat_tasks()
        except Exception as e:
            self.btn_cvat_login.setEnabled(True)
            self.mascot.set_state(DancingSvgMascot.STATE_IDLE)
            self.lbl_cvat_status.setText(f"Error: {e}")
            QMessageBox.critical(self, "CVAT Login Error", str(e))

    def _fetch_cvat_tasks(self):
        self.active_worker = CvatFetchWorker(self.cvat_client)
        self.active_worker.finished_signal.connect(self._on_tasks_fetched)
        self.active_worker.error_signal.connect(self._on_tasks_fetch_error)
        self.active_worker.start()

    def _on_tasks_fetched(self, projects: list[dict], tasks: list[dict]):
        self.btn_cvat_login.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_IDLE)

        # Populate projects dropdown
        self.cmb_proj_filter.blockSignals(True)
        self.cmb_proj_filter.clear()
        self.cmb_proj_filter.addItem("All Projects", None)
        for p in projects:
            self.cmb_proj_filter.addItem(f"[{p['id']}] {p['name']}", p['id'])
        self.cmb_proj_filter.blockSignals(False)

        self.all_tasks_data = tasks
        self._populate_tasks_table(self.all_tasks_data)
        self.lbl_cvat_status.setText(f"Loaded {len(tasks)} tasks across {len(projects)} projects.")

    def _on_tasks_fetch_error(self, err_msg: str):
        self.btn_cvat_login.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_IDLE)
        self.lbl_cvat_status.setText(f"Fetch error: {err_msg}")
        QMessageBox.warning(self, "Fetch Tasks Error", err_msg)

    def _apply_task_filters(self):
        selected_proj_id = self.cmb_proj_filter.currentData()
        stage_sel = self.cmb_stage_filter.currentText().lower()
        state_sel = self.cmb_state_filter.currentText().lower()
        search_query = self.txt_task_search.text().strip().lower()

        filtered = []
        for t in self.all_tasks_data:
            if selected_proj_id is not None and t.get("project_id") != selected_proj_id:
                continue

            t_stage = str(t.get("stage", "")).lower()
            if stage_sel != "all" and stage_sel not in t_stage:
                continue

            t_state = str(t.get("state", "")).lower()
            if state_sel != "all" and state_sel not in t_state:
                continue

            if search_query:
                t_name = str(t.get("name", "")).lower()
                if search_query not in t_name and str(t.get("id")) != search_query:
                    continue

            filtered.append(t)

        self._populate_tasks_table(filtered)

    def _populate_tasks_table(self, tasks: list[dict]):
        self.tbl_tasks.setRowCount(len(tasks))
        for row, t in enumerate(tasks):
            # Checkbox Item
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk_item.setCheckState(Qt.Checked)
            chk_item.setData(Qt.UserRole, t)
            self.tbl_tasks.setItem(row, 0, chk_item)

            self.tbl_tasks.setItem(row, 1, QTableWidgetItem(str(t.get("id"))))
            self.tbl_tasks.setItem(row, 2, QTableWidgetItem(str(t.get("name"))))
            self.tbl_tasks.setItem(row, 3, QTableWidgetItem(str(t.get("project_name"))))
            self.tbl_tasks.setItem(row, 4, QTableWidgetItem(str(t.get("stage"))))
            self.tbl_tasks.setItem(row, 5, QTableWidgetItem(str(t.get("state"))))
            self.tbl_tasks.setItem(row, 6, QTableWidgetItem(str(t.get("status"))))
            self.tbl_tasks.setItem(row, 7, QTableWidgetItem(str(t.get("size"))))

    def _set_all_tasks_checked(self, checked: bool):
        state = Qt.Checked if checked else Qt.Unchecked
        for row in range(self.tbl_tasks.rowCount()):
            item = self.tbl_tasks.item(row, 0)
            if item:
                item.setCheckState(state)

    def _get_selected_tasks(self) -> list[dict]:
        selected = []
        for row in range(self.tbl_tasks.rowCount()):
            item = self.tbl_tasks.item(row, 0)
            if item and item.checkState() == Qt.Checked:
                task_data = item.data(Qt.UserRole)
                if task_data:
                    selected.append(task_data)
        return selected

    def _on_download_selected_tasks(self):
        tasks = self._get_selected_tasks()
        if not tasks:
            QMessageBox.warning(self, "No Tasks Selected", "Please select at least one task to download.")
            return

        dest_dir = Path(self.txt_dl_folder.text().strip())
        dest_dir.mkdir(parents=True, exist_ok=True)

        self.btn_download_tasks.setEnabled(False)
        self.mascot.set_state(DancingSvgMascot.STATE_DOWNLOADING)
        self.lbl_global_status.setText(f"Downloading {len(tasks)} tasks...")

        self.active_worker = CvatDownloadWorker(self.cvat_client, tasks, dest_dir)
        self.active_worker.progress_signal.connect(self._on_download_progress)
        self.active_worker.task_completed_signal.connect(self._on_task_downloaded)
        self.active_worker.all_completed_signal.connect(self._on_all_downloads_completed)
        self.active_worker.error_signal.connect(self._on_download_error)
        self.active_worker.start()

    def _on_download_progress(self, current: int, total: int, msg: str):
        pct = int((current / total) * 100) if total > 0 else 0
        self.prog_dl.setValue(pct)
        self.lbl_cvat_status.setText(msg)
        self.lbl_global_status.setText(msg)

    def _on_task_downloaded(self, task_id: int, task_name: str, folder_path: str):
        pass

    def _on_all_downloads_completed(self, completed_folders: list[Path]):
        self.btn_download_tasks.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_SUCCESS)
        self.prog_dl.setValue(100)
        self.lbl_cvat_status.setText(f"All {len(completed_folders)} tasks downloaded & cleaned successfully!")
        self.lbl_global_status.setText("🟢 Download complete! Images flattened from /images/default.")

        # Auto populate Tab 2 source
        self.txt_merge_src.setText(str(self.downloads_dir))
        QMessageBox.information(
            self, "Download Complete",
            f"Successfully downloaded {len(completed_folders)} tasks in COCO 1.0 format.\n"
            f"All images extracted and moved from /images/default directly to /images!\n"
            f"Ready for Step 2: Merge & Convert."
        )

    def _on_download_error(self, err_msg: str):
        self.btn_download_tasks.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_IDLE)
        self.lbl_cvat_status.setText(f"Download Error: {err_msg}")
        self.lbl_global_status.setText("🔴 Download Error")
        QMessageBox.critical(self, "Download Error", f"An error occurred during download:\n{err_msg}")

    # -----------------------------------------------------------------
    # Merge & Convert Slots
    # -----------------------------------------------------------------
    def _on_run_merge(self):
        src_path = Path(self.txt_merge_src.text().strip())
        dst_path = Path(self.txt_merge_dst.text().strip())

        if not src_path.exists():
            QMessageBox.warning(self, "Error", f"Source directory does not exist: {src_path}")
            return

        self.btn_run_merge.setEnabled(False)
        self.mascot.set_state(DancingSvgMascot.STATE_PROCESSING)
        self.lbl_global_status.setText("Merging COCO datasets...")

        self.active_worker = MergeWorker([src_path], dst_path)
        self.active_worker.progress_signal.connect(self._on_merge_progress)
        self.active_worker.finished_signal.connect(self._on_merge_finished)
        self.active_worker.error_signal.connect(self._on_merge_error)
        self.active_worker.start()

    def _on_merge_progress(self, current: int, total: int, msg: str):
        pct = int((current / total) * 100) if total > 0 else 0
        self.prog_merge_conv.setValue(pct)
        self.lbl_merge_stats.setText(msg)

    def _on_merge_finished(self, res: dict):
        self.btn_run_merge.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_SUCCESS)
        self.prog_merge_conv.setValue(100)

        classes_str = ", ".join(c["name"] for c in res["categories"])
        stats_text = (
            f"✅ Merge Complete! Datasets: {res['jsons_count']} | Total Images: {res['images_count']} | "
            f"Total Annotations: {res['annotations_count']} | Classes ({len(res['categories'])}): [{classes_str}]"
        )
        self.lbl_merge_stats.setText(stats_text)
        self.lbl_global_status.setText("🟢 Datasets merged successfully.")

        # Update Split tab paths
        self.txt_split_img_src.setText(str(res["images_dir"]))
        self.txt_split_lbl_src.setText(str(self.merged_dir / "labels"))

    def _on_merge_error(self, err_msg: str):
        self.btn_run_merge.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_IDLE)
        self.lbl_merge_stats.setText(f"Merge Error: {err_msg}")
        QMessageBox.critical(self, "Merge Error", err_msg)

    def _on_task_mode_changed(self):
        mode = "detect" if self.rb_mode_det.isChecked() else "segment"
        self.cmb_train_task.setCurrentText(mode)

    def _on_run_convert(self):
        coco_json = Path(self.txt_merge_dst.text().strip()) / "annotations" / "instances_default.json"
        if not coco_json.exists():
            coco_json = Path(self.txt_merge_dst.text().strip()) / "annotations" / "instances_merged.json"
        if not coco_json.exists():
            QMessageBox.warning(self, "File Not Found", f"COCO JSON not found at:\n{coco_json}\nPlease run Merge first.")
            return

        out_labels = Path(self.txt_conv_labels_dst.text().strip())
        mode = "detect" if self.rb_mode_det.isChecked() else "segment"

        self.btn_run_convert.setEnabled(False)
        self.mascot.set_state(DancingSvgMascot.STATE_PROCESSING)
        self.lbl_global_status.setText(f"Converting COCO to YOLO ({mode})...")

        self.active_worker = ConvertWorker(coco_json, out_labels, mode)
        self.active_worker.progress_signal.connect(self._on_convert_progress)
        self.active_worker.finished_signal.connect(self._on_convert_finished)
        self.active_worker.error_signal.connect(self._on_convert_error)
        self.active_worker.start()

    def _on_convert_progress(self, current: int, total: int, msg: str):
        pct = int((current / total) * 100) if total > 0 else 0
        self.prog_merge_conv.setValue(pct)
        self.lbl_conv_stats.setText(msg)

    def _on_convert_finished(self, res: dict):
        self.btn_run_convert.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_SUCCESS)
        self.prog_merge_conv.setValue(100)

        classes_str = ", ".join(res["classes"])
        stats_text = (
            f"✅ Converted for [{res['mode'].upper()}]: Total Images: {res['total_images']} | "
            f"Labeled Images: {res['images_with_labels']} | Annotations: {res['total_annotations_written']} | "
            f"Classes ({res['num_classes']}): [{classes_str}]"
        )
        self.lbl_conv_stats.setText(stats_text)
        self.lbl_global_status.setText("🟢 Conversion to YOLO format complete.")

        # Update Split tab labels path
        self.txt_split_lbl_src.setText(str(res["output_dir"]))

    def _on_convert_error(self, err_msg: str):
        self.btn_run_convert.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_IDLE)
        self.lbl_conv_stats.setText(f"Conversion Error: {err_msg}")
        QMessageBox.critical(self, "Conversion Error", err_msg)

    # -----------------------------------------------------------------
    # Dataset Split Slots
    # -----------------------------------------------------------------
    def _on_run_split(self):
        img_src = Path(self.txt_split_img_src.text().strip())
        lbl_src = Path(self.txt_split_lbl_src.text().strip())
        dst_dir = Path(self.txt_split_dst.text().strip())

        if not img_src.exists():
            QMessageBox.warning(self, "Error", f"Images directory does not exist: {img_src}")
            return
        if not lbl_src.exists():
            QMessageBox.warning(self, "Error", f"Labels directory does not exist: {lbl_src}")
            return

        coco_json = Path(self.txt_merge_dst.text().strip()) / "annotations" / "instances_default.json"

        r_train = self.spin_train.value() / 100.0
        r_val = self.spin_val.value() / 100.0
        r_test = self.spin_test.value() / 100.0

        self.btn_run_split.setEnabled(False)
        self.mascot.set_state(DancingSvgMascot.STATE_PROCESSING)
        self.lbl_global_status.setText("Splitting dataset into train/val/test...")

        self.active_worker = SplitWorker(
            images_dir=img_src,
            labels_dir=lbl_src,
            out_dir=dst_dir,
            coco_json=coco_json if coco_json.exists() else None,
            train_r=r_train,
            val_r=r_val,
            test_r=r_test,
        )
        self.active_worker.progress_signal.connect(self._on_split_progress)
        self.active_worker.finished_signal.connect(self._on_split_finished)
        self.active_worker.error_signal.connect(self._on_split_error)
        self.active_worker.start()

    def _on_split_progress(self, current: int, total: int, msg: str):
        pct = int((current / total) * 100) if total > 0 else 0
        self.prog_split.setValue(pct)
        self.lbl_split_stats.setText(msg)

    def _on_split_finished(self, res: dict):
        self.btn_run_split.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_SUCCESS)
        self.prog_split.setValue(100)

        classes_str = ", ".join(res["classes"])
        stats_text = (
            f"✅ Split Complete! Total Images: {res['total_images']} (Labeled: {res['labeled_images']}, "
            f"Backgrounds: {res['background_images']})\n"
            f"Partition: Train = {res['train_count']} | Val = {res['val_count']} | Test = {res['test_count']}\n"
            f"Classes ({res['num_classes']}): [{classes_str}] -> data.yaml created successfully!"
        )
        self.lbl_split_stats.setText(stats_text)
        self.lbl_global_status.setText("🟢 Split complete. All images included.")

        # Update Training tab data.yaml path
        self.txt_data_yaml.setText(str(res["data_yaml_path"]))
        self._refresh_yaml_preview()

    def _on_split_error(self, err_msg: str):
        self.btn_run_split.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_IDLE)
        self.lbl_split_stats.setText(f"Split Error: {err_msg}")
        QMessageBox.critical(self, "Split Error", err_msg)

    # -----------------------------------------------------------------
    # YOLO Trainer Slots
    # -----------------------------------------------------------------
    def _update_model_presets(self):
        task = self.cmb_train_task.currentText()
        self.cmb_model_preset.clear()

        if task == "detect":
            presets = [
                "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
                "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
                "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt",
            ]
        else:
            presets = [
                "yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt",
                "yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt",
                "yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt", "yolo26l-seg.pt",
            ]
        self.cmb_model_preset.addItems(presets)
        self._update_run_preview()

    def _update_run_preview(self):
        if not hasattr(self, "txt_custom_run_name") or not hasattr(self, "cmb_train_task"):
            return
        task = self.cmb_train_task.currentText()
        custom_name = self.txt_custom_run_name.text().strip() or "yolo_model"
        base_dir = Path(self.txt_runs_base.text().strip()) / task

        next_version_name, _ = get_next_run_version(base_dir, custom_name)
        preview_text = (
            f"Target Directory: runs/{task}/{next_version_name}/\n"
            f"Weights Saved to: runs/{task}/{next_version_name}/weights/best.pt AND weights/{custom_name}.pt"
        )
        self.lbl_run_preview.setText(preview_text)

    def _on_auto_tune_clicked(self):
        # Read dataset statistics from images dir
        dataset_path = Path(self.txt_data_yaml.text().strip()).parent
        images_dir = dataset_path / "images" / "train"

        num_images = 50
        if images_dir.exists():
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            num_images = len([p for p in images_dir.iterdir() if p.suffix.lower() in exts])

        is_gpu = torch.cuda.is_available()
        recommended = auto_tune_hyperparameters(
            num_images=num_images,
            num_annotations=num_images * 3,
            num_classes=2,
            is_gpu=is_gpu,
        )

        # Apply to GUI fields
        self.spin_epochs.setValue(recommended["epochs"])
        self.spin_batch.setValue(recommended["batch"])
        self.spin_imgsz.setValue(recommended["imgsz"])
        self.spin_patience.setValue(recommended["patience"])
        self.spin_lr0.setValue(recommended["lr0"])
        self.cmb_optimizer.setCurrentText(recommended["optimizer"])
        self.txt_device.setText(recommended["device"])
        self.spin_workers.setValue(recommended["workers"])

        QMessageBox.information(
            self, "Auto-Tune Applied",
            f"⚡ Hyperparameters tuned based on {num_images} training images:\n"
            f"• Epochs: {recommended['epochs']}\n"
            f"• Batch Size: {recommended['batch']}\n"
            f"• Early Stopping (Patience): {recommended['patience']} epochs\n"
            f"• Optimizer: {recommended['optimizer']}\n"
            f"• Device: {recommended['device']}\n\n"
            f"You can modify any of these values before training."
        )

    def _on_start_training(self):
        data_yaml = Path(self.txt_data_yaml.text().strip())
        if not data_yaml.exists():
            QMessageBox.warning(self, "File Not Found", f"data.yaml not found at:\n{data_yaml}\nPlease run Split step first.")
            return

        task = self.cmb_train_task.currentText()
        custom_model = self.txt_custom_model.text().strip()
        model_name = custom_model if custom_model else self.cmb_model_preset.currentText()
        custom_run_name = self.txt_custom_run_name.text().strip() or "yolo_model"
        base_runs = Path(self.txt_runs_base.text().strip())

        # Build Hyperparameters
        batch_val = self.spin_batch.value()
        hyperparams = {
            "epochs": self.spin_epochs.value(),
            "batch": batch_val if batch_val > 0 else -1,
            "imgsz": self.spin_imgsz.value(),
            "patience": self.spin_patience.value(),  # Early stop
            "lr0": self.spin_lr0.value(),
            "optimizer": self.cmb_optimizer.currentText(),
            "device": self.txt_device.text().strip(),
            "workers": self.spin_workers.value(),
        }

        self.btn_start_train.setEnabled(False)
        self.btn_stop_train.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_TRAINING)
        self.lbl_global_status.setText("Training YOLO model...")
        self.txt_train_log.clear()

        self.active_worker = TrainWorker(
            engine=self.trainer_engine,
            task=task,
            model_name=model_name,
            data_yaml=data_yaml,
            run_name=custom_run_name,
            base_dir=base_runs,
            hyperparams=hyperparams,
        )
        self.active_worker.log_signal.connect(self._on_train_log)
        self.active_worker.epoch_signal.connect(self._on_train_epoch)
        self.active_worker.finished_signal.connect(self._on_train_finished)
        self.active_worker.error_signal.connect(self._on_train_error)
        self.active_worker.start()

    def _on_stop_training(self):
        self.trainer_engine.stop_training()
        self.lbl_global_status.setText("Stopping training gracefully...")
        self.txt_train_log.appendPlainText("⏹️ User requested training stop. Finalizing current epoch...")

    def _on_train_log(self, msg: str):
        self.txt_train_log.appendPlainText(msg)
        # Auto scroll to bottom
        sb = self.txt_train_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_train_epoch(self, epoch: int, total_epochs: int, metrics: dict):
        pct = int((epoch / total_epochs) * 100) if total_epochs > 0 else 0
        self.prog_train.setValue(pct)

        self.lbl_metric_epoch.setText(f"Epoch: {epoch}/{total_epochs}")
        loss_val = metrics.get("loss", 0.0)
        self.lbl_metric_loss.setText(f"Loss: {loss_val:.4f}")

        fitness_val = metrics.get("fitness", 0.0)
        best_fitness = metrics.get("best_fitness", 0.0)
        self.lbl_metric_fitness.setText(f"Fitness: {fitness_val:.4f} (Best: {best_fitness:.4f})")

    def _on_train_finished(self, res: dict):
        self.btn_start_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
        self.mascot.set_state(DancingSvgMascot.STATE_SUCCESS)
        self.prog_train.setValue(100)
        self.lbl_global_status.setText("🟢 Training session completed!")

        self._update_run_preview()

        run_dir = res.get("run_dir")
        best_pt = res.get("best_pt")
        custom_pt = res.get("custom_pt")

        summary = (
            f"🎉 YOLO Training Finished Successfully!\n\n"
            f"• Output Folder: {run_dir}\n"
            f"• Best Weights: {best_pt}\n"
            f"• Custom Named Weights: {custom_pt}\n\n"
            f"Check Tab 5 (Runs & Explorer) to view saved weights."
        )
        QMessageBox.information(self, "Training Complete", summary)

    def _on_train_error(self, err_msg: str):
        self.btn_start_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
        self.mascot.set_state(DancingSvgMascot.STATE_IDLE)
        self.lbl_global_status.setText("🔴 Training Error")
        self.txt_train_log.appendPlainText(f"\n❌ Training Error:\n{err_msg}")
        QMessageBox.critical(self, "Training Error", err_msg)

    # -----------------------------------------------------------------
    # Tab 5 Explorer Slots
    # -----------------------------------------------------------------
    def _refresh_yaml_preview(self):
        yaml_path = Path(self.txt_data_yaml.text().strip())
        if yaml_path.exists():
            try:
                content = yaml_path.read_text(encoding="utf-8")
                self.txt_yaml_preview.setPlainText(content)
            except Exception as e:
                self.txt_yaml_preview.setPlainText(f"Error reading data.yaml: {e}")
        else:
            self.txt_yaml_preview.setPlainText("data.yaml does not exist yet. Run Split step in Tab 3.")
