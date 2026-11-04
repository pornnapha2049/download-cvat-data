"""
Main Window for the CVAT-to-YOLO GUI application.
Faithfully designed in the Pinterest Pastel Wellness & Mindfulness style:
https://i.pinimg.com/1200x/69/a3/a3/69a3a365c2017d08d2c1ce4795633dd5.jpg

Features:
- Dedicated standalone pages with smooth forward 'Next' and '< Back' navigation (no top stepper buttons)
- Top app bar with circular back button, centered title, bell notification, wave pill, and dancing mascot
- Soft pastel wellness palette (slate-cream, pure white cards, deep slate navy #0f3747, matcha mint, lavender pink, butter yellow)
- Single unified 'data/' folder architecture for all dataset operations
- Blank CVAT server URL by default (no hardcoded initial URL)
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
    QStackedWidget, QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox,
    QDoubleSpinBox, QSlider, QCheckBox, QRadioButton, QButtonGroup,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox,
    QProgressBar, QPlainTextEdit, QFrame, QGroupBox, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize

from core.cvat_client import CvatClientWrapper, sanitize_filename
from core.merger import CocoDatasetMerger
from core.converter import CocoToYoloConverter
from core.splitter import DatasetSplitter
from core.trainer import YoloTrainerEngine, auto_tune_hyperparameters, get_next_run_version
from ui.dancing_svg import DancingSvgMascot, AnimatedWavePill
from ui.theme import DARK_STYLESHEET
from ui.svg_icons import get_svg_icon, get_svg_pixmap


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


class CvatDownloadWorker(QThread):
    progress_signal = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)

    def __init__(self, client: CvatClientWrapper, tasks: list[dict], dest_folder: Path):
        super().__init__()
        self.client = client
        self.tasks = tasks
        self.dest_folder = dest_folder

    def run(self):
        results = []
        total_tasks = len(self.tasks)
        try:
            for idx, task in enumerate(self.tasks):
                task_id = task["id"]
                task_name = task["name"]

                def cb(cur, tot, desc):
                    overall_pct = int(((idx + (cur / max(tot, 1))) / total_tasks) * 100)
                    self.progress_signal.emit(overall_pct, 100, f"Task [{task_id}] {task_name}: {desc}")

                out_dir = self.client.download_and_extract_task(
                    task_id=task_id,
                    task_name=task_name,
                    dest_folder=self.dest_folder,
                    progress_callback=cb
                )
                results.append({"task": task, "dir": str(out_dir)})

            self.progress_signal.emit(100, 100, "All selected tasks downloaded and extracted!")
            self.finished_signal.emit(results)
        except Exception as e:
            self.error_signal.emit(str(e))


class MergeWorker(QThread):
    progress_signal = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, merger: CocoDatasetMerger, src_dir: Path, out_dir: Path):
        super().__init__()
        self.merger = merger
        self.src_dir = src_dir
        self.out_dir = out_dir

    def run(self):
        try:
            def cb(cur, tot, desc):
                self.progress_signal.emit(cur, tot, desc)

            stats = self.merger.merge_datasets(self.src_dir, self.out_dir, progress_callback=cb)
            self.finished_signal.emit(stats)
        except Exception as e:
            self.error_signal.emit(str(e))


class ConvertWorker(QThread):
    progress_signal = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, converter: CocoToYoloConverter, coco_json: Path, out_labels: Path, mode: str):
        super().__init__()
        self.converter = converter
        self.coco_json = coco_json
        self.out_labels = out_labels
        self.mode = mode

    def run(self):
        try:
            def cb(cur, tot, desc):
                self.progress_signal.emit(cur, tot, desc)

            stats = self.converter.convert(
                self.coco_json, self.out_labels, mode=self.mode, progress_callback=cb
            )
            self.finished_signal.emit(stats)
        except Exception as e:
            self.error_signal.emit(str(e))


class SplitWorker(QThread):
    progress_signal = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(
        self,
        splitter: DatasetSplitter,
        img_dir: Path,
        lbl_dir: Path,
        out_dir: Path,
        r_train: float,
        r_val: float,
        r_test: float,
        coco_json: Path = None
    ):
        super().__init__()
        self.splitter = splitter
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.out_dir = out_dir
        self.r_train = r_train
        self.r_val = r_val
        self.r_test = r_test
        self.coco_json = coco_json

    def run(self):
        try:
            def cb(cur, tot, desc):
                self.progress_signal.emit(cur, tot, desc)

            res = self.splitter.split_dataset(
                images_dir=self.img_dir,
                labels_dir=self.lbl_dir,
                output_dataset_dir=self.out_dir,
                coco_json_path=self.coco_json,
                train_ratio=self.r_train,
                val_ratio=self.r_val,
                test_ratio=self.r_test,
                copy_files=True,
                progress_callback=cb
            )
            self.finished_signal.emit(res)
        except Exception as e:
            self.error_signal.emit(str(e))


class TrainWorker(QThread):
    log_signal = pyqtSignal(str)
    epoch_signal = pyqtSignal(int, int, dict)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, engine: YoloTrainerEngine, task: str, model_name: str, data_yaml: Path, run_name: str, base_dir: Path, hyperparams: dict):
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
            def log_cb(msg):
                self.log_signal.emit(msg)

            def epoch_cb(ep, tot, metrics):
                self.epoch_signal.emit(ep, tot, metrics)

            res = self.engine.train(
                task=self.task,
                model_name=self.model_name,
                data_yaml=self.data_yaml,
                run_name=self.run_name,
                base_dir=self.base_dir,
                hyperparams=self.hyperparams,
                log_callback=log_cb,
                epoch_callback=epoch_cb
            )
            self.finished_signal.emit(res)
        except Exception as e:
            self.error_signal.emit(str(e))


# =====================================================================
# Main GUI Window (Mobile / Tablet Wellness Style)
# =====================================================================

class MainWindow(QMainWindow):
    PAGE_TITLES = [
        "1. CVAT Downloader",
        "2. Merge Datasets",
        "3. COCO to YOLO Converter",
        "4. Dataset Splitter",
        "5. YOLO Auto-Trainer"
    ]

    PAGE_TAGS = [
        ("TagPink", "CVAT Connect & Download"),
        ("TagYellow", "COCO 1.0 Data Consolidation"),
        ("TagGreen", "YOLO Format Converter"),
        ("TagBlue", "Train / Val / Test Partition"),
        ("TagPurple", "Model Training & Weights"),
    ]

    PAGE_SUBTITLES = [
        "Enter CVAT server credentials, query tasks, and auto-flatten image directories.",
        "Consolidate multiple downloaded tasks into data/ with automatic MD5 deduplication.",
        "Convert COCO annotations into YOLO .txt format for Detection (BBox) or Segmentation (Polygon).",
        "Partition dataset with 100% images guarantee (including unannotated background images).",
        "Intelligently auto-tune hyperparameters, train YOLO model, and inspect outputs in data/ and runs/."
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle(".☘︎ ݁˖ CVAT to YOLO Studio • Cozy AI Pipeline")
        self.resize(1120, 840)
        self.setMinimumSize(920, 620)

        # Single Unified Data Folder Architecture
        self.app_dir = Path("C:/Articulus/cvat").resolve()
        self.data_dir = self.app_dir / "data"
        self.runs_dir = self.app_dir / "runs"
        self.weights_dir = self.app_dir / "weights"

        for d in [self.data_dir, self.runs_dir, self.weights_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # State Engines
        self.cvat_client = CvatClientWrapper()
        self.merger_engine = CocoDatasetMerger()
        self.converter_engine = CocoToYoloConverter()
        self.splitter_engine = DatasetSplitter()
        self.trainer_engine = YoloTrainerEngine()

        self.all_tasks_data: list[dict] = []
        self.active_worker = None

        self._init_ui()
        self.setStyleSheet(DARK_STYLESHEET)
        self._go_to_step(0)

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
        root_layout.setContentsMargins(18, 12, 18, 12)
        root_layout.setSpacing(10)

        # 1. Top App Bar (< Back | Title | Wave | Bell | Mascot)
        root_layout.addWidget(self._build_top_app_bar())

        # 2. Page Greeting Header (Mental Health / Wellness style)
        root_layout.addWidget(self._build_page_header_card())

        # 3. Dedicated Standalone Pages Container (QStackedWidget)
        self.pages = QStackedWidget()
        self.pages.addWidget(self._wrap_page_scroll(self._build_page_cvat()))
        self.pages.addWidget(self._wrap_page_scroll(self._build_page_merge()))
        self.pages.addWidget(self._wrap_page_scroll(self._build_page_convert()))
        self.pages.addWidget(self._wrap_page_scroll(self._build_page_split()))
        self.pages.addWidget(self._wrap_page_scroll(self._build_page_train()))
        root_layout.addWidget(self.pages, stretch=1)

        # 4. Bottom Navigation Footer (< Previous | Step Pills 1-5 | Next >)
        root_layout.addWidget(self._build_bottom_nav_footer())

    # -----------------------------------------------------------------
    # Top App Bar (Sleek Circle Navigation & Mascot)
    # -----------------------------------------------------------------
    def _build_top_app_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("topAppBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(4, 2, 4, 2)

        # Circular Back Button (like `< Mental Health` in the pin)
        self.btn_top_back = QPushButton()
        self.btn_top_back.setObjectName("circleNavButton")
        self.btn_top_back.setIcon(get_svg_icon("arrow_left", 16))
        self.btn_top_back.setIconSize(QSize(16, 16))
        self.btn_top_back.setCursor(Qt.PointingHandCursor)
        self.btn_top_back.setToolTip("Go to Previous Page")
        self.btn_top_back.clicked.connect(self._on_prev_step)
        layout.addWidget(self.btn_top_back)

        layout.addSpacing(10)

        # Top Page Title
        self.lbl_top_title = QLabel("CVAT Downloader")
        self.lbl_top_title.setObjectName("appTitleLabel")
        layout.addWidget(self.lbl_top_title)

        layout.addSpacing(12)

        # Pulsing Wave Pill
        self.wave_indicator = AnimatedWavePill(36, 18)
        layout.addWidget(self.wave_indicator)

        layout.addStretch()

        # Circular Bell Button with pink notification dot (from the pin!)
        self.btn_bell = QPushButton()
        self.btn_bell.setObjectName("circleNavButton")
        self.btn_bell.setIcon(get_svg_icon("bell", 20))
        self.btn_bell.setIconSize(QSize(20, 20))
        self.btn_bell.setCursor(Qt.PointingHandCursor)
        self.btn_bell.setToolTip("Open data/ Folder")
        self.btn_bell.clicked.connect(lambda: self._open_dir(self.data_dir))
        layout.addWidget(self.btn_bell)

        layout.addSpacing(6)

        # Cozy Mascot
        self.mascot = DancingSvgMascot(size=58)
        layout.addWidget(self.mascot)

        return bar

    # -----------------------------------------------------------------
    # Greeting Header Card (Wellness Style)
    # -----------------------------------------------------------------
    def _build_page_header_card(self) -> QFrame:
        card = QFrame()
        card.setProperty("class", "CardFrame")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(18, 12, 18, 12)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(3)

        tag_row = QHBoxLayout()
        self.lbl_page_tag = QLabel("CVAT Connect & Download")
        self.lbl_page_tag.setProperty("class", "TagPink")
        tag_row.addWidget(self.lbl_page_tag)
        tag_row.addStretch()
        text_layout.addLayout(tag_row)

        self.lbl_greeting_title = QLabel("Good Morning, AI Researcher 👋")
        self.lbl_greeting_title.setObjectName("greetingTitle")
        text_layout.addWidget(self.lbl_greeting_title)

        self.lbl_greeting_sub = QLabel("Select what you want more for now • Seamless computer vision pipeline")
        self.lbl_greeting_sub.setObjectName("greetingSubtitle")
        text_layout.addWidget(self.lbl_greeting_sub)

        layout.addLayout(text_layout, stretch=1)

        # Quick Data Folder Link Pill
        btn_data_link = QPushButton("data/ Folder")
        btn_data_link.setProperty("class", "SecondaryNavButton")
        btn_data_link.setIcon(get_svg_icon("folder", 16))
        btn_data_link.setIconSize(QSize(16, 16))
        btn_data_link.setCursor(Qt.PointingHandCursor)
        btn_data_link.clicked.connect(lambda: self._open_dir(self.data_dir))
        layout.addWidget(btn_data_link)

        return card

    # -----------------------------------------------------------------
    # Bottom Navigation Footer (Carousel Dots & Next Button)
    # -----------------------------------------------------------------
    def _build_bottom_nav_footer(self) -> QFrame:
        footer = QFrame()
        footer.setObjectName("bottomNavFooter")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(14, 6, 14, 6)

        # Left: Prev Button
        self.btn_footer_prev = QPushButton("Previous")
        self.btn_footer_prev.setProperty("class", "SecondaryNavButton")
        self.btn_footer_prev.setIcon(get_svg_icon("arrow_left", 14))
        self.btn_footer_prev.setIconSize(QSize(14, 14))
        self.btn_footer_prev.clicked.connect(self._on_prev_step)
        layout.addWidget(self.btn_footer_prev)

        layout.addStretch()

        # Center: 5 Carousel Step Pills (Active is Deep Navy #0f3747)
        self.step_pills: list[QPushButton] = []
        for i in range(5):
            pill = QPushButton(f"Step {i+1}")
            pill.setProperty("class", "StepPillActive" if i == 0 else "StepPill")
            pill.setCursor(Qt.PointingHandCursor)
            pill.clicked.connect(lambda chk, idx=i: self._go_to_step(idx))
            layout.addWidget(pill)
            self.step_pills.append(pill)

        layout.addStretch()

        # Right: Next Button
        self.btn_footer_next = QPushButton("Next Step")
        self.btn_footer_next.setProperty("class", "StepNavButton")
        self.btn_footer_next.setIcon(get_svg_icon("arrow_right", 14))
        self.btn_footer_next.setIconSize(QSize(14, 14))
        self.btn_footer_next.clicked.connect(self._on_next_step)
        layout.addWidget(self.btn_footer_next)

        return footer

    # -----------------------------------------------------------------
    # PAGE 1: CVAT Downloader (Dedicated Standalone Screen)
    # -----------------------------------------------------------------
    def _build_page_cvat(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(12)

        # Connection Card (NO DEFAULT URL!)
        conn_card = QFrame()
        conn_card.setProperty("class", "CardFrame")
        conn_layout = QGridLayout(conn_card)
        conn_layout.setContentsMargins(16, 14, 16, 14)
        conn_layout.setHorizontalSpacing(12)
        conn_layout.setVerticalSpacing(8)

        conn_layout.addWidget(QLabel("CVAT Server URL:"), 0, 0)
        self.txt_login_url = QLineEdit("")
        self.txt_login_url.setPlaceholderText("http://<your-cvat-server>:8080")
        conn_layout.addWidget(self.txt_login_url, 0, 1)

        conn_layout.addWidget(QLabel("Username:"), 0, 2)
        self.txt_login_user = QLineEdit("")
        self.txt_login_user.setPlaceholderText("CVAT username")
        conn_layout.addWidget(self.txt_login_user, 0, 3)

        conn_layout.addWidget(QLabel("Password:"), 0, 4)
        self.txt_login_pass = QLineEdit("")
        self.txt_login_pass.setEchoMode(QLineEdit.Password)
        self.txt_login_pass.setPlaceholderText("CVAT password")
        self.txt_login_pass.returnPressed.connect(self._on_do_login)
        conn_layout.addWidget(self.txt_login_pass, 0, 5)

        self.btn_do_login = QPushButton("Connect & Fetch")
        self.btn_do_login.setProperty("class", "PrimaryButton")
        self.btn_do_login.setIcon(get_svg_icon("key", 18))
        self.btn_do_login.setIconSize(QSize(18, 18))
        self.btn_do_login.clicked.connect(self._on_do_login)
        conn_layout.addWidget(self.btn_do_login, 0, 6)

        self.lbl_login_status = QLabel("Enter your CVAT Server URL and credentials above, then click 'Connect & Fetch'.")
        self.lbl_login_status.setStyleSheet("color: #64748b; font-weight: 600; min-height: 20px;")
        conn_layout.addWidget(self.lbl_login_status, 1, 0, 1, 6)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.setIcon(get_svg_icon("refresh", 14))
        btn_refresh.setIconSize(QSize(14, 14))
        btn_refresh.clicked.connect(self._fetch_cvat_tasks)
        conn_layout.addWidget(btn_refresh, 1, 6)

        layout.addWidget(conn_card)

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
        self.tbl_tasks.setMinimumHeight(240)
        self.tbl_tasks.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.tbl_tasks, stretch=1)

        # Download Card (Target data/raw_tasks)
        dl_card = QFrame()
        dl_card.setProperty("class", "CardFrame")
        dl_layout = QHBoxLayout(dl_card)
        dl_layout.setContentsMargins(16, 12, 16, 12)

        dl_layout.addWidget(QLabel("Download Folder:"))
        self.txt_dl_folder = QLineEdit(str(self.data_dir / "raw_tasks"))
        dl_layout.addWidget(self.txt_dl_folder, stretch=1)

        btn_browse_dl = QPushButton("Browse...")
        btn_browse_dl.setIcon(get_svg_icon("folder", 16))
        btn_browse_dl.setIconSize(QSize(16, 16))
        btn_browse_dl.clicked.connect(lambda: self._browse_folder(self.txt_dl_folder))
        dl_layout.addWidget(btn_browse_dl)

        self.btn_download_tasks = QPushButton("Download & Extract Selected (COCO 1.0)")
        self.btn_download_tasks.setProperty("class", "PrimaryButton")
        self.btn_download_tasks.setIcon(get_svg_icon("download", 18))
        self.btn_download_tasks.setIconSize(QSize(18, 18))
        self.btn_download_tasks.clicked.connect(self._on_download_selected_tasks)
        dl_layout.addWidget(self.btn_download_tasks)

        layout.addWidget(dl_card)

        # Progress & Status
        self.prog_dl = QProgressBar()
        self.prog_dl.setValue(0)
        self.prog_dl.setTextVisible(True)
        layout.addWidget(self.prog_dl)

        status_row = QHBoxLayout()
        self.lbl_cvat_status = QLabel("Select tasks above and click Download.")
        self.lbl_cvat_status.setStyleSheet("color: #64748b; font-weight: 600;")
        status_row.addWidget(self.lbl_cvat_status, stretch=1)

        self.btn_page1_next = QPushButton("Next: Merge Datasets ➔")
        self.btn_page1_next.setProperty("class", "StepNavButton")
        self.btn_page1_next.clicked.connect(lambda: self._go_to_step(1))
        status_row.addWidget(self.btn_page1_next)

        layout.addLayout(status_row)
        return page

    # -----------------------------------------------------------------
    # PAGE 2: Merge Datasets (Dedicated Standalone Screen)
    # -----------------------------------------------------------------
    def _build_page_merge(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(14)

        # Zen Stones Quote Card (Saint Augustine Style from Pin)
        quote_card = QFrame()
        quote_card.setProperty("class", "HeroQuoteCard")
        quote_layout = QHBoxLayout(quote_card)
        quote_layout.setContentsMargins(18, 16, 18, 16)

        quote_text_box = QVBoxLayout()
        quote_lbl = QLabel('"Patience is the companion of wisdom."')
        quote_lbl.setObjectName("heroQuoteText")
        quote_text_box.addWidget(quote_lbl)

        author_lbl = QLabel("― Consolidate multiple tasks into data/ with automatic MD5 image collision resolution.")
        author_lbl.setObjectName("heroQuoteAuthor")
        quote_text_box.addWidget(author_lbl)
        quote_layout.addLayout(quote_text_box, stretch=1)

        zen_icon_lbl = QLabel()
        zen_icon_lbl.setPixmap(get_svg_pixmap("zen_stones", 48, 48))
        quote_layout.addWidget(zen_icon_lbl)

        layout.addWidget(quote_card)

        # Paths Card
        paths_card = QFrame()
        paths_card.setProperty("class", "CardFrame")
        paths_layout = QVBoxLayout(paths_card)
        paths_layout.setContentsMargins(18, 16, 18, 16)
        paths_layout.setSpacing(12)

        row_src = QHBoxLayout()
        row_src.addWidget(QLabel("Tasks Source Directory:"))
        self.txt_merge_src = QLineEdit(str(self.data_dir / "raw_tasks"))
        row_src.addWidget(self.txt_merge_src, stretch=1)
        btn_src = QPushButton("Browse...")
        btn_src.setIcon(get_svg_icon("folder", 16))
        btn_src.setIconSize(QSize(16, 16))
        btn_src.clicked.connect(lambda: self._browse_folder(self.txt_merge_src))
        row_src.addWidget(btn_src)
        paths_layout.addLayout(row_src)

        row_dst = QHBoxLayout()
        row_dst.addWidget(QLabel("Merged Output Directory:"))
        self.txt_merge_dst = QLineEdit(str(self.data_dir))
        row_dst.addWidget(self.txt_merge_dst, stretch=1)
        btn_dst = QPushButton("Browse...")
        btn_dst.setIcon(get_svg_icon("folder", 16))
        btn_dst.setIconSize(QSize(16, 16))
        btn_dst.clicked.connect(lambda: self._browse_folder(self.txt_merge_dst))
        row_dst.addWidget(btn_dst)
        paths_layout.addLayout(row_dst)

        self.btn_run_merge = QPushButton("Merge Datasets (Deduplicate Images & Unify Annotations)")
        self.btn_run_merge.setProperty("class", "PrimaryButton")
        self.btn_run_merge.setIcon(get_svg_icon("step_merge", 18))
        self.btn_run_merge.setIconSize(QSize(18, 18))
        self.btn_run_merge.clicked.connect(self._on_run_merge)
        paths_layout.addWidget(self.btn_run_merge)

        self.prog_merge = QProgressBar()
        self.prog_merge.setValue(0)
        paths_layout.addWidget(self.prog_merge)

        self.lbl_merge_stats = QLabel("No merged dataset yet.")
        self.lbl_merge_stats.setStyleSheet("color: #1e293b; font-weight: 600; min-height: 24px;")
        paths_layout.addWidget(self.lbl_merge_stats)

        layout.addWidget(paths_card)

        advance_row = QHBoxLayout()
        advance_row.addStretch()
        self.btn_page2_next = QPushButton("Next: Convert to YOLO ➔")
        self.btn_page2_next.setProperty("class", "StepNavButton")
        self.btn_page2_next.clicked.connect(lambda: self._go_to_step(2))
        advance_row.addWidget(self.btn_page2_next)
        layout.addLayout(advance_row)

        layout.addStretch()
        return page

    # -----------------------------------------------------------------
    # PAGE 3: COCO to YOLO Converter (Dedicated Standalone Screen)
    # -----------------------------------------------------------------
    def _build_page_convert(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(14)

        # 2 Large Choice Cards side-by-side (like "Hang out" and "Laugh" in Phone 1 of the pin!)
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
        card_det_layout.addLayout(c_det_top)

        lbl_det_title = QLabel("YOLO Detection")
        lbl_det_title.setProperty("class", "ChoiceTitleNavy")
        card_det_layout.addWidget(lbl_det_title)

        lbl_det_sub = QLabel("Converts polygons to normalized bounding boxes (xc, yc, w, h).")
        lbl_det_sub.setProperty("class", "ChoiceSubNavy")
        lbl_det_sub.setWordWrap(True)
        card_det_layout.addWidget(lbl_det_sub)

        self.rb_mode_det = QRadioButton("Select Detection Mode")
        self.rb_mode_det.setChecked(True)
        self.rb_mode_det.setStyleSheet("color: #ffffff; font-weight: 700;")
        self.rb_mode_det.toggled.connect(self._on_task_mode_changed)
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
        card_seg_layout.addLayout(c_seg_top)

        lbl_seg_title = QLabel("YOLO Segmentation")
        lbl_seg_title.setProperty("class", "ChoiceTitlePink")
        card_seg_layout.addWidget(lbl_seg_title)

        lbl_seg_sub = QLabel("Preserves exact polygon vertex contours (cls x1 y1 x2 y2...).")
        lbl_seg_sub.setProperty("class", "ChoiceSubPink")
        lbl_seg_sub.setWordWrap(True)
        card_seg_layout.addWidget(lbl_seg_sub)

        self.rb_mode_seg = QRadioButton("Select Segmentation Mode")
        self.rb_mode_seg.setStyleSheet("color: #9f1239; font-weight: 700;")
        card_seg_layout.addWidget(self.rb_mode_seg)

        choice_row.addWidget(self.card_seg, stretch=1)
        layout.addLayout(choice_row)

        # Conversion Paths Card
        conv_paths = QFrame()
        conv_paths.setProperty("class", "CardFrame")
        cp_layout = QVBoxLayout(conv_paths)
        cp_layout.setContentsMargins(18, 16, 18, 16)
        cp_layout.setSpacing(12)

        row_json = QHBoxLayout()
        row_json.addWidget(QLabel("COCO JSON File:"))
        self.txt_conv_json = QLineEdit(str(self.data_dir / "annotations" / "instances_default.json"))
        row_json.addWidget(self.txt_conv_json, stretch=1)
        btn_json = QPushButton("Browse...")
        btn_json.setIcon(get_svg_icon("folder", 16))
        btn_json.setIconSize(QSize(16, 16))
        btn_json.clicked.connect(lambda: self._browse_file(self.txt_conv_json, "JSON Files (*.json)"))
        row_json.addWidget(btn_json)
        cp_layout.addLayout(row_json)

        row_lbl_dst = QHBoxLayout()
        row_lbl_dst.addWidget(QLabel("Labels Output Folder:"))
        self.txt_conv_labels_dst = QLineEdit(str(self.data_dir / "labels"))
        row_lbl_dst.addWidget(self.txt_conv_labels_dst, stretch=1)
        btn_lbl_dst = QPushButton("Browse...")
        btn_lbl_dst.setIcon(get_svg_icon("folder", 16))
        btn_lbl_dst.setIconSize(QSize(16, 16))
        btn_lbl_dst.clicked.connect(lambda: self._browse_folder(self.txt_conv_labels_dst))
        row_lbl_dst.addWidget(btn_lbl_dst)
        cp_layout.addLayout(row_lbl_dst)

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
        self.lbl_conv_stats.setStyleSheet("color: #1e293b; font-weight: 600; min-height: 24px;")
        cp_layout.addWidget(self.lbl_conv_stats)

        layout.addWidget(conv_paths)

        advance_row = QHBoxLayout()
        advance_row.addStretch()
        self.btn_page3_next = QPushButton("Next: Split Dataset ➔")
        self.btn_page3_next.setProperty("class", "StepNavButton")
        self.btn_page3_next.clicked.connect(lambda: self._go_to_step(3))
        advance_row.addWidget(self.btn_page3_next)
        layout.addLayout(advance_row)

        layout.addStretch()
        return page

    # -----------------------------------------------------------------
    # PAGE 4: Dataset Splitter (Dedicated Standalone Screen)
    # -----------------------------------------------------------------
    def _build_page_split(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(14)

        # Paths Card
        paths_card = QFrame()
        paths_card.setProperty("class", "CardFrame")
        box_paths_layout = QGridLayout(paths_card)
        box_paths_layout.setContentsMargins(18, 16, 18, 16)
        box_paths_layout.setHorizontalSpacing(12)
        box_paths_layout.setVerticalSpacing(10)

        box_paths_layout.addWidget(QLabel("Images Source Folder:"), 0, 0)
        self.txt_split_img_src = QLineEdit(str(self.data_dir / "images"))
        box_paths_layout.addWidget(self.txt_split_img_src, 0, 1)
        btn_img_src = QPushButton("Browse...")
        btn_img_src.setIcon(get_svg_icon("folder", 16))
        btn_img_src.setIconSize(QSize(16, 16))
        btn_img_src.clicked.connect(lambda: self._browse_folder(self.txt_split_img_src))
        box_paths_layout.addWidget(btn_img_src, 0, 2)

        box_paths_layout.addWidget(QLabel("Labels Source Folder:"), 1, 0)
        self.txt_split_lbl_src = QLineEdit(str(self.data_dir / "labels"))
        box_paths_layout.addWidget(self.txt_split_lbl_src, 1, 1)
        btn_lbl_src = QPushButton("Browse...")
        btn_lbl_src.setIcon(get_svg_icon("folder", 16))
        btn_lbl_src.setIconSize(QSize(16, 16))
        btn_lbl_src.clicked.connect(lambda: self._browse_folder(self.txt_split_lbl_src))
        box_paths_layout.addWidget(btn_lbl_src, 1, 2)

        box_paths_layout.addWidget(QLabel("Split Dataset Destination:"), 2, 0)
        self.txt_split_dst = QLineEdit(str(self.data_dir))
        box_paths_layout.addWidget(self.txt_split_dst, 2, 1)
        btn_split_dst = QPushButton("Browse...")
        btn_split_dst.setIcon(get_svg_icon("folder", 16))
        btn_split_dst.setIconSize(QSize(16, 16))
        btn_split_dst.clicked.connect(lambda: self._browse_folder(self.txt_split_dst))
        box_paths_layout.addWidget(btn_split_dst, 2, 2)

        layout.addWidget(paths_card)

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

        self.slider_train.valueChanged.connect(self.spin_train.setValue)
        self.spin_train.valueChanged.connect(self.slider_train.setValue)
        self.slider_val.valueChanged.connect(self.spin_val.setValue)
        self.spin_val.valueChanged.connect(self.slider_val.setValue)
        self.slider_test.valueChanged.connect(self.spin_test.setValue)
        self.spin_test.valueChanged.connect(self.slider_test.setValue)

        self.chk_all_images = QCheckBox("Include ALL images in dataset (background images without annotations receive empty label files)")
        self.chk_all_images.setChecked(True)
        self.chk_all_images.setEnabled(False)
        box_ratios_layout.addWidget(self.chk_all_images, 3, 0, 1, 3)

        self.btn_run_split = QPushButton("Split Dataset & Generate data.yaml")
        self.btn_run_split.setProperty("class", "PrimaryButton")
        self.btn_run_split.setIcon(get_svg_icon("step_split", 18))
        self.btn_run_split.setIconSize(QSize(18, 18))
        self.btn_run_split.clicked.connect(self._on_run_split)
        box_ratios_layout.addWidget(self.btn_run_split, 4, 0, 1, 3)

        self.prog_split = QProgressBar()
        self.prog_split.setValue(0)
        box_ratios_layout.addWidget(self.prog_split, 5, 0, 1, 3)

        self.lbl_split_stats = QLabel("No dataset split yet.")
        self.lbl_split_stats.setStyleSheet("color: #1e293b; font-weight: 600; min-height: 24px;")
        box_ratios_layout.addWidget(self.lbl_split_stats, 6, 0, 1, 3)

        layout.addWidget(ratios_card)

        advance_row = QHBoxLayout()
        advance_row.addStretch()
        self.btn_page4_next = QPushButton("Next: YOLO Auto-Trainer ➔")
        self.btn_page4_next.setProperty("class", "StepNavButton")
        self.btn_page4_next.clicked.connect(lambda: self._go_to_step(4))
        advance_row.addWidget(self.btn_page4_next)
        layout.addLayout(advance_row)

        layout.addStretch()
        return page

    # -----------------------------------------------------------------
    # PAGE 5: YOLO Auto-Trainer & Explorer (Dedicated Standalone Screen)
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

        h_layout.addWidget(QLabel("data.yaml Path:"), 1, 0)
        self.txt_data_yaml = QLineEdit(str(self.data_dir / "data.yaml"))
        h_layout.addWidget(self.txt_data_yaml, 1, 1, 1, 5)
        btn_browse_yaml = QPushButton("Browse...")
        btn_browse_yaml.setIcon(get_svg_icon("folder", 16))
        btn_browse_yaml.setIconSize(QSize(16, 16))
        btn_browse_yaml.clicked.connect(self._browse_data_yaml)
        h_layout.addWidget(btn_browse_yaml, 1, 6)

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
        self.txt_runs_base = QLineEdit(str(self.runs_dir))
        h_layout.addWidget(self.txt_runs_base, 5, 4, 1, 2)
        btn_runs_base = QPushButton("Browse...")
        btn_runs_base.setIcon(get_svg_icon("folder", 16))
        btn_runs_base.setIconSize(QSize(16, 16))
        btn_runs_base.clicked.connect(lambda: self._browse_folder(self.txt_runs_base))
        h_layout.addWidget(btn_runs_base, 5, 6)

        self.lbl_run_preview = QLabel("Output Preview: runs/detect/my_yolo_model_v1/weights/my_yolo_model.pt")
        self.lbl_run_preview.setStyleSheet("color: #166534; font-weight: 700; min-height: 28px;")
        h_layout.addWidget(self.lbl_run_preview, 6, 0, 1, 7)

        layout.addWidget(train_card)

        self._update_model_presets()
        self._update_run_preview()

        # Action Row (Start & Stop)
        action_row = QHBoxLayout()
        self.btn_start_train = QPushButton("Start YOLO Training")
        self.btn_start_train.setProperty("class", "PrimaryButton")
        self.btn_start_train.setIcon(get_svg_icon("step_train", 20))
        self.btn_start_train.setIconSize(QSize(20, 20))
        self.btn_start_train.clicked.connect(self._on_start_training)
        action_row.addWidget(self.btn_start_train, stretch=2)

        self.btn_stop_train = QPushButton("Stop Training")
        self.btn_stop_train.setProperty("class", "DangerButton")
        self.btn_stop_train.setIcon(get_svg_icon("stop", 18))
        self.btn_stop_train.setIconSize(QSize(18, 18))
        self.btn_stop_train.setEnabled(False)
        self.btn_stop_train.clicked.connect(self._on_stop_training)
        action_row.addWidget(self.btn_stop_train, stretch=1)

        layout.addLayout(action_row)

        # Real-time Metrics Card (Sleep/Stress indicator style)
        metrics_card = QFrame()
        metrics_card.setProperty("class", "CardFrame")
        metrics_layout = QHBoxLayout(metrics_card)
        metrics_layout.setContentsMargins(18, 10, 18, 10)
        metrics_layout.setSpacing(14)

        def make_metric_badge(icon_name, text_lbl):
            b_layout = QHBoxLayout()
            b_layout.setSpacing(6)
            icon_lbl = QLabel()
            icon_lbl.setPixmap(get_svg_pixmap(icon_name, 18, 18))
            b_layout.addWidget(icon_lbl)
            text_lbl.setStyleSheet("font-weight: 700; font-size: 13px; color: #1e293b; min-height: 24px;")
            b_layout.addWidget(text_lbl)
            return b_layout

        self.lbl_metric_epoch = QLabel("Epoch: --/--")
        self.lbl_metric_loss = QLabel("Loss: --")
        self.lbl_metric_fitness = QLabel("Best Fitness / mAP: --")
        self.lbl_metric_early_stop = QLabel("Early Stopping: Active")

        metrics_layout.addLayout(make_metric_badge("clock", self.lbl_metric_epoch))
        metrics_layout.addStretch()
        metrics_layout.addLayout(make_metric_badge("chart", self.lbl_metric_loss))
        metrics_layout.addStretch()
        metrics_layout.addLayout(make_metric_badge("target", self.lbl_metric_fitness))
        metrics_layout.addStretch()
        metrics_layout.addLayout(make_metric_badge("zen_stones", self.lbl_metric_early_stop))

        layout.addWidget(metrics_card)

        self.prog_train = QProgressBar()
        self.prog_train.setValue(0)
        layout.addWidget(self.prog_train)

        self.txt_train_log = QPlainTextEdit()
        self.txt_train_log.setReadOnly(True)
        self.txt_train_log.setMinimumHeight(140)
        layout.addWidget(self.txt_train_log, stretch=1)

        # Quick Explorer Box
        quick_card = QFrame()
        quick_card.setProperty("class", "CardFrame")
        quick_layout = QVBoxLayout(quick_card)
        quick_layout.setContentsMargins(16, 12, 16, 12)

        quick_btns = QHBoxLayout()
        btn_open_data = QPushButton("Open data/ Folder")
        btn_open_data.setIcon(get_svg_icon("folder", 16))
        btn_open_data.setIconSize(QSize(16, 16))
        btn_open_data.clicked.connect(lambda: self._open_dir(self.data_dir))
        quick_btns.addWidget(btn_open_data)

        btn_open_runs = QPushButton("Open runs/ Folder")
        btn_open_runs.setIcon(get_svg_icon("lotus", 16))
        btn_open_runs.setIconSize(QSize(16, 16))
        btn_open_runs.clicked.connect(lambda: self._open_dir(self.runs_dir))
        quick_btns.addWidget(btn_open_runs)

        btn_refresh_yaml = QPushButton("Refresh data.yaml Preview")
        btn_refresh_yaml.setIcon(get_svg_icon("refresh", 16))
        btn_refresh_yaml.setIconSize(QSize(16, 16))
        btn_refresh_yaml.clicked.connect(self._refresh_yaml_preview)
        quick_btns.addWidget(btn_refresh_yaml)

        quick_layout.addLayout(quick_btns)

        self.txt_yaml_preview = QPlainTextEdit()
        self.txt_yaml_preview.setReadOnly(True)
        self.txt_yaml_preview.setMaximumHeight(100)
        quick_layout.addWidget(self.txt_yaml_preview)

        layout.addWidget(quick_card)
        return page

    # =================================================================
    # Common Event Handlers & Slots
    # =================================================================

    def _browse_folder(self, line_edit: QLineEdit):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", line_edit.text())
        if folder:
            line_edit.setText(str(Path(folder).resolve()))

    def _browse_file(self, line_edit: QLineEdit, filter_str: str = "All Files (*.*)"):
        f, _ = QFileDialog.getOpenFileName(self, "Select File", line_edit.text(), filter_str)
        if f:
            line_edit.setText(str(Path(f).resolve()))

    def _open_dir(self, dir_path: Path):
        dir_path.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(dir_path))
        else:
            subprocess.Popen(["xdg-open", str(dir_path)])

    def _go_to_step(self, step_idx: int):
        if step_idx < 0 or step_idx >= self.pages.count():
            return

        self.pages.setCurrentIndex(step_idx)

        # Update Top App Title
        self.lbl_top_title.setText(self.PAGE_TITLES[step_idx])
        self.btn_top_back.setEnabled(step_idx > 0)

        # Update Greeting Header
        tag_class, tag_text = self.PAGE_TAGS[step_idx]
        self.lbl_page_tag.setProperty("class", tag_text if False else tag_class)
        self.lbl_page_tag.setText(tag_text)
        self.lbl_page_tag.style().unpolish(self.lbl_page_tag)
        self.lbl_page_tag.style().polish(self.lbl_page_tag)

        self.lbl_greeting_title = self.findChild(QLabel, "greetingTitle")
        if self.lbl_greeting_title:
            self.lbl_greeting_title.setText(f"{self.PAGE_TITLES[step_idx]} 👋")

        self.lbl_greeting_sub = self.findChild(QLabel, "greetingSubtitle")
        if self.lbl_greeting_sub:
            self.lbl_greeting_sub.setText(self.PAGE_SUBTITLES[step_idx])

        # Update Footer Carousel Step Pills
        for idx, pill in enumerate(self.step_pills):
            if idx == step_idx:
                pill.setProperty("class", "StepPillActive")
            else:
                pill.setProperty("class", "StepPill")
            pill.style().unpolish(pill)
            pill.style().polish(pill)

        self.btn_footer_prev.setEnabled(step_idx > 0)
        self.btn_footer_next.setEnabled(step_idx < self.pages.count() - 1)

        # If navigating to Step 5, refresh yaml preview
        if step_idx == 4:
            self._refresh_yaml_preview()

    def _on_prev_step(self):
        cur = self.pages.currentIndex()
        if cur > 0:
            self._go_to_step(cur - 1)

    def _on_next_step(self):
        cur = self.pages.currentIndex()
        if cur < self.pages.count() - 1:
            self._go_to_step(cur + 1)

    # -----------------------------------------------------------------
    # CVAT Downloader Slots
    # -----------------------------------------------------------------
    def _on_do_login(self):
        url = self.txt_login_url.text().strip()
        user = self.txt_login_user.text().strip()
        pwd = self.txt_login_pass.text()

        if not url:
            QMessageBox.warning(self, "Input Required", "Please enter your CVAT Server URL (e.g. http://<server-ip>:8080).")
            return

        self.btn_do_login.setEnabled(False)
        self.lbl_login_status.setText("Connecting to CVAT server...")
        self.mascot.set_state(DancingSvgMascot.STATE_DOWNLOADING)

        self.active_worker = CvatAuthWorker(self.cvat_client, url, user, pwd)
        self.active_worker.finished_signal.connect(self._on_auth_success)
        self.active_worker.error_signal.connect(self._on_auth_error)
        self.active_worker.start()

    def _on_auth_success(self, login_info: dict, projects: list[dict], tasks: list[dict]):
        self.btn_do_login.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_SUCCESS)
        username = login_info.get("username") or "User"
        server_url = login_info.get("server_url")
        self.lbl_login_status.setText(f"Connected successfully as {username}!")

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

    def _on_auth_error(self, err_msg: str):
        self.btn_do_login.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_IDLE)
        self.lbl_login_status.setText(f"Connection Failed: {err_msg}")
        QMessageBox.critical(self, "Connection Error", f"Unable to connect to CVAT server:\n{err_msg}")

    def _fetch_cvat_tasks(self):
        if not self.cvat_client.is_authenticated():
            self._on_do_login()
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
            QMessageBox.warning(self, "Not Connected", "Please connect to CVAT before downloading tasks.")
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
        self.btn_download_tasks.setEnabled(False)
        self.prog_dl.setValue(0)
        self.mascot.set_state(DancingSvgMascot.STATE_DOWNLOADING)

        self.active_worker = CvatDownloadWorker(self.cvat_client, selected_tasks, dest_folder)
        self.active_worker.progress_signal.connect(self._on_download_progress)
        self.active_worker.finished_signal.connect(self._on_download_finished)
        self.active_worker.error_signal.connect(self._on_download_error)
        self.active_worker.start()

    def _on_download_progress(self, cur: int, tot: int, desc: str):
        self.prog_dl.setMaximum(tot)
        self.prog_dl.setValue(cur)
        self.lbl_cvat_status.setText(desc)

    def _on_download_finished(self, results: list):
        self.btn_download_tasks.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_SUCCESS)
        self.prog_dl.setValue(self.prog_dl.maximum())
        self.lbl_cvat_status.setText(f"Successfully downloaded and extracted {len(results)} tasks into data/raw_tasks!")

        QMessageBox.information(
            self, "Download Complete",
            f"Successfully downloaded {len(results)} tasks!\n\n"
            f"All images extracted and flattened (default folder removed).\n"
            f"Location: {self.txt_dl_folder.text()}\n\n"
            f"Click 'Next: Merge Datasets' below to proceed."
        )

    def _on_download_error(self, err_msg: str):
        self.btn_download_tasks.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_IDLE)
        self.lbl_cvat_status.setText(f"Download Error: {err_msg}")
        QMessageBox.critical(self, "Download Error", f"An error occurred during download:\n{err_msg}")

    # -----------------------------------------------------------------
    # Merge Datasets Slots
    # -----------------------------------------------------------------
    def _on_run_merge(self):
        src_dir = Path(self.txt_merge_src.text().strip())
        dst_dir = Path(self.txt_merge_dst.text().strip())

        if not src_dir.exists():
            QMessageBox.warning(self, "Directory Not Found", f"Source directory does not exist:\n{src_dir}")
            return

        self.btn_run_merge.setEnabled(False)
        self.prog_merge.setValue(0)
        self.lbl_merge_stats.setText("Merging datasets...")
        self.mascot.set_state(DancingSvgMascot.STATE_PROCESSING)

        self.active_worker = MergeWorker(self.merger_engine, src_dir, dst_dir)
        self.active_worker.progress_signal.connect(self._on_merge_progress)
        self.active_worker.finished_signal.connect(self._on_merge_finished)
        self.active_worker.error_signal.connect(self._on_merge_error)
        self.active_worker.start()

    def _on_merge_progress(self, cur: int, tot: int, desc: str):
        self.prog_merge.setMaximum(tot)
        self.prog_merge.setValue(cur)
        self.lbl_merge_stats.setText(desc)

    def _on_merge_finished(self, stats: dict):
        self.btn_run_merge.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_SUCCESS)
        self.prog_merge.setValue(self.prog_merge.maximum())

        msg = (
            f"Merge Complete!\n"
            f"• Merged Images: {stats['total_images']}\n"
            f"• Merged Annotations: {stats['total_annotations']}\n"
            f"• Unified Categories: {stats['total_categories']}\n"
            f"• Duplicate Filenames Resolved: {stats['duplicates_resolved']}\n"
            f"• Output COCO JSON: {stats['output_annotation_path']}"
        )
        self.lbl_merge_stats.setText(msg)
        QMessageBox.information(self, "Merge Succeeded", msg)

    def _on_merge_error(self, err_msg: str):
        self.btn_run_merge.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_IDLE)
        self.lbl_merge_stats.setText(f"Merge Error: {err_msg}")
        QMessageBox.critical(self, "Merge Error", f"Failed to merge datasets:\n{err_msg}")

    # -----------------------------------------------------------------
    # COCO to YOLO Converter Slots
    # -----------------------------------------------------------------
    def _on_task_mode_changed(self):
        mode = "detect" if self.rb_mode_det.isChecked() else "segment"
        self.cmb_train_task.setCurrentText(mode)

    def _on_run_convert(self):
        coco_json = Path(self.txt_conv_json.text().strip())
        if not coco_json.exists():
            alt = self.data_dir / "annotations" / "instances_default.json"
            if alt.exists():
                coco_json = alt
        if not coco_json.exists():
            QMessageBox.warning(self, "File Not Found", f"COCO JSON not found at:\n{coco_json}\nPlease run Merge first or select an existing COCO JSON file.")
            return

        out_labels = Path(self.txt_conv_labels_dst.text().strip())
        mode = "detect" if self.rb_mode_det.isChecked() else "segment"

        self.btn_run_convert.setEnabled(False)
        self.prog_convert.setValue(0)
        self.lbl_conv_stats.setText(f"Converting COCO 1.0 to YOLO {mode.upper()}...")
        self.mascot.set_state(DancingSvgMascot.STATE_PROCESSING)

        self.active_worker = ConvertWorker(self.converter_engine, coco_json, out_labels, mode)
        self.active_worker.progress_signal.connect(self._on_convert_progress)
        self.active_worker.finished_signal.connect(self._on_convert_finished)
        self.active_worker.error_signal.connect(self._on_convert_error)
        self.active_worker.start()

    def _on_convert_progress(self, cur: int, tot: int, desc: str):
        self.prog_convert.setMaximum(tot)
        self.prog_convert.setValue(cur)
        self.lbl_conv_stats.setText(desc)

    def _on_convert_finished(self, stats: dict):
        self.btn_run_convert.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_SUCCESS)
        self.prog_convert.setValue(self.prog_convert.maximum())

        msg = (
            f"Conversion Complete! (Mode: {stats['mode'].upper()})\n"
            f"• Total Images Processed: {stats['total_images']}\n"
            f"• Label Files Written: {stats['converted_images']}\n"
            f"• Total Annotations: {stats['total_annotations_written']}\n"
            f"• Classes: {len(stats['classes'])} ({', '.join(stats['classes'][:5])}...)\n"
            f"• Output Folder: {stats['output_labels_dir']}"
        )
        self.lbl_conv_stats.setText(msg)
        QMessageBox.information(self, "Conversion Succeeded", msg)

    def _on_convert_error(self, err_msg: str):
        self.btn_run_convert.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_IDLE)
        self.lbl_conv_stats.setText(f"Conversion Error: {err_msg}")
        QMessageBox.critical(self, "Conversion Error", f"Failed to convert annotations:\n{err_msg}")

    # -----------------------------------------------------------------
    # Dataset Splitter Slots
    # -----------------------------------------------------------------
    def _on_run_split(self):
        img_dir = Path(self.txt_split_img_src.text().strip())
        lbl_dir = Path(self.txt_split_lbl_src.text().strip())
        out_dst = Path(self.txt_split_dst.text().strip())

        if not img_dir.exists():
            QMessageBox.warning(self, "Folder Not Found", f"Images folder does not exist:\n{img_dir}")
            return
        if not lbl_dir.exists():
            QMessageBox.warning(self, "Folder Not Found", f"Labels folder does not exist:\n{lbl_dir}")
            return

        r_train = self.spin_train.value() / 100.0
        r_val = self.spin_val.value() / 100.0
        r_test = self.spin_test.value() / 100.0

        coco_json = self.data_dir / "annotations" / "instances_default.json"

        self.btn_run_split.setEnabled(False)
        self.prog_split.setValue(0)
        self.lbl_split_stats.setText("Splitting dataset and generating data.yaml...")
        self.mascot.set_state(DancingSvgMascot.STATE_PROCESSING)

        self.active_worker = SplitWorker(
            splitter=self.splitter_engine,
            img_dir=img_dir,
            lbl_dir=lbl_dir,
            out_dir=out_dst,
            r_train=r_train,
            r_val=r_val,
            r_test=r_test,
            coco_json=coco_json if coco_json.exists() else None
        )
        self.active_worker.progress_signal.connect(self._on_split_progress)
        self.active_worker.finished_signal.connect(self._on_split_finished)
        self.active_worker.error_signal.connect(self._on_split_error)
        self.active_worker.start()

    def _on_split_progress(self, cur: int, tot: int, desc: str):
        self.prog_split.setMaximum(tot)
        self.prog_split.setValue(cur)
        self.lbl_split_stats.setText(desc)

    def _on_split_finished(self, stats: dict):
        self.btn_run_split.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_SUCCESS)
        self.prog_split.setValue(self.prog_split.maximum())

        yaml_path = stats.get("data_yaml")
        if yaml_path:
            self.txt_data_yaml.setText(str(yaml_path))

        msg = (
            f"Dataset Split Complete!\n"
            f"• Total Images (100%): {stats['total_images']}\n"
            f"• Train Set: {stats['train_count']} images\n"
            f"• Validation Set: {stats['val_count']} images\n"
            f"• Test Set: {stats['test_count']} images\n"
            f"• Labeled Images: {stats['labeled_images']} | Background Images: {stats['background_images']}\n"
            f"• Created: {yaml_path}"
        )
        self.lbl_split_stats.setText(msg)
        self._refresh_yaml_preview()
        QMessageBox.information(self, "Split Succeeded", msg)

    def _on_split_error(self, err_msg: str):
        self.btn_run_split.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_IDLE)
        self.lbl_split_stats.setText(f"Split Error: {err_msg}")
        QMessageBox.critical(self, "Split Error", f"Failed to split dataset:\n{err_msg}")

    # -----------------------------------------------------------------
    # YOLO Trainer Slots
    # -----------------------------------------------------------------
    def _browse_model_pt(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model Weights", "", "PyTorch Weights (*.pt)")
        if f:
            self.txt_custom_model.setText(str(Path(f).resolve()))

    def _browse_data_yaml(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select data.yaml", str(self.data_dir), "YAML Files (*.yaml *.yml)")
        if f:
            self.txt_data_yaml.setText(str(Path(f).resolve()))
            self._refresh_yaml_preview()

    def _update_model_presets(self):
        task = self.cmb_train_task.currentText()
        self.cmb_model_preset.clear()
        if task == "detect":
            presets = [
                "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
                "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt",
                "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt"
            ]
        else:
            presets = [
                "yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt",
                "yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt",
                "yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt"
            ]
        self.cmb_model_preset.addItems(presets)
        self._update_run_preview()

    def _update_run_preview(self):
        if not hasattr(self, "txt_custom_run_name") or not hasattr(self, "cmb_train_task") or not hasattr(self, "txt_runs_base"):
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
        dataset_path = Path(self.txt_data_yaml.text().strip()).parent
        images_dir = dataset_path / "images" / "train"

        num_images = 50
        if images_dir.exists():
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            num_images = len([p for p in images_dir.iterdir() if p.suffix.lower() in exts])

        recommended = auto_tune_hyperparameters(num_images, num_classes=1)
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

        batch_val = self.spin_batch.value()
        hyperparams = {
            "epochs": self.spin_epochs.value(),
            "batch": batch_val if batch_val > 0 else -1,
            "imgsz": self.spin_imgsz.value(),
            "patience": self.spin_patience.value(),
            "lr0": self.spin_lr0.value(),
            "optimizer": self.cmb_optimizer.currentText(),
            "device": self.txt_device.text().strip(),
            "workers": self.spin_workers.value(),
        }

        self.btn_start_train.setEnabled(False)
        self.btn_stop_train.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_TRAINING)
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
        self.txt_train_log.appendPlainText("⏹️ User requested training stop. Finalizing current epoch...")

    def _on_train_log(self, msg: str):
        self.txt_train_log.appendPlainText(msg)
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
        self._update_run_preview()

        run_dir = res.get("run_dir")
        best_pt = res.get("best_pt")
        custom_pt = res.get("custom_pt")

        summary = (
            f"🎉 YOLO Training Finished Successfully!\n\n"
            f"• Output Folder: {run_dir}\n"
            f"• Best Weights: {best_pt}\n"
            f"• Custom Named Weights: {custom_pt}\n\n"
            f"Weights and logs are saved in runs/ and weights/."
        )
        QMessageBox.information(self, "Training Complete", summary)

    def _on_train_error(self, err_msg: str):
        self.btn_start_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
        self.mascot.set_state(DancingSvgMascot.STATE_IDLE)
        self.txt_train_log.appendPlainText(f"\n❌ Training Error: {err_msg}")
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
