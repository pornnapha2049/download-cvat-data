"""
Main Window for the CVAT-to-YOLO GUI application.
Faithfully designed in the Pinterest Pastel Wellness & Mindfulness style:
https://i.pinimg.com/1200x/69/a3/a3/69a3a365c2017d08d2c1ce4795633dd5.jpg

Features:
1. Dedicated CVAT Login Page (Standalone Step 1)
2. Dedicated CVAT Downloader Page (Standalone Step 2)
3. Dedicated Merge Datasets Page (Standalone Step 3)
4. Dedicated COCO to YOLO Converter Page (Standalone Step 4)
5. Dedicated Dataset Splitter Page (Standalone Step 5)
6. Dedicated YOLO Auto-Trainer Page (Standalone Step 6) with weights directly renamed to [custom_name]_v[num].pt inside runs/.../weights/
7. Dedicated Test Model / Inference Page (Standalone Step 7) for single image, folder, or video with live preview and save options
- No top stepper buttons: clean mobile/tablet app bar with circular < Back, centered title, bell notification, mascot
- Bottom carousel step pills (1 to 7) with < Previous and Next Step >
- Single unified 'data/' directory architecture
- Blank CVAT server URL by default
"""

import os
import sys
import json
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
    QProgressBar, QPlainTextEdit, QFrame, QGroupBox, QScrollArea, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap

from core.cvat_client import CvatClientWrapper, sanitize_filename
from core.merger import CocoDatasetMerger
from core.converter import CocoToYoloConverter
from core.splitter import DatasetSplitter
from core.trainer import YoloTrainerEngine, auto_tune_hyperparameters, get_next_run_version
from core.tester import YoloTestEngine, IMAGE_EXTS, VIDEO_EXTS
from ui.dancing_svg import DancingSvgMascot, AnimatedWavePill
from ui.theme import DARK_STYLESHEET
from ui.svg_icons import get_svg_icon, get_svg_pixmap


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
        ratios: tuple[float, float, float],
        copy_mode: bool
    ):
        super().__init__()
        self.splitter = splitter
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.out_dir = out_dir
        self.ratios = ratios
        self.copy_mode = copy_mode

    def run(self):
        try:
            def cb(cur, tot, desc):
                self.progress_signal.emit(cur, tot, desc)

            stats = self.splitter.split_dataset(
                images_dir=self.img_dir,
                labels_dir=self.lbl_dir,
                output_dir=self.out_dir,
                ratios=self.ratios,
                copy_files=self.copy_mode,
                progress_callback=cb
            )
            self.finished_signal.emit(stats)
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

            res = self.engine.train_model(
                task=self.task,
                model_name_or_path=self.model_name,
                data_yaml_path=self.data_yaml,
                custom_run_name=self.run_name,
                base_runs_dir=self.base_dir,
                hyperparams=self.hyperparams,
                log_callback=log_cb,
                epoch_callback=epoch_cb
            )
            self.finished_signal.emit(res)
        except Exception as e:
            self.error_signal.emit(str(e))


class TestInferenceWorker(QThread):
    progress_signal = pyqtSignal(int, int, str)
    frame_signal = pyqtSignal(object, int, int, dict)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, engine: YoloTestEngine, mode: str, model_path: Path, src_path: Path, conf: float, iou: float, save: bool, save_dir: Path):
        super().__init__()
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
                self.finished_signal.emit({"mode": "image", "data": res})
            elif self.mode == "folder":
                def cb(cur, tot, desc):
                    self.progress_signal.emit(cur, tot, desc)
                res = self.engine.predict_folder(
                    model_path=self.model_path,
                    folder_path=self.src_path,
                    conf=self.conf,
                    iou=self.iou,
                    save=self.save,
                    save_dir=self.save_dir,
                    progress_callback=cb,
                )
                self.finished_signal.emit({"mode": "folder", "data": res})
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


# =====================================================================
# Main GUI Window (Mobile / Tablet Wellness Style)
# =====================================================================

class MainWindow(QMainWindow):
    PAGE_TITLES = [
        "1. CVAT Login",
        "2. CVAT Downloader",
        "3. Merge Datasets",
        "4. COCO to YOLO Converter",
        "5. Dataset Splitter",
        "6. YOLO Auto-Trainer",
        "7. Test Model"
    ]

    PAGE_TAGS = [
        ("TagPink", "CVAT Authentication"),
        ("TagPink", "Select & Download Tasks"),
        ("TagYellow", "COCO 1.0 Data Consolidation"),
        ("TagGreen", "YOLO Format Converter"),
        ("TagBlue", "Train / Val / Test Partition"),
        ("TagPurple", "Model Training & Weights"),
        ("TagTeal", "Inference & Validation"),
    ]

    PAGE_SUBTITLES = [
        "Enter your CVAT Server URL and credentials to authenticate securely.",
        "Filter and select tasks from your CVAT server to download and auto-flatten images.",
        "Consolidate multiple downloaded tasks into data/ with automatic MD5 deduplication.",
        "Convert COCO annotations into YOLO .txt format for Detection (BBox) or Segmentation (Polygon).",
        "Partition dataset with 100% images guarantee (including unannotated background images).",
        "Intelligently auto-tune hyperparameters, train YOLO model, and inspect outputs in runs/.",
        "Evaluate trained YOLO weights on single images, folders, or video files with visual preview."
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle(".☘︎ ݁˖ CVAT to YOLO Studio • Cozy AI Pipeline")
        self.resize(1160, 860)
        self.setMinimumSize(940, 640)

        # Single Unified Data Folder Architecture
        self.app_dir = Path("C:/Articulus/cvat").resolve()
        self.data_dir = self.app_dir / "data"
        self.runs_dir = self.app_dir / "runs"
        self.test_results_dir = self.data_dir / "test_results"

        for d in [self.data_dir, self.runs_dir, self.test_results_dir]:
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
        self.last_trained_weight: Path | None = None

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
        self.pages.addWidget(self._wrap_page_scroll(self._build_page_login()))      # Step 1: Login
        self.pages.addWidget(self._wrap_page_scroll(self._build_page_cvat()))       # Step 2: Tasks Downloader
        self.pages.addWidget(self._wrap_page_scroll(self._build_page_merge()))      # Step 3: Merge
        self.pages.addWidget(self._wrap_page_scroll(self._build_page_convert()))    # Step 4: Convert
        self.pages.addWidget(self._wrap_page_scroll(self._build_page_split()))      # Step 5: Split
        self.pages.addWidget(self._wrap_page_scroll(self._build_page_train()))      # Step 6: Train
        self.pages.addWidget(self._wrap_page_scroll(self._build_page_test()))       # Step 7: Test Model
        root_layout.addWidget(self.pages, stretch=1)

        # 4. Bottom Navigation Footer (< Previous | Step Pills 1-7 | Next >)
        root_layout.addWidget(self._build_bottom_nav_footer())

    # -----------------------------------------------------------------
    # Top App Bar (Sleek Circle Navigation & Mascot)
    # -----------------------------------------------------------------
    def _build_top_app_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("topAppBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(4, 2, 4, 2)

        # Circular Back Button
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
        self.lbl_top_title = QLabel("1. CVAT Login")
        self.lbl_top_title.setObjectName("appTitleLabel")
        layout.addWidget(self.lbl_top_title)

        layout.addSpacing(12)

        # Pulsing Wave Pill
        self.wave_indicator = AnimatedWavePill(36, 18)
        layout.addWidget(self.wave_indicator)

        layout.addStretch()

        # Circular Bell Button with pink notification dot (opens data/ folder)
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
        self.lbl_page_tag = QLabel("CVAT Authentication")
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

        # Center: 7 Carousel Step Pills
        self.step_pills: list[QPushButton] = []
        for i in range(len(self.PAGE_TITLES)):
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
    # PAGE 1: Dedicated CVAT Login Screen (Standalone Screen)
    # -----------------------------------------------------------------
    def _build_page_login(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(16)

        # Wellness Hero Card for Login
        hero = QFrame()
        hero.setProperty("class", "HeroQuoteCard")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(22, 18, 22, 18)

        htext = QVBoxLayout()
        htext.setSpacing(4)
        lbl_h1 = QLabel('"Begin with clarity and focused intention."')
        lbl_h1.setObjectName("heroQuoteText")
        htext.addWidget(lbl_h1)

        lbl_h2 = QLabel("Connect securely to your CVAT Server to fetch annotations, tasks, and image datasets.")
        lbl_h2.setObjectName("heroQuoteAuthor")
        htext.addWidget(lbl_h2)
        hero_layout.addLayout(htext, stretch=1)

        lotus_icon = QLabel()
        lotus_icon.setPixmap(get_svg_pixmap("lotus", 52, 52))
        hero_layout.addWidget(lotus_icon)
        layout.addWidget(hero)

        # Login Card Container
        card = QFrame()
        card.setProperty("class", "CardFrame")
        c_layout = QVBoxLayout(card)
        c_layout.setContentsMargins(28, 24, 28, 24)
        c_layout.setSpacing(16)

        card_title_row = QHBoxLayout()
        card_icon = QLabel()
        card_icon.setPixmap(get_svg_pixmap("key", 28, 28))
        card_title_row.addWidget(card_icon)

        card_title = QLabel("CVAT Server Authentication")
        card_title.setStyleSheet("font-size: 18px; font-weight: 800; color: #0f3747;")
        card_title_row.addWidget(card_title)
        card_title_row.addStretch()
        c_layout.addLayout(card_title_row)

        form_grid = QGridLayout()
        form_grid.setHorizontalSpacing(16)
        form_grid.setVerticalSpacing(14)

        # Server URL (BLANK BY DEFAULT!)
        form_grid.addWidget(QLabel("CVAT Server URL:"), 0, 0)
        self.txt_login_url = QLineEdit("")
        self.txt_login_url.setPlaceholderText("http://<your-cvat-server>:8080")
        form_grid.addWidget(self.txt_login_url, 0, 1)

        # Username
        form_grid.addWidget(QLabel("Username:"), 1, 0)
        self.txt_login_user = QLineEdit("")
        self.txt_login_user.setPlaceholderText("Enter your CVAT username")
        form_grid.addWidget(self.txt_login_user, 1, 1)

        # Password
        form_grid.addWidget(QLabel("Password:"), 2, 0)
        self.txt_login_pass = QLineEdit("")
        self.txt_login_pass.setEchoMode(QLineEdit.Password)
        self.txt_login_pass.setPlaceholderText("Enter your CVAT password")
        self.txt_login_pass.returnPressed.connect(self._on_do_login)
        form_grid.addWidget(self.txt_login_pass, 2, 1)

        c_layout.addLayout(form_grid)

        # Login Button
        self.btn_do_login = QPushButton("Connect to CVAT Server")
        self.btn_do_login.setProperty("class", "PrimaryButton")
        self.btn_do_login.setIcon(get_svg_icon("key", 20))
        self.btn_do_login.setIconSize(QSize(20, 20))
        self.btn_do_login.setMinimumHeight(44)
        self.btn_do_login.clicked.connect(self._on_do_login)
        c_layout.addWidget(self.btn_do_login)

        # Status text
        self.lbl_login_status = QLabel("Enter your CVAT Server URL and credentials above, then click 'Connect to CVAT Server'.")
        self.lbl_login_status.setStyleSheet("color: #64748b; font-weight: 600; min-height: 24px;")
        c_layout.addWidget(self.lbl_login_status)

        # Connected Info Box
        self.box_auth_info = QFrame()
        self.box_auth_info.setStyleSheet("background-color: #f8fafc; border: 1.5px dashed #cbd5e1; border-radius: 12px; padding: 12px;")
        self.box_auth_info.setVisible(False)
        bai_layout = QGridLayout(self.box_auth_info)
        bai_layout.addWidget(QLabel("<b>Connection Status:</b>"), 0, 0)
        self.lbl_auth_conn_status = QLabel("Connected")
        self.lbl_auth_conn_status.setStyleSheet("color: #16a34a; font-weight: 700;")
        bai_layout.addWidget(self.lbl_auth_conn_status, 0, 1)

        bai_layout.addWidget(QLabel("<b>Logged in as:</b>"), 0, 2)
        self.lbl_auth_user = QLabel("-")
        bai_layout.addWidget(self.lbl_auth_user, 0, 3)

        bai_layout.addWidget(QLabel("<b>Server:</b>"), 1, 0)
        self.lbl_auth_server = QLabel("-")
        bai_layout.addWidget(self.lbl_auth_server, 1, 1)

        bai_layout.addWidget(QLabel("<b>Tasks Available:</b>"), 1, 2)
        self.lbl_auth_tasks_count = QLabel("0 tasks")
        bai_layout.addWidget(self.lbl_auth_tasks_count, 1, 3)

        c_layout.addWidget(self.box_auth_info)
        layout.addWidget(card)

        # Advance navigation row
        advance_row = QHBoxLayout()
        advance_row.addStretch()
        self.btn_login_next = QPushButton("Next: Select & Download Tasks ➔")
        self.btn_login_next.setProperty("class", "StepNavButton")
        self.btn_login_next.clicked.connect(lambda: self._go_to_step(1))
        advance_row.addWidget(self.btn_login_next)
        layout.addLayout(advance_row)

        layout.addStretch()
        return page

    # -----------------------------------------------------------------
    # PAGE 2: CVAT Downloader (Dedicated Standalone Screen)
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
        self.btn_page1_next.clicked.connect(lambda: self._go_to_step(2))
        status_row.addWidget(self.btn_page1_next)

        layout.addLayout(status_row)
        return page

    # -----------------------------------------------------------------
    # PAGE 3: Merge Datasets (Dedicated Standalone Screen)
    # -----------------------------------------------------------------
    def _build_page_merge(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(14)

        # Zen Stones Quote Card
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
        self.btn_page2_next.clicked.connect(lambda: self._go_to_step(3))
        advance_row.addWidget(self.btn_page2_next)
        layout.addLayout(advance_row)

        layout.addStretch()
        return page

    # -----------------------------------------------------------------
    # PAGE 4: COCO to YOLO Converter (Dedicated Standalone Screen)
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
        self.btn_page3_next.clicked.connect(lambda: self._go_to_step(4))
        advance_row.addWidget(self.btn_page3_next)
        layout.addLayout(advance_row)

        layout.addStretch()
        return page

    # -----------------------------------------------------------------
    # PAGE 5: Dataset Splitter (Dedicated Standalone Screen)
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
        self.btn_page4_next.clicked.connect(lambda: self._go_to_step(5))
        advance_row.addWidget(self.btn_page4_next)
        layout.addLayout(advance_row)

        layout.addStretch()
        return page

    # -----------------------------------------------------------------
    # PAGE 6: YOLO Auto-Trainer (Dedicated Standalone Screen)
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

        # Output preview notice (NO EXTRA WEIGHTS FOLDER!)
        self.lbl_run_preview = QLabel("Output Preview: runs/detect/my_yolo_model_v1/weights/my_yolo_model_v1.pt")
        self.lbl_run_preview.setStyleSheet("color: #0f766e; font-weight: 700; background: #f0fdfa; padding: 6px 12px; border-radius: 8px;")
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
        self.lbl_train_metrics.setStyleSheet("color: #1e293b; font-weight: 600;")
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

        advance_row = QHBoxLayout()
        advance_row.addStretch()
        self.btn_page5_next = QPushButton("Next: Test Model ➔")
        self.btn_page5_next.setProperty("class", "StepNavButton")
        self.btn_page5_next.clicked.connect(lambda: self._go_to_step(6))
        advance_row.addWidget(self.btn_page5_next)
        layout.addLayout(advance_row)

        self._update_model_presets()
        return page

    # -----------------------------------------------------------------
    # PAGE 7: Test Model / Inference (Dedicated Standalone Screen)
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
        self.rb_src_image.setChecked(True)

        self.btn_grp_source = QButtonGroup(self)
        self.btn_grp_source.addButton(self.rb_src_image, 0)
        self.btn_grp_source.addButton(self.rb_src_folder, 1)
        self.btn_grp_source.addButton(self.rb_src_video, 2)
        self.btn_grp_source.buttonToggled.connect(self._on_test_source_mode_changed)

        src_mode_box.addWidget(self.rb_src_image)
        src_mode_box.addWidget(self.rb_src_folder)
        src_mode_box.addWidget(self.rb_src_video)
        src_mode_box.addStretch()
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

        # 5. Save results checkbox & destination folder
        self.chk_save_test = QCheckBox("Save Test Prediction Results")
        self.chk_save_test.setChecked(True)
        self.chk_save_test.setStyleSheet("font-weight: 700; color: #0f3747;")
        cg_layout.addWidget(self.chk_save_test, 4, 0)

        self.txt_test_save_dir = QLineEdit(str(self.test_results_dir))
        cg_layout.addWidget(self.txt_test_save_dir, 4, 1, 1, 4)
        btn_browse_save_dir = QPushButton("Browse...")
        btn_browse_save_dir.setIcon(get_svg_icon("folder", 16))
        btn_browse_save_dir.setIconSize(QSize(16, 16))
        btn_browse_save_dir.clicked.connect(lambda: self._browse_folder(self.txt_test_save_dir))
        cg_layout.addWidget(btn_browse_save_dir, 4, 5)

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

        cg_layout.addLayout(act_row, 5, 0, 1, 6)

        self.prog_test = QProgressBar()
        self.prog_test.setValue(0)
        cg_layout.addWidget(self.prog_test, 6, 0, 1, 6)

        self.lbl_test_status = QLabel("Select model weights and input source above, then click 'Run Model Test / Predict'.")
        self.lbl_test_status.setStyleSheet("color: #64748b; font-weight: 600;")
        cg_layout.addWidget(self.lbl_test_status, 7, 0, 1, 6)

        layout.addWidget(cfg_card)

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
        lbl_canvas_title.setStyleSheet("font-weight: 700; color: #0f3747; font-size: 14px;")
        canvas_top.addWidget(lbl_canvas_title)
        canvas_top.addStretch()

        # Folder Carousel Navigator
        self.btn_prev_test_img = QPushButton("< Prev")
        self.btn_prev_test_img.setEnabled(False)
        self.btn_prev_test_img.clicked.connect(self._on_prev_test_img)
        canvas_top.addWidget(self.btn_prev_test_img)

        self.lbl_test_img_counter = QLabel("0 / 0")
        self.lbl_test_img_counter.setStyleSheet("font-weight: 700; color: #475569;")
        canvas_top.addWidget(self.lbl_test_img_counter)

        self.btn_next_test_img = QPushButton("Next >")
        self.btn_next_test_img.setEnabled(False)
        self.btn_next_test_img.clicked.connect(self._on_next_test_img)
        canvas_top.addWidget(self.btn_next_test_img)

        canvas_col.addLayout(canvas_top)

        # Canvas Frame
        canvas_frame = QFrame()
        canvas_frame.setStyleSheet("background-color: #0f172a; border-radius: 12px; border: 1.5px solid #334155;")
        cf_layout = QVBoxLayout(canvas_frame)
        cf_layout.setContentsMargins(4, 4, 4, 4)

        self.lbl_test_preview = QLabel("Visual evaluation result will appear here")
        self.lbl_test_preview.setAlignment(Qt.AlignCenter)
        self.lbl_test_preview.setStyleSheet("color: #94a3b8; font-weight: 600;")
        self.lbl_test_preview.setMinimumSize(480, 320)
        cf_layout.addWidget(self.lbl_test_preview)
        canvas_col.addWidget(canvas_frame, stretch=1)

        ic_layout.addLayout(canvas_col, stretch=3)

        # Right: Detection Summary & Table
        table_col = QVBoxLayout()
        lbl_det_table_title = QLabel("Detected Objects & Metrics")
        lbl_det_table_title.setStyleSheet("font-weight: 700; color: #0f3747; font-size: 14px;")
        table_col.addWidget(lbl_det_table_title)

        self.lbl_test_classes_badge = QLabel("No detections yet")
        self.lbl_test_classes_badge.setStyleSheet("color: #64748b; font-weight: 600; padding: 4px;")
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

    # -----------------------------------------------------------------
    # Step Navigation Logic (1 to 7)
    # -----------------------------------------------------------------
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

        # Update Top App Title
        self.lbl_top_title.setText(self.PAGE_TITLES[step_idx])
        self.btn_top_back.setEnabled(step_idx > 0)

        # Update Greeting Header
        tag_class, tag_text = self.PAGE_TAGS[step_idx]
        self.lbl_page_tag.setProperty("class", tag_class)
        self.lbl_page_tag.setText(tag_text)
        self.lbl_page_tag.style().unpolish(self.lbl_page_tag)
        self.lbl_page_tag.style().polish(self.lbl_page_tag)

        self.lbl_greeting_title.setText(f"{self.PAGE_TITLES[step_idx]} 👋")
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

        # Refresh preview if going to Train (Step 6 / index 5)
        if step_idx == 5:
            self._refresh_yaml_preview()
            self._update_run_preview()

        # Pre-fill model path if going to Test (Step 7 / index 6)
        if step_idx == 6 and self.last_trained_weight and self.last_trained_weight.exists():
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
    # Step 1: CVAT Authentication Slots
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
        username = login_info.get("username") or self.txt_login_user.text().strip() or "User"
        server_url = login_info.get("server_url") or self.txt_login_url.text().strip()
        self.lbl_login_status.setText(f"Connected successfully as '{username}'!")

        # Update info box
        self.box_auth_info.setVisible(True)
        self.lbl_auth_conn_status.setText("Online & Authenticated")
        self.lbl_auth_user.setText(username)
        self.lbl_auth_server.setText(server_url)
        self.lbl_auth_tasks_count.setText(f"{len(tasks)} tasks across {len(projects)} projects")

        # Populate projects dropdown in Step 2
        self.cmb_proj_filter.blockSignals(True)
        self.cmb_proj_filter.clear()
        self.cmb_proj_filter.addItem("All Projects", None)
        for p in projects:
            self.cmb_proj_filter.addItem(f"[{p['id']}] {p['name']}", p['id'])
        self.cmb_proj_filter.blockSignals(False)

        self.all_tasks_data = tasks
        self._populate_tasks_table(self.all_tasks_data)
        self.lbl_cvat_status.setText(f"Loaded {len(tasks)} tasks across {len(projects)} projects.")

        QMessageBox.information(self, "Connected", f"Connected to CVAT Server successfully!\nLoaded {len(tasks)} tasks.")

    def _on_auth_error(self, err_msg: str):
        self.btn_do_login.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_IDLE)
        self.lbl_login_status.setText(f"Connection Failed: {err_msg}")
        QMessageBox.critical(self, "Connection Error", f"Unable to connect to CVAT server:\n{err_msg}")

    # -----------------------------------------------------------------
    # Step 2: CVAT Downloader Slots
    # -----------------------------------------------------------------
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
            QMessageBox.warning(self, "Not Connected", "Please connect to CVAT in Step 1 before downloading tasks.")
            self._go_to_step(0)
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
        self.lbl_cvat_status.setText(f"Downloaded and extracted {len(results)} tasks into data/raw_tasks!")
        QMessageBox.information(
            self,
            "Download Finished",
            f"Successfully downloaded and extracted {len(results)} task datasets into:\n{self.txt_dl_folder.text()}"
        )

    def _on_download_error(self, err_msg: str):
        self.btn_download_tasks.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_IDLE)
        self.lbl_cvat_status.setText(f"Error: {err_msg}")
        QMessageBox.critical(self, "Download Error", f"Task download failed:\n{err_msg}")

    # -----------------------------------------------------------------
    # Step 3: Merge Datasets Slots
    # -----------------------------------------------------------------
    def _on_run_merge(self):
        src = Path(self.txt_merge_src.text().strip())
        dst = Path(self.txt_merge_dst.text().strip())

        if not src.exists():
            QMessageBox.warning(self, "Path Not Found", f"Source directory does not exist:\n{src}")
            return

        self.btn_run_merge.setEnabled(False)
        self.prog_merge.setValue(0)
        self.lbl_merge_stats.setText("Merging datasets...")
        self.mascot.set_state(DancingSvgMascot.STATE_DOWNLOADING)

        self.active_worker = MergeWorker(self.merger_engine, src, dst)
        self.active_worker.progress_signal.connect(lambda c, t, d: (self.prog_merge.setMaximum(t), self.prog_merge.setValue(c), self.lbl_merge_stats.setText(d)))
        self.active_worker.finished_signal.connect(self._on_merge_finished)
        self.active_worker.error_signal.connect(self._on_merge_error)
        self.active_worker.start()

    def _on_merge_finished(self, stats: dict):
        self.btn_run_merge.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_SUCCESS)
        self.prog_merge.setValue(100)

        summary = (
            f"Merged: {stats.get('total_tasks', 0)} tasks | "
            f"Images: {stats.get('total_images', 0)} | "
            f"Annotations: {stats.get('total_annotations', 0)} | "
            f"Categories: {stats.get('total_categories', 0)}"
        )
        self.lbl_merge_stats.setText(summary)
        QMessageBox.information(self, "Merge Successful", f"Dataset Merge Completed!\n\n{summary}\nOutput: {self.txt_merge_dst.text()}")

    def _on_merge_error(self, err_msg: str):
        self.btn_run_merge.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_IDLE)
        self.lbl_merge_stats.setText(f"Merge Error: {err_msg}")
        QMessageBox.critical(self, "Merge Error", f"Merge failed:\n{err_msg}")

    # -----------------------------------------------------------------
    # Step 4: COCO to YOLO Converter Slots
    # -----------------------------------------------------------------
    def _on_task_mode_changed(self):
        is_det = self.rb_mode_det.isChecked()
        self.card_det.setProperty("class", "ChoiceCardNavy" if is_det else "ChoiceCardPink")
        self.card_seg.setProperty("class", "ChoiceCardPink" if is_det else "ChoiceCardNavy")
        self.card_det.style().unpolish(self.card_det)
        self.card_det.style().polish(self.card_det)
        self.card_seg.style().unpolish(self.card_seg)
        self.card_seg.style().polish(self.card_seg)

    def _on_run_convert(self):
        json_path = Path(self.txt_conv_json.text().strip())
        labels_dst = Path(self.txt_conv_labels_dst.text().strip())
        mode = "detect" if self.rb_mode_det.isChecked() else "segment"

        if not json_path.exists():
            QMessageBox.warning(self, "File Not Found", f"COCO JSON file not found:\n{json_path}")
            return

        self.btn_run_convert.setEnabled(False)
        self.prog_convert.setValue(0)
        self.lbl_conv_stats.setText(f"Converting in [{mode.upper()}] mode...")
        self.mascot.set_state(DancingSvgMascot.STATE_DOWNLOADING)

        self.active_worker = ConvertWorker(self.converter_engine, json_path, labels_dst, mode)
        self.active_worker.progress_signal.connect(lambda c, t, d: (self.prog_convert.setMaximum(t), self.prog_convert.setValue(c), self.lbl_conv_stats.setText(d)))
        self.active_worker.finished_signal.connect(self._on_convert_finished)
        self.active_worker.error_signal.connect(self._on_convert_error)
        self.active_worker.start()

    def _on_convert_finished(self, stats: dict):
        self.btn_run_convert.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_SUCCESS)
        self.prog_convert.setValue(100)

        summary = (
            f"Converted: {stats.get('total_labels', 0)} label files | "
            f"Annotations: {stats.get('total_annotations', 0)} | "
            f"Mode: {stats.get('mode', 'unknown')}"
        )
        self.lbl_conv_stats.setText(summary)
        QMessageBox.information(self, "Conversion Complete", f"COCO to YOLO Conversion Completed!\n\n{summary}\nOutput: {self.txt_conv_labels_dst.text()}")

    def _on_convert_error(self, err_msg: str):
        self.btn_run_convert.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_IDLE)
        self.lbl_conv_stats.setText(f"Conversion Error: {err_msg}")
        QMessageBox.critical(self, "Conversion Error", f"Conversion failed:\n{err_msg}")

    # -----------------------------------------------------------------
    # Step 5: Dataset Splitter Slots
    # -----------------------------------------------------------------
    def _on_run_split(self):
        img_src = Path(self.txt_split_img_src.text().strip())
        lbl_src = Path(self.txt_split_lbl_src.text().strip())
        dst = Path(self.txt_split_dst.text().strip())

        if not img_src.exists():
            QMessageBox.warning(self, "Images Not Found", f"Images directory not found:\n{img_src}")
            return

        r_train = self.spin_train.value() / 100.0
        r_val = self.spin_val.value() / 100.0
        r_test = self.spin_test.value() / 100.0
        total_r = r_train + r_val + r_test

        if abs(total_r - 1.0) > 0.01:
            QMessageBox.warning(self, "Invalid Ratios", f"Train + Val + Test ratios must sum to 100% (currently {int(total_r*100)}%).")
            return

        self.btn_run_split.setEnabled(False)
        self.prog_split.setValue(0)
        self.lbl_split_stats.setText("Partitioning dataset (100% images included)...")
        self.mascot.set_state(DancingSvgMascot.STATE_DOWNLOADING)

        self.active_worker = SplitWorker(
            self.splitter_engine, img_src, lbl_src, dst, (r_train, r_val, r_test), copy_mode=True
        )
        self.active_worker.progress_signal.connect(lambda c, t, d: (self.prog_split.setMaximum(t), self.prog_split.setValue(c), self.lbl_split_stats.setText(d)))
        self.active_worker.finished_signal.connect(self._on_split_finished)
        self.active_worker.error_signal.connect(self._on_split_error)
        self.active_worker.start()

    def _on_split_finished(self, stats: dict):
        self.btn_run_split.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_SUCCESS)
        self.prog_split.setValue(100)

        summary = (
            f"Train: {stats.get('train_count', 0)} | "
            f"Val: {stats.get('val_count', 0)} | "
            f"Test: {stats.get('test_count', 0)} | "
            f"Total: {stats.get('total_images', 0)} images | "
            f"data.yaml generated!"
        )
        self.lbl_split_stats.setText(summary)
        self._refresh_yaml_preview()
        QMessageBox.information(self, "Split Complete", f"Dataset Partition Completed!\n\n{summary}\nOutput: {self.txt_split_dst.text()}")

    def _on_split_error(self, err_msg: str):
        self.btn_run_split.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_IDLE)
        self.lbl_split_stats.setText(f"Split Error: {err_msg}")
        QMessageBox.critical(self, "Split Error", f"Split failed:\n{err_msg}")

    # -----------------------------------------------------------------
    # Step 6: YOLO Auto-Trainer Slots
    # -----------------------------------------------------------------
    def _update_model_presets(self):
        task = self.cmb_train_task.currentText()
        self.cmb_model_preset.clear()
        if task == "detect":
            presets = [
                "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt",
                "yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
                "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt"
            ]
        else:
            presets = [
                "yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt",
                "yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt",
                "yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt", "yolo26l-seg.pt"
            ]
        self.cmb_model_preset.addItems(presets)
        self._update_run_preview()

    def _update_run_preview(self):
        task = self.cmb_train_task.currentText()
        custom_name = self.txt_custom_run_name.text().strip() or "my_yolo_model"
        base_runs = Path(self.txt_runs_base.text().strip()) / task
        next_version_name, _ = get_next_run_version(base_runs, custom_name)

        # EXACT USER REQUIREMENT:
        # Renamed inside weights folder: runs/detect/[custom_name]_v[run_num]/weights/[custom_name]_v[run_num].pt
        # NO separate root weights folder!
        self.lbl_run_preview.setText(
            f"Weights Destination: runs/{task}/{next_version_name}/weights/{next_version_name}.pt (and last.pt)"
        )

    def _on_auto_tune_clicked(self):
        yaml_path = Path(self.txt_data_yaml.text().strip())
        if not yaml_path.exists():
            QMessageBox.warning(self, "File Not Found", f"data.yaml not found at:\n{yaml_path}")
            return

        params = auto_tune_hyperparameters(yaml_path)
        self.spin_epochs.setValue(params["epochs"])
        self.spin_batch.setValue(params["batch"])
        self.spin_imgsz.setValue(params["imgsz"])
        self.spin_patience.setValue(params["patience"])
        self.spin_lr0.setValue(params["lr0"])

        QMessageBox.information(
            self,
            "Auto-Tune Completed",
            f"Recommended Hyperparameters applied based on dataset size:\n\n"
            f"• Epochs: {params['epochs']}\n"
            f"• Batch Size: {params['batch']}\n"
            f"• Image Size: {params['imgsz']}\n"
            f"• Early Stopping (Patience): {params['patience']}\n"
            f"• Learning Rate (lr0): {params['lr0']}"
        )

    def _on_start_training(self):
        yaml_path = Path(self.txt_data_yaml.text().strip())
        if not yaml_path.exists():
            QMessageBox.warning(self, "Missing data.yaml", f"Cannot find dataset configuration at:\n{yaml_path}")
            return

        task = self.cmb_train_task.currentText()
        custom_pt = self.txt_custom_model.text().strip()
        model_name = custom_pt if (custom_pt and Path(custom_pt).exists()) else self.cmb_model_preset.currentText()
        run_name = self.txt_custom_run_name.text().strip() or "yolo_model"
        base_runs_dir = Path(self.txt_runs_base.text().strip())

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
        self.mascot.set_state(DancingSvgMascot.STATE_TRAINING)

        self.active_worker = TrainWorker(
            self.trainer_engine, task, model_name, yaml_path, run_name, base_runs_dir, hyperparams
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
        self.mascot.set_state(DancingSvgMascot.STATE_SUCCESS)
        self.prog_train.setValue(self.prog_train.maximum())
        self._update_run_preview()

        run_dir = res.get("run_dir")
        best_pt = res.get("best_pt")
        last_pt = res.get("last_pt")

        if best_pt:
            self.last_trained_weight = Path(best_pt)
            self.txt_test_model.setText(str(best_pt))

        summary = (
            f"🎉 YOLO Training Finished Successfully!\n\n"
            f"• Output Run Folder: {run_dir}\n"
            f"• Renamed Best Weights: {best_pt}\n"
            f"• Last Checkpoint: {last_pt}\n\n"
            f"The best weights have been renamed inside {run_dir / 'weights'}."
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

    # -----------------------------------------------------------------
    # Step 7: Model Test / Inference Slots
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

    def _on_run_test(self):
        model_p = Path(self.txt_test_model.text().strip())
        src_p = Path(self.txt_test_source.text().strip())
        save_enabled = self.chk_save_test.isChecked()
        save_dir = Path(self.txt_test_save_dir.text().strip())

        if not model_p.exists():
            QMessageBox.warning(self, "Model Not Found", f"Model weights not found:\n{model_p}")
            return

        if not src_p.exists():
            QMessageBox.warning(self, "Source Not Found", f"Input test source not found:\n{src_p}")
            return

        mode_id = self.btn_grp_source.checkedId()
        mode_str = "image" if mode_id == 0 else ("folder" if mode_id == 1 else "video")

        conf = self.spin_test_conf.value()
        iou = self.spin_test_iou.value()

        self.btn_run_test.setEnabled(False)
        self.btn_stop_test.setEnabled(True)
        self.prog_test.setValue(0)
        self.lbl_test_status.setText(f"Running inference ({mode_str.upper()})...")
        self.mascot.set_state(DancingSvgMascot.STATE_DOWNLOADING)

        self.active_worker = TestInferenceWorker(
            self.tester_engine, mode_str, model_p, src_p, conf, iou, save_enabled, save_dir
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
        self.prog_test.setMaximum(tot)
        self.prog_test.setValue(cur)
        self.lbl_test_status.setText(desc)

    def _on_test_frame(self, frame_bgr: np.ndarray, cur: int, tot: int, summary: dict):
        pix = bgr_to_qpixmap(frame_bgr)
        if not pix.isNull():
            scaled = pix.scaled(self.lbl_test_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_test_preview.setPixmap(scaled)

        # Update counter
        self.lbl_test_img_counter.setText(f"Frame {cur} / {tot}")

        # Update summary text
        badge_text = ", ".join([f"{k}: {v}" for k, v in summary.items()]) or "No objects detected in frame"
        self.lbl_test_classes_badge.setText(f"Active Objects: {badge_text}")

    def _on_test_finished(self, out: dict):
        self.btn_run_test.setEnabled(True)
        self.btn_stop_test.setEnabled(False)
        self.mascot.set_state(DancingSvgMascot.STATE_SUCCESS)
        self.prog_test.setValue(self.prog_test.maximum())

        mode = out.get("mode")
        data = out.get("data", {})

        if mode == "image":
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

        QMessageBox.information(self, "Test Complete", f"Model prediction completed successfully!\nMode: {mode.upper()}")

    def _show_folder_test_item(self, idx: int):
        if not self.folder_test_results or idx < 0 or idx >= len(self.folder_test_results):
            return

        self.current_folder_test_idx = idx
        item = self.folder_test_results[idx]

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

    def _on_test_error(self, err_msg: str):
        self.btn_run_test.setEnabled(True)
        self.btn_stop_test.setEnabled(False)
        self.mascot.set_state(DancingSvgMascot.STATE_IDLE)
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
        f, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model Weights", "", "PyTorch Weights (*.pt)")
        if f:
            self.txt_custom_model.setText(f)

    def _browse_data_yaml(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select data.yaml", self.txt_data_yaml.text(), "YAML Files (*.yaml *.yml)")
        if f:
            self.txt_data_yaml.setText(f)
            self._refresh_yaml_preview()
