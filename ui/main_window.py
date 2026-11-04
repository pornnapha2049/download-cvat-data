"""
Main Window for the CVAT-to-YOLO GUI application.
All Pages are 100% Standalone Tools with a Single Unified 'data' Directory:
1. CVAT Downloader (Server connection with NO default URL, filtering, downloading & cleanup of /images/default)
2. Merge Datasets (Deduplicates images with MD5 and unifies COCO 1.0 annotations)
3. COCO to YOLO Converter (Detection Bounding Box or Segmentation Polygon)
4. Dataset Splitter (Train/Val/Test partitions including all background images, generates data.yaml)
5. YOLO Auto-Trainer & Explorer (Auto-Tune, Early Stopping, versioned runs, and custom weights)
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
    finished_signal = pyqtSignal(dict, list, list)  # login_info, projects, tasks
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
# Main GUI Window (100% Standalone Tools Architecture)
# =====================================================================

class MainWindow(QMainWindow):
    STEP_DESCRIPTIONS = [
        ("CVAT Downloader", "เชื่อมต่อและดาวน์โหลด Tasks จาก CVAT (COCO 1.0 + Clean /images/default)"),
        ("Merge Datasets", "รวม Datasets หลายๆ Task เข้าด้วยกัน และแก้ชื่อภาพซ้ำด้วย MD5"),
        ("COCO to YOLO", "แปลง Annotation COCO 1.0 เป็น YOLO Detection (.txt) หรือ Segmentation"),
        ("Dataset Splitter", "แบ่ง Train/Val/Test โดยนำทุกรูปภาพเข้าร่วม 100% (รวม Backgrounds)"),
        ("YOLO Auto-Trainer", "Auto-Tune โมเดล, Train พร้อม Early Stopping และจัดการ Weights"),
    ]

    STEP_GUIDES = [
        ".☘︎ ݁˖ เครื่องมือที่ 1 (CVAT Downloader): กรอก URL, Username, Password แล้วคลิก 'Connect & Fetch' เพื่อดึงรายการ Tasks จากนั้นเลือก Tasks แล้วกดดาวน์โหลด ระบบจะแตกไฟล์และย้ายรูปออกจาก /images/default ให้เองครับ",
        ".☘︎ ݁˖ เครื่องมือที่ 2 (Merge Datasets): เลือกโฟลเดอร์ต้นทางที่มีหลายๆ Task แล้วกด Merge ระบบจะรวมไฟล์ภาพและ Annotations เข้าเป็น Dataset เดียวกันในโฟลเดอร์ data/ พร้อมแก้ชื่อไฟล์ซ้ำด้วย MD5",
        ".☘︎ ݁˖ เครื่องมือที่ 3 (COCO to YOLO): เลือกไฟล์ COCO JSON และโหมด Detection (แปลง Polygon เป็น Bounding Box) หรือ Segmentation จากนั้นกด Convert เพื่อสร้างไฟล์ Label .txt",
        ".☘︎ ݁˖ เครื่องมือที่ 4 (Dataset Splitter): กำหนดสัดส่วน Train / Val / Test (รูปภาพทั้งหมด 100% จะถูกนำมาใช้งาน รวมถึงภาพ Background ที่ไม่มี Label) จากนั้นกด Split เพื่อสร้าง data.yaml",
        ".☘︎ ݁˖ เครื่องมือที่ 5 (YOLO Auto-Trainer): กดปุ่ม '⚡ Auto-Tune Hyperparameters' เพื่อตั้งค่าโมเดลให้เข้ากับข้อมูล ตั้งชื่อโฟลเดอร์รัน แล้วเริ่ม Train โมเดล YOLO ได้ทันที",
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle(".☘︎ ݁˖ CVAT to YOLO Studio • Standalone AI Pipeline")
        self.resize(1260, 860)
        self.setMinimumSize(940, 620)

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
        self.step_completed = [False, False, False, False, False]
        self.active_worker = None

        self._init_ui()
        self.setStyleSheet(DARK_STYLESHEET)
        self._go_to_step(0)

    def _wrap_page_scroll(self, widget: QWidget) -> QScrollArea:
        """Wraps any standalone page inside a responsive QScrollArea so content never clips."""
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
        root_layout.setContentsMargins(16, 12, 16, 12)
        root_layout.setSpacing(10)

        # 1. Header with Animated Mascot, Hardware info & Quick Folder Action
        root_layout.addWidget(self._build_studio_header())

        # 2. Standalone Page Navigation Bar (5 Tools)
        root_layout.addWidget(self._build_stepper_bar())

        # 3. Dynamic Guide Box
        root_layout.addWidget(self._build_guide_box())

        # 4. Multi-Page Container (QStackedWidget with Responsive QScrollArea)
        self.pages = QStackedWidget()
        self.pages.addWidget(self._wrap_page_scroll(self._build_page_cvat()))
        self.pages.addWidget(self._wrap_page_scroll(self._build_page_merge()))
        self.pages.addWidget(self._wrap_page_scroll(self._build_page_convert()))
        self.pages.addWidget(self._wrap_page_scroll(self._build_page_split()))
        self.pages.addWidget(self._wrap_page_scroll(self._build_page_train()))
        root_layout.addWidget(self.pages, stretch=1)

        # 5. Bottom Stepper Footer Navigation (< Previous Tool | Tool X of 5 | Next Tool >)
        root_layout.addWidget(self._build_stepper_footer())

        # 6. Global Status Bar
        root_layout.addWidget(self._build_global_status_bar())

    # -----------------------------------------------------------------
    # Studio Header
    # -----------------------------------------------------------------
    def _build_studio_header(self) -> QFrame:
        card = QFrame()
        card.setProperty("class", "HeaderCard")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(18, 10, 18, 10)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(3)

        title_row = QHBoxLayout()
        title_lbl = QLabel(".☘︎ ݁˖ CVAT to YOLO Studio • Standalone AI Pipeline")
        title_lbl.setObjectName("titleLabel")
        title_row.addWidget(title_lbl)

        self.wave_indicator = AnimatedWavePill(42, 22)
        title_row.addWidget(self.wave_indicator)
        title_row.addStretch()
        text_layout.addLayout(title_row)

        subtitle_lbl = QLabel(
            "All-in-One Standalone Pipeline: CVAT Downloader ➔ Dataset Merger ➔ "
            "COCO-to-YOLO (Det/Seg) ➔ Full Dataset Split ➔ YOLO Auto-Trainer"
        )
        subtitle_lbl.setObjectName("subtitleLabel")
        text_layout.addWidget(subtitle_lbl)

        profile_row = QHBoxLayout()
        self.lbl_server_icon = QLabel()
        self.lbl_server_icon.setPixmap(get_svg_pixmap("server", 18, 18))
        profile_row.addWidget(self.lbl_server_icon)

        self.lbl_server_info = QLabel("CVAT Server: Not Connected")
        self.lbl_server_info.setStyleSheet("color: #64748b; font-weight: 700; font-size: 12px; min-height: 20px;")
        profile_row.addWidget(self.lbl_server_info)

        btn_open_data = QPushButton("Open data/ Folder")
        btn_open_data.setProperty("class", "LogoutButton")
        btn_open_data.setIcon(get_svg_icon("folder", 15))
        btn_open_data.setIconSize(QSize(15, 15))
        btn_open_data.setCursor(Qt.PointingHandCursor)
        btn_open_data.clicked.connect(lambda: self._open_dir(self.data_dir))
        profile_row.addWidget(btn_open_data)

        profile_row.addSpacing(14)

        self.lbl_hw_icon = QLabel()
        self.lbl_hw_icon.setPixmap(get_svg_pixmap("hardware", 18, 18))
        profile_row.addWidget(self.lbl_hw_icon)

        cuda_ok = torch.cuda.is_available()
        hw_text = f"{'NVIDIA GPU (' + torch.cuda.get_device_name(0) + ')' if cuda_ok else 'CPU Mode'}"
        self.hw_badge = QLabel(hw_text)
        self.hw_badge.setProperty("class", "BadgeSuccess" if cuda_ok else "BadgeInfo")
        profile_row.addWidget(self.hw_badge)
        profile_row.addStretch()

        text_layout.addLayout(profile_row)
        layout.addLayout(text_layout, stretch=1)

        # Mascot Box
        mascot_box = QVBoxLayout()
        mascot_box.setAlignment(Qt.AlignCenter)

        self.mascot = DancingSvgMascot(size=105)
        mascot_box.addWidget(self.mascot, alignment=Qt.AlignCenter)

        self.chk_mascot = QCheckBox("SVG Mascot")
        self.chk_mascot.setChecked(True)
        self.chk_mascot.toggled.connect(self.mascot.toggle_animation)
        mascot_box.addWidget(self.chk_mascot, alignment=Qt.AlignCenter)

        layout.addLayout(mascot_box)
        return card

    # -----------------------------------------------------------------
    # Stepper Navigation Bar (5 Standalone Tool Buttons)
    # -----------------------------------------------------------------
    def _build_stepper_bar(self) -> QFrame:
        bar = QFrame()
        bar.setProperty("class", "StepperContainer")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.step_buttons: list[QPushButton] = []
        step_config = [
            ("step_cvat", "1. CVAT Downloader", "Connect & Download Tasks"),
            ("step_merge", "2. Merge Datasets", "Deduplicate & Unify COCO"),
            ("step_convert", "3. COCO to YOLO", "Det (BBox) / Seg (Poly)"),
            ("step_split", "4. Dataset Splitter", "Train/Val/Test (All Imgs)"),
            ("step_train", "5. YOLO Auto-Trainer", "Train, Auto-Tune & Explorer"),
        ]

        for idx, (icon_name, title, sub) in enumerate(step_config):
            btn = QPushButton()
            btn.setProperty("class", "StepButton")
            btn.setIcon(get_svg_icon(icon_name, 26))
            btn.setIconSize(QSize(26, 26))
            btn.setText(f"{title}\n{sub}")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, step_idx=idx: self._go_to_step(step_idx))
            layout.addWidget(btn, stretch=1)
            self.step_buttons.append(btn)

        return bar

    # -----------------------------------------------------------------
    # Guide Box
    # -----------------------------------------------------------------
    def _build_guide_box(self) -> QFrame:
        box = QFrame()
        box.setProperty("class", "GuideBox")
        layout = QHBoxLayout(box)
        layout.setContentsMargins(12, 6, 12, 6)

        self.lbl_guide = QLabel()
        self.lbl_guide.setObjectName("guideLabel")
        self.lbl_guide.setWordWrap(True)
        layout.addWidget(self.lbl_guide, stretch=1)
        return box

    # -----------------------------------------------------------------
    # Page 1: CVAT Downloader (Standalone)
    # -----------------------------------------------------------------
    def _build_page_cvat(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # 1. Connection Card (NO DEFAULT URL!)
        conn_box = QGroupBox("CVAT Server Authentication & Connection")
        conn_layout = QGridLayout(conn_box)
        conn_layout.setHorizontalSpacing(10)
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
        self.lbl_login_status.setStyleSheet("color: #64748b; font-weight: 600; min-height: 22px;")
        conn_layout.addWidget(self.lbl_login_status, 1, 0, 1, 6)

        btn_refresh_tasks = QPushButton("Refresh Tasks")
        btn_refresh_tasks.setIcon(get_svg_icon("refresh", 16))
        btn_refresh_tasks.setIconSize(QSize(16, 16))
        btn_refresh_tasks.clicked.connect(self._fetch_cvat_tasks)
        conn_layout.addWidget(btn_refresh_tasks, 1, 6)

        layout.addWidget(conn_box)

        # 2. Filter Card (Project, Stage, State, Search)
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

        # 3. Tasks Table
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

        # 4. Download Destination & Action Bar (Unified data/raw_tasks)
        dl_card = QFrame()
        dl_card.setProperty("class", "CardFrame")
        dl_layout = QHBoxLayout(dl_card)
        dl_layout.setContentsMargins(14, 10, 14, 10)

        dl_layout.addWidget(QLabel("Download Folder:"))
        self.txt_dl_folder = QLineEdit(str(self.data_dir / "raw_tasks"))
        dl_layout.addWidget(self.txt_dl_folder, stretch=1)

        btn_browse_dl = QPushButton("Browse...")
        btn_browse_dl.setIcon(get_svg_icon("folder", 16))
        btn_browse_dl.setIconSize(QSize(16, 16))
        btn_browse_dl.clicked.connect(lambda: self._browse_folder(self.txt_dl_folder))
        dl_layout.addWidget(btn_browse_dl)

        self.btn_download_tasks = QPushButton("Download & Auto-Extract Tasks (COCO 1.0)")
        self.btn_download_tasks.setProperty("class", "PrimaryButton")
        self.btn_download_tasks.setIcon(get_svg_icon("download", 18))
        self.btn_download_tasks.setIconSize(QSize(18, 18))
        self.btn_download_tasks.clicked.connect(self._on_download_selected_tasks)
        dl_layout.addWidget(self.btn_download_tasks)

        layout.addWidget(dl_card)

        # Download Progress & Status
        self.prog_dl = QProgressBar()
        self.prog_dl.setValue(0)
        self.prog_dl.setTextVisible(True)
        layout.addWidget(self.prog_dl)

        status_row = QHBoxLayout()
        self.lbl_cvat_status = QLabel("Select tasks above to download.")
        status_row.addWidget(self.lbl_cvat_status, stretch=1)

        self.btn_next_step_1 = QPushButton("Proceed to Step 2: Merge Datasets")
        self.btn_next_step_1.setProperty("class", "StepNavButton")
        self.btn_next_step_1.setIcon(get_svg_icon("arrow_right", 16))
        self.btn_next_step_1.setIconSize(QSize(16, 16))
        self.btn_next_step_1.clicked.connect(lambda: self._go_to_step(1))
        status_row.addWidget(self.btn_next_step_1)

        layout.addLayout(status_row)
        return page

    # -----------------------------------------------------------------
    # Page 2: Merge Datasets (Standalone)
    # -----------------------------------------------------------------
    def _build_page_merge(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        merge_box = QGroupBox("Standalone Tool: Merge Multiple Tasks into Unified COCO 1.0 Dataset")
        merge_layout = QVBoxLayout(merge_box)
        merge_layout.setSpacing(12)

        desc = QLabel(
            "Combines multiple downloaded CVAT tasks (images & instances_default.json) into a single dataset in the data/ folder. "
            "Automatically resolves duplicate image filenames using MD5 hashes, re-indexes category IDs, and consolidates annotations."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #64748b; font-size: 12.5px; line-height: 1.4;")
        merge_layout.addWidget(desc)

        row_src = QHBoxLayout()
        row_src.addWidget(QLabel("Tasks Source Directory:"))
        self.txt_merge_src = QLineEdit(str(self.data_dir / "raw_tasks"))
        row_src.addWidget(self.txt_merge_src, stretch=1)
        btn_src = QPushButton("Browse...")
        btn_src.setIcon(get_svg_icon("folder", 16))
        btn_src.setIconSize(QSize(16, 16))
        btn_src.clicked.connect(lambda: self._browse_folder(self.txt_merge_src))
        row_src.addWidget(btn_src)
        merge_layout.addLayout(row_src)

        row_dst = QHBoxLayout()
        row_dst.addWidget(QLabel("Merged Output Directory:"))
        self.txt_merge_dst = QLineEdit(str(self.data_dir))
        row_dst.addWidget(self.txt_merge_dst, stretch=1)
        btn_dst = QPushButton("Browse...")
        btn_dst.setIcon(get_svg_icon("folder", 16))
        btn_dst.setIconSize(QSize(16, 16))
        btn_dst.clicked.connect(lambda: self._browse_folder(self.txt_merge_dst))
        row_dst.addWidget(btn_dst)
        merge_layout.addLayout(row_dst)

        merge_action_row = QHBoxLayout()
        self.btn_run_merge = QPushButton("Merge Datasets (Deduplicate Images & Unify Annotations)")
        self.btn_run_merge.setProperty("class", "PrimaryButton")
        self.btn_run_merge.setIcon(get_svg_icon("step_merge", 18))
        self.btn_run_merge.setIconSize(QSize(18, 18))
        self.btn_run_merge.clicked.connect(self._on_run_merge)
        merge_action_row.addWidget(self.btn_run_merge)
        merge_layout.addLayout(merge_action_row)

        self.prog_merge = QProgressBar()
        self.prog_merge.setValue(0)
        merge_layout.addWidget(self.prog_merge)

        self.lbl_merge_stats = QLabel("No merged dataset yet.")
        self.lbl_merge_stats.setStyleSheet("color: #1e293b; font-weight: 600; min-height: 24px;")
        merge_layout.addWidget(self.lbl_merge_stats)

        layout.addWidget(merge_box)

        advance_row = QHBoxLayout()
        advance_row.addStretch()
        self.btn_next_step_2 = QPushButton("Proceed to Step 3: COCO to YOLO Converter")
        self.btn_next_step_2.setProperty("class", "StepNavButton")
        self.btn_next_step_2.setIcon(get_svg_icon("arrow_right", 16))
        self.btn_next_step_2.setIconSize(QSize(16, 16))
        self.btn_next_step_2.clicked.connect(lambda: self._go_to_step(2))
        advance_row.addWidget(self.btn_next_step_2)
        layout.addLayout(advance_row)

        layout.addStretch()
        return page

    # -----------------------------------------------------------------
    # Page 3: COCO to YOLO Converter (Standalone)
    # -----------------------------------------------------------------
    def _build_page_convert(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        conv_box = QGroupBox("Standalone Tool: Convert COCO 1.0 Annotations to YOLO Format (.txt)")
        conv_layout = QVBoxLayout(conv_box)
        conv_layout.setSpacing(12)

        desc = QLabel(
            "Parses any COCO 1.0 JSON and converts all annotations into YOLO normalized format (.txt). "
            "For Detection, segmentation polygons are automatically converted to normalized bounding boxes (xc, yc, w, h). "
            "For Segmentation, exact polygon vertex coordinates are preserved."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #64748b; font-size: 12.5px; line-height: 1.4;")
        conv_layout.addWidget(desc)

        row_json = QHBoxLayout()
        row_json.addWidget(QLabel("COCO JSON File:"))
        self.txt_conv_json = QLineEdit(str(self.data_dir / "annotations" / "instances_default.json"))
        row_json.addWidget(self.txt_conv_json, stretch=1)
        btn_json = QPushButton("Browse...")
        btn_json.setIcon(get_svg_icon("folder", 16))
        btn_json.setIconSize(QSize(16, 16))
        btn_json.clicked.connect(lambda: self._browse_file(self.txt_conv_json, "JSON Files (*.json)"))
        row_json.addWidget(btn_json)
        conv_layout.addLayout(row_json)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("YOLO Task Mode:"))
        self.rb_mode_det = QRadioButton("Detection (Converts Segmentation Polygons to Normalized Bounding Boxes)")
        self.rb_mode_seg = QRadioButton("Segmentation (Preserves Exact Polygon Coordinates)")
        self.rb_mode_det.setChecked(True)
        self.rb_mode_det.toggled.connect(self._on_task_mode_changed)

        mode_row.addWidget(self.rb_mode_det)
        mode_row.addWidget(self.rb_mode_seg)
        mode_row.addStretch()
        conv_layout.addLayout(mode_row)

        row_lbl_dst = QHBoxLayout()
        row_lbl_dst.addWidget(QLabel("Labels Output Folder:"))
        self.txt_conv_labels_dst = QLineEdit(str(self.data_dir / "labels"))
        row_lbl_dst.addWidget(self.txt_conv_labels_dst, stretch=1)
        btn_lbl_dst = QPushButton("Browse...")
        btn_lbl_dst.setIcon(get_svg_icon("folder", 16))
        btn_lbl_dst.setIconSize(QSize(16, 16))
        btn_lbl_dst.clicked.connect(lambda: self._browse_folder(self.txt_conv_labels_dst))
        row_lbl_dst.addWidget(btn_lbl_dst)
        conv_layout.addLayout(row_lbl_dst)

        conv_action_row = QHBoxLayout()
        self.btn_run_convert = QPushButton("Convert COCO to YOLO (.txt)")
        self.btn_run_convert.setProperty("class", "SuccessButton")
        self.btn_run_convert.setIcon(get_svg_icon("step_convert", 18))
        self.btn_run_convert.setIconSize(QSize(18, 18))
        self.btn_run_convert.clicked.connect(self._on_run_convert)
        conv_action_row.addWidget(self.btn_run_convert)
        conv_layout.addLayout(conv_action_row)

        self.prog_convert = QProgressBar()
        self.prog_convert.setValue(0)
        conv_layout.addWidget(self.prog_convert)

        self.lbl_conv_stats = QLabel("No conversion performed yet.")
        self.lbl_conv_stats.setStyleSheet("color: #1e293b; font-weight: 600; min-height: 24px;")
        conv_layout.addWidget(self.lbl_conv_stats)

        layout.addWidget(conv_box)

        advance_row = QHBoxLayout()
        advance_row.addStretch()
        self.btn_next_step_3 = QPushButton("Proceed to Step 4: Dataset Splitter")
        self.btn_next_step_3.setProperty("class", "StepNavButton")
        self.btn_next_step_3.setIcon(get_svg_icon("arrow_right", 16))
        self.btn_next_step_3.setIconSize(QSize(16, 16))
        self.btn_next_step_3.clicked.connect(lambda: self._go_to_step(3))
        advance_row.addWidget(self.btn_next_step_3)
        layout.addLayout(advance_row)

        layout.addStretch()
        return page

    # -----------------------------------------------------------------
    # Page 4: Dataset Splitter (Standalone)
    # -----------------------------------------------------------------
    def _build_page_split(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        box_paths = QGroupBox("Standalone Tool: Dataset Input Sources & Destination (data/)")
        box_paths_layout = QGridLayout(box_paths)
        box_paths_layout.setHorizontalSpacing(10)
        box_paths_layout.setVerticalSpacing(8)

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

        layout.addWidget(box_paths)

        # Split Ratio Configuration
        box_ratios = QGroupBox("Partition Ratios (Train / Val / Test)")
        box_ratios_layout = QGridLayout(box_ratios)
        box_ratios_layout.setHorizontalSpacing(12)
        box_ratios_layout.setVerticalSpacing(8)

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

        self.chk_all_images = QCheckBox("Include ALL images in dataset (background images without annotations will receive empty label files)")
        self.chk_all_images.setChecked(True)
        self.chk_all_images.setEnabled(False)
        box_ratios_layout.addWidget(self.chk_all_images, 3, 0, 1, 3)

        layout.addWidget(box_ratios)

        self.btn_run_split = QPushButton("Split Dataset & Generate data.yaml")
        self.btn_run_split.setProperty("class", "PrimaryButton")
        self.btn_run_split.setIcon(get_svg_icon("step_split", 18))
        self.btn_run_split.setIconSize(QSize(18, 18))
        self.btn_run_split.clicked.connect(self._on_run_split)
        layout.addWidget(self.btn_run_split)

        self.prog_split = QProgressBar()
        self.prog_split.setValue(0)
        layout.addWidget(self.prog_split)

        self.lbl_split_stats = QLabel("No dataset split yet.")
        self.lbl_split_stats.setStyleSheet("color: #1e293b; font-weight: 600; min-height: 24px;")
        layout.addWidget(self.lbl_split_stats)

        advance_row = QHBoxLayout()
        advance_row.addStretch()
        self.btn_next_step_4 = QPushButton("Proceed to Step 5: YOLO Auto-Trainer")
        self.btn_next_step_4.setProperty("class", "StepNavButton")
        self.btn_next_step_4.setIcon(get_svg_icon("arrow_right", 16))
        self.btn_next_step_4.setIconSize(QSize(16, 16))
        self.btn_next_step_4.clicked.connect(lambda: self._go_to_step(4))
        advance_row.addWidget(self.btn_next_step_4)
        layout.addLayout(advance_row)

        layout.addStretch()
        return page

    # -----------------------------------------------------------------
    # Page 5: YOLO Auto-Trainer & Explorer (Standalone)
    # -----------------------------------------------------------------
    def _build_page_train(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

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
        m_layout.addWidget(self.cmb_model_preset, 0, 3)

        m_layout.addWidget(QLabel("Or Custom .pt:"), 0, 4)
        self.txt_custom_model = QLineEdit()
        self.txt_custom_model.setPlaceholderText("Optional path to custom .pt weights")
        m_layout.addWidget(self.txt_custom_model, 0, 5)
        btn_browse_pt = QPushButton("Browse...")
        btn_browse_pt.setIcon(get_svg_icon("folder", 16))
        btn_browse_pt.setIconSize(QSize(16, 16))
        btn_browse_pt.clicked.connect(self._browse_model_pt)
        m_layout.addWidget(btn_browse_pt, 0, 6)

        m_layout.addWidget(QLabel("data.yaml Path:"), 1, 0)
        self.txt_data_yaml = QLineEdit(str(self.data_dir / "data.yaml"))
        m_layout.addWidget(self.txt_data_yaml, 1, 1, 1, 5)
        btn_browse_yaml = QPushButton("Browse...")
        btn_browse_yaml.setIcon(get_svg_icon("folder", 16))
        btn_browse_yaml.setIconSize(QSize(16, 16))
        btn_browse_yaml.clicked.connect(self._browse_data_yaml)
        m_layout.addWidget(btn_browse_yaml, 1, 6)

        layout.addWidget(box_model)

        box_hparams = QGroupBox("Training Hyperparameters (Intelligently Auto-Tuned & Fully Editable)")
        h_layout = QGridLayout(box_hparams)
        h_layout.setHorizontalSpacing(12)
        h_layout.setVerticalSpacing(8)

        self.btn_auto_tune = QPushButton("Auto-Tune Hyperparameters for Dataset")
        self.btn_auto_tune.setProperty("class", "SuccessButton")
        self.btn_auto_tune.setIcon(get_svg_icon("brain", 18))
        self.btn_auto_tune.setIconSize(QSize(18, 18))
        self.btn_auto_tune.clicked.connect(self._on_auto_tune_clicked)
        h_layout.addWidget(self.btn_auto_tune, 0, 0, 1, 6)

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

        h_layout.addWidget(QLabel("Early Stop (Patience):"), 2, 0)
        self.spin_patience = QSpinBox()
        self.spin_patience.setRange(1, 100)
        self.spin_patience.setValue(15)
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
        btn_runs_base.setIcon(get_svg_icon("folder", 16))
        btn_runs_base.setIconSize(QSize(16, 16))
        btn_runs_base.clicked.connect(lambda: self._browse_folder(self.txt_runs_base))
        naming_layout.addWidget(btn_runs_base, 0, 4)

        self.lbl_run_preview = QLabel("Output Preview: runs/detect/my_yolo_model_v1/weights/my_yolo_model.pt")
        self.lbl_run_preview.setStyleSheet("color: #166534; font-weight: 700; min-height: 36px; line-height: 1.4;")
        naming_layout.addWidget(self.lbl_run_preview, 1, 0, 1, 5)

        layout.addWidget(box_naming)

        self._update_model_presets()
        self._update_run_preview()

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

        metrics_card = QFrame()
        metrics_card.setProperty("class", "CardFrame")
        metrics_layout = QHBoxLayout(metrics_card)
        metrics_layout.setContentsMargins(14, 8, 14, 8)
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
        self.txt_train_log.setMinimumHeight(150)
        layout.addWidget(self.txt_train_log, stretch=1)

        # Workspace Explorer Box (Integrated in Tool 5)
        box_quick = QGroupBox("Quick Open Workspaces & data.yaml Preview")
        quick_layout = QVBoxLayout(box_quick)

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
        self.txt_yaml_preview.setMaximumHeight(120)
        quick_layout.addWidget(self.txt_yaml_preview)

        layout.addWidget(box_quick)

        return page

    # -----------------------------------------------------------------
    # Stepper Bottom Footer Navigation Bar
    # -----------------------------------------------------------------
    def _build_stepper_footer(self) -> QFrame:
        footer = QFrame()
        footer.setProperty("class", "StepperFooter")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(8, 4, 8, 4)

        self.btn_prev_step = QPushButton("Previous Tool")
        self.btn_prev_step.setProperty("class", "StepNavButton")
        self.btn_prev_step.setIcon(get_svg_icon("arrow_left", 18))
        self.btn_prev_step.setIconSize(QSize(18, 18))
        self.btn_prev_step.clicked.connect(self._on_prev_step)
        layout.addWidget(self.btn_prev_step)

        layout.addStretch()

        self.lbl_step_indicator = QLabel("Tool 1 of 5: CVAT Downloader")
        self.lbl_step_indicator.setStyleSheet("font-weight: 700; color: #166534; font-size: 13px; min-height: 22px;")
        layout.addWidget(self.lbl_step_indicator)

        layout.addStretch()

        self.btn_next_step = QPushButton("Next Tool")
        self.btn_next_step.setProperty("class", "StepNavButton")
        self.btn_next_step.setIcon(get_svg_icon("arrow_right", 18))
        self.btn_next_step.setIconSize(QSize(18, 18))
        self.btn_next_step.clicked.connect(self._on_next_step)
        layout.addWidget(self.btn_next_step)

        return footer

    # -----------------------------------------------------------------
    # Global Status Bar
    # -----------------------------------------------------------------
    def _build_global_status_bar(self) -> QWidget:
        bar = QFrame()
        bar.setProperty("class", "CardFrame")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 6, 12, 6)

        self.lbl_status_icon = QLabel()
        self.lbl_status_icon.setPixmap(get_svg_pixmap("check", 16, 16))
        layout.addWidget(self.lbl_status_icon)

        self.lbl_global_status = QLabel("Ready")
        self.lbl_global_status.setStyleSheet("font-weight: 600; color: #166534;")
        layout.addWidget(self.lbl_global_status, stretch=1)

        lbl_workspace = QLabel(f"Data Directory: {self.data_dir}")
        lbl_workspace.setStyleSheet("color: #64748b; font-size: 11px;")
        layout.addWidget(lbl_workspace)

        return bar

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

        # Update button styles in Stepper Bar
        for idx, btn in enumerate(self.step_buttons):
            if idx == step_idx:
                btn.setProperty("class", "StepButtonActive")
            elif self.step_completed[idx]:
                btn.setProperty("class", "StepButtonCompleted")
            else:
                btn.setProperty("class", "StepButton")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        # Update Footer
        self.btn_prev_step.setEnabled(step_idx > 0)
        self.btn_next_step.setEnabled(step_idx < self.pages.count() - 1)
        self.lbl_step_indicator.setText(f"Tool {step_idx + 1} of {self.pages.count()}: {self.STEP_DESCRIPTIONS[step_idx][0]}")

        # Update Dynamic Guidance Box
        self.lbl_guide.setText(self.STEP_GUIDES[step_idx])

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
        self.lbl_server_info.setText(f"Connected: {server_url} (User: {username})")

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
        self.step_completed[0] = True

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
        self.step_completed[0] = True

        QMessageBox.information(
            self, "Download Complete",
            f"Successfully downloaded {len(results)} tasks!\n\n"
            f"All images extracted and flattened (default folder removed).\n"
            f"Location: {self.txt_dl_folder.text()}\n\n"
            f"You can now switch to '2. Merge Datasets' to unify them."
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
        self.step_completed[1] = True

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
        self.step_completed[2] = True

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
        self.step_completed[3] = True

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
        self.lbl_global_status.setText("Training session completed!")
        self.step_completed[4] = True
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
        self.lbl_global_status.setText("Training completed successfully!")
        QMessageBox.information(self, "Training Complete", summary)

    def _on_train_error(self, err_msg: str):
        self.btn_start_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
        self.mascot.set_state(DancingSvgMascot.STATE_IDLE)
        self.lbl_global_status.setText(f"Training Failed: {err_msg}")
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
