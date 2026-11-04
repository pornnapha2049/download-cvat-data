"""
Main Window for the CVAT-to-YOLO GUI application.
Features a dedicated Authentication Screen (Login Page) followed by
an enterprise 5-Step Wizard Workflow:
1. CVAT Task Explorer & Download (COCO 1.0 + unzipping + cleanup default folder)
2. Task Merging & COCO-to-YOLO Conversion (Detection & Segmentation)
3. Train/Val/Test Splitter (including all background images)
4. YOLO Training with Auto-Tuning, Early Stopping, and Auto-Versioning
5. Runs & Weights Explorer
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
    QProgressBar, QPlainTextEdit, QFrame, QGroupBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

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
            login_info = self.client.login(self.url, self.username, self.password)
            projects = self.client.refresh_projects()
            tasks = self.client.get_tasks()
            self.finished_signal.emit(login_info, projects, tasks)
        except Exception as e:
            self.error_signal.emit(str(e))


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
    STEP_DESCRIPTIONS = [
        ("Step 1: Task Explorer", "☘︎ สำรวจและดาวน์โหลด Tasks (COCO 1.0 + Clean /images/default)"),
        ("Step 2: Merge & Convert", "🧩 รวม Datasets เข้าด้วยกัน & แปลง COCO เป็น YOLO (Det/Seg)"),
        ("Step 3: Split Dataset", "✂️ แบ่ง Train/Val/Test โดยนำทุกรูปภาพเข้าร่วม (รวม Backgrounds)"),
        ("Step 4: YOLO Training", "🧠 ปรับ Auto-Tune Hyperparameters & เริ่ม Train พร้อม Early Stopping"),
        ("Step 5: Runs & Explorer", "☕ สำรวจผลลัพธ์การเทรน ตรวจสอบ weights และ data.yaml"),
    ]

    STEP_GUIDES = [
        ".☘︎ ݁˖ ขั้นตอนที่ 1: เลือก Tasks ที่ต้องการดาวน์โหลด หรือฟิลเตอร์ตาม stage/state แล้วกด 'Download & Auto-Extract Tasks' ระบบจะแตกไฟล์และย้ายรูปออกจาก /images/default ให้เองครับ",
        ".☘︎ ݁˖ ขั้นตอนที่ 2: กด '🧩 Merge Datasets' เพื่อรวมข้อมูลและแก้ชื่อภาพซ้ำ จากนั้นเลือกว่าจะแปลงเป็น Detection (Bounding Box) หรือ Segmentation (Polygon) แล้วกด Convert",
        ".☘︎ ݁˖ ขั้นตอนที่ 3: ปรับสัดส่วนเปอร์เซ็นต์ Train / Val / Test ตามต้องการ (รูปภาพทุกรูปจะถูกนำมารวม 100% แม้ไม่มี Label ก็ตาม) แล้วกด Split",
        ".☘︎ ݁˖ ขั้นตอนที่ 4: กดปุ่ม '⚡ Auto-Tune Hyperparameters' เพื่อตั้งค่าโมเดลให้เหมาะกับจำนวนภาพ ตั้งชื่อโฟลเดอร์รัน แล้วกด Start YOLO Training ได้เลย!",
        ".☘︎ ݁˖ ขั้นตอนที่ 5: ตรวจสอบโฟลเดอร์ผลลัพธ์ โมเดล Weights (.pt) ที่ตั้งชื่อตามที่คุณกำหนด และกราฟผลการเทรนอย่างละเอียด",
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle(".☘︎ ݁˖ CVAT to YOLO Studio • Cozy AI Pipeline")
        self.resize(1260, 880)
        self.setMinimumSize(1020, 740)

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
        self.step_completed = [False, False, False, False, False]

        # Active Workers
        self.active_worker = None

        self._init_ui()
        self.setStyleSheet(DARK_STYLESHEET)

        # Start on dedicated Login Page
        self.root_stack.setCurrentIndex(0)

    def _init_ui(self):
        central_widget = QWidget(self)
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Root Stack: 0 -> Dedicated Login Page, 1 -> Studio Wizard Workflow
        self.root_stack = QStackedWidget()
        self.view_login = self._build_login_view()
        self.view_studio = self._build_studio_view()

        self.root_stack.addWidget(self.view_login)
        self.root_stack.addWidget(self.view_studio)

        root_layout.addWidget(self.root_stack)

    # =================================================================
    # VIEW 0: DEDICATED LOGIN PAGE
    # =================================================================

    def _build_login_view(self) -> QWidget:
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(20, 20, 20, 20)

        # Centering layout with vertical stretches
        layout.addStretch(1)

        center_row = QHBoxLayout()
        center_row.addStretch(1)

        # Glowing Login Card
        card = QFrame()
        card.setProperty("class", "LoginCard")
        card.setFixedWidth(500)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(32, 28, 32, 28)
        card_layout.setSpacing(14)

        # Dancing Mascot at top of Login Card
        mascot_box = QHBoxLayout()
        self.login_mascot = DancingSvgMascot(size=120)
        mascot_box.addStretch()
        mascot_box.addWidget(self.login_mascot)
        mascot_box.addStretch()
        card_layout.addLayout(mascot_box)

        # Title & Subtitle
        lbl_title = QLabel(".☘︎ ݁˖ CVAT Portal Sign-In")
        lbl_title.setObjectName("loginTitle")
        lbl_title.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(lbl_title)

        lbl_sub = QLabel("Cozy AI Studio • Enter credentials to access datasets, images & annotations.")
        lbl_sub.setObjectName("loginSubtitle")
        lbl_sub.setAlignment(Qt.AlignCenter)
        lbl_sub.setWordWrap(True)
        card_layout.addWidget(lbl_sub)

        card_layout.addSpacing(6)

        # Error / Alert Banner (Hidden by default)
        self.lbl_login_alert = QLabel()
        self.lbl_login_alert.setProperty("class", "AlertError")
        self.lbl_login_alert.setWordWrap(True)
        self.lbl_login_alert.hide()
        card_layout.addWidget(self.lbl_login_alert)

        # Form Inputs
        lbl_url = QLabel("CVAT Server URL:")
        lbl_url.setStyleSheet("font-weight: 600; color: #736154;")
        card_layout.addWidget(lbl_url)
        self.txt_login_url = QLineEdit("http://1.10.131.222:8080/projects")
        self.txt_login_url.setProperty("class", "LoginInput")
        card_layout.addWidget(self.txt_login_url)

        lbl_user = QLabel("Username:")
        lbl_user.setStyleSheet("font-weight: 600; color: #736154;")
        card_layout.addWidget(lbl_user)
        self.txt_login_user = QLineEdit()
        self.txt_login_user.setPlaceholderText("Enter your CVAT username")
        self.txt_login_user.setProperty("class", "LoginInput")
        card_layout.addWidget(self.txt_login_user)

        lbl_pass = QLabel("Password:")
        lbl_pass.setStyleSheet("font-weight: 600; color: #736154;")
        card_layout.addWidget(lbl_pass)
        self.txt_login_pass = QLineEdit()
        self.txt_login_pass.setEchoMode(QLineEdit.Password)
        self.txt_login_pass.setPlaceholderText("Enter your password")
        self.txt_login_pass.setProperty("class", "LoginInput")
        self.txt_login_pass.returnPressed.connect(self._on_do_login)
        card_layout.addWidget(self.txt_login_pass)

        # Status Label
        self.lbl_login_status = QLabel("")
        self.lbl_login_status.setAlignment(Qt.AlignCenter)
        self.lbl_login_status.setStyleSheet("color: #5e8460; font-weight: 700;")
        card_layout.addWidget(self.lbl_login_status)

        # Login Action Button
        self.btn_do_login = QPushButton("☘︎ Sign In & Launch Studio")
        self.btn_do_login.setProperty("class", "LoginButton")
        self.btn_do_login.setCursor(Qt.PointingHandCursor)
        self.btn_do_login.clicked.connect(self._on_do_login)
        card_layout.addWidget(self.btn_do_login)

        # Offline / Local Dataset Skip Button
        card_layout.addSpacing(4)
        btn_skip = QPushButton("📁 Work with Local Datasets (Skip Login / Offline Mode)")
        btn_skip.setProperty("class", "SkipLoginButton")
        btn_skip.setCursor(Qt.PointingHandCursor)
        btn_skip.clicked.connect(self._on_skip_login)
        card_layout.addWidget(btn_skip, alignment=Qt.AlignCenter)

        center_row.addWidget(card)
        center_row.addStretch(1)

        layout.addLayout(center_row)
        layout.addStretch(1)

        # Version stamp
        lbl_foot = QLabel("CVAT to YOLO Studio v2.0 • Enterprise Edition")
        lbl_foot.setStyleSheet("color: #475569; font-size: 11px;")
        lbl_foot.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl_foot)

        return view

    # =================================================================
    # VIEW 1: MAIN STUDIO WIZARD WORKFLOW (5 PAGES)
    # =================================================================

    def _build_studio_view(self) -> QWidget:
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        # 1. Header with Animated Mascot & User Profile Info
        layout.addWidget(self._build_studio_header())

        # 2. Interactive Stepper Bar (Step 1 -> 5)
        layout.addWidget(self._build_stepper_bar())

        # 3. Dynamic Guide Box
        layout.addWidget(self._build_guide_box())

        # 4. Multi-Page Container (QStackedWidget)
        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_page_cvat())
        self.pages.addWidget(self._build_page_merge_convert())
        self.pages.addWidget(self._build_page_split())
        self.pages.addWidget(self._build_page_train())
        self.pages.addWidget(self._build_page_explorer())
        layout.addWidget(self.pages, stretch=1)

        # 5. Bottom Stepper Footer Navigation (< Previous | Step X of 5 | Next >)
        layout.addWidget(self._build_stepper_footer())

        # 6. Global Status Bar
        layout.addWidget(self._build_global_status_bar())

        return view

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
        title_lbl = QLabel(".☘︎ ݁˖ CVAT to YOLO Studio")
        title_lbl.setObjectName("titleLabel")
        title_row.addWidget(title_lbl)

        # Pulsing Activity Indicator
        self.wave_indicator = AnimatedWavePill(42, 22)
        title_row.addWidget(self.wave_indicator)
        title_row.addStretch()

        text_layout.addLayout(title_row)

        subtitle_lbl = QLabel(
            "Cozy AI Pipeline: CVAT Downloader ➔ Auto-Cleanup ➔ COCO Merger ➔ "
            "Polygon-to-YOLO (Det/Seg) ➔ Full Dataset Split ➔ YOLO Auto-Trainer"
        )
        subtitle_lbl.setObjectName("subtitleLabel")
        text_layout.addWidget(subtitle_lbl)

        # Profile & Status Row
        profile_row = QHBoxLayout()
        self.lbl_user_badge = QLabel("👤 Connected: Anonymous")
        self.lbl_user_badge.setStyleSheet("color: #38bdf8; font-weight: 700; font-size: 11.5px;")
        profile_row.addWidget(self.lbl_user_badge)

        btn_logout = QPushButton("🚪 Logout")
        btn_logout.setProperty("class", "LogoutButton")
        btn_logout.setCursor(Qt.PointingHandCursor)
        btn_logout.clicked.connect(self._on_logout)
        profile_row.addWidget(btn_logout)

        profile_row.addSpacing(12)

        cuda_ok = torch.cuda.is_available()
        hw_text = f"Hardware: {'⚡ NVIDIA GPU (' + torch.cuda.get_device_name(0) + ')' if cuda_ok else '💻 CPU Mode'}"
        self.hw_badge = QLabel(hw_text)
        self.hw_badge.setProperty("class", "BadgeSuccess" if cuda_ok else "BadgeInfo")
        profile_row.addWidget(self.hw_badge)
        profile_row.addStretch()

        text_layout.addLayout(profile_row)

        layout.addLayout(text_layout, stretch=1)

        # Studio Mascot Container
        mascot_box = QVBoxLayout()
        mascot_box.setAlignment(Qt.AlignCenter)

        self.mascot = DancingSvgMascot(size=105)
        mascot_box.addWidget(self.mascot, alignment=Qt.AlignCenter)

        self.chk_mascot = QCheckBox("✨ SVG Mascot")
        self.chk_mascot.setChecked(True)
        self.chk_mascot.toggled.connect(self.mascot.toggle_animation)
        mascot_box.addWidget(self.chk_mascot, alignment=Qt.AlignCenter)

        layout.addLayout(mascot_box)
        return card

    # -----------------------------------------------------------------
    # Stepper Navigation Bar
    # -----------------------------------------------------------------
    def _build_stepper_bar(self) -> QFrame:
        bar = QFrame()
        bar.setProperty("class", "StepperContainer")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.step_buttons: list[QPushButton] = []
        step_titles = [
            ("☘︎ 1. Task Explorer", "Fetch & Flatten Images"),
            ("🧩 2. Merge & Convert", "COCO 1.0 to YOLO"),
            ("✂️ 3. Dataset Splitter", "Train/Val/Test All Imgs"),
            ("🧠 4. YOLO Auto-Trainer", "Auto-Tune & Train Model"),
            ("☕ 5. Runs & Explorer", "Explore Weights & Logs"),
        ]

        for idx, (title, sub) in enumerate(step_titles):
            btn = QPushButton()
            btn.setProperty("class", "StepButton")
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
    # Page 1: CVAT Task Explorer & Download
    # -----------------------------------------------------------------
    def _build_page_cvat(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # Connection Info & Quick Refresh Bar
        conn_card = QFrame()
        conn_card.setProperty("class", "CardFrame")
        conn_layout = QHBoxLayout(conn_card)
        conn_layout.setContentsMargins(14, 8, 14, 8)

        self.lbl_server_info = QLabel("Connected Server: http://1.10.131.222:8080")
        self.lbl_server_info.setStyleSheet("color: #38bdf8; font-weight: 600;")
        conn_layout.addWidget(self.lbl_server_info, stretch=1)

        btn_refresh_tasks = QPushButton("🔄 Refresh Tasks from Server")
        btn_refresh_tasks.clicked.connect(self._fetch_cvat_tasks)
        conn_layout.addWidget(btn_refresh_tasks)

        layout.addWidget(conn_card)

        # Filter Card (Project, Stage, State, Search)
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

        status_row = QHBoxLayout()
        self.lbl_cvat_status = QLabel("Select tasks above to download.")
        status_row.addWidget(self.lbl_cvat_status, stretch=1)

        self.btn_next_step_1 = QPushButton("Proceed to Step 2: Merge & Convert ➔")
        self.btn_next_step_1.setProperty("class", "StepNavButton")
        self.btn_next_step_1.clicked.connect(lambda: self._go_to_step(1))
        status_row.addWidget(self.btn_next_step_1)

        layout.addLayout(status_row)
        return page

    # -----------------------------------------------------------------
    # Page 2: Merge & Convert (COCO → YOLO)
    # -----------------------------------------------------------------
    def _build_page_merge_convert(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Group 1: Merge Downloaded Tasks
        merge_box = QGroupBox("Part A: Merge Multiple Downloaded CVAT Tasks into Single Dataset")
        merge_layout = QVBoxLayout(merge_box)
        merge_layout.setSpacing(10)

        row_src = QHBoxLayout()
        row_src.addWidget(QLabel("Tasks Source Directory:"))
        self.txt_merge_src = QLineEdit(str(self.downloads_dir))
        row_src.addWidget(self.txt_merge_src, stretch=1)
        btn_src = QPushButton("Browse...")
        btn_src.clicked.connect(lambda: self._browse_folder(self.txt_merge_src))
        row_src.addWidget(btn_src)
        merge_layout.addLayout(row_src)

        row_dst = QHBoxLayout()
        row_dst.addWidget(QLabel("Merged Output Directory:"))
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

        # Group 2: Convert COCO 1.0 to YOLO Format
        conv_box = QGroupBox("Part B: Convert Unified COCO 1.0 to YOLO Format (.txt)")
        conv_layout = QVBoxLayout(conv_box)
        conv_layout.setSpacing(10)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("YOLO Task Mode:"))
        self.rb_mode_det = QRadioButton("🎯 Detection (Converts Segmentation Polygons to Normalized Bounding Boxes)")
        self.rb_mode_seg = QRadioButton("🔷 Segmentation (Preserves Exact Polygon Coordinates)")
        self.rb_mode_det.setChecked(True)
        self.rb_mode_det.toggled.connect(self._on_task_mode_changed)

        mode_row.addWidget(self.rb_mode_det)
        mode_row.addWidget(self.rb_mode_seg)
        mode_row.addStretch()
        conv_layout.addLayout(mode_row)

        row_lbl_dst = QHBoxLayout()
        row_lbl_dst.addWidget(QLabel("Labels Output Folder:"))
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

        # Progress bar for Page 2
        self.prog_merge_conv = QProgressBar()
        self.prog_merge_conv.setValue(0)
        layout.addWidget(self.prog_merge_conv)

        advance_row = QHBoxLayout()
        advance_row.addStretch()
        self.btn_next_step_2 = QPushButton("Proceed to Step 3: Dataset Splitter ➔")
        self.btn_next_step_2.setProperty("class", "StepNavButton")
        self.btn_next_step_2.clicked.connect(lambda: self._go_to_step(2))
        advance_row.addWidget(self.btn_next_step_2)
        layout.addLayout(advance_row)

        layout.addStretch()
        return page

    # -----------------------------------------------------------------
    # Page 3: Dataset Splitter
    # -----------------------------------------------------------------
    def _build_page_split(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        box_paths = QGroupBox("Dataset Input Sources & Split Output Destination")
        box_paths_layout = QGridLayout(box_paths)
        box_paths_layout.setHorizontalSpacing(10)
        box_paths_layout.setVerticalSpacing(8)

        box_paths_layout.addWidget(QLabel("Images Source Folder:"), 0, 0)
        self.txt_split_img_src = QLineEdit(str(self.merged_dir / "images"))
        box_paths_layout.addWidget(self.txt_split_img_src, 0, 1)
        btn_img_src = QPushButton("Browse...")
        btn_img_src.clicked.connect(lambda: self._browse_folder(self.txt_split_img_src))
        box_paths_layout.addWidget(btn_img_src, 0, 2)

        box_paths_layout.addWidget(QLabel("Labels Source Folder:"), 1, 0)
        self.txt_split_lbl_src = QLineEdit(str(self.merged_dir / "labels"))
        box_paths_layout.addWidget(self.txt_split_lbl_src, 1, 1)
        btn_lbl_src = QPushButton("Browse...")
        btn_lbl_src.clicked.connect(lambda: self._browse_folder(self.txt_split_lbl_src))
        box_paths_layout.addWidget(btn_lbl_src, 1, 2)

        box_paths_layout.addWidget(QLabel("Split Dataset Destination:"), 2, 0)
        self.txt_split_dst = QLineEdit(str(self.dataset_dir))
        box_paths_layout.addWidget(self.txt_split_dst, 2, 1)
        btn_split_dst = QPushButton("Browse...")
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

        self.chk_all_images = QCheckBox("☑️ Include ALL images in dataset (background images without annotations will receive empty label files)")
        self.chk_all_images.setChecked(True)
        self.chk_all_images.setEnabled(False)
        box_ratios_layout.addWidget(self.chk_all_images, 3, 0, 1, 3)

        layout.addWidget(box_ratios)

        self.btn_run_split = QPushButton("📦 Split Dataset & Generate data.yaml")
        self.btn_run_split.setProperty("class", "PrimaryButton")
        self.btn_run_split.clicked.connect(self._on_run_split)
        layout.addWidget(self.btn_run_split)

        self.prog_split = QProgressBar()
        self.prog_split.setValue(0)
        layout.addWidget(self.prog_split)

        self.lbl_split_stats = QLabel("No dataset split yet.")
        layout.addWidget(self.lbl_split_stats)

        advance_row = QHBoxLayout()
        advance_row.addStretch()
        self.btn_next_step_3 = QPushButton("Proceed to Step 4: YOLO Training ➔")
        self.btn_next_step_3.setProperty("class", "StepNavButton")
        self.btn_next_step_3.clicked.connect(lambda: self._go_to_step(3))
        advance_row.addWidget(self.btn_next_step_3)
        layout.addLayout(advance_row)

        layout.addStretch()
        return page

    # -----------------------------------------------------------------
    # Page 4: YOLO Auto-Trainer
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
        btn_browse_pt.clicked.connect(self._browse_model_pt)
        m_layout.addWidget(btn_browse_pt, 0, 6)

        m_layout.addWidget(QLabel("data.yaml Path:"), 1, 0)
        self.txt_data_yaml = QLineEdit(str(self.dataset_dir / "data.yaml"))
        m_layout.addWidget(self.txt_data_yaml, 1, 1, 1, 5)
        btn_browse_yaml = QPushButton("Browse...")
        btn_browse_yaml.clicked.connect(self._browse_data_yaml)
        m_layout.addWidget(btn_browse_yaml, 1, 6)

        layout.addWidget(box_model)

        box_hparams = QGroupBox("Training Hyperparameters (Intelligently Auto-Tuned & Fully Editable)")
        h_layout = QGridLayout(box_hparams)
        h_layout.setHorizontalSpacing(12)
        h_layout.setVerticalSpacing(8)

        self.btn_auto_tune = QPushButton("⚡ Auto-Tune Hyperparameters for Dataset")
        self.btn_auto_tune.setProperty("class", "SuccessButton")
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
        btn_runs_base.clicked.connect(lambda: self._browse_folder(self.txt_runs_base))
        naming_layout.addWidget(btn_runs_base, 0, 4)

        self.lbl_run_preview = QLabel("Output Preview: runs/detect/my_yolo_model_v1/weights/my_yolo_model.pt")
        self.lbl_run_preview.setStyleSheet("color: #38bdf8; font-weight: 600;")
        naming_layout.addWidget(self.lbl_run_preview, 1, 0, 1, 5)

        layout.addWidget(box_naming)

        self._update_model_presets()
        self._update_run_preview()

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

        self.prog_train = QProgressBar()
        self.prog_train.setValue(0)
        layout.addWidget(self.prog_train)

        self.txt_train_log = QPlainTextEdit()
        self.txt_train_log.setReadOnly(True)
        layout.addWidget(self.txt_train_log, stretch=1)

        advance_row = QHBoxLayout()
        advance_row.addStretch()
        self.btn_next_step_4 = QPushButton("View Results in Step 5 ➔")
        self.btn_next_step_4.setProperty("class", "StepNavButton")
        self.btn_next_step_4.clicked.connect(lambda: self._go_to_step(4))
        advance_row.addWidget(self.btn_next_step_4)
        layout.addLayout(advance_row)

        return page

    # -----------------------------------------------------------------
    # Page 5: Runs & Explorer
    # -----------------------------------------------------------------
    def _build_page_explorer(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        box_quick = QGroupBox("Quick Open Workspaces in Windows File Explorer")
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

        box_yaml = QGroupBox("Active data.yaml Configuration Preview")
        yaml_layout = QVBoxLayout(box_yaml)
        self.txt_yaml_preview = QPlainTextEdit()
        self.txt_yaml_preview.setReadOnly(True)
        yaml_layout.addWidget(self.txt_yaml_preview)

        btn_refresh_yaml = QPushButton("🔄 Refresh data.yaml Preview")
        btn_refresh_yaml.clicked.connect(self._refresh_yaml_preview)
        yaml_layout.addWidget(btn_refresh_yaml)

        layout.addWidget(box_yaml, stretch=1)

        return page

    # -----------------------------------------------------------------
    # Stepper Bottom Footer Navigation Bar
    # -----------------------------------------------------------------
    def _build_stepper_footer(self) -> QFrame:
        footer = QFrame()
        footer.setProperty("class", "StepperFooter")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(8, 4, 8, 4)

        self.btn_prev_step = QPushButton("⬅️ Previous Step")
        self.btn_prev_step.setProperty("class", "StepNavButton")
        self.btn_prev_step.clicked.connect(self._on_prev_step)
        layout.addWidget(self.btn_prev_step)

        layout.addStretch()

        self.lbl_step_indicator = QLabel("Step 1 of 5: Task Explorer")
        self.lbl_step_indicator.setStyleSheet("font-weight: 700; color: #38bdf8; font-size: 13px;")
        layout.addWidget(self.lbl_step_indicator)

        layout.addStretch()

        self.btn_next_step = QPushButton("Next Step ➡️")
        self.btn_next_step.setProperty("class", "StepNavButton")
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

        self.lbl_global_status = QLabel("🟢 Ready")
        layout.addWidget(self.lbl_global_status, stretch=1)

        lbl_workspace = QLabel(f"Workspace: {self.app_dir}")
        lbl_workspace.setStyleSheet("color: #64748b; font-size: 11px;")
        layout.addWidget(lbl_workspace)

        return bar

    # =================================================================
    # Login & Authentication Slots
    # =================================================================

    def _on_do_login(self):
        url = self.txt_login_url.text().strip()
        user = self.txt_login_user.text().strip()
        pwd = self.txt_login_pass.text()

        if not url:
            self._show_login_alert("Please provide a valid CVAT URL.")
            return

        self.lbl_login_alert.hide()
        self.btn_do_login.setEnabled(False)
        self.lbl_login_status.setText("Connecting to CVAT server...")
        self.login_mascot.set_state(DancingSvgMascot.STATE_DOWNLOADING)

        self.active_worker = CvatAuthWorker(self.cvat_client, url, user, pwd)
        self.active_worker.finished_signal.connect(self._on_auth_success)
        self.active_worker.error_signal.connect(self._on_auth_error)
        self.active_worker.start()

    def _on_auth_success(self, login_info: dict, projects: list[dict], tasks: list[dict]):
        self.btn_do_login.setEnabled(True)
        self.login_mascot.set_state(DancingSvgMascot.STATE_SUCCESS)
        self.lbl_login_status.setText("Authentication successful!")

        # Update Studio Header
        username = login_info.get("username") or "User"
        server_url = login_info.get("server_url")
        self.lbl_user_badge.setText(f"👤 Logged in as: {username} | Host: {server_url}")
        self.lbl_server_info.setText(f"Connected Server: {server_url} (v{login_info.get('server_version')})")

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

        # Transition to Studio Wizard View
        self.root_stack.setCurrentIndex(1)
        self._go_to_step(0)

    def _on_auth_error(self, err_msg: str):
        self.btn_do_login.setEnabled(True)
        self.login_mascot.set_state(DancingSvgMascot.STATE_IDLE)
        self.lbl_login_status.setText("")
        self._show_login_alert(f"Authentication Failed: {err_msg}")

    def _show_login_alert(self, message: str):
        self.lbl_login_alert.setText(f"⚠️ {message}")
        self.lbl_login_alert.show()

    def _on_skip_login(self):
        """Allows user to bypass CVAT login to work directly on local datasets/models."""
        self.lbl_user_badge.setText("👤 Mode: Local Workspaces (Offline)")
        self.lbl_server_info.setText("CVAT Status: Not connected (Local Offline Mode)")
        self.root_stack.setCurrentIndex(1)
        # Go directly to Step 2 (Merge & Convert) or Step 1
        self._go_to_step(1)

    def _on_logout(self):
        ret = QMessageBox.question(
            self, "Confirm Logout", "Are you sure you want to log out of CVAT and return to the Sign-In screen?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        if ret == QMessageBox.Yes:
            self.cvat_client.logout()
            self.all_tasks_data.clear()
            self.tbl_tasks.setRowCount(0)
            self.txt_login_pass.clear()
            self.lbl_login_status.setText("")
            self.lbl_login_alert.hide()
            self.root_stack.setCurrentIndex(0)

    # =================================================================
    # Step Wizard Controller
    # =================================================================

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
        self.lbl_step_indicator.setText(f"Step {step_idx + 1} of {self.pages.count()}: {self.STEP_DESCRIPTIONS[step_idx][0]}")

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

    # =================================================================
    # Common Utilities
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
    # Step 1: CVAT Slots
    # -----------------------------------------------------------------
    def _fetch_cvat_tasks(self):
        if not self.cvat_client.is_logged_in():
            QMessageBox.information(self, "Not Logged In", "Please sign in to CVAT first.")
            self.root_stack.setCurrentIndex(0)
            return

        self.lbl_cvat_status.setText("Fetching updated tasks from CVAT...")
        self.active_worker = CvatFetchWorker(self.cvat_client)
        self.active_worker.finished_signal.connect(self._on_tasks_fetched)
        self.active_worker.error_signal.connect(self._on_tasks_fetch_error)
        self.active_worker.start()

    def _on_tasks_fetched(self, projects: list[dict], tasks: list[dict]):
        self.mascot.set_state(DancingSvgMascot.STATE_IDLE)

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
        self.step_completed[0] = True

        self.txt_merge_src.setText(str(self.downloads_dir))

        ret = QMessageBox.question(
            self, "Download Complete",
            f"Successfully downloaded {len(completed_folders)} tasks in COCO 1.0 format.\n"
            f"All images extracted and moved from /images/default directly to /images!\n\n"
            f"Would you like to proceed to Step 2 (Merge & Convert) now?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        if ret == QMessageBox.Yes:
            self._go_to_step(1)

    def _on_download_error(self, err_msg: str):
        self.btn_download_tasks.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_IDLE)
        self.lbl_cvat_status.setText(f"Download Error: {err_msg}")
        self.lbl_global_status.setText("🔴 Download Error")
        QMessageBox.critical(self, "Download Error", f"An error occurred during download:\n{err_msg}")

    # -----------------------------------------------------------------
    # Step 2: Merge & Convert Slots
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
        self.step_completed[1] = True

        classes_str = ", ".join(res["classes"])
        stats_text = (
            f"✅ Converted for [{res['mode'].upper()}]: Total Images: {res['total_images']} | "
            f"Labeled Images: {res['images_with_labels']} | Annotations: {res['total_annotations_written']} | "
            f"Classes ({res['num_classes']}): [{classes_str}]"
        )
        self.lbl_conv_stats.setText(stats_text)
        self.lbl_global_status.setText("🟢 Conversion to YOLO format complete.")

        self.txt_split_lbl_src.setText(str(res["output_dir"]))

        ret = QMessageBox.question(
            self, "Conversion Complete",
            f"{stats_text}\n\nWould you like to proceed to Step 3 (Dataset Splitter)?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        if ret == QMessageBox.Yes:
            self._go_to_step(2)

    def _on_convert_error(self, err_msg: str):
        self.btn_run_convert.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_IDLE)
        self.lbl_conv_stats.setText(f"Conversion Error: {err_msg}")
        QMessageBox.critical(self, "Conversion Error", err_msg)

    # -----------------------------------------------------------------
    # Step 3: Dataset Split Slots
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
        self.step_completed[2] = True

        classes_str = ", ".join(res["classes"])
        stats_text = (
            f"✅ Split Complete! Total Images: {res['total_images']} (Labeled: {res['labeled_images']}, "
            f"Backgrounds: {res['background_images']})\n"
            f"Partition: Train = {res['train_count']} | Val = {res['val_count']} | Test = {res['test_count']}\n"
            f"Classes ({res['num_classes']}): [{classes_str}] -> data.yaml created successfully!"
        )
        self.lbl_split_stats.setText(stats_text)
        self.lbl_global_status.setText("🟢 Split complete. All images included.")

        self.txt_data_yaml.setText(str(res["data_yaml_path"]))
        self._refresh_yaml_preview()

        ret = QMessageBox.question(
            self, "Split Complete",
            f"{stats_text}\n\nWould you like to proceed to Step 4 (YOLO Auto-Trainer)?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        if ret == QMessageBox.Yes:
            self._go_to_step(3)

    def _on_split_error(self, err_msg: str):
        self.btn_run_split.setEnabled(True)
        self.mascot.set_state(DancingSvgMascot.STATE_IDLE)
        self.lbl_split_stats.setText(f"Split Error: {err_msg}")
        QMessageBox.critical(self, "Split Error", err_msg)

    # -----------------------------------------------------------------
    # Step 4: YOLO Trainer Slots
    # -----------------------------------------------------------------
    def _update_model_presets(self):
        if not hasattr(self, "cmb_train_task") or not hasattr(self, "cmb_model_preset"):
            return
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

        is_gpu = torch.cuda.is_available()
        recommended = auto_tune_hyperparameters(
            num_images=num_images,
            num_annotations=num_images * 3,
            num_classes=2,
            is_gpu=is_gpu,
        )

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
        self.lbl_global_status.setText("🟢 Training session completed!")
        self.step_completed[3] = True

        self._update_run_preview()

        run_dir = res.get("run_dir")
        best_pt = res.get("best_pt")
        custom_pt = res.get("custom_pt")

        summary = (
            f"🎉 YOLO Training Finished Successfully!\n\n"
            f"• Output Folder: {run_dir}\n"
            f"• Best Weights: {best_pt}\n"
            f"• Custom Named Weights: {custom_pt}\n\n"
            f"Would you like to view the outputs in Step 5 (Runs & Explorer)?"
        )
        ret = QMessageBox.information(self, "Training Complete", summary, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if ret == QMessageBox.Yes:
            self._go_to_step(4)

    def _on_train_error(self, err_msg: str):
        self.btn_start_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
        self.mascot.set_state(DancingSvgMascot.STATE_IDLE)
        self.lbl_global_status.setText("🔴 Training Error")
        self.txt_train_log.appendPlainText(f"\n❌ Training Error:\n{err_msg}")
        QMessageBox.critical(self, "Training Error", err_msg)

    # -----------------------------------------------------------------
    # Step 5: Explorer Slots
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
            self.txt_yaml_preview.setPlainText("data.yaml does not exist yet. Run Split step in Step 3.")
