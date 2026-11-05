"""
Fullscreen Preview Viewer for Model Prediction & Inference (Images, Folders, and Videos).
Provides a dedicated cinema-mode view with real-time frame streaming, navigation,
font resizing, and label mode switching.
"""

from __future__ import annotations

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QSizePolicy, QButtonGroup, QRadioButton, QSpinBox, QShortcut
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QImage, QKeyEvent, QKeySequence

from ui.svg_icons import get_svg_icon


def bgr_to_qpixmap(bgr_array: np.ndarray) -> QPixmap:
    """Converts OpenCV BGR numpy array to a QPixmap."""
    if bgr_array is None or bgr_array.size == 0:
        return QPixmap()
    h, w, ch = bgr_array.shape
    bytes_per_line = ch * w
    rgb = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class ScaledImageLabel(QLabel):
    """
    A QLabel that smoothly rescales its pixmap to fill its dimensions while preserving aspect ratio.
    Double-clicking emits or triggers dialog close.
    """

    def __init__(self, parent=None, on_double_click=None):
        super().__init__(parent)
        self._pixmap: QPixmap = QPixmap()
        self.on_double_click = on_double_click
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #0b1014; color: #6b7f88; font-size: 15px; font-weight: 600;")
        self.setText("Visual evaluation result will appear here")

    def set_pixmap(self, pix: QPixmap):
        self._pixmap = pix
        self._update_display()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def _update_display(self):
        if self._pixmap.isNull():
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        super().setPixmap(scaled)

    def mouseDoubleClickEvent(self, event):
        if self.on_double_click:
            self.on_double_click()
        else:
            super().mouseDoubleClickEvent(event)


class TestFullscreenDialog(QDialog):
    """
    Full-screen visual viewer for YOLO test predictions.
    Supports real-time frame streaming, carousel navigation, font resizing, and label mode switching.
    """

    def __init__(self, main_window, parent=None):
        super().__init__(parent or main_window)
        self.main_window = main_window
        self.setWindowTitle("Prediction Visual Preview - Full Screen")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        # Dark Cinema Style
        self.setStyleSheet("""
            QDialog {
                background-color: #0b1014;
            }
            QFrame#topBar {
                background-color: rgba(15, 23, 36, 0.94);
                border-bottom: 1.5px solid rgba(255, 255, 255, 0.08);
                padding: 6px 14px;
            }
            QLabel#titleLbl {
                color: #ffffff;
                font-size: 14px;
                font-weight: 700;
            }
            QLabel#infoLbl {
                color: #9fb6d4;
                font-size: 13px;
                font-weight: 600;
                padding: 2px 8px;
                background-color: rgba(255, 255, 255, 0.06);
                border-radius: 4px;
            }
            QLabel#badgeLbl {
                color: #a5d3b0;
                font-size: 12px;
                font-weight: 600;
            }
            QRadioButton {
                color: #e2e8f0;
                font-size: 12px;
                font-weight: 600;
                spacing: 6px;
            }
            QRadioButton::indicator {
                width: 15px;
                height: 15px;
                border: 1.5px solid #4a5d6e;
                border-radius: 8px;
                background-color: #1a2430;
            }
            QRadioButton::indicator:checked {
                background-color: #38bdf8;
                border: 2px solid #38bdf8;
            }
            QPushButton.fsNavBtn {
                background-color: rgba(255, 255, 255, 0.08);
                color: #ffffff;
                border: 1px solid rgba(255, 255, 255, 0.14);
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton.fsNavBtn:hover {
                background-color: rgba(255, 255, 255, 0.16);
            }
            QPushButton.fsNavBtn:disabled {
                color: #4a5d6e;
                background-color: rgba(255, 255, 255, 0.03);
                border-color: rgba(255, 255, 255, 0.05);
            }
            QPushButton.fsCloseBtn {
                background-color: #e11d48;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 5px 12px;
                font-size: 12px;
                font-weight: 700;
            }
            QPushButton.fsCloseBtn:hover {
                background-color: #f43f5e;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Top Control Bar
        top_bar = QFrame()
        top_bar.setObjectName("topBar")
        tb_layout = QHBoxLayout(top_bar)
        tb_layout.setContentsMargins(12, 8, 12, 8)
        tb_layout.setSpacing(12)

        lbl_title = QLabel("Prediction Visual Preview")
        lbl_title.setObjectName("titleLbl")
        tb_layout.addWidget(lbl_title)

        self.lbl_info = QLabel("0 / 0")
        self.lbl_info.setObjectName("infoLbl")
        tb_layout.addWidget(self.lbl_info)

        self.lbl_badge = QLabel("")
        self.lbl_badge.setObjectName("badgeLbl")
        tb_layout.addWidget(self.lbl_badge)

        tb_layout.addStretch()

        # Carousel Navigator
        self.btn_prev = QPushButton("< Prev")
        self.btn_prev.setProperty("class", "fsNavBtn")
        self.btn_prev.clicked.connect(self._on_prev_clicked)
        tb_layout.addWidget(self.btn_prev)

        self.btn_next = QPushButton("Next >")
        self.btn_next.setProperty("class", "fsNavBtn")
        self.btn_next.clicked.connect(self._on_next_clicked)
        tb_layout.addWidget(self.btn_next)

        tb_layout.addSpacing(8)

        # Label Mode Radio Group
        lbl_mode_title = QLabel("Label:")
        lbl_mode_title.setStyleSheet("color: #94a3b8; font-size: 12px; font-weight: 600;")
        tb_layout.addWidget(lbl_mode_title)

        self.rb_mode_name = QRadioButton("Name")
        self.rb_mode_id = QRadioButton("ID")
        self.rb_mode_none = QRadioButton("None")
        self.rb_mode_name.setChecked(True)

        self.btn_grp_mode = QButtonGroup(self)
        self.btn_grp_mode.addButton(self.rb_mode_name, 0)
        self.btn_grp_mode.addButton(self.rb_mode_id, 1)
        self.btn_grp_mode.addButton(self.rb_mode_none, 2)
        self.btn_grp_mode.buttonToggled.connect(self._on_label_mode_toggled)

        tb_layout.addWidget(self.rb_mode_name)
        tb_layout.addWidget(self.rb_mode_id)
        tb_layout.addWidget(self.rb_mode_none)

        tb_layout.addSpacing(8)

        # Font Adjuster
        lbl_font_title = QLabel("Font:")
        lbl_font_title.setStyleSheet("color: #94a3b8; font-size: 12px; font-weight: 600;")
        tb_layout.addWidget(lbl_font_title)

        self.btn_fs_dec = QPushButton("A-")
        self.btn_fs_dec.setProperty("class", "fsNavBtn")
        self.btn_fs_dec.setFixedWidth(32)
        self.btn_fs_dec.setToolTip("Decrease Font Size (-2 px) [Key: -]")
        self.btn_fs_dec.clicked.connect(self._dec_font)
        tb_layout.addWidget(self.btn_fs_dec)

        self.btn_fs_inc = QPushButton("A+")
        self.btn_fs_inc.setProperty("class", "fsNavBtn")
        self.btn_fs_inc.setFixedWidth(32)
        self.btn_fs_inc.setToolTip("Increase Font Size (+2 px) [Key: +]")
        self.btn_fs_inc.clicked.connect(self._inc_font)
        tb_layout.addWidget(self.btn_fs_inc)

        self.spin_fs = QSpinBox()
        self.spin_fs.setRange(8, 48)
        self.spin_fs.setValue(14)
        self.spin_fs.setSuffix(" px")
        self.spin_fs.setStyleSheet("""
            QSpinBox {
                background-color: rgba(255, 255, 255, 0.08);
                color: #ffffff;
                border: 1px solid rgba(255, 255, 255, 0.14);
                border-radius: 4px;
                padding: 2px 6px;
                font-size: 12px;
            }
        """)
        self.spin_fs.valueChanged.connect(self._on_font_spin_changed)
        tb_layout.addWidget(self.spin_fs)

        tb_layout.addSpacing(12)

        # Exit Fullscreen Button
        self.btn_close = QPushButton("Exit Fullscreen (Esc)")
        self.btn_close.setProperty("class", "fsCloseBtn")
        self.btn_close.setIcon(get_svg_icon("fullscreen", 14, inverse=True))
        self.btn_close.setIconSize(QSize(14, 14))
        self.btn_close.clicked.connect(self.close)
        tb_layout.addWidget(self.btn_close)

        layout.addWidget(top_bar)

        # Center Scaled Image Label
        self.image_canvas = ScaledImageLabel(self, on_double_click=self.close)
        self.image_canvas.setToolTip("Double-click to Exit Fullscreen")
        layout.addWidget(self.image_canvas, stretch=1)

        # Keyboard shortcuts
        QShortcut(QKeySequence(Qt.Key_Escape), self, self.close)
        QShortcut(QKeySequence(Qt.Key_F11), self, self.close)
        QShortcut(QKeySequence(Qt.Key_Left), self, self._on_prev_clicked)
        QShortcut(QKeySequence(Qt.Key_Right), self, self._on_next_clicked)
        QShortcut(QKeySequence(Qt.Key_Plus), self, self._inc_font)
        QShortcut(QKeySequence(Qt.Key_Equal), self, self._inc_font)
        QShortcut(QKeySequence(Qt.Key_Minus), self, self._dec_font)

        self._updating_from_main = False

    def sync_from_main(self):
        """Synchronizes label mode, font size, and navigation state from MainWindow."""
        self._updating_from_main = True
        try:
            cur_mode = self.main_window._get_current_test_label_mode()
            if cur_mode == "id":
                self.rb_mode_id.setChecked(True)
            elif cur_mode == "none":
                self.rb_mode_none.setChecked(True)
            else:
                self.rb_mode_name.setChecked(True)

            fs = self.main_window.slider_test_font.value()
            self.spin_fs.setValue(fs)
        finally:
            self._updating_from_main = False

        self.refresh_current_view()

    def set_frame(self, frame_bgr: np.ndarray, cur: int, tot: int, summary: dict):
        """Called live when a new prediction frame arrives from video or folder streaming."""
        pix = bgr_to_qpixmap(frame_bgr)
        self.image_canvas.set_pixmap(pix)

        mode_name = getattr(self.main_window.active_worker, "mode", "frame")
        prefix = "Image" if mode_name in ("folder", "split") else "Frame"
        self.lbl_info.setText(f"{prefix} {cur} / {tot}")

        badge_text = ", ".join([f"{k}: {v}" for k, v in summary.items()]) or "No objects detected"
        self.lbl_badge.setText(f"Objects: {badge_text}")

        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)

    def refresh_current_view(self):
        """Re-renders and displays whichever test item is active in MainWindow."""
        mw = self.main_window

        # 1. Single image mode
        if getattr(mw, "single_test_result", None):
            res = mw.single_test_result.get("raw_result")
            annotated_bgr = mw.single_test_result.get("annotated_bgr")
            if res is not None:
                mode = mw._get_current_test_label_mode()
                fs = mw.slider_test_font.value()
                annotated_bgr = mw.tester_engine.plot_result(res, label_mode=mode, font_size=fs)
                mw.single_test_result["annotated_bgr"] = annotated_bgr
            pix = bgr_to_qpixmap(annotated_bgr)
            self.image_canvas.set_pixmap(pix)
            self.lbl_info.setText(f"Image 1 / 1 ({mw.single_test_result.get('file_name', '')})")
            cls_sum = mw.single_test_result.get("classes_summary", {})
            badge_str = ", ".join([f"{k} ({v})" for k, v in cls_sum.items()]) or "No objects detected"
            self.lbl_badge.setText(f"Objects: {badge_str}")
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            return

        # 2. Folder / Split mode
        if mw.folder_test_results and 0 <= mw.current_folder_test_idx < len(mw.folder_test_results):
            idx = mw.current_folder_test_idx
            item = mw.folder_test_results[idx]
            res = item.get("raw_result")
            annotated_bgr = item.get("annotated_bgr")
            if res is not None:
                mode = mw._get_current_test_label_mode()
                fs = mw.slider_test_font.value()
                annotated_bgr = mw.tester_engine.plot_result(res, label_mode=mode, font_size=fs)
                item["annotated_bgr"] = annotated_bgr
            pix = bgr_to_qpixmap(annotated_bgr)
            self.image_canvas.set_pixmap(pix)
            tot = len(mw.folder_test_results)
            self.lbl_info.setText(f"Image {idx + 1} / {tot} ({item.get('file_name', '')})")
            cls_sum = item.get("classes_summary", {})
            badge_str = ", ".join([f"{k} ({v})" for k, v in cls_sum.items()]) or "No objects detected"
            self.lbl_badge.setText(f"Objects: {badge_str}")
            self.btn_prev.setEnabled(idx > 0)
            self.btn_next.setEnabled(idx < tot - 1)
            return

        # 3. Video mode
        if getattr(mw, "last_video_result", None) or getattr(mw, "last_video_raw_res", None):
            res = getattr(mw, "last_video_raw_res", None)
            if res is not None:
                mode = mw._get_current_test_label_mode()
                fs = mw.slider_test_font.value()
                annotated_bgr = mw.tester_engine.plot_result(res, label_mode=mode, font_size=fs)
            else:
                annotated_bgr = mw.last_video_result.get("last_annotated")
            pix = bgr_to_qpixmap(annotated_bgr)
            self.image_canvas.set_pixmap(pix)
            self.lbl_info.setText("Video Prediction")
            cls_sum = mw.last_video_result.get("classes_summary", {}) if mw.last_video_result else {}
            badge_str = ", ".join([f"{k}: {v}" for k, v in cls_sum.items()]) or "No objects detected"
            self.lbl_badge.setText(f"Video Summary: {badge_str}")
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            return

        # 4. Fallback to whatever is in lbl_test_preview
        if mw.lbl_test_preview.pixmap() and not mw.lbl_test_preview.pixmap().isNull():
            self.image_canvas.set_pixmap(mw.lbl_test_preview.pixmap())
            self.lbl_info.setText(mw.lbl_test_img_counter.text())
            self.lbl_badge.setText(mw.lbl_test_classes_badge.text())
            self.btn_prev.setEnabled(mw.btn_prev_test_img.isEnabled())
            self.btn_next.setEnabled(mw.btn_next_test_img.isEnabled())

    def _on_prev_clicked(self):
        self.main_window._on_prev_test_img()
        self.refresh_current_view()

    def _on_next_clicked(self):
        self.main_window._on_next_test_img()
        self.refresh_current_view()

    def _on_label_mode_toggled(self):
        if self._updating_from_main:
            return
        idx = self.btn_grp_mode.checkedId()
        # Sync to main window
        if idx == 0:
            self.main_window.rb_label_name.setChecked(True)
        elif idx == 1:
            self.main_window.rb_label_id.setChecked(True)
        elif idx == 2:
            self.main_window.rb_label_none.setChecked(True)
        self.refresh_current_view()

    def _dec_font(self):
        new_val = max(8, self.main_window.slider_test_font.value() - 2)
        self.main_window.slider_test_font.setValue(new_val)
        self.spin_fs.setValue(new_val)
        self.refresh_current_view()

    def _inc_font(self):
        new_val = min(48, self.main_window.slider_test_font.value() + 2)
        self.main_window.slider_test_font.setValue(new_val)
        self.spin_fs.setValue(new_val)
        self.refresh_current_view()

    def _on_font_spin_changed(self, val: int):
        if self._updating_from_main:
            return
        self.main_window.slider_test_font.setValue(val)
        self.refresh_current_view()
