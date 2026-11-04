"""
Theme and styling definitions for the CVAT-to-YOLO PyQt5 GUI application.
Modern Dark Cyberpunk / Slate styling with vibrant neon accents.
"""

DARK_STYLESHEET = """
/* Base Application Window */
QMainWindow, QWidget#centralWidget {
    background-color: #0b1120;
    color: #f1f5f9;
    font-family: 'Segoe UI', 'SF Pro Display', -apple-system, sans-serif;
    font-size: 13px;
}

/* Card Containers */
QFrame.CardFrame, QWidget.CardFrame {
    background-color: #151f32;
    border: 1px solid #1e293b;
    border-radius: 12px;
}

QFrame.HeaderCard {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #111827, stop:1 #1e1b4b);
    border: 1px solid #312e81;
    border-radius: 14px;
}

/* Headings & Labels */
QLabel {
    color: #e2e8f0;
}

QLabel#titleLabel {
    font-size: 22px;
    font-weight: 800;
    color: #f8fafc;
}

QLabel#subtitleLabel {
    font-size: 12px;
    color: #94a3b8;
}

QLabel.SectionHeader {
    font-size: 15px;
    font-weight: 700;
    color: #38bdf8;
    padding-bottom: 4px;
}

QLabel.BadgeSuccess {
    background-color: #064e3b;
    color: #34d399;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
}

QLabel.BadgeInfo {
    background-color: #1e3a8a;
    color: #60a5fa;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
}

/* Modern Tab Widget */
QTabWidget::pane {
    border: 1px solid #1e293b;
    border-radius: 10px;
    background-color: #0f172a;
    top: -1px;
}

QTabBar::tab {
    background-color: #172033;
    color: #94a3b8;
    padding: 10px 20px;
    margin-right: 4px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    border: 1px solid #1e293b;
    border-bottom: none;
    font-weight: 600;
    font-size: 13px;
}

QTabBar::tab:selected {
    background-color: #0f172a;
    color: #38bdf8;
    border-top: 3px solid #0284c7;
    border-left: 1px solid #334155;
    border-right: 1px solid #334155;
}

QTabBar::tab:hover:!selected {
    background-color: #1e293b;
    color: #e2e8f0;
}

/* Form Controls: LineEdit, ComboBox, SpinBox */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #0d1527;
    color: #f8fafc;
    border: 1px solid #293548;
    border-radius: 7px;
    padding: 7px 12px;
    selection-background-color: #6366f1;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #38bdf8;
    background-color: #0f172a;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 25px;
    border-left: 1px solid #293548;
}

QComboBox QAbstractItemView {
    background-color: #151f32;
    color: #f1f5f9;
    border: 1px solid #334155;
    selection-background-color: #312e81;
    selection-color: #38bdf8;
    outline: none;
    padding: 4px;
}

/* Push Buttons */
QPushButton {
    background-color: #1e293b;
    color: #f1f5f9;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #273549;
    border-color: #475569;
}

QPushButton:pressed {
    background-color: #0f172a;
}

/* Primary Action Button (Electric Cyan / Indigo) */
QPushButton.PrimaryButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f46e5, stop:1 #06b6d4);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 10px 22px;
    font-weight: 700;
    font-size: 13px;
}

QPushButton.PrimaryButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6366f1, stop:1 #22d3ee);
}

QPushButton.PrimaryButton:pressed {
    background: #3730a3;
}

QPushButton.PrimaryButton:disabled {
    background: #1e293b;
    color: #64748b;
    border: 1px solid #334155;
}

/* Success Action Button (Neon Emerald) */
QPushButton.SuccessButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #059669, stop:1 #10b981);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 700;
}

QPushButton.SuccessButton:hover {
    background: #10b981;
}

/* Danger / Stop Button (Rose Red) */
QPushButton.DangerButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #b91c1c, stop:1 #f43f5e);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 700;
}

QPushButton.DangerButton:hover {
    background: #f43f5e;
}

/* Tables & Tree Views */
QTableWidget, QTableView {
    background-color: #0c1322;
    color: #f1f5f9;
    gridline-color: #1e293b;
    border: 1px solid #1e293b;
    border-radius: 8px;
    selection-background-color: #1e293b;
    selection-color: #38bdf8;
}

QHeaderView::section {
    background-color: #131d31;
    color: #94a3b8;
    padding: 8px 10px;
    border: none;
    border-right: 1px solid #1e293b;
    border-bottom: 2px solid #24324d;
    font-weight: 700;
    font-size: 12px;
}

/* Progress Bars */
QProgressBar {
    background-color: #0d1527;
    border: 1px solid #1e293b;
    border-radius: 7px;
    text-align: center;
    color: #f8fafc;
    font-weight: 600;
    font-size: 11px;
    min-height: 18px;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:0.5 #8b5cf6, stop:1 #06b6d4);
    border-radius: 6px;
}

/* Checkboxes & Radio Buttons */
QCheckBox, QRadioButton {
    color: #e2e8f0;
    spacing: 8px;
    font-size: 13px;
}

QCheckBox::indicator, QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid #475569;
    border-radius: 4px;
    background-color: #0f172a;
}

QRadioButton::indicator {
    border-radius: 9px;
}

QCheckBox::indicator:hover, QRadioButton::indicator:hover {
    border-color: #38bdf8;
}

QCheckBox::indicator:checked {
    background-color: #0284c7;
    border-color: #38bdf8;
    image: none;
}

QRadioButton::indicator:checked {
    background-color: #0284c7;
    border-color: #38bdf8;
}

/* Text Edit / Console Log */
QPlainTextEdit, QTextEdit {
    background-color: #080d1a;
    color: #34d399;
    border: 1px solid #1e293b;
    border-radius: 8px;
    font-family: 'Cascadia Code', 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    padding: 8px;
}

/* Sliders */
QSlider::groove:horizontal {
    height: 6px;
    background: #1e293b;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6366f1, stop:1 #06b6d4);
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #38bdf8;
    border: 2px solid #ffffff;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #ffffff;
    border-color: #38bdf8;
}

/* Scrollbars */
QScrollBar:vertical {
    background: #0f172a;
    width: 10px;
    margin: 0px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: #334155;
    min-height: 25px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #475569;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background: #0f172a;
    height: 10px;
    margin: 0px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background: #334155;
    min-width: 25px;
    border-radius: 5px;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}
"""
