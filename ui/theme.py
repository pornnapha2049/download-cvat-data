"""
Theme and styling definitions for the CVAT-to-YOLO PyQt5 GUI application.
Inspired by the modern mindfulness, pastel wellness, and study aesthetic:
https://i.pinimg.com/1200x/69/a3/a3/69a3a365c2017d08d2c1ce4795633dd5.jpg

Features:
- Calm, soothing palette (soft slate-cream #f0f4f8, crisp white cards #ffffff, deep slate navy #132c3f, matcha mint #daf0dd, lavender rose #fde4ed, butter yellow #fef3d1, periwinkle #e0e8f8)
- Rounded cards (16px radius), sleek pill buttons, and responsive vertical line-height
- High DPI scaling compatibility with zero text clipping in the Y-axis.
"""

AESTHETIC_STYLESHEET = """
/* Base Application Window (Soft Modern Slate-Cream) */
QMainWindow, QWidget#centralWidget {
    background-color: #f0f4f8;
    color: #1e293b;
    font-family: 'Segoe UI', 'Nunito', 'Leelawadee UI', 'Tahoma', -apple-system, sans-serif;
    font-size: 13px;
}

/* ========================================================= */
/* Scroll Areas                                              */
/* ========================================================= */
QScrollArea {
    background-color: transparent;
    border: none;
}

QScrollArea > QWidget > QWidget {
    background-color: transparent;
}

/* ========================================================= */
/* Dedicated Login Page (Mindfulness Wellness Card)          */
/* ========================================================= */
QFrame.LoginCard {
    background-color: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 20px;
    padding: 26px;
}

QLabel#loginTitle {
    font-size: 24px;
    font-weight: 800;
    color: #132c3f;
    min-height: 32px;
}

QLabel#loginSubtitle {
    font-size: 13px;
    color: #64748b;
    min-height: 20px;
}

QLineEdit.LoginInput {
    background-color: #f8fafc;
    color: #1e293b;
    border: 1.5px solid #cbd5e1;
    border-radius: 12px;
    padding: 8px 14px;
    font-size: 13.5px;
    min-height: 30px;
}

QLineEdit.LoginInput:focus {
    border: 2px solid #132c3f;
    background-color: #ffffff;
}

/* Primary Action Button (Deep Slate Navy / Matcha) */
QPushButton.LoginButton {
    background-color: #132c3f;
    color: #ffffff;
    border: none;
    border-radius: 12px;
    padding: 10px 24px;
    font-weight: 800;
    font-size: 14px;
    min-height: 34px;
}

QPushButton.LoginButton:hover {
    background-color: #1d3e58;
}

QPushButton.LoginButton:pressed {
    background-color: #0b1a26;
}

QPushButton.LogoutButton {
    background-color: #fde4ed;
    color: #9f1239;
    border: 1.2px solid #fbcfe8;
    border-radius: 10px;
    padding: 5px 12px;
    font-weight: 700;
    font-size: 11.5px;
    min-height: 24px;
}

QPushButton.LogoutButton:hover {
    background-color: #f43f5e;
    color: #ffffff;
}

QPushButton.SkipLoginButton {
    background-color: transparent;
    color: #64748b;
    border: none;
    font-size: 12.5px;
    text-decoration: underline;
    min-height: 24px;
}

QPushButton.SkipLoginButton:hover {
    color: #132c3f;
}

QLabel.AlertError {
    background-color: #fef2f2;
    color: #991b1b;
    border: 1.5px solid #fecaca;
    border-radius: 10px;
    padding: 8px 14px;
    font-size: 12px;
    font-weight: 600;
    min-height: 22px;
}

QLabel.AlertSuccess {
    background-color: #f0fdf4;
    color: #166534;
    border: 1.5px solid #bbf7d0;
    border-radius: 10px;
    padding: 8px 14px;
    font-size: 12px;
    font-weight: 600;
    min-height: 22px;
}

/* ========================================================= */
/* Card Containers & Header                                  */
/* ========================================================= */
QFrame.CardFrame, QWidget.CardFrame {
    background-color: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 16px;
}

QFrame.HeaderCard {
    background-color: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 18px;
}

/* Headings & Labels */
QLabel {
    color: #1e293b;
}

QLabel#titleLabel {
    font-size: 21px;
    font-weight: 800;
    color: #132c3f;
    min-height: 28px;
}

QLabel#subtitleLabel {
    font-size: 12.5px;
    color: #64748b;
    min-height: 18px;
}

QLabel.SectionHeader {
    font-size: 15px;
    font-weight: 700;
    color: #132c3f;
    padding-bottom: 4px;
    min-height: 22px;
}

QLabel.BadgeSuccess {
    background-color: #daf0dd;
    color: #166534;
    padding: 4px 10px;
    border-radius: 9px;
    font-size: 11.5px;
    font-weight: 700;
    border: 1px solid #c4e7c7;
    min-height: 18px;
}

QLabel.BadgeInfo {
    background-color: #e0e8f8;
    color: #1e3a8a;
    padding: 4px 10px;
    border-radius: 9px;
    font-size: 11.5px;
    font-weight: 700;
    border: 1px solid #cbd9f4;
    min-height: 18px;
}

/* ========================================================= */
/* Stepper Navigation Bar (Pastel Card Buttons)              */
/* ========================================================= */
QFrame.StepperContainer {
    background-color: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 16px;
    padding: 6px;
}

QPushButton.StepButton {
    background-color: #f8fafc;
    color: #475569;
    border: 1.5px solid #e2e8f0;
    border-radius: 12px;
    padding: 6px 12px;
    text-align: left;
    font-size: 12px;
    font-weight: 600;
    min-height: 48px;
}

QPushButton.StepButton:hover {
    background-color: #ffffff;
    color: #1e293b;
    border-color: #cbd5e1;
}

/* Active Step: Soft Matcha Mint Highlight */
QPushButton.StepButtonActive {
    background-color: #daf0dd;
    color: #14532d;
    border: 2px solid #22c55e;
    border-radius: 12px;
    padding: 6px 12px;
    text-align: left;
    font-size: 12px;
    font-weight: 800;
    min-height: 48px;
}

/* Completed Step: Soft Pastel Cream with Green Border */
QPushButton.StepButtonCompleted {
    background-color: #f0fdf4;
    color: #166534;
    border: 1.5px solid #86efac;
    border-radius: 12px;
    padding: 6px 12px;
    text-align: left;
    font-size: 12px;
    font-weight: 700;
    min-height: 48px;
}

/* Step Wizard Guidance Card (Day Advice / Quote Style) */
QFrame.GuideBox {
    background-color: #fefce8;
    border: 1.5px solid #fef08a;
    border-radius: 14px;
    padding: 8px 14px;
    min-height: 28px;
}

QLabel#guideLabel {
    color: #713f12;
    font-size: 12.5px;
    font-weight: 600;
    line-height: 1.4;
}

/* Bottom Stepper Footer */
QFrame.StepperFooter {
    background-color: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 14px;
    padding: 6px 14px;
    min-height: 38px;
}

/* ========================================================= */
/* Form Controls: LineEdit, ComboBox, SpinBox                */
/* ========================================================= */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #f8fafc;
    color: #1e293b;
    border: 1.5px solid #cbd5e1;
    border-radius: 10px;
    padding: 5px 12px;
    min-height: 28px;
    selection-background-color: #132c3f;
    selection-color: #ffffff;
    font-size: 13px;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 2px solid #132c3f;
    background-color: #ffffff;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 25px;
    border-left: 1px solid #cbd5e1;
}

QComboBox QAbstractItemView {
    background-color: #ffffff;
    color: #1e293b;
    border: 1.5px solid #cbd5e1;
    selection-background-color: #daf0dd;
    selection-color: #14532d;
    outline: none;
    padding: 4px;
    font-size: 13px;
}

/* Buttons */
QPushButton {
    background-color: #f8fafc;
    color: #1e293b;
    border: 1.5px solid #cbd5e1;
    border-radius: 10px;
    padding: 6px 16px;
    min-height: 28px;
    font-weight: 700;
    font-size: 12.5px;
}

QPushButton:hover {
    background-color: #f1f5f9;
    border-color: #94a3b8;
}

QPushButton:pressed {
    background-color: #e2e8f0;
}

/* Primary Action Button (Deep Slate Navy) */
QPushButton.PrimaryButton {
    background-color: #132c3f;
    color: #ffffff;
    border: none;
    border-radius: 11px;
    padding: 8px 20px;
    min-height: 32px;
    font-weight: 800;
    font-size: 13px;
}

QPushButton.PrimaryButton:hover {
    background-color: #1e405a;
}

QPushButton.PrimaryButton:pressed {
    background-color: #0d1e2b;
}

QPushButton.PrimaryButton:disabled {
    background-color: #e2e8f0;
    color: #94a3b8;
    border: 1px solid #cbd5e1;
}

/* Success Action Button (Forest Matcha Green) */
QPushButton.SuccessButton {
    background-color: #166534;
    color: #ffffff;
    border: none;
    border-radius: 11px;
    padding: 8px 20px;
    min-height: 32px;
    font-weight: 800;
    font-size: 13px;
}

QPushButton.SuccessButton:hover {
    background-color: #15803d;
}

QPushButton.SuccessButton:pressed {
    background-color: #14532d;
}

/* Danger / Stop Button (Rose Red) */
QPushButton.DangerButton {
    background-color: #e11d48;
    color: #ffffff;
    border: none;
    border-radius: 11px;
    padding: 8px 20px;
    min-height: 32px;
    font-weight: 800;
    font-size: 13px;
}

QPushButton.DangerButton:hover {
    background-color: #be123c;
}

/* Step Flow Nav Button */
QPushButton.StepNavButton {
    background-color: #ffffff;
    color: #1e293b;
    border: 1.5px solid #cbd5e1;
    border-radius: 10px;
    padding: 7px 18px;
    min-height: 30px;
    font-weight: 800;
    font-size: 12.5px;
}

QPushButton.StepNavButton:hover {
    background-color: #f1f5f9;
    border-color: #94a3b8;
}

QPushButton.StepNavButton:disabled {
    background-color: #f8fafc;
    color: #94a3b8;
    border: 1px solid #e2e8f0;
}

/* Tables */
QTableWidget, QTableView {
    background-color: #ffffff;
    color: #1e293b;
    gridline-color: #f1f5f9;
    border: 1.5px solid #e2e8f0;
    border-radius: 12px;
    selection-background-color: #daf0dd;
    selection-color: #14532d;
    font-size: 12.5px;
}

QHeaderView::section {
    background-color: #f8fafc;
    color: #334155;
    padding: 6px 10px;
    min-height: 28px;
    border: none;
    border-right: 1px solid #e2e8f0;
    border-bottom: 2px solid #cbd5e1;
    font-weight: 800;
    font-size: 12px;
}

/* Progress Bars */
QProgressBar {
    background-color: #e2e8f0;
    border: 1.5px solid #cbd5e1;
    border-radius: 9px;
    text-align: center;
    color: #1e293b;
    font-weight: 800;
    font-size: 11.5px;
    min-height: 22px;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #166534, stop:0.5 #3b82f6, stop:1 #e11d48);
    border-radius: 7px;
}

/* Checkboxes & Radio Buttons */
QCheckBox, QRadioButton {
    color: #1e293b;
    spacing: 8px;
    font-size: 13px;
    font-weight: 600;
    min-height: 22px;
}

QCheckBox::indicator, QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border: 1.5px solid #94a3b8;
    border-radius: 5px;
    background-color: #f8fafc;
}

QRadioButton::indicator {
    border-radius: 10px;
}

QCheckBox::indicator:hover, QRadioButton::indicator:hover {
    border-color: #132c3f;
}

QCheckBox::indicator:checked {
    background-color: #132c3f;
    border-color: #132c3f;
    image: none;
}

QRadioButton::indicator:checked {
    background-color: #132c3f;
    border-color: #132c3f;
}

/* Text Edit / Console Log */
QPlainTextEdit, QTextEdit {
    background-color: #0f172a;
    color: #4ade80;
    border: 1.5px solid #1e293b;
    border-radius: 12px;
    font-family: 'Cascadia Code', 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    padding: 8px;
}

/* Sliders */
QSlider::groove:horizontal {
    height: 7px;
    background: #e2e8f0;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #166534, stop:1 #3b82f6);
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #132c3f;
    border: 2px solid #ffffff;
    width: 18px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 9px;
}

QSlider::handle:horizontal:hover {
    background: #1e405a;
    border-color: #ffffff;
}

/* GroupBox */
QGroupBox {
    border: 1.5px solid #e2e8f0;
    border-radius: 14px;
    margin-top: 12px;
    padding-top: 14px;
    padding-bottom: 10px;
    padding-left: 10px;
    padding-right: 10px;
    font-weight: 800;
    color: #132c3f;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
}

/* Scrollbars */
QScrollBar:vertical {
    background: #f0f4f8;
    width: 10px;
    margin: 0px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: #cbd5e1;
    min-height: 25px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #94a3b8;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background: #f0f4f8;
    height: 10px;
    margin: 0px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background: #cbd5e1;
    min-width: 25px;
    border-radius: 5px;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}
"""

DARK_STYLESHEET = AESTHETIC_STYLESHEET
