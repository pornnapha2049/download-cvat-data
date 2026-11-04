"""
Theme and styling definitions for the CVAT-to-YOLO PyQt5 GUI application.
Inspired by Aesthetic Printable Stickers, Cozy Stationery, and Journal Layouts (.☘︎ ݁˖).
Optimized for all screen resolutions and High DPI displays with generous vertical breathing room.
"""

AESTHETIC_STYLESHEET = """
/* Base Application Window (Warm Linen / Cream Paper) */
QMainWindow, QWidget#centralWidget {
    background-color: #fbf8f2;
    color: #332822;
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
/* Dedicated Login Page (Cozy Aesthetic Stationery Card)    */
/* ========================================================= */
QFrame.LoginCard {
    background-color: #ffffff;
    border: 2px solid #e2d5c3;
    border-radius: 20px;
    padding: 24px;
}

QLabel#loginTitle {
    font-size: 24px;
    font-weight: 800;
    color: #2e221b;
    min-height: 32px;
}

QLabel#loginSubtitle {
    font-size: 13px;
    color: #736154;
    min-height: 20px;
}

QLineEdit.LoginInput {
    background-color: #faf6ef;
    color: #2e221b;
    border: 1.5px solid #dcd0bf;
    border-radius: 10px;
    padding: 8px 14px;
    font-size: 13.5px;
    min-height: 30px;
}

QLineEdit.LoginInput:focus {
    border: 2px solid #5e8460;
    background-color: #ffffff;
}

/* Primary Action Button (Matcha Sage Green ☘︎) */
QPushButton.LoginButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #587b5a, stop:1 #476649);
    color: #ffffff;
    border: none;
    border-radius: 12px;
    padding: 10px 24px;
    font-weight: 800;
    font-size: 14px;
    min-height: 34px;
}

QPushButton.LoginButton:hover {
    background: #4a6c4c;
}

QPushButton.LoginButton:pressed {
    background: #39523b;
}

QPushButton.LogoutButton {
    background-color: #fce8e6;
    color: #c85a54;
    border: 1px solid #f3c2be;
    border-radius: 8px;
    padding: 5px 12px;
    font-weight: 700;
    font-size: 11.5px;
    min-height: 22px;
}

QPushButton.LogoutButton:hover {
    background-color: #c85a54;
    color: #ffffff;
}

QPushButton.SkipLoginButton {
    background-color: transparent;
    color: #7a6a5e;
    border: none;
    font-size: 12.5px;
    text-decoration: underline;
    min-height: 24px;
}

QPushButton.SkipLoginButton:hover {
    color: #5e8460;
}

QLabel.AlertError {
    background-color: #fef2f2;
    color: #b91c1c;
    border: 1.5px solid #fca5a5;
    border-radius: 10px;
    padding: 8px 14px;
    font-size: 12px;
    font-weight: 600;
    min-height: 22px;
}

QLabel.AlertSuccess {
    background-color: #f0fdf4;
    color: #166534;
    border: 1.5px solid #86efac;
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
    border: 1.5px solid #e5dbcd;
    border-radius: 14px;
}

QFrame.HeaderCard {
    background-color: #ffffff;
    border: 1.5px solid #dfd2c0;
    border-radius: 16px;
}

/* Headings & Labels */
QLabel {
    color: #382c24;
}

QLabel#titleLabel {
    font-size: 21px;
    font-weight: 800;
    color: #2b1f18;
    min-height: 28px;
}

QLabel#subtitleLabel {
    font-size: 12.5px;
    color: #786659;
    min-height: 18px;
}

QLabel.SectionHeader {
    font-size: 15px;
    font-weight: 700;
    color: #5e8460;
    padding-bottom: 4px;
    min-height: 22px;
}

QLabel.BadgeSuccess {
    background-color: #eaf3ea;
    color: #3b633d;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 11.5px;
    font-weight: 700;
    border: 1px solid #cce2cc;
    min-height: 18px;
}

QLabel.BadgeInfo {
    background-color: #eef4f8;
    color: #4a6c82;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 11.5px;
    font-weight: 700;
    border: 1px solid #d2e3ee;
    min-height: 18px;
}

/* ========================================================= */
/* Stepper Navigation Bar (Sticker Tag Style)                */
/* ========================================================= */
QFrame.StepperContainer {
    background-color: #f3ece0;
    border: 1.5px solid #e0d4c2;
    border-radius: 14px;
    padding: 5px;
}

QPushButton.StepButton {
    background-color: #ffffff;
    color: #6d5b4e;
    border: 1.5px solid #e2d6c4;
    border-radius: 10px;
    padding: 6px 12px;
    text-align: left;
    font-size: 12px;
    font-weight: 600;
    min-height: 48px;
}

QPushButton.StepButton:hover {
    background-color: #faf6ee;
    color: #2e221b;
    border-color: #c9bbae;
}

/* Active Step: Sage Matcha Highlight (.☘︎ ݁˖) */
QPushButton.StepButtonActive {
    background-color: #edf4ed;
    color: #355737;
    border: 2px solid #5e8460;
    border-radius: 10px;
    padding: 6px 12px;
    text-align: left;
    font-size: 12px;
    font-weight: 800;
    min-height: 48px;
}

/* Completed Step: Soft Cream with Green Edge */
QPushButton.StepButtonCompleted {
    background-color: #fbf8f0;
    color: #4a6e4c;
    border: 1.5px solid #7ea880;
    border-radius: 10px;
    padding: 6px 12px;
    text-align: left;
    font-size: 12px;
    font-weight: 700;
    min-height: 48px;
}

/* Step Wizard Guidance Card (Cozy Washi Tape Box) */
QFrame.GuideBox {
    background-color: #fefaf3;
    border: 1.5px dashed #d5c4ae;
    border-radius: 12px;
    padding: 8px 14px;
    min-height: 28px;
}

QLabel#guideLabel {
    color: #4d3d32;
    font-size: 12.5px;
    font-weight: 600;
    line-height: 1.4;
}

/* Bottom Stepper Footer */
QFrame.StepperFooter {
    background-color: #ffffff;
    border: 1.5px solid #e5dbcd;
    border-radius: 12px;
    padding: 6px 14px;
    min-height: 38px;
}

/* ========================================================= */
/* Form Controls: LineEdit, ComboBox, SpinBox                */
/* ========================================================= */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #faf6f0;
    color: #2e221b;
    border: 1.5px solid #ded3c2;
    border-radius: 8px;
    padding: 5px 12px;
    min-height: 28px;
    selection-background-color: #5e8460;
    selection-color: #ffffff;
    font-size: 13px;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 2px solid #5e8460;
    background-color: #ffffff;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 25px;
    border-left: 1px solid #ded3c2;
}

QComboBox QAbstractItemView {
    background-color: #ffffff;
    color: #332822;
    border: 1.5px solid #ded3c2;
    selection-background-color: #edf4ed;
    selection-color: #355737;
    outline: none;
    padding: 4px;
    font-size: 13px;
}

/* Buttons */
QPushButton {
    background-color: #f4ecdf;
    color: #45372d;
    border: 1.5px solid #d8cbba;
    border-radius: 9px;
    padding: 6px 16px;
    min-height: 28px;
    font-weight: 700;
    font-size: 12.5px;
}

QPushButton:hover {
    background-color: #e9decb;
    border-color: #c5b6a3;
}

QPushButton:pressed {
    background-color: #dcd0bd;
}

/* Primary Action Button (Matcha Sage Green ☘︎) */
QPushButton.PrimaryButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #587b5a, stop:1 #476649);
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 8px 20px;
    min-height: 32px;
    font-weight: 800;
    font-size: 13px;
}

QPushButton.PrimaryButton:hover {
    background: #4a6c4c;
}

QPushButton.PrimaryButton:pressed {
    background: #39523b;
}

QPushButton.PrimaryButton:disabled {
    background: #e5dbcd;
    color: #9c8e82;
    border: 1px solid #d8cbba;
}

/* Success Action Button (Honey Butter Yellow) */
QPushButton.SuccessButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #e09f3e, stop:1 #cc8b2b);
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 8px 20px;
    min-height: 32px;
    font-weight: 800;
    font-size: 13px;
}

QPushButton.SuccessButton:hover {
    background: #d49332;
}

/* Danger / Stop Button (Terracotta / Red Apple 🍎) */
QPushButton.DangerButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #c85a54, stop:1 #b34640);
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 8px 20px;
    min-height: 32px;
    font-weight: 800;
    font-size: 13px;
}

QPushButton.DangerButton:hover {
    background: #b34640;
}

/* Step Flow Button */
QPushButton.StepNavButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #5e8460, stop:1 #4d6f4f);
    color: #ffffff;
    border: none;
    border-radius: 9px;
    padding: 7px 18px;
    min-height: 30px;
    font-weight: 800;
    font-size: 12.5px;
}

QPushButton.StepNavButton:hover {
    background: #4a6c4c;
}

QPushButton.StepNavButton:disabled {
    background: #f0e6d6;
    color: #a89a8c;
    border: 1px solid #dfd2bf;
}

/* Tables */
QTableWidget, QTableView {
    background-color: #ffffff;
    color: #332822;
    gridline-color: #efe5d6;
    border: 1.5px solid #dfd2c0;
    border-radius: 10px;
    selection-background-color: #edf4ed;
    selection-color: #355737;
    font-size: 12.5px;
}

QHeaderView::section {
    background-color: #f7efe2;
    color: #5c4b3f;
    padding: 6px 10px;
    min-height: 28px;
    border: none;
    border-right: 1px solid #e8dcce;
    border-bottom: 2px solid #ddd0c0;
    font-weight: 800;
    font-size: 12px;
}

/* Progress Bars */
QProgressBar {
    background-color: #f2e9dc;
    border: 1.5px solid #ded2c1;
    border-radius: 9px;
    text-align: center;
    color: #382c24;
    font-weight: 800;
    font-size: 11.5px;
    min-height: 22px;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #5e8460, stop:0.5 #e8b056, stop:1 #c85a54);
    border-radius: 7px;
}

/* Checkboxes & Radio Buttons */
QCheckBox, QRadioButton {
    color: #3a2e26;
    spacing: 8px;
    font-size: 13px;
    font-weight: 600;
    min-height: 22px;
}

QCheckBox::indicator, QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border: 1.5px solid #baa996;
    border-radius: 5px;
    background-color: #faf6f0;
}

QRadioButton::indicator {
    border-radius: 10px;
}

QCheckBox::indicator:hover, QRadioButton::indicator:hover {
    border-color: #5e8460;
}

QCheckBox::indicator:checked {
    background-color: #5e8460;
    border-color: #4a6c4c;
    image: none;
}

QRadioButton::indicator:checked {
    background-color: #5e8460;
    border-color: #4a6c4c;
}

/* Text Edit / Console Log (Vintage Paper Log) */
QPlainTextEdit, QTextEdit {
    background-color: #2b211a;
    color: #86efac;
    border: 1.5px solid #4a3b32;
    border-radius: 10px;
    font-family: 'Cascadia Code', 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    padding: 8px;
}

/* Sliders */
QSlider::groove:horizontal {
    height: 7px;
    background: #e8dccb;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #5e8460, stop:1 #e09f3e);
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #5e8460;
    border: 2px solid #ffffff;
    width: 18px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 9px;
}

QSlider::handle:horizontal:hover {
    background: #476649;
    border-color: #ffffff;
}

/* GroupBox */
QGroupBox {
    border: 1.5px solid #ded1bf;
    border-radius: 12px;
    margin-top: 12px;
    padding-top: 14px;
    padding-bottom: 10px;
    padding-left: 10px;
    padding-right: 10px;
    font-weight: 800;
    color: #587b5a;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
    background-color: #ffffff;
    border: 1px solid #e0d4c2;
    border-radius: 6px;
}

/* Scrollbars */
QScrollBar:vertical {
    background: #fbf8f2;
    width: 10px;
    margin: 0px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: #d8cbbb;
    min-height: 25px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #b5a492;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background: #fbf8f2;
    height: 10px;
    margin: 0px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background: #d8cbbb;
    min-width: 25px;
    border-radius: 5px;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}
"""

DARK_STYLESHEET = AESTHETIC_STYLESHEET
