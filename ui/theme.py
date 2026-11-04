"""
Theme and styling definitions for the CVAT-to-YOLO PyQt5 GUI application.
Inspired by the modern mindfulness, pastel wellness, and study aesthetic:
https://i.pinimg.com/1200x/69/a3/a3/69a3a365c2017d08d2c1ce4795633dd5.jpg

Features:
- Calm, soothing palette (soft slate-cream #eef2f6, crisp white cards #ffffff, deep slate teal/navy #0f3747, matcha mint #daf0dd, lavender rose #fde2ec, butter yellow #fef4cc, periwinkle #e0e8f8, lilac #eee3fc)
- Rounded cards (20px radius), sleek pill buttons, circular navigation buttons, and responsive vertical line-height.
- High DPI scaling compatibility with zero text clipping in the Y-axis.
"""

AESTHETIC_STYLESHEET = """
/* Base Application Window (Soft Modern Slate-Cream) */
QMainWindow, QWidget#centralWidget {
    background-color: #eef2f6;
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
/* Top App Header (Clean Modern Mobile / Tablet Style)       */
/* ========================================================= */
QFrame#topAppBar {
    background-color: transparent;
    border: none;
}

QPushButton#circleNavButton {
    background-color: #ffffff;
    border: 1.5px solid #dce4ec;
    border-radius: 19px;
    min-width: 38px;
    min-height: 38px;
    max-width: 38px;
    max-height: 38px;
    padding: 0px;
}

QPushButton#circleNavButton:hover {
    background-color: #f1f5f9;
    border-color: #0f3747;
}

QPushButton#circleNavButton:pressed {
    background-color: #e2e8f0;
}

QLabel#appTitleLabel {
    font-size: 20px;
    font-weight: 800;
    color: #0f3747;
    min-height: 28px;
}

QLabel#greetingTitle {
    font-size: 22px;
    font-weight: 800;
    color: #0f3747;
    min-height: 28px;
}

QLabel#greetingSubtitle {
    font-size: 13px;
    color: #64748b;
    min-height: 18px;
}

/* ========================================================= */
/* Capsule Tag Badges (Pinterest Day Advice / Category Style)*/
/* ========================================================= */
QLabel.TagPink {
    background-color: #fde2ec;
    color: #9f1239;
    border-radius: 12px;
    padding: 4px 14px;
    font-weight: 700;
    font-size: 12px;
    min-height: 20px;
}

QLabel.TagYellow {
    background-color: #fef4cc;
    color: #92400e;
    border-radius: 12px;
    padding: 4px 14px;
    font-weight: 700;
    font-size: 12px;
    min-height: 20px;
}

QLabel.TagGreen {
    background-color: #daf0dd;
    color: #166534;
    border-radius: 12px;
    padding: 4px 14px;
    font-weight: 700;
    font-size: 12px;
    min-height: 20px;
}

QLabel.TagBlue {
    background-color: #e0e8f8;
    color: #1e3a8a;
    border-radius: 12px;
    padding: 4px 14px;
    font-weight: 700;
    font-size: 12px;
    min-height: 20px;
}

QLabel.TagPurple {
    background-color: #eee3fc;
    color: #6b21a8;
    border-radius: 12px;
    padding: 4px 14px;
    font-weight: 700;
    font-size: 12px;
    min-height: 20px;
}

QLabel.TagNavy {
    background-color: #0f3747;
    color: #ffffff;
    border-radius: 12px;
    padding: 4px 14px;
    font-weight: 700;
    font-size: 12px;
    min-height: 20px;
}

/* ========================================================= */
/* Card Containers & Quote Cards                             */
/* ========================================================= */
QFrame.CardFrame, QWidget.CardFrame {
    background-color: #ffffff;
    border: 1.5px solid #dce4ec;
    border-radius: 20px;
}

QFrame.HeroQuoteCard {
    background-color: #ffffff;
    border: 1.5px solid #dce4ec;
    border-radius: 20px;
    padding: 16px;
}

QLabel#heroQuoteText {
    font-size: 18px;
    font-weight: 800;
    color: #0f3747;
    line-height: 1.3;
}

QLabel#heroQuoteAuthor {
    font-size: 12.5px;
    color: #64748b;
    font-style: italic;
}

/* Two Large Choice Cards (Like "Hang out" and "Laugh" in Phone 1) */
QFrame.ChoiceCardPink {
    background-color: #fde2ec;
    border: 2px solid #fbcfe8;
    border-radius: 20px;
    padding: 16px;
}

QFrame.ChoiceCardNavy {
    background-color: #0f3747;
    border: 2px solid #0f3747;
    border-radius: 20px;
    padding: 16px;
}

QLabel.ChoiceTitleNavy {
    color: #ffffff;
    font-size: 16px;
    font-weight: 800;
}

QLabel.ChoiceSubNavy {
    color: #94a3b8;
    font-size: 12px;
}

QLabel.ChoiceTitlePink {
    color: #9f1239;
    font-size: 16px;
    font-weight: 800;
}

QLabel.ChoiceSubPink {
    color: #be123c;
    font-size: 12px;
}

/* ========================================================= */
/* Form Controls: LineEdit, ComboBox, SpinBox                */
/* ========================================================= */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #ffffff;
    color: #1e293b;
    border: 1.5px solid #cbd5e1;
    border-radius: 12px;
    padding: 6px 14px;
    min-height: 30px;
    selection-background-color: #0f3747;
    selection-color: #ffffff;
    font-size: 13px;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 2px solid #0f3747;
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
    selection-color: #166534;
    outline: none;
    padding: 4px;
    font-size: 13px;
}

/* ========================================================= */
/* Buttons                                                   */
/* ========================================================= */
QPushButton {
    background-color: #ffffff;
    color: #1e293b;
    border: 1.5px solid #dce4ec;
    border-radius: 12px;
    padding: 7px 16px;
    min-height: 30px;
    font-weight: 700;
    font-size: 12.5px;
}

QPushButton:hover {
    background-color: #f8fafc;
    border-color: #94a3b8;
}

QPushButton:pressed {
    background-color: #e2e8f0;
}

/* Primary Action Button (Deep Slate Teal / Navy) */
QPushButton.PrimaryButton {
    background-color: #0f3747;
    color: #ffffff;
    border: none;
    border-radius: 14px;
    padding: 9px 22px;
    min-height: 34px;
    font-weight: 800;
    font-size: 13.5px;
}

QPushButton.PrimaryButton:hover {
    background-color: #174b5f;
}

QPushButton.PrimaryButton:pressed {
    background-color: #092430;
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
    border-radius: 14px;
    padding: 9px 22px;
    min-height: 34px;
    font-weight: 800;
    font-size: 13.5px;
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
    border-radius: 14px;
    padding: 9px 22px;
    min-height: 34px;
    font-weight: 800;
    font-size: 13.5px;
}

QPushButton.DangerButton:hover {
    background-color: #be123c;
}

/* Navigation Buttons */
QPushButton.StepNavButton {
    background-color: #0f3747;
    color: #ffffff;
    border: none;
    border-radius: 14px;
    padding: 8px 20px;
    min-height: 32px;
    font-weight: 800;
    font-size: 13px;
}

QPushButton.StepNavButton:hover {
    background-color: #174b5f;
}

QPushButton.StepNavButton:disabled {
    background-color: #e2e8f0;
    color: #94a3b8;
    border: 1px solid #cbd5e1;
}

QPushButton.SecondaryNavButton {
    background-color: #ffffff;
    color: #0f3747;
    border: 1.5px solid #dce4ec;
    border-radius: 14px;
    padding: 8px 18px;
    min-height: 32px;
    font-weight: 800;
    font-size: 13px;
}

QPushButton.SecondaryNavButton:hover {
    background-color: #f1f5f9;
    border-color: #0f3747;
}

QPushButton.SecondaryNavButton:disabled {
    background-color: #f8fafc;
    color: #94a3b8;
    border: 1px solid #e2e8f0;
}

/* ========================================================= */
/* Bottom Navigation Footer & Step Indicator Pills           */
/* ========================================================= */
QFrame#bottomNavFooter {
    background-color: #ffffff;
    border: 1.5px solid #dce4ec;
    border-radius: 18px;
    padding: 6px 14px;
    min-height: 42px;
}

QPushButton.StepPill {
    background-color: #f8fafc;
    color: #64748b;
    border: 1.5px solid #dce4ec;
    border-radius: 12px;
    min-width: 42px;
    min-height: 28px;
    font-weight: 700;
    font-size: 12.5px;
    padding: 2px 10px;
}

QPushButton.StepPill:hover {
    background-color: #ffffff;
    border-color: #94a3b8;
    color: #0f3747;
}

QPushButton.StepPillActive {
    background-color: #0f3747;
    color: #ffffff;
    border: 1.5px solid #0f3747;
    border-radius: 12px;
    min-width: 42px;
    min-height: 28px;
    font-weight: 800;
    font-size: 12.5px;
    padding: 2px 10px;
}

/* ========================================================= */
/* Tables & Lists                                            */
/* ========================================================= */
QTableWidget, QTableView {
    background-color: #ffffff;
    color: #1e293b;
    gridline-color: #f1f5f9;
    border: 1.5px solid #dce4ec;
    border-radius: 16px;
    selection-background-color: #daf0dd;
    selection-color: #166534;
    font-size: 12.5px;
}

QHeaderView::section {
    background-color: #f8fafc;
    color: #334155;
    padding: 7px 12px;
    min-height: 28px;
    border: none;
    border-right: 1px solid #e2e8f0;
    border-bottom: 2px solid #cbd5e1;
    font-weight: 800;
    font-size: 12px;
}

/* ========================================================= */
/* Progress Bars                                             */
/* ========================================================= */
QProgressBar {
    background-color: #e2e8f0;
    border: 1.5px solid #cbd5e1;
    border-radius: 10px;
    text-align: center;
    color: #1e293b;
    font-weight: 800;
    font-size: 11.5px;
    min-height: 22px;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #166534, stop:0.5 #3b82f6, stop:1 #e11d48);
    border-radius: 8px;
}

/* ========================================================= */
/* Checkboxes & Radio Buttons                                */
/* ========================================================= */
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
    border-radius: 6px;
    background-color: #ffffff;
}

QRadioButton::indicator {
    border-radius: 10px;
}

QCheckBox::indicator:hover, QRadioButton::indicator:hover {
    border-color: #0f3747;
}

QCheckBox::indicator:checked {
    background-color: #0f3747;
    border-color: #0f3747;
}

QRadioButton::indicator:checked {
    background-color: #0f3747;
    border-color: #0f3747;
}

/* ========================================================= */
/* Text Edit / Console Log                                   */
/* ========================================================= */
QPlainTextEdit, QTextEdit {
    background-color: #0f172a;
    color: #4ade80;
    border: 1.5px solid #1e293b;
    border-radius: 14px;
    font-family: 'Cascadia Code', 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    padding: 8px;
}

/* ========================================================= */
/* Sliders                                                   */
/* ========================================================= */
QSlider::groove:horizontal {
    height: 7px;
    background: #e2e8f0;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0f3747, stop:1 #3b82f6);
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #0f3747;
    border: 2px solid #ffffff;
    width: 18px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 9px;
}

QSlider::handle:horizontal:hover {
    background: #174b5f;
    border-color: #ffffff;
}

/* ========================================================= */
/* GroupBox                                                  */
/* ========================================================= */
QGroupBox {
    border: 1.5px solid #dce4ec;
    border-radius: 16px;
    margin-top: 12px;
    padding-top: 14px;
    padding-bottom: 10px;
    padding-left: 12px;
    padding-right: 12px;
    font-weight: 800;
    color: #0f3747;
    background-color: #ffffff;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
}

/* ========================================================= */
/* Scrollbars                                                */
/* ========================================================= */
QScrollBar:vertical {
    background: #eef2f6;
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
    background: #eef2f6;
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
