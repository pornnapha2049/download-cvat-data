"""
Theme for CVAT to YOLO Studio — "Navy Enterprise".

Formal, data-first look:
- Flat cool-grey background (#eef1f5), white cards with thin borders and small corner radii
- Navy primary (#1f3a5f) for primary buttons, the active step and the selected step box
- Step tabs underlined like an enterprise app; Tip box with a navy left bar
- Neutral, muted tags; single-colour navy progress bar; semibold (600) type in Segoe UI

All colours live in PALETTE so they can be reused from Python (inline styles, icons, mascot).
Key names are kept from the earlier theme: 'teal' = primary colour, 'pink' = highlight colour.
"""

import tempfile
from pathlib import Path

PALETTE = {
    "bg_top": "#eef1f5",   # flat background
    "bg_bottom": "#eef1f5",
    "card": "#ffffff",
    "card_border": "#dde3ea",
    "soft": "#f3f5f8",
    "line": "#cfd7e0",
    "ink": "#152238",
    "text": "#27354b",
    "muted": "#5f6c80",
    "faint": "#97a2b2",
    "teal": "#1f3a5f",   # primary (navy)
    "teal_hover": "#2a4a76",
    "teal_press": "#172d4a",
    "teal_dark": "#14233a",
    "pink": "#e5ebf3",   # highlight (light navy)
    "pink_hover": "#d8e1ed",
    "pink_strong": "#8fa8c8",
    "pink_ink": "#1f3a5f",
    "lilac": "#e9ecf2",
    "lilac_ink": "#3d4a5c",
    "lavender_bar": "#6f8fb8",
    "sky": "#e6eef3",
    "sky_ink": "#2b4a5a",
    "lemon": "#f3f0dc",
    "lemon_ink": "#5c5530",
    "sage": "#e3efe6",
    "sage_ink": "#2f5e3d",
    "rose": "#c8414f",
    "rose_hover": "#b13542",
}


def _write_arrow_assets(color: str) -> str:
    """Qt stylesheets can't draw CSS triangles, so write tiny chevron SVGs to a temp folder."""
    folder = Path(tempfile.gettempdir()) / "cvat_yolo_studio_theme"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        chevrons = {
            "chevron_down.svg": "M 3 6 L 8 11 L 13 6",
            "chevron_up.svg": "M 3 10 L 8 5 L 13 10",
        }
        for name, d in chevrons.items():
            (folder / name).write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16">'
                f'<path d="{d}" stroke="{color}" stroke-width="2.2" fill="none" '
                'stroke-linecap="round" stroke-linejoin="round"/></svg>',
                encoding="utf-8",
            )
    except OSError:
        pass
    return folder.as_posix()


FONT_STACK = "'Segoe UI', 'Leelawadee UI', 'Tahoma', sans-serif"


def _build_stylesheet(c: dict) -> str:
    assets = _write_arrow_assets(c['teal'])
    return f"""
/* ========================================================= */
/* Base window — flat cool grey   */
/* ========================================================= */
QMainWindow {{
    background-color: {c['bg_top']};
}}
QWidget#centralWidget {{
    background: qlineargradient(x1:0, y1:0, x2:0.35, y2:1, stop:0 {c['bg_top']}, stop:1 {c['bg_bottom']});
    color: {c['text']};
    font-family: {FONT_STACK};
    font-size: 13px;
}}
QWidget {{
    color: {c['text']};
    font-family: {FONT_STACK};
}}
QLabel {{
    background: transparent;
}}
QToolTip {{
    background-color: {c['teal']};
    color: #ffffff;
    border: none;
    border-radius: 3px;
    padding: 6px 10px;
}}

QScrollArea, QScrollArea > QWidget > QWidget, QStackedWidget {{
    background: transparent;
    border: none;
}}
QSplitter::handle {{
    background: transparent;
    width: 10px;
}}

/* ========================================================= */
/* Login page                                                */
/* ========================================================= */
QFrame.LoginCard {{
    background-color: {c['card']};
    border: 1px solid {c['card_border']};
    border-radius: 7px;
}}
QLabel#loginTitle {{
    font-size: 26px;
    font-weight: 600;
    color: {c['ink']};
}}
QLabel#loginSubtitle {{
    font-size: 13px;
    color: {c['muted']};
}}
QLabel.FieldLabel {{
    font-weight: 700;
    font-size: 12.5px;
    color: {c['ink']};
    padding-left: 6px;
}}
QLineEdit.LoginInput {{
    background-color: {c['soft']};
    color: {c['ink']};
    border: 1.5px solid {c['soft']};
    border-radius: 5px;
    padding: 10px 18px;
    min-height: 22px;
    font-size: 13px;
}}
QLineEdit.LoginInput:focus {{
    border: 1.5px solid {c['teal']};
    background-color: #ffffff;
}}
QPushButton.LoginButton {{
    background-color: {c['teal']};
    color: #ffffff;
    border: none;
    border-radius: 5px;
    padding: 12px 24px;
    font-weight: 600;
    font-size: 14px;
    min-height: 24px;
}}
QPushButton.LoginButton:hover {{ background-color: {c['teal_hover']}; }}
QPushButton.LoginButton:pressed {{ background-color: {c['teal_press']}; }}

QPushButton#btnTogglePassword {{
    background-color: {c['soft']};
    border: 1.5px solid {c['soft']};
    border-radius: 5px;
    padding: 4px;
    min-width: 42px; max-width: 42px;
    min-height: 42px; max-height: 42px;
}}
QPushButton#btnTogglePassword:hover {{
    background-color: {c['pink']};
    border-color: {c['pink']};
}}

QPushButton.SkipLoginButton {{
    background-color: transparent;
    color: {c['muted']};
    border: none;
    font-size: 12px;
    text-decoration: underline;
    font-weight: 600;
}}
QPushButton.SkipLoginButton:hover {{ color: {c['teal']}; }}

QPushButton.LogoutButton {{
    background-color: {c['pink']};
    color: {c['pink_ink']};
    border: none;
    border-radius: 4px;
    padding: 6px 16px;
    min-height: 24px;
    font-weight: 700;
    font-size: 12px;
}}
QPushButton.LogoutButton:hover {{
    background-color: {c['rose']};
    color: #ffffff;
}}

QLabel.AlertError {{
    background-color: #fde8ef;
    color: #9b1c47;
    border: none;
    border-radius: 4px;
    padding: 10px 16px;
    font-size: 12px;
    font-weight: 600;
}}
QLabel.AlertSuccess {{
    background-color: {c['sage']};
    color: {c['sage_ink']};
    border: none;
    border-radius: 4px;
    padding: 8px 14px;
    font-size: 12px;
    font-weight: 600;
}}

/* ========================================================= */
/* Header / greeting                                         */
/* ========================================================= */
QLabel#appTitleLabel {{
    font-size: 21px;
    font-weight: 600;
    color: {c['ink']};
}}
QLabel#greetingTitle {{
    font-size: 24px;
    font-weight: 600;
    color: {c['ink']};
}}
QLabel#greetingSubtitle {{
    font-size: 14px;
    font-weight: 500;
    color: {c['muted']};
}}
QLabel#userBadge {{
    color: {c['muted']};
    font-weight: 600;
    font-size: 12px;
}}
QPushButton#circleNavButton, QLabel.CircleBadge {{
    background-color: #ffffff;
    border: 1px solid {c['card_border']};
    border-radius: 5px;
    min-width: 44px; min-height: 44px;
    max-width: 44px; max-height: 44px;
    padding: 0px;
}}
QPushButton#circleNavButton:hover {{ background-color: {c['soft']}; }}

/* Round arrow badge (→ in the corner of choice cards) */
QLabel.ArrowBadgePink {{
    background-color: rgba(255, 255, 255, 0.55);
    color: {c['ink']};
    border-radius: 4px;
    min-width: 36px; min-height: 36px;
    max-width: 36px; max-height: 36px;
    font-size: 17px;
    font-weight: 700;
}}
QLabel.ArrowBadgeNavy {{
    background-color: rgba(255, 255, 255, 0.16);
    color: #ffffff;
    border-radius: 4px;
    min-width: 36px; min-height: 36px;
    max-width: 36px; max-height: 36px;
    font-size: 17px;
    font-weight: 700;
}}

/* ========================================================= */
/* Stepper buttons (restyled as underline tabs at the end)  */
/* ========================================================= */
QFrame.StepperContainer {{
    background-color: transparent;
    border: none;
}}
QPushButton.StepButton {{
    background-color: rgba(255, 255, 255, 0.55);
    color: {c['text']};
    border: 1.5px solid #ffffff;
    border-radius: 5px;
    padding: 7px 12px;
    min-height: 30px;
    font-size: 12px;
    font-weight: 600;
    text-align: center;
}}
QPushButton.StepButton:hover {{
    background-color: #ffffff;
    color: {c['ink']};
}}
QPushButton.StepButtonActive {{
    background-color: {c['pink']};
    color: {c['ink']};
    border: 1.5px solid {c['pink']};
    border-radius: 5px;
    padding: 7px 12px;
    min-height: 30px;
    font-size: 12px;
    font-weight: 600;
    text-align: center;
}}

/* Guide box (navy left bar override at the end) */
QFrame.GuideBox {{
    background-color: {c['pink']};
    border: none;
    border-radius: 5px;
}}
QLabel#guideLabel {{
    color: {c['ink']};
    font-weight: 600;
    font-size: 12.5px;
}}

/* ========================================================= */
/* Capsule tags                                              */
/* ========================================================= */
QLabel.TagPink, QLabel.TagYellow, QLabel.TagGreen, QLabel.TagBlue,
QLabel.TagPurple, QLabel.TagNavy, QLabel.TagTeal {{
    border-radius: 3px;
    padding: 5px 16px;
    font-weight: 700;
    font-size: 12px;
    min-height: 18px;
}}
QLabel.TagPink   {{ background-color: {c['pink']};   color: {c['pink_ink']}; }}
QLabel.TagYellow {{ background-color: {c['lemon']};  color: {c['lemon_ink']}; }}
QLabel.TagGreen  {{ background-color: {c['sage']};   color: {c['sage_ink']}; }}
QLabel.TagBlue   {{ background-color: {c['sky']};    color: {c['sky_ink']}; }}
QLabel.TagPurple {{ background-color: {c['lilac']};  color: {c['lilac_ink']}; }}
QLabel.TagNavy   {{ background-color: {c['teal']};   color: #ffffff; }}
QLabel.TagTeal   {{ background-color: {c['sky']};    color: {c['teal']}; }}

/* ========================================================= */
/* Cards                                                     */
/* ========================================================= */
QFrame.CardFrame, QWidget.CardFrame, QFrame.HeroQuoteCard {{
    background-color: {c['card']};
    border: 1px solid {c['card_border']};
    border-radius: 6px;
}}
QFrame.SoftCard {{
    background-color: {c['soft']};
    border: none;
    border-radius: 4px;
}}
QLabel#heroQuoteText {{
    font-size: 22px;
    font-weight: 600;
    color: {c['ink']};
}}
QLabel#heroQuoteAuthor {{
    font-size: 12.5px;
    color: {c['muted']};
}}
QLabel.SectionTitle {{
    font-size: 15px;
    font-weight: 600;
    color: {c['ink']};
}}
QLabel.MutedText {{
    color: {c['muted']};
    font-weight: 600;
}}
QLabel.StatText {{
    color: {c['ink']};
    font-weight: 600;
}}
QLabel.RunPreview {{
    color: {c['teal']};
    font-weight: 700;
    background: {c['sky']};
    padding: 7px 14px;
    border-radius: 3px;
}}

/* Choice cards — light (ChoiceCardPink) & navy (ChoiceCardNavy) */
QFrame.ChoiceCardPink {{
    background-color: {c['pink']};
    border: none;
    border-radius: 6px;
}}
QFrame.ChoiceCardPink:hover {{ background-color: {c['pink_hover']}; }}
QFrame.ChoiceCardNavy {{
    background-color: {c['teal']};
    border: none;
    border-radius: 6px;
}}
QFrame.ChoiceCardNavy:hover {{ background-color: {c['teal_hover']}; }}

/* Text inside choice cards follows the card, so swapping classes recolours everything */
QFrame.ChoiceCardNavy QLabel, QFrame.ChoiceCardNavy QRadioButton {{ color: #ffffff; }}
QFrame.ChoiceCardNavy QLabel.ChoiceSubNavy, QFrame.ChoiceCardNavy QLabel.ChoiceSubPink {{ color: #b8c7db; }}
QFrame.ChoiceCardPink QLabel, QFrame.ChoiceCardPink QRadioButton {{ color: {c['ink']}; }}
QFrame.ChoiceCardPink QLabel.ChoiceSubPink, QFrame.ChoiceCardPink QLabel.ChoiceSubNavy {{ color: {c['pink_ink']}; }}

QLabel.ChoiceTitleNavy, QLabel.ChoiceTitlePink {{
    font-size: 15px;
    font-weight: 600;
}}
QLabel.ChoiceSubNavy, QLabel.ChoiceSubPink {{
    font-size: 12px;
}}
QFrame.ChoiceCardNavy QRadioButton, QFrame.ChoiceCardPink QRadioButton {{
    font-weight: 700;
    background: transparent;
}}

/* ========================================================= */
/* Inputs        */
/* ========================================================= */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: {c['soft']};
    color: {c['ink']};
    border: 1.5px solid {c['soft']};
    border-radius: 4px;
    padding: 6px 16px;
    min-height: 24px;
    selection-background-color: {c['pink_strong']};
    selection-color: {c['ink']};
    font-size: 13px;
}}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {c['line']};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1.5px solid {c['teal']};
    background-color: #ffffff;
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: {c['faint']};
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 28px;
    border: none;
}}
QComboBox::down-arrow {{
    image: url({assets}/chevron_down.svg);
    width: 12px;
    height: 12px;
    margin-right: 12px;
}}
QComboBox QAbstractItemView {{
    background-color: #ffffff;
    color: {c['ink']};
    border: 1px solid {c['card_border']};
    border-radius: 3px;
    selection-background-color: {c['pink']};
    selection-color: {c['ink']};
    outline: none;
    padding: 6px;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 24px;
    border: none;
    background: transparent;
    margin-top: 3px;
    margin-right: 6px;
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 24px;
    border: none;
    background: transparent;
    margin-bottom: 3px;
    margin-right: 6px;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url({assets}/chevron_up.svg);
    width: 10px;
    height: 10px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url({assets}/chevron_down.svg);
    width: 10px;
    height: 10px;
}}

/* ========================================================= */
/* Buttons                                           */
/* ========================================================= */
QPushButton {{
    background-color: #ffffff;
    color: {c['ink']};
    border: 1.5px solid {c['line']};
    border-radius: 4px;
    padding: 7px 18px;
    min-height: 22px;
    font-weight: 700;
    font-size: 12.5px;
}}
QPushButton:hover {{
    background-color: {c['soft']};
    border-color: {c['faint']};
}}
QPushButton:pressed {{ background-color: {c['line']}; }}
QPushButton:disabled {{
    color: {c['faint']};
    background-color: {c['soft']};
    border-color: {c['soft']};
}}

QPushButton.PrimaryButton, QPushButton.StepNavButton {{
    background-color: {c['teal']};
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 9px 24px;
    min-height: 22px;
    font-weight: 600;
    font-size: 13px;
}}
QPushButton.PrimaryButton:hover, QPushButton.StepNavButton:hover {{ background-color: {c['teal_hover']}; }}
QPushButton.PrimaryButton:pressed, QPushButton.StepNavButton:pressed {{ background-color: {c['teal_press']}; }}
QPushButton.PrimaryButton:disabled, QPushButton.StepNavButton:disabled {{
    background-color: {c['line']};
    color: {c['faint']};
}}

/* Secondary action button (muted) */
QPushButton.SuccessButton {{
    background-color: {c['lilac']};
    color: {c['lilac_ink']};
    border: none;
    border-radius: 4px;
    padding: 9px 24px;
    min-height: 22px;
    font-weight: 600;
    font-size: 13px;
}}
QPushButton.SuccessButton:hover {{ background-color: #cdc0f1; }}
QPushButton.SuccessButton:pressed {{ background-color: #bfb0ec; }}
QPushButton.SuccessButton:disabled {{ background-color: {c['line']}; color: {c['faint']}; }}

QPushButton.DangerButton {{
    background-color: {c['rose']};
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 9px 24px;
    min-height: 22px;
    font-weight: 600;
    font-size: 13px;
}}
QPushButton.DangerButton:hover {{ background-color: {c['rose_hover']}; }}
QPushButton.DangerButton:disabled {{ background-color: {c['line']}; color: {c['faint']}; }}

QPushButton.SecondaryNavButton {{
    background-color: #ffffff;
    color: {c['ink']};
    border: 1.5px solid {c['line']};
    border-radius: 4px;
    padding: 8px 20px;
    min-height: 22px;
    font-weight: 700;
    font-size: 12.5px;
}}
QPushButton.SecondaryNavButton:hover {{
    background-color: {c['pink']};
    border-color: {c['pink']};
}}
QPushButton.SecondaryNavButton:disabled {{
    background-color: transparent;
    color: {c['faint']};
    border-color: {c['line']};
}}

/* ========================================================= */
/* Footer — numbered step boxes        */
/* ========================================================= */
QFrame#bottomNavFooter {{
    background-color: #ffffff;
    border: 1px solid {c['card_border']};
    border-radius: 6px;
}}
QPushButton.StepPill {{
    background-color: #ffffff;
    color: {c['ink']};
    border: 1.5px solid {c['line']};
    border-radius: 4px;
    min-width: 38px; max-width: 38px;
    min-height: 38px; max-height: 38px;
    padding: 0px;
    font-weight: 600;
    font-size: 13px;
}}
QPushButton.StepPill:hover {{
    background-color: {c['pink']};
    border-color: {c['pink']};
}}
QPushButton.StepPillActive {{
    background-color: {c['teal']};
    color: #ffffff;
    border: 1.5px solid {c['teal']};
    border-radius: 4px;
    min-width: 38px; max-width: 38px;
    min-height: 38px; max-height: 38px;
    padding: 0px;
    font-weight: 600;
    font-size: 13px;
}}
QLabel#stepIndicator {{
    font-weight: 700;
    color: {c['muted']};
    font-size: 12px;
}}

/* ========================================================= */
/* Tables                                                    */
/* ========================================================= */
QTableWidget, QTableView {{
    background-color: #ffffff;
    alternate-background-color: {c['soft']};
    color: {c['ink']};
    gridline-color: transparent;
    border: 1px solid {c['card_border']};
    border-radius: 4px;
    selection-background-color: {c['pink']};
    selection-color: {c['ink']};
    font-size: 12.5px;
    padding: 4px;
}}
QTableWidget::item, QTableView::item {{
    padding: 4px 8px;
    border-bottom: 1px solid {c['soft']};
}}
QHeaderView {{ background: transparent; }}
QHeaderView::section {{
    background-color: {c['soft']};
    color: {c['muted']};
    padding: 8px 12px;
    min-height: 22px;
    border: none;
    font-weight: 700;
    font-size: 12px;
}}
QHeaderView::section:first {{ border-top-left-radius: 3px; border-bottom-left-radius: 3px; }}
QHeaderView::section:last  {{ border-top-right-radius: 3px; border-bottom-right-radius: 3px; }}
QTableCornerButton::section {{ background: transparent; border: none; }}

/* ========================================================= */
/* Progress — single navy bar    */
/* ========================================================= */
QProgressBar {{
    background-color: {c['soft']};
    border: none;
    border-radius: 3px;
    text-align: center;
    color: {c['ink']};
    font-weight: 700;
    font-size: 11.5px;
    min-height: 22px;
    max-height: 22px;
}}
QProgressBar::chunk {{
    background: {c['teal']};
    border-radius: 3px;
}}

/* ========================================================= */
/* Checkboxes & radios                                       */
/* ========================================================= */
QCheckBox, QRadioButton {{
    color: {c['ink']};
    spacing: 8px;
    font-size: 13px;
    font-weight: 600;
    min-height: 22px;
    background: transparent;
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 18px;
    height: 18px;
    border: 1.5px solid {c['faint']};
    border-radius: 3px;
    background-color: #ffffff;
}}
QRadioButton::indicator {{ border-radius: 10px; }}   /* radios stay round (18px + border) */
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {c['teal']}; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {c['teal']};
    border: 4px solid {c['pink']};
}}
QRadioButton::indicator:checked {{
    background-color: {c['teal']};
    border: 2px solid {c['teal']};
    border-radius: 11px;
}}
QFrame.ChoiceCardNavy QRadioButton::indicator {{ border-color: #ffffff; background: transparent; }}
QFrame.ChoiceCardNavy QRadioButton::indicator:checked {{ background: #ffffff; border: 2px solid #ffffff; border-radius: 11px; }}

/* ========================================================= */
/* Console / log                                             */
/* ========================================================= */
QPlainTextEdit, QTextEdit {{
    background-color: {c['teal_dark']};
    color: #d6e2f0;
    border: none;
    border-radius: 4px;
    font-family: 'Cascadia Code', 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    padding: 10px;
    selection-background-color: {c['pink_strong']};
    selection-color: {c['ink']};
}}

/* Test canvas */
QFrame#testCanvas {{
    background-color: {c['teal_dark']};
    border: none;
    border-radius: 5px;
}}
QLabel#testPreview {{
    color: #8fa1b8;
    font-weight: 600;
}}

/* ========================================================= */
/* Sliders                                                   */
/* ========================================================= */
QSlider::groove:horizontal {{
    height: 8px;
    background: {c['soft']};
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {c['pink_strong']}, stop:1 {c['lavender_bar']});
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {c['teal']};
    border: 3px solid #ffffff;
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 10px;   /* round handle */
}}
QSlider::handle:horizontal:hover {{ background: {c['teal_hover']}; }}

/* ========================================================= */
/* GroupBox                                                  */
/* ========================================================= */
QGroupBox {{
    border: 1px solid {c['card_border']};
    border-radius: 5px;
    margin-top: 16px;
    padding: 18px 12px 10px 12px;
    font-weight: 700;
    color: {c['ink']};
    background-color: #ffffff;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 18px;
    padding: 3px 12px;
    background-color: {c['lilac']};
    color: {c['lilac_ink']};
    border-radius: 3px;
}}

/* ========================================================= */
/* Message boxes                                             */
/* ========================================================= */
QMessageBox {{ background-color: #ffffff; }}
QMessageBox QLabel {{ color: {c['ink']}; font-size: 13px; }}
QMessageBox QPushButton {{ min-width: 80px; }}

/* ========================================================= */
/* Scrollbars — thin & round                                 */
/* ========================================================= */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 4px 2px;
}}
QScrollBar::handle:vertical {{
    background: #c3ccd8;
    min-height: 30px;
    border-radius: 3px;
}}
QScrollBar::handle:vertical:hover {{ background: {c['faint']}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px 4px;
}}
QScrollBar::handle:horizontal {{
    background: #c3ccd8;
    min-width: 30px;
    border-radius: 3px;
}}
QScrollBar::handle:horizontal:hover {{ background: {c['faint']}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0px; }}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: transparent; }}
/* ========================================================= */
/* Enterprise overrides — underline step tabs, Tip left bar   */
/* ========================================================= */
QFrame.StepperContainer {{
    background: {c['card']};
    border: 1px solid {c['card_border']};
    border-radius: 4px;
}}
QPushButton.StepButton {{
    background: transparent;
    border: none;
    border-bottom: 3px solid transparent;
    border-radius: 0px;
    color: {c['muted']};
}}
QPushButton.StepButton:hover {{
    background: {c['pink']};
    color: {c['ink']};
}}
QPushButton.StepButtonActive {{
    background: {c['card']};
    border: none;
    border-bottom: 3px solid {c['teal']};
    border-radius: 0px;
    color: {c['teal']};
}}
QFrame.GuideBox {{
    background: {c['card']};
    border: 1px solid {c['card_border']};
    border-left: 4px solid {c['teal']};
}}
"""


AESTHETIC_STYLESHEET = _build_stylesheet(PALETTE)
DARK_STYLESHEET = AESTHETIC_STYLESHEET  # backwards-compatible name