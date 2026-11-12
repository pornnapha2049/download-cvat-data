"""
Theme for CVAT to YOLO Studio — "Console".

Clean, light, data-first look:
- Light grey canvas (#f7f8fa), white cards with thin borders and 10–12 px corners
- One indigo accent (#3b4ce0) for primary buttons, focus rings and the active step
- Left sidebar with the pipeline; tinted-tile icons (see ui/svg_icons.py)
- Semibold (600) type in Inter / Segoe UI

All colours live in PALETTE so they can be reused from Python (inline styles, icons, reports).
Key names are kept from the earlier themes: 'teal' = primary colour, 'pink' = highlight colour.
"""

import tempfile
from pathlib import Path

PALETTE = {
    "bg_top": "#f7f8fa",   # flat background
    "bg_bottom": "#f7f8fa",
    "card": "#ffffff",
    "card_border": "#e4e7ec",
    "soft": "#f9fafb",
    "line": "#d0d5dd",
    "ink": "#101828",
    "text": "#344054",
    "muted": "#5b6474",
    "faint": "#98a2b3",
    "teal": "#3b4ce0",   # primary (indigo accent)
    "teal_hover": "#3341c8",
    "teal_press": "#2a37b0",
    "teal_dark": "#101828",   # dark surfaces: console, test canvas, login brand panel
    "pink": "#eef0ff",   # highlight (light indigo)
    "pink_hover": "#e0e4ff",
    "pink_strong": "#a4b0ff",
    "pink_ink": "#2a37b0",
    "lilac": "#f2f4f7",
    "lilac_ink": "#344054",
    "lavender_bar": "#7a8bff",
    "sky": "#eff4ff",
    "sky_ink": "#1849a9",
    "lemon": "#fff6e5",
    "lemon_ink": "#b54708",
    "sage": "#ecfdf3",
    "sage_ink": "#067647",
    "rose": "#d92d20",
    "rose_hover": "#b42318",
    "selected_row": "#f2f4f7",
}


def _write_arrow_assets(color: str) -> str:
    """Qt stylesheets can't draw CSS triangles, so write tiny chevron SVGs to a temp folder."""
    folder = Path(tempfile.gettempdir()) / "cvat_yolo_studio_theme"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        chevrons = {
            "chevron_down.svg": "M 4 6 L 8 10 L 12 6",
            "chevron_up.svg": "M 4 10 L 8 6 L 12 10",
        }
        for name, d in chevrons.items():
            (folder / name).write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16">'
                f'<path d="{d}" stroke="{color}" stroke-width="1.8" fill="none" '
                'stroke-linecap="round" stroke-linejoin="round"/></svg>',
                encoding="utf-8",
            )
        (folder / "check_white.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16">'
            '<path d="M3.5 8.5 L6.5 11.5 L12.5 5" stroke="#ffffff" stroke-width="2" fill="none" '
            'stroke-linecap="round" stroke-linejoin="round"/></svg>',
            encoding="utf-8",
        )
    except OSError:
        pass
    return folder.as_posix()


FONT_STACK = "'Inter', 'Segoe UI', 'Leelawadee UI', 'Tahoma', sans-serif"
MONO_STACK = "'JetBrains Mono', 'Cascadia Code', 'Consolas', 'Courier New', monospace"


def _build_stylesheet(c: dict) -> str:
    assets = _write_arrow_assets(c['muted'])
    check = f"{assets}/check_white.svg"
    return f"""
/* ========================================================= */
/* Base window                                               */
/* ========================================================= */
QMainWindow {{
    background-color: {c['bg_top']};
}}
QWidget#centralWidget {{
    background-color: {c['bg_top']};
    color: {c['text']};
    font-family: {FONT_STACK};
    font-size: 13px;
}}
QWidget {{
    color: {c['text']};
    font-family: {FONT_STACK};
    font-size: 13px;
}}
QLabel {{
    background: transparent;
}}
QToolTip {{
    background-color: {c['ink']};
    color: #ffffff;
    border: none;
    border-radius: 6px;
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
/* Login — dark brand panel + form                           */
/* ========================================================= */
QFrame#loginBrandPanel {{
    background-color: {c['teal_dark']};
    border: none;
}}
QFrame#loginBrandPanel QLabel {{ color: #ffffff; }}
QLabel#loginBrandName {{
    font-size: 17px;
    font-weight: 700;
}}
QLabel#loginHero {{
    font-size: 32px;
    font-weight: 700;
}}
QFrame#loginBrandPanel QLabel.LoginStepText {{
    color: #d0d5dd;
    font-size: 14px;
}}
QFrame#loginBrandPanel QLabel.LoginStepNum {{
    color: #ffffff;
    border: 1px solid #475467;
    border-radius: 14px;
    min-width: 28px; max-width: 28px;
    min-height: 28px; max-height: 28px;
    font-size: 12px;
    font-weight: 600;
}}
QFrame#loginBrandPanel QLabel#loginVersion {{
    color: {c['faint']};
    font-size: 12px;
}}
QFrame.LoginCard {{
    background-color: transparent;
    border: none;
}}
QLabel#loginTitle {{
    font-size: 28px;
    font-weight: 700;
    color: {c['ink']};
}}
QLabel#loginSubtitle {{
    font-size: 14px;
    color: {c['muted']};
}}
QLabel.FieldLabel {{
    font-weight: 600;
    font-size: 13px;
    color: {c['ink']};
}}
QLineEdit.LoginInput {{
    background-color: #ffffff;
    color: {c['ink']};
    border: 1px solid {c['line']};
    border-radius: 8px;
    padding: 10px 12px;
    min-height: 22px;
    font-size: 14px;
}}
QLineEdit.LoginInput:focus {{
    border: 1px solid {c['teal']};
}}
QPushButton.LoginButton {{
    background-color: {c['teal']};
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-weight: 600;
    font-size: 15px;
    min-height: 24px;
}}
QPushButton.LoginButton:hover {{ background-color: {c['teal_hover']}; }}
QPushButton.LoginButton:pressed {{ background-color: {c['teal_press']}; }}

QPushButton.OfflineButton {{
    background-color: #ffffff;
    color: {c['ink']};
    border: 1px solid {c['line']};
    border-radius: 8px;
    padding: 12px 24px;
    font-weight: 600;
    font-size: 15px;
    min-height: 24px;
}}
QPushButton.OfflineButton:hover {{ background-color: {c['soft']}; }}

QLabel#orDivider {{
    color: {c['muted']};
    font-size: 13px;
}}
QFrame.DividerLine {{
    background-color: {c['card_border']};
    border: none;
    min-height: 1px; max-height: 1px;
}}

QPushButton#btnTogglePassword {{
    background-color: #ffffff;
    border: 1px solid {c['line']};
    border-radius: 8px;
    padding: 4px;
    min-width: 44px; max-width: 44px;
    min-height: 44px; max-height: 44px;
}}
QPushButton#btnTogglePassword:hover {{
    background-color: {c['soft']};
}}

QPushButton.SkipLoginButton {{
    background-color: transparent;
    color: {c['teal']};
    border: none;
    font-size: 13px;
    font-weight: 600;
    padding: 4px 0px;
}}
QPushButton.SkipLoginButton:hover {{ color: {c['teal_press']}; text-decoration: underline; }}

QPushButton.LogoutButton {{
    background-color: transparent;
    border: none;
    border-radius: 8px;
    min-width: 36px; max-width: 36px;
    min-height: 36px; max-height: 36px;
    padding: 0px;
}}
QPushButton.LogoutButton:hover {{
    background-color: #fef3f2;
}}

QLabel.AlertError {{
    background-color: #fef3f2;
    color: #b42318;
    border: 1px solid #fecdca;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 13px;
    font-weight: 600;
}}
QLabel.AlertSuccess {{
    background-color: {c['sage']};
    color: {c['sage_ink']};
    border: 1px solid #abefc6;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 13px;
    font-weight: 600;
}}

/* ========================================================= */
/* Sidebar                                                   */
/* ========================================================= */
QFrame#sidebar {{
    background-color: #ffffff;
    border: none;
    border-right: 1px solid {c['card_border']};
}}
QLabel#brandTitle {{
    font-size: 15px;
    font-weight: 700;
    color: {c['ink']};
}}
QLabel#brandSub {{
    font-size: 12px;
    color: {c['muted']};
}}
QLabel.SidebarSection {{
    font-size: 11px;
    font-weight: 600;
    color: {c['muted']};
    letter-spacing: 1px;
    padding-left: 8px;
}}
QPushButton#workspaceButton {{
    background-color: {c['soft']};
    border: 1px solid {c['card_border']};
    border-radius: 8px;
    padding: 8px 10px;
    min-height: 28px;
    text-align: left;
    color: {c['ink']};
    font-size: 12px;
    font-weight: 600;
}}
QPushButton#workspaceButton:hover {{ background-color: {c['lilac']}; }}

/* Pipeline rows (kept the StepButton / StepButtonActive class names) */
QPushButton.StepButton, QPushButton.StepButtonActive {{
    background-color: transparent;
    color: {c['ink']};
    border: none;
    border-radius: 8px;
    padding: 6px 8px;
    min-height: 40px;
    text-align: left;
    font-size: 13px;
    font-weight: 600;
}}
QPushButton.StepButton:hover {{ background-color: {c['soft']}; }}
QPushButton.StepButtonActive {{ background-color: {c['selected_row']}; }}

QFrame#userCard {{
    background: transparent;
    border: none;
    border-top: 1px solid {c['card_border']};
}}
QLabel#userAvatar {{
    background-color: {c['pink']};
    color: {c['teal']};
    border-radius: 16px;
    min-width: 32px; max-width: 32px;
    min-height: 32px; max-height: 32px;
    font-weight: 700;
    font-size: 13px;
}}
QLabel#userName {{
    font-size: 13px;
    font-weight: 600;
    color: {c['ink']};
}}
QLabel#userBadge {{
    color: {c['muted']};
    font-size: 12px;
}}

/* ========================================================= */
/* Page header                                               */
/* ========================================================= */
QLabel#pageCrumb {{
    font-size: 13px;
    color: {c['muted']};
}}
QLabel#pageTitle {{
    font-size: 26px;
    font-weight: 700;
    color: {c['ink']};
}}
QLabel#guideLabel {{
    font-size: 14px;
    color: {c['muted']};
}}
QLabel#appTitleLabel {{
    font-size: 21px;
    font-weight: 700;
    color: {c['ink']};
}}
QLabel#greetingTitle {{
    font-size: 24px;
    font-weight: 700;
    color: {c['ink']};
}}
QLabel#greetingSubtitle {{
    font-size: 14px;
    color: {c['muted']};
}}
QPushButton#circleNavButton, QLabel.CircleBadge {{
    background-color: #ffffff;
    border: 1px solid {c['line']};
    border-radius: 8px;
    min-width: 44px; min-height: 44px;
    max-width: 44px; max-height: 44px;
    padding: 0px;
}}
QPushButton#circleNavButton:hover {{ background-color: {c['soft']}; }}

/* Status pill (replaces the old mascot) */
QLabel#statusPill {{
    border-radius: 11px;
    padding: 3px 10px;
    font-size: 12px;
    font-weight: 600;
}}

QLabel.ArrowBadgePink, QLabel.ArrowBadgeNavy {{
    background-color: {c['pink']};
    color: {c['teal']};
    border-radius: 8px;
    min-width: 36px; min-height: 36px;
    max-width: 36px; max-height: 36px;
    font-size: 17px;
    font-weight: 700;
}}

/* Old stepper / guide containers (no longer shown, kept harmless) */
QFrame.StepperContainer {{ background: transparent; border: none; }}
QFrame.GuideBox {{ background: transparent; border: none; }}

/* ========================================================= */
/* Tags                                                      */
/* ========================================================= */
QLabel.TagPink, QLabel.TagYellow, QLabel.TagGreen, QLabel.TagBlue,
QLabel.TagPurple, QLabel.TagNavy, QLabel.TagTeal {{
    border-radius: 11px;
    padding: 3px 10px;
    font-weight: 600;
    font-size: 12px;
    min-height: 16px;
}}
QLabel.TagPink   {{ background-color: {c['pink']};   color: {c['pink_ink']}; }}
QLabel.TagYellow {{ background-color: {c['lemon']};  color: {c['lemon_ink']}; }}
QLabel.TagGreen  {{ background-color: {c['sage']};   color: {c['sage_ink']}; }}
QLabel.TagBlue   {{ background-color: {c['sky']};    color: {c['sky_ink']}; }}
QLabel.TagPurple {{ background-color: #f4f0ff;       color: #6941c6; }}
QLabel.TagNavy   {{ background-color: {c['teal']};   color: #ffffff; }}
QLabel.TagTeal   {{ background-color: {c['pink']};   color: {c['teal']}; }}

/* ========================================================= */
/* Cards                                                     */
/* ========================================================= */
QFrame.CardFrame, QWidget.CardFrame, QFrame.HeroQuoteCard {{
    background-color: {c['card']};
    border: 1px solid {c['card_border']};
    border-radius: 12px;
}}
QFrame.SoftCard {{
    background-color: {c['soft']};
    border: 1px solid #eef0f3;
    border-radius: 10px;
}}
QLabel#heroQuoteText {{
    font-size: 22px;
    font-weight: 600;
    color: {c['ink']};
}}
QLabel#heroQuoteAuthor {{
    font-size: 13px;
    color: {c['muted']};
}}
QLabel.SectionTitle {{
    font-size: 15px;
    font-weight: 600;
    color: {c['ink']};
}}
QLabel.MutedText {{
    color: {c['muted']};
}}
QLabel.StatText {{
    color: {c['ink']};
    font-weight: 600;
}}
QLabel.RunPreview {{
    color: {c['text']};
    font-family: {MONO_STACK};
    font-size: 12px;
    background: {c['soft']};
    border: 1px dashed {c['line']};
    padding: 8px 12px;
    border-radius: 8px;
}}

/* Choice cards — selectable option cards (format, source, …) */
QFrame.ChoiceCardPink {{
    background-color: #ffffff;
    border: 1px solid {c['line']};
    border-radius: 10px;
}}
QFrame.ChoiceCardPink:hover {{ background-color: {c['soft']}; }}
QFrame.ChoiceCardNavy {{
    background-color: #f5f6ff;
    border: 2px solid {c['teal']};
    border-radius: 10px;
}}
QFrame.ChoiceCardNavy QLabel, QFrame.ChoiceCardNavy QRadioButton,
QFrame.ChoiceCardPink QLabel, QFrame.ChoiceCardPink QRadioButton {{ color: {c['ink']}; }}
QFrame.ChoiceCardNavy QLabel.ChoiceSubNavy, QFrame.ChoiceCardNavy QLabel.ChoiceSubPink,
QFrame.ChoiceCardPink QLabel.ChoiceSubPink, QFrame.ChoiceCardPink QLabel.ChoiceSubNavy {{ color: {c['muted']}; }}

QLabel.ChoiceTitleNavy, QLabel.ChoiceTitlePink {{
    font-size: 15px;
    font-weight: 600;
}}
QLabel.ChoiceSubNavy, QLabel.ChoiceSubPink {{
    font-size: 13px;
}}
QFrame.ChoiceCardNavy QRadioButton, QFrame.ChoiceCardPink QRadioButton {{
    font-weight: 600;
    background: transparent;
}}

/* ========================================================= */
/* Inputs                                                    */
/* ========================================================= */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: #ffffff;
    color: {c['ink']};
    border: 1px solid {c['line']};
    border-radius: 8px;
    padding: 7px 12px;
    min-height: 24px;
    selection-background-color: {c['pink_strong']};
    selection-color: {c['ink']};
    font-size: 13px;
}}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {c['faint']};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {c['teal']};
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: {c['faint']};
    background-color: {c['soft']};
}}
QLineEdit:read-only {{
    background-color: {c['soft']};
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
    margin-right: 10px;
}}
QComboBox QAbstractItemView {{
    background-color: #ffffff;
    color: {c['ink']};
    border: 1px solid {c['card_border']};
    border-radius: 8px;
    selection-background-color: {c['selected_row']};
    selection-color: {c['ink']};
    outline: none;
    padding: 4px;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 22px;
    border: none;
    background: transparent;
    margin-top: 3px;
    margin-right: 6px;
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 22px;
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
/* Buttons                                                   */
/* ========================================================= */
QPushButton {{
    background-color: #ffffff;
    color: {c['ink']};
    border: 1px solid {c['line']};
    border-radius: 8px;
    padding: 8px 16px;
    min-height: 22px;
    font-weight: 600;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: {c['soft']};
}}
QPushButton:pressed {{ background-color: {c['lilac']}; }}
QPushButton:disabled {{
    color: {c['faint']};
    background-color: {c['soft']};
    border-color: {c['card_border']};
}}

QPushButton.PrimaryButton, QPushButton.StepNavButton {{
    background-color: {c['teal']};
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 9px 20px;
    min-height: 22px;
    font-weight: 600;
    font-size: 13px;
}}
QPushButton.PrimaryButton:hover, QPushButton.StepNavButton:hover {{ background-color: {c['teal_hover']}; }}
QPushButton.PrimaryButton:pressed, QPushButton.StepNavButton:pressed {{ background-color: {c['teal_press']}; }}
QPushButton.PrimaryButton:disabled, QPushButton.StepNavButton:disabled {{
    background-color: {c['card_border']};
    color: {c['faint']};
}}

/* Secondary action button */
QPushButton.SuccessButton {{
    background-color: #ffffff;
    color: {c['ink']};
    border: 1px solid {c['line']};
    border-radius: 8px;
    padding: 9px 20px;
    min-height: 22px;
    font-weight: 600;
    font-size: 13px;
}}
QPushButton.SuccessButton:hover {{ background-color: {c['soft']}; }}
QPushButton.SuccessButton:pressed {{ background-color: {c['lilac']}; }}
QPushButton.SuccessButton:disabled {{ background-color: {c['soft']}; color: {c['faint']}; }}

QPushButton.DangerButton {{
    background-color: {c['rose']};
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 9px 20px;
    min-height: 22px;
    font-weight: 600;
    font-size: 13px;
}}
QPushButton.DangerButton:hover {{ background-color: {c['rose_hover']}; }}
QPushButton.DangerButton:disabled {{ background-color: {c['card_border']}; color: {c['faint']}; }}

QPushButton.SecondaryNavButton {{
    background-color: #ffffff;
    color: {c['ink']};
    border: 1px solid {c['line']};
    border-radius: 8px;
    padding: 8px 16px;
    min-height: 22px;
    font-weight: 600;
    font-size: 13px;
}}
QPushButton.SecondaryNavButton:hover {{ background-color: {c['soft']}; }}
QPushButton.SecondaryNavButton:disabled {{
    background-color: transparent;
    color: {c['faint']};
    border-color: {c['card_border']};
}}

/* ========================================================= */
/* Footer — previous / next                                  */
/* ========================================================= */
QFrame#bottomNavFooter {{
    background-color: transparent;
    border: none;
    border-top: 1px solid {c['card_border']};
}}
QLabel#stepIndicator {{
    color: {c['muted']};
    font-size: 13px;
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
    border-radius: 10px;
    selection-background-color: {c['pink']};
    selection-color: {c['ink']};
    font-size: 13px;
    padding: 0px;
}}
QTableWidget::item, QTableView::item {{
    padding: 6px 10px;
    border-bottom: 1px solid #eef0f3;
}}
QHeaderView {{ background: transparent; }}
QHeaderView::section {{
    background-color: {c['soft']};
    color: {c['muted']};
    padding: 8px 10px;
    min-height: 22px;
    border: none;
    border-bottom: 1px solid {c['card_border']};
    font-weight: 600;
    font-size: 12px;
}}
QTableCornerButton::section {{ background: transparent; border: none; }}

/* ========================================================= */
/* Progress                                                  */
/* ========================================================= */
QProgressBar {{
    background-color: {c['lilac']};
    border: none;
    border-radius: 6px;
    text-align: center;
    color: {c['ink']};
    font-weight: 600;
    font-size: 12px;
    min-height: 20px;
    max-height: 20px;
}}
QProgressBar::chunk {{
    background: {c['teal']};
    border-radius: 6px;
}}

/* ========================================================= */
/* Checkboxes & radios                                       */
/* ========================================================= */
QCheckBox, QRadioButton {{
    color: {c['ink']};
    spacing: 8px;
    font-size: 13px;
    min-height: 22px;
    background: transparent;
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {c['line']};
    border-radius: 4px;
    background-color: #ffffff;
}}
QRadioButton::indicator {{ border-radius: 9px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {c['teal']}; }}
QCheckBox::indicator:checked {{
    background-color: {c['teal']};
    border: 1px solid {c['teal']};
    image: url({check});
}}
QRadioButton::indicator:checked {{
    width: 8px;
    height: 8px;
    background-color: #ffffff;
    border: 5px solid {c['teal']};
    border-radius: 9px;
}}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    background-color: {c['lilac']};
    border-color: {c['card_border']};
}}

/* ========================================================= */
/* Console / log                                             */
/* ========================================================= */
QPlainTextEdit, QTextEdit {{
    background-color: {c['teal_dark']};
    color: #d0d5dd;
    border: none;
    border-radius: 10px;
    font-family: {MONO_STACK};
    font-size: 12px;
    padding: 10px;
    selection-background-color: {c['pink_strong']};
    selection-color: {c['ink']};
}}

/* Test canvas */
QFrame#testCanvas {{
    background-color: #eef0f3;
    border: none;
    border-radius: 10px;
}}
QLabel#testPreview {{
    color: {c['muted']};
    font-weight: 600;
}}

/* ========================================================= */
/* Sliders                                                   */
/* ========================================================= */
QSlider::groove:horizontal {{
    height: 6px;
    background: {c['card_border']};
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: {c['teal']};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: #ffffff;
    border: 2px solid {c['teal']};
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 9px;
}}
QSlider::handle:horizontal:hover {{ background: {c['pink']}; }}

/* ========================================================= */
/* GroupBox — renders like a card with a plain title         */
/* ========================================================= */
QGroupBox {{
    border: 1px solid {c['card_border']};
    border-radius: 12px;
    margin-top: 14px;
    padding: 22px 16px 14px 16px;
    font-weight: 600;
    font-size: 14px;
    color: {c['ink']};
    background-color: #ffffff;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 16px;
    top: 4px;
    padding: 0px 6px;
    background-color: #ffffff;
    color: {c['ink']};
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
    background: {c['line']};
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
    background: {c['line']};
    min-width: 30px;
    border-radius: 3px;
}}
QScrollBar::handle:horizontal:hover {{ background: {c['faint']}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0px; }}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: transparent; }}
"""


AESTHETIC_STYLESHEET = _build_stylesheet(PALETTE)
DARK_STYLESHEET = AESTHETIC_STYLESHEET  # backwards-compatible name