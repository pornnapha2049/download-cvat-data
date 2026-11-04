"""
Notification center behind the bell button.

- NotificationCenter.add(level, title, message, action) records an event
  (download / merge / convert / split / training / test finished or failed).
- Unread count is shown as a red badge on every attached bell button.
- Clicking a bell opens a popup list; clicking an item runs its action
  (open the training / accuracy report, a report.md file or a folder).
- The last 50 notifications are kept between app restarts (QSettings).
- When the window is not in front, a Windows tray balloon is shown as well,
  so a long training run can be left in the background.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QWidget, QSystemTrayIcon,
    QApplication, QAbstractButton,
)
from PyQt5.QtCore import Qt, QObject, QPoint, QSettings, QEvent
from PyQt5.QtGui import QIcon

from ui.theme import PALETTE
from ui import svg_icons
from ui.svg_icons import get_svg_pixmap, get_svg_icon

# Bell without the built-in red dot — the unread badge shows the real state instead.
if "bell_plain" not in svg_icons.ICONS:
    _hue, _els = svg_icons.ICONS["bell"]
    svg_icons.ICONS["bell_plain"] = (_hue, [e for e in _els if e[1] != "accent"])
from ui.dancing_svg import get_cat_pixmap

MAX_ITEMS = 50
LEVEL_ICON = {"success": "check", "error": "stop", "warning": "warning", "info": "bell_plain"}
LEVEL_COLOR = {"success": PALETTE["sage_ink"], "error": PALETTE["rose"], "warning": "#8a5a08", "info": PALETTE["teal"]}


def _ago(ts: float) -> str:
    d = max(0, int(time.time() - ts))
    if d < 60:
        return "just now"
    if d < 3600:
        return f"{d // 60} min ago"
    if d < 86400:
        return f"{d // 3600} h ago"
    return datetime.fromtimestamp(ts).strftime("%d %b %H:%M")


class _Badge(QLabel):
    """Small red unread counter drawn on top of a bell button."""

    def __init__(self, button: QAbstractButton):
        super().__init__(button)
        self.button = button
        self.setAlignment(Qt.AlignCenter)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setStyleSheet(f"background: {PALETTE['rose']}; color: #ffffff; border: 2px solid #ffffff;"
                           f" border-radius: 9px; font-size: 10px; font-weight: 700; padding: 0 4px;")
        self.setFixedHeight(18)
        self.hide()
        button.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if obj is self.button and ev.type() in (QEvent.Resize, QEvent.Show):
            self._place()
        return False

    def set_count(self, n: int):
        if n <= 0:
            self.hide()
            return
        self.setText(str(n) if n < 100 else "99+")
        self.setMinimumWidth(18)
        self.adjustSize()
        self._place()
        self.show()
        self.raise_()

    def _place(self):
        self.move(self.button.width() - self.width() + 2, -2)


class _Row(QFrame):
    def __init__(self, item: dict, on_click, parent=None):
        super().__init__(parent)
        self.item = item
        unread = not item.get("read")
        self.setCursor(Qt.PointingHandCursor if item.get("action") else Qt.ArrowCursor)
        self.setObjectName("notifRow")
        self.setStyleSheet(
            f"QFrame#notifRow {{ background: {PALETTE['pink'] if unread else PALETTE['card']}; border: none;"
            f" border-bottom: 1px solid {PALETTE['card_border']}; }}"
            f"QFrame#notifRow:hover {{ background: {PALETTE['soft']}; }}"
            "QLabel { background: transparent; border: none; }")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(10)
        ic = QLabel()
        ic.setPixmap(get_svg_pixmap(LEVEL_ICON.get(item.get("level"), "bell"), 26, 26))
        ic.setAlignment(Qt.AlignTop)
        lay.addWidget(ic, 0, Qt.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(2)
        t = QLabel(item.get("title", ""))
        t.setStyleSheet(f"color: {LEVEL_COLOR.get(item.get('level'), PALETTE['ink'])}; font-weight: 700; font-size: 12.5px;")
        m = QLabel(item.get("message", ""))
        m.setWordWrap(True)
        m.setStyleSheet(f"color: {PALETTE['text']}; font-size: 12px;")
        when = _ago(item.get("ts", time.time()))
        hint = "  ·  click to open" if item.get("action") else ""
        w = QLabel(when + hint)
        w.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 10.5px;")
        col.addWidget(t)
        col.addWidget(m)
        col.addWidget(w)
        lay.addLayout(col, 1)
        if unread:
            dot = QLabel()
            dot.setFixedSize(8, 8)
            dot.setStyleSheet(f"background: {PALETTE['rose']}; border-radius: 4px;")
            lay.addWidget(dot, 0, Qt.AlignTop)
        self._on_click = on_click

    def mousePressEvent(self, e):
        self._on_click(self.item)
        super().mousePressEvent(e)


class NotificationPopup(QFrame):
    def __init__(self, center: "NotificationCenter"):
        super().__init__(None, Qt.Popup | Qt.FramelessWindowHint)
        self.center = center
        self.setObjectName("notifPopup")
        self.setFixedWidth(400)
        # top-level popup: inherit the app stylesheet explicitly
        self.setStyleSheet(center.window.styleSheet() +
                           f"\nQFrame#notifPopup {{ background: {PALETTE['card']}; border: 1px solid {PALETTE['line']};"
                           f" border-radius: 6px; }}")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        head = QHBoxLayout()
        head.setContentsMargins(14, 10, 10, 10)
        title = QLabel("Notifications")
        title.setStyleSheet(f"font-size: 15px; font-weight: 700; color: {PALETTE['ink']}; border: none;")
        head.addWidget(title)
        head.addStretch()
        for text, fn in (("Mark all read", center.mark_all_read), ("Clear", center.clear)):
            b = QPushButton(text)
            b.setProperty("class", "SkipLoginButton")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _, f=fn: (f(), self.rebuild()))
            head.addWidget(b)
        hw = QWidget()
        hw.setObjectName("notifHead")
        hw.setLayout(head)
        hw.setStyleSheet(f"QWidget#notifHead {{ border-bottom: 1px solid {PALETTE['card_border']}; }}")
        root.addWidget(hw)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        root.addWidget(self.scroll)
        self.rebuild()

    def rebuild(self):
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        items = self.center.items
        if not items:
            empty = QVBoxLayout()
            face = QLabel()
            face.setPixmap(get_cat_pixmap(70, "navy", "sleep"))
            face.setAlignment(Qt.AlignCenter)
            txt = QLabel("No notifications yet.\nYou'll see downloads, training and test results here.")
            txt.setAlignment(Qt.AlignCenter)
            txt.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 12px;")
            lay.addSpacing(24)
            lay.addWidget(face)
            lay.addWidget(txt)
            lay.addSpacing(24)
        for it in items:
            lay.addWidget(_Row(it, self._clicked))
        lay.addStretch()
        self.scroll.setWidget(body)
        rows = max(1, len(items))
        self.setFixedHeight(min(480, 52 + (78 * rows if items else 170)))

    def _clicked(self, item):
        self.hide()
        self.center.run_action(item)


class NotificationCenter(QObject):
    def __init__(self, window, settings: QSettings):
        super().__init__(window)
        self.window = window
        self.settings = settings
        self.items: list[dict] = self._load()
        self._live: dict[str, object] = {}   # in-memory payloads (e.g. full report dicts) for this session
        self.badges: list[_Badge] = []
        self.popup: NotificationPopup | None = None
        self.tray: QSystemTrayIcon | None = None

    # ---- persistence -----------------------------------------------------
    def _load(self) -> list[dict]:
        try:
            data = json.loads(self.settings.value("notifications/items", "[]", type=str) or "[]")
            return [d for d in data if isinstance(d, dict)][:MAX_ITEMS]
        except Exception:
            return []

    def _save(self):
        self.settings.setValue("notifications/items", json.dumps(self.items[:MAX_ITEMS], ensure_ascii=False))
        self.settings.sync()

    # ---- public API ------------------------------------------------------
    def attach_button(self, button: QAbstractButton):
        size = button.iconSize().width() or 22
        button.setIcon(get_svg_icon("bell_plain", size))
        button.setToolTip("Notifications")
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(lambda: self.show_popup(button))
        self.badges.append(_Badge(button))
        self._refresh_badges()

    def add(self, level: str, title: str, message: str = "", action: dict | None = None, payload=None,
            toast: bool = True) -> dict:
        """level: success | error | warning | info. action: {"type": "report"|"file"|"folder", "path": ...}"""
        item = {"id": uuid.uuid4().hex, "ts": time.time(), "level": level, "title": title,
                "message": message, "action": action, "read": False}
        if payload is not None:
            self._live[item["id"]] = payload
        self.items.insert(0, item)
        del self.items[MAX_ITEMS:]
        self._save()
        self._refresh_badges()
        if self.popup and self.popup.isVisible():
            self.popup.rebuild()
        if toast:
            self._toast(item)
        return item

    def unread(self) -> int:
        return sum(1 for i in self.items if not i.get("read"))

    def mark_all_read(self):
        for i in self.items:
            i["read"] = True
        self._save()
        self._refresh_badges()

    def clear(self):
        self.items.clear()
        self._live.clear()
        self._save()
        self._refresh_badges()

    def show_popup(self, anchor: QAbstractButton):
        if self.popup is None:
            self.popup = NotificationPopup(self)
        self.popup.rebuild()
        pos = anchor.mapToGlobal(QPoint(anchor.width(), anchor.height() + 6))
        x = pos.x() - self.popup.width()
        screen = QApplication.screenAt(pos) or QApplication.primaryScreen()
        if screen:
            g = screen.availableGeometry()
            x = max(g.left() + 8, min(x, g.right() - self.popup.width() - 8))
        self.popup.move(x, pos.y())
        self.popup.show()

    def run_action(self, item: dict):
        item["read"] = True
        self._save()
        self._refresh_badges()
        act = item.get("action") or {}
        payload = self._live.get(item["id"])
        if act.get("type") == "report" and payload is not None and hasattr(self.window, "_show_report"):
            self.window._show_report(payload)
            return
        path = act.get("path") or act.get("fallback")
        if path and Path(path).exists():
            from ui.report_dialog import open_path
            open_path(path)

    # ---- internals -------------------------------------------------------
    def _refresh_badges(self):
        n = self.unread()
        for b in self.badges:
            b.set_count(n)

    def _toast(self, item: dict):
        """Tray balloon only when the user is looking somewhere else."""
        if self.window.isActiveWindow() or not QSystemTrayIcon.isSystemTrayAvailable():
            return
        if self.tray is None:
            self.tray = QSystemTrayIcon(QIcon(get_cat_pixmap(64, "navy")), self.window)
            self.tray.setToolTip("CVAT to YOLO Studio")
            self.tray.messageClicked.connect(self._bring_to_front)
            self.tray.activated.connect(lambda *_: self._bring_to_front())
            self.tray.show()
        icon = {"error": QSystemTrayIcon.Critical, "warning": QSystemTrayIcon.Warning}.get(item["level"],
                                                                                          QSystemTrayIcon.Information)
        self.tray.showMessage(item["title"], item.get("message", ""), icon, 8000)

    def _bring_to_front(self):
        self.window.showNormal()
        self.window.raise_()
        self.window.activateWindow()