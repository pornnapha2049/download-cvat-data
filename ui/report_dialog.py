"""
Training / evaluation report window (Navy Enterprise style).

ReportDialog(report) shows:
- summary cards (images, labels, classes, epochs, time …)
- accuracy table (Precision, Recall, F1, mAP50, mAP50-95, Accuracy) for validation / test / external data
- tabs: training curves (drawn natively from results.csv), confusion matrix, PR / F1 curves,
  per-class table, dataset overview, raw Ultralytics plots

`report` is a plain dict built by main_window (see _build_train_report / _build_eval_report).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QFrame, QPushButton, QTabWidget,
    QWidget, QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView, QComboBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QPointF, QRectF, QSize
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QPainterPath

from ui.theme import PALETTE
from ui.svg_icons import get_svg_icon
from ui.dancing_svg import get_cat_pixmap

SERIES_COLORS = ["#1f3a5f", "#6f8fb8", "#c8414f", "#2f6b3d", "#8a5a08", "#4a3c8c"]


def pct(v) -> str:
    return "-" if v is None else f"{v * 100:.1f}%"


def fmt_secs(s) -> str:
    if s is None:
        return "-"
    s = int(s)
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}h {m:02d}m" if h else (f"{m}m {sec:02d}s" if m else f"{sec}s")


def open_path(path):
    p = Path(path)
    if not p.exists():
        return
    if sys.platform == "win32":
        os.startfile(str(p))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(p)])
    else:
        subprocess.Popen(["xdg-open", str(p)])


# =====================================================================
# Small widgets
# =====================================================================
class StatCard(QFrame):
    """Small metric card: caption, big value, optional sub-line."""

    def __init__(self, caption: str, value: str, sub: str = "", accent: bool = False, parent=None):
        super().__init__(parent)
        bg = PALETTE["teal"] if accent else PALETTE["card"]
        fg = "#ffffff" if accent else PALETTE["ink"]
        mt = "#c9d6e8" if accent else PALETTE["muted"]
        self.setStyleSheet(f"QFrame {{ background: {bg}; border: 1px solid {PALETTE['card_border'] if not accent else bg};"
                           f" border-radius: 6px; }} QLabel {{ border: none; background: transparent; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(1)
        c = QLabel(caption.upper())
        c.setStyleSheet(f"color: {mt}; font-size: 10.5px; font-weight: 600; letter-spacing: 0.5px;")
        v = QLabel(value)
        v.setStyleSheet(f"color: {fg}; font-size: 20px; font-weight: 700;")
        lay.addWidget(c)
        lay.addWidget(v)
        if sub:
            s = QLabel(sub)
            s.setWordWrap(True)
            s.setStyleSheet(f"color: {mt}; font-size: 11px;")
            lay.addWidget(s)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(86)


class LineChart(QWidget):
    """Minimal line chart painted with QPainter (no matplotlib needed)."""

    def __init__(self, title: str, series: dict[str, list], y_label: str = "", percent: bool = False,
                 best_x: int | None = None, parent=None):
        super().__init__(parent)
        self.title, self.series, self.y_label, self.percent, self.best_x = title, series, y_label, percent, best_x
        self.setMinimumHeight(280)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(PALETTE["card"]))
        left, right, top, bottom = 56, 22, 34, 62
        pw, ph = w - left - right, h - top - bottom

        f = QFont(self.font())
        f.setPointSizeF(10)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(PALETTE["ink"]))
        p.drawText(left, 22, self.title)

        series = {k: [v for v in vals] for k, vals in self.series.items() if any(x is not None for x in vals)}
        if not series or pw < 50 or ph < 50:
            p.setPen(QColor(PALETTE["muted"]))
            p.drawText(self.rect(), Qt.AlignCenter, "No data")
            return
        n = max(len(v) for v in series.values())
        ys = [y for v in series.values() for y in v if y is not None]
        y_min, y_max = (0.0, 1.0) if self.percent else (min(ys), max(ys))
        if self.percent:
            y_max = max(1.0, max(ys))
        if y_max - y_min < 1e-9:
            y_max = y_min + 1.0
        pad = (y_max - y_min) * 0.06
        y_min, y_max = (y_min, y_max) if self.percent else (y_min - pad, y_max + pad)

        def X(i):
            return left + (pw * i / max(1, n - 1))

        def Y(v):
            return top + ph - (v - y_min) / (y_max - y_min) * ph

        # grid + y labels
        f.setBold(False)
        f.setPointSizeF(8.5)
        p.setFont(f)
        for k in range(5):
            v = y_min + (y_max - y_min) * k / 4
            y = Y(v)
            p.setPen(QPen(QColor(PALETTE["soft"] if k else PALETTE["line"]), 1))
            p.drawLine(QPointF(left, y), QPointF(left + pw, y))
            p.setPen(QColor(PALETTE["muted"]))
            txt = f"{v * 100:.0f}%" if self.percent else f"{v:.2f}"
            p.drawText(QRectF(0, y - 8, left - 8, 16), Qt.AlignRight | Qt.AlignVCenter, txt)
        # x labels (epochs)
        step = max(1, n // 8)
        for i in range(0, n, step):
            p.drawText(QRectF(X(i) - 20, top + ph + 4, 40, 16), Qt.AlignCenter, str(i + 1))
        p.drawText(QRectF(left, top + ph + 18, pw, 14), Qt.AlignCenter, "epoch")

        # best epoch marker
        if self.best_x is not None and 1 <= self.best_x <= n:
            x = X(self.best_x - 1)
            pen = QPen(QColor(PALETTE["rose"]), 1, Qt.DashLine)
            p.setPen(pen)
            p.drawLine(QPointF(x, top), QPointF(x, top + ph))
            lbl_r = QRectF(x + 4, top, 80, 14) if x + 84 < w else QRectF(x - 84, top, 80, 14)
            p.drawText(lbl_r, Qt.AlignRight if x + 84 >= w else Qt.AlignLeft, f"best {self.best_x}")

        # series
        legend_x = left
        for si, (name, vals) in enumerate(series.items()):
            col = QColor(SERIES_COLORS[si % len(SERIES_COLORS)])
            path = QPainterPath()
            started = False
            for i, v in enumerate(vals):
                if v is None:
                    continue
                pt = QPointF(X(i), Y(v))
                if not started:
                    path.moveTo(pt)
                    started = True
                else:
                    path.lineTo(pt)
            p.setPen(QPen(col, 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
            # legend
            ly = h - 8
            p.fillRect(QRectF(legend_x, ly - 5, 14, 4), col)
            p.setPen(QColor(PALETTE["text"]))
            tw = p.fontMetrics().horizontalAdvance(name)
            p.drawText(QPointF(legend_x + 18, ly), name)
            legend_x += 18 + tw + 18


class ConfusionMatrixWidget(QWidget):
    """Readable confusion matrix (rows = predicted, columns = true, like Ultralytics)."""

    def __init__(self, matrix: list[list], names: list[str], normalized: bool = True, parent=None):
        super().__init__(parent)
        self.m, self.names, self.normalized = matrix, names, normalized
        self.setMinimumSize(420, 320)

    def set_normalized(self, on: bool):
        self.normalized = on
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(PALETTE["card"]))
        n = len(self.m)
        if not n:
            return
        col_tot = [sum(self.m[r][c] for r in range(n)) for c in range(n)]
        f = QFont(self.font())
        f.setPointSizeF(9)
        p.setFont(f)
        fm = p.fontMetrics()
        label_w = min(160, max(fm.horizontalAdvance(x) for x in self.names) + 16)
        left, top = label_w + 34, 30
        avail_w, avail_h = self.width() - left - 20, self.height() - top - label_w - 30
        cell = max(18, min(90, int(min(avail_w, avail_h) / n)))
        grid_w = cell * n
        ox = left + max(0, (avail_w - grid_w) // 2)

        base = QColor(PALETTE["teal"])
        for r in range(n):
            for c in range(n):
                v = self.m[r][c]
                frac = (v / col_tot[c]) if col_tot[c] else 0.0
                rect = QRectF(ox + c * cell, top + r * cell, cell - 2, cell - 2)
                col = QColor(base)
                col.setAlphaF(0.06 + 0.94 * frac if v else 0.03)
                p.fillRect(rect, col)
                if v:
                    p.setPen(QColor("#ffffff" if frac > 0.5 else PALETTE["ink"]))
                    txt = f"{frac:.2f}" if self.normalized else str(v)
                    if cell >= 28:
                        p.drawText(rect, Qt.AlignCenter, txt)
        # labels
        p.setPen(QColor(PALETTE["text"]))
        for i, name in enumerate(self.names):
            el = fm.elidedText(name, Qt.ElideRight, label_w - 8)
            p.drawText(QRectF(ox - label_w - 6, top + i * cell, label_w, cell), Qt.AlignRight | Qt.AlignVCenter, el)
            p.save()
            p.translate(ox + i * cell + cell / 2 + 4, top + grid_w + 6)
            p.rotate(60)
            p.drawText(QRectF(0, -8, label_w, 16), Qt.AlignLeft | Qt.AlignVCenter, el)
            p.restore()
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(PALETTE["muted"]))
        p.drawText(QRectF(ox + grid_w / 2 - 300, 6, 600, 18), Qt.AlignCenter,
                   "True class  →  (columns)   ·   " + ("normalized by true class" if self.normalized else "counts"))
        p.save()
        p.translate(14, top + grid_w / 2)
        p.rotate(-90)
        p.drawText(QRectF(-grid_w / 2, -8, grid_w, 16), Qt.AlignCenter, "Predicted  →")
        p.restore()


class ImagePanel(QScrollArea):
    """Shows a plot image scaled to the panel width; click opens the file."""

    def __init__(self, path: str | None, empty_text: str = "Plot not available", parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.path = path
        self.lbl = QLabel()
        self.lbl.setAlignment(Qt.AlignCenter)
        self.lbl.setStyleSheet(f"background: {PALETTE['card']}; color: {PALETTE['muted']};")
        self.setWidget(self.lbl)
        self.pix = QPixmap(path) if path and Path(path).exists() else QPixmap()
        if self.pix.isNull():
            self.lbl.setText(empty_text)
        else:
            self.lbl.setCursor(Qt.PointingHandCursor)
            self.lbl.setToolTip("Click to open full size")
            self.lbl.mousePressEvent = lambda e: open_path(self.path)

    def set_path(self, path):
        self.path = path
        self.pix = QPixmap(path) if path and Path(path).exists() else QPixmap()
        if self.pix.isNull():
            self.lbl.setPixmap(QPixmap())
            self.lbl.setText("Plot not available")
        else:
            self.lbl.setText("")
            self._rescale()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale()

    def _rescale(self):
        if self.pix.isNull():
            return
        w = max(200, self.viewport().width() - 8)
        h = max(200, self.viewport().height() - 8)
        self.lbl.setPixmap(self.pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))


def _table(headers: list[str], rows: list[list], stretch_first=True) -> QTableWidget:
    t = QTableWidget(len(rows), len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QTableWidget.NoEditTriggers)
    t.setSelectionBehavior(QTableWidget.SelectRows)
    t.setAlternatingRowColors(True)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            it = QTableWidgetItem("" if val is None else str(val))
            it.setTextAlignment(Qt.AlignVCenter | (Qt.AlignLeft if c == 0 else Qt.AlignCenter))
            t.setItem(r, c, it)
    hh = t.horizontalHeader()
    for c in range(len(headers)):
        hh.setSectionResizeMode(c, QHeaderView.Stretch if (c == 0 and stretch_first) else QHeaderView.ResizeToContents)
    return t


# =====================================================================
# Report dialog
# =====================================================================
class ReportDialog(QDialog):
    def __init__(self, report: dict, parent=None):
        super().__init__(parent)
        self.report = report
        self.setWindowTitle(report.get("title", "Report"))
        self.resize(1180, 820)
        self.setStyleSheet(f"""
            QDialog {{ background: {PALETTE['bg_top']}; }}
            QTabWidget::pane {{ border: 1px solid {PALETTE['card_border']}; background: {PALETTE['card']}; border-radius: 4px; top: -1px; }}
            QTabBar::tab {{ background: transparent; color: {PALETTE['muted']}; padding: 8px 16px; border: none;
                           border-bottom: 3px solid transparent; font-weight: 600; }}
            QTabBar::tab:selected {{ color: {PALETTE['teal']}; border-bottom: 3px solid {PALETTE['teal']}; }}
            QTabBar::tab:hover {{ color: {PALETTE['ink']}; }}
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        # ---- header
        head = QHBoxLayout()
        face = QLabel()
        face.setPixmap(get_cat_pixmap(54, "navy", "happy" if report.get("ok", True) else "open"))
        head.addWidget(face)
        tcol = QVBoxLayout()
        tcol.setSpacing(2)
        t = QLabel(report.get("title", "Report"))
        t.setStyleSheet(f"font-size: 20px; font-weight: 700; color: {PALETTE['ink']};")
        s = QLabel(report.get("subtitle", ""))
        s.setWordWrap(True)
        s.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 12px;")
        tcol.addWidget(t)
        tcol.addWidget(s)
        head.addLayout(tcol, 1)
        if report.get("folder"):
            b = QPushButton("Open Folder")
            b.setProperty("class", "SecondaryNavButton")
            b.setIcon(get_svg_icon("folder", 16))
            b.setIconSize(QSize(16, 16))
            b.clicked.connect(lambda: open_path(report["folder"]))
            head.addWidget(b)
        if report.get("report_md"):
            b = QPushButton("Open report.md")
            b.setProperty("class", "SecondaryNavButton")
            b.clicked.connect(lambda: open_path(report["report_md"]))
            head.addWidget(b)
        root.addLayout(head)

        # ---- stat cards
        cards = report.get("cards", [])
        if cards:
            grid = QGridLayout()
            grid.setSpacing(10)
            for i, c in enumerate(cards):
                grid.addWidget(StatCard(c[0], c[1], c[2] if len(c) > 2 else "", accent=(len(c) > 3 and c[3])),
                               i // 6, i % 6)
            root.addLayout(grid)

        # ---- tabs
        tabs = QTabWidget()
        summary = QWidget()
        sl = QVBoxLayout(summary)
        sl.setContentsMargins(10, 10, 10, 10)
        sl.setSpacing(8)
        # ---- accuracy table
        acc_rows = report.get("accuracy_rows", [])
        if acc_rows:
            lbl = QLabel("Accuracy")
            lbl.setProperty("class", "SectionTitle")
            sl.addWidget(lbl)
            tbl = _table(["Data", "Images", "Instances", "Precision", "Recall", "F1", "mAP50", "mAP50-95", "Accuracy*"],
                         acc_rows)
            tbl.setFixedHeight(48 + 38 * len(acc_rows))
            tbl.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            sl.addWidget(tbl)
            note = QLabel("* Accuracy = TP / (TP + FP + FN) from the confusion matrix (conf 0.25 after training; the conf you set on the Test page). "
                          "mAP50 is the standard headline metric for object detection.")
            note.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 11px;")
            sl.addWidget(note)
        for w in report.get("warnings", []):
            wl = QLabel(f"⚠  {w}")
            wl.setWordWrap(True)
            wl.setStyleSheet(f"background: {PALETTE['lemon']}; color: {PALETTE['lemon_ink']}; padding: 6px 10px; border-radius: 4px;")
            sl.addWidget(wl)

        pc = report.get("per_class")
        if pc:
            lbl = QLabel("Per-class")
            lbl.setProperty("class", "SectionTitle")
            sl.addWidget(lbl)
            sl.addWidget(_table(pc["headers"], pc["rows"]), 1)
        else:
            sl.addStretch(1)
        tabs.addTab(summary, "Summary")

        plots = report.get("plots", {})
        curves = report.get("curves")
        if curves:
            w = QWidget()
            lay = QHBoxLayout(w)
            lay.setContentsMargins(8, 8, 8, 8)
            lay.addWidget(LineChart("Loss", curves.get("loss", {}), best_x=curves.get("best_epoch")), 1)
            lay.addWidget(LineChart("Validation metrics", curves.get("metrics", {}), percent=True,
                                    best_x=curves.get("best_epoch")), 1)
            tabs.addTab(w, "Training Curves")

        cm_n, cm_r = plots.get("confusion_matrix_normalized"), plots.get("confusion_matrix")
        conf = report.get("confusion") or {}
        if conf.get("matrix"):
            w = QWidget()
            lay = QVBoxLayout(w)
            lay.setContentsMargins(8, 8, 8, 8)
            top_row = QHBoxLayout()
            sel = QComboBox()
            sel.addItems(["Normalized", "Counts"])
            sel.setFixedWidth(160)
            top_row.addWidget(sel)
            info = QLabel(f"TP {int(conf.get('tp', 0))} of {int(conf.get('total_events', 0))} matched/unmatched objects  ·  "
                          "off-diagonal = wrong class, background row/column = missed or extra boxes")
            info.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 11px;")
            top_row.addWidget(info, 1)
            if cm_n or cm_r:
                b = QPushButton("Open Ultralytics image")
                b.setProperty("class", "SecondaryNavButton")
                b.clicked.connect(lambda: open_path(cm_n or cm_r))
                top_row.addWidget(b)
            lay.addLayout(top_row)
            cmw = ConfusionMatrixWidget(conf["matrix"], conf.get("names", []))
            sel.currentIndexChanged.connect(lambda i: cmw.set_normalized(i == 0))
            lay.addWidget(cmw, 1)
            tabs.addTab(w, "Confusion Matrix")
        elif cm_n or cm_r:
            w = QWidget()
            lay = QVBoxLayout(w)
            lay.setContentsMargins(8, 8, 8, 8)
            sel = QComboBox()
            opts = [(n, p) for n, p in (("Normalized", cm_n), ("Counts", cm_r)) if p]
            for n, p in opts:
                sel.addItem(n, p)
            sel.setFixedWidth(180)
            panel = ImagePanel(opts[0][1])
            sel.currentIndexChanged.connect(lambda i: panel.set_path(sel.itemData(i)))
            lay.addWidget(sel)
            lay.addWidget(panel, 1)
            tabs.addTab(w, "Confusion Matrix")

        curve_opts = [(n, plots.get(k)) for n, k in (("Precision-Recall", "pr_curve"), ("F1-Confidence", "f1_curve"),
                                                     ("Precision-Confidence", "p_curve"), ("Recall-Confidence", "r_curve"),
                                                     ("Mask PR", "mask_pr_curve")) if plots.get(k)]
        if curve_opts:
            w = QWidget()
            lay = QVBoxLayout(w)
            lay.setContentsMargins(8, 8, 8, 8)
            sel = QComboBox()
            for n, p in curve_opts:
                sel.addItem(n, p)
            sel.setFixedWidth(220)
            panel = ImagePanel(curve_opts[0][1])
            sel.currentIndexChanged.connect(lambda i, s=sel, pn=panel: pn.set_path(s.itemData(i)))
            lay.addWidget(sel)
            lay.addWidget(panel, 1)
            tabs.addTab(w, "PR / F1 Curves")

        if plots.get("labels"):
            tabs.addTab(ImagePanel(plots["labels"]), "Dataset Labels")
        if plots.get("val_pred"):
            tabs.addTab(ImagePanel(plots["val_pred"]), "Sample Predictions")
        if plots.get("results"):
            tabs.addTab(ImagePanel(plots["results"]), "results.png")

        if tabs.count():
            root.addWidget(tabs, 1)
        else:
            root.addStretch(1)

        foot = QHBoxLayout()
        foot.addStretch()
        close = QPushButton("Close")
        close.setProperty("class", "PrimaryButton")
        close.clicked.connect(self.accept)
        foot.addWidget(close)
        root.addLayout(foot)