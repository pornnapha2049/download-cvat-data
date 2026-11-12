"""
CVAT to YOLO Studio & Auto-Trainer
Main Entry Point.

Features:
- Dedicated Login View with CVAT Authentication
- 5-Step Wizard Workflow with Responsive QScrollArea for all screen sizes
- CVAT Download with auto-unzip & cleanup of /images/default
- Filter by Project, Stage (acceptance), State (completed), Status
- COCO 1.0 Dataset Merge with duplicate image MD5 resolution
- COCO to YOLO Converter (Detection polygon->bbox, Segmentation polygon)
- Train/Val/Test Splitter including ALL images (with & without annotations)
- YOLO Auto-Trainer (Detection & Segmentation) with Auto-Tuning
- Early Stopping (Patience) when performance stagnates
- Auto-versioned runs folder (_v1, _v2, ...) and custom weights naming
- Aesthetic Printable Stickers & Study Stationery Theme (.☘︎ ݁˖) with animated dancing mascot
"""

import os
import shutil
import sys

# Remove any OpenCV bundled Qt plugins that crash PyQt5's xcb platform plugin
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
for _dir in sys.path + ["/usr/local/lib/python3.10/site-packages"]:
    _bad_cv2_qt = os.path.join(_dir, "cv2", "qt")
    if os.path.exists(_bad_cv2_qt):
        try:
            shutil.rmtree(_bad_cv2_qt)
        except Exception:
            pass

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from ui.main_window import MainWindow


def main():
    # Enable High DPI display support for all screen resolutions
    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    if hasattr(QApplication, "setHighDpiScaleFactorRoundingPolicy"):
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

    app = QApplication(sys.argv)
    app.setApplicationName("CVAT to YOLO Studio")
    app.setOrganizationName("Articulus")

    # Set default readable font with proper vertical line-height
    font = app.font()
    font.setFamily("Segoe UI")
    font.setPointSize(10)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
