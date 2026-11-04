"""
CVAT to YOLO Studio & Auto-Trainer
Main Entry Point.

Features:
- CVAT Download with auto-unzip & cleanup of /images/default
- Filter by Project, Stage (acceptance), State (completed), Status
- COCO 1.0 Dataset Merge with duplicate image MD5 resolution
- COCO to YOLO Converter (Detection polygon->bbox, Segmentation polygon)
- Train/Val/Test Splitter including ALL images (with & without annotations)
- YOLO Auto-Trainer (Detection & Segmentation) with Auto-Tuning
- Early Stopping (Patience) when performance stagnates
- Auto-versioned runs folder (_v1, _v2, ...) and custom weights naming
- Modern Dark GUI with dynamic animated dancing SVG mascot
"""

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from ui.main_window import MainWindow


def main():
    # Enable High DPI display support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("CVAT to YOLO Studio")
    app.setOrganizationName("Articulus")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
