"""
SVG Icon Library for CVAT to YOLO Studio.
Inspired by the modern mindfulness / wellness pastel study aesthetic:
https://i.pinimg.com/1200x/69/a3/a3/69a3a365c2017d08d2c1ce4795633dd5.jpg

Features:
- Clean outline curves (stroke #1e293b / #233240, stroke-linecap="round", stroke-linejoin="round")
- Soft pastel fills (sage mint #daf0dd, lavender rose #fde4ed, butter yellow #fef3d1, periwinkle #e0e8f8, lilac #eee3fc)
- Icons modeled after the reference pin: Lotus bowl, Zen stones, Mind detox brain, Gratitude sunburst, Sleek chevron arrows, Notification bell, Bar charts.
- High-resolution SVG rendering to QIcon and QPixmap with in-memory caching.
"""

from PyQt5 import QtSvg
from PyQt5.QtGui import QIcon, QPixmap, QPainter
from PyQt5.QtCore import QByteArray, Qt, QSize


# =====================================================================
# SVG Vector Definitions (24x24 and 32x32 viewports)
# =====================================================================

SVG_DEFS = {
    # -----------------------------------------------------------------
    # Reference Pin Highlights: Lotus, Zen Stones, Brain, Sparkle, Bell
    # -----------------------------------------------------------------
    "lotus": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="32" height="32">
        <!-- Bowl -->
        <path d="M 6 19 C 6 26, 26 26, 26 19 Z" fill="#ffd5be" stroke="#233240" stroke-width="1.8" stroke-linejoin="round"/>
        <!-- Water line -->
        <path d="M 7 19 Q 16 20 25 19" stroke="#68b4d8" stroke-width="1.5" stroke-linecap="round"/>
        <!-- Center Lotus Petal -->
        <path d="M 16 9 C 12 13, 13 18, 16 19 C 19 18, 20 13, 16 9 Z" fill="#fca5a5" stroke="#233240" stroke-width="1.6" stroke-linejoin="round"/>
        <!-- Left Petal -->
        <path d="M 16 19 C 11 18, 8 13, 10 11 C 12 11, 14 15, 16 19 Z" fill="#fbcfe8" stroke="#233240" stroke-width="1.6" stroke-linejoin="round"/>
        <!-- Right Petal -->
        <path d="M 16 19 C 21 18, 24 13, 22 11 C 20 11, 18 15, 16 19 Z" fill="#fbcfe8" stroke="#233240" stroke-width="1.6" stroke-linejoin="round"/>
    </svg>
    """,

    "zen_stones": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="32" height="32">
        <!-- Bottom Large Stone -->
        <ellipse cx="16" cy="24" rx="11" ry="5.5" fill="#475569" stroke="#1e293b" stroke-width="1.8"/>
        <ellipse cx="16" cy="23" rx="9" ry="3.5" fill="#64748b" opacity="0.6"/>
        <!-- Middle Stone -->
        <ellipse cx="16" cy="16" rx="8" ry="4.5" fill="#64748b" stroke="#1e293b" stroke-width="1.8"/>
        <ellipse cx="16" cy="15" rx="6.5" ry="3" fill="#94a3b8" opacity="0.6"/>
        <!-- Top Small Stone -->
        <ellipse cx="16" cy="9.5" rx="5" ry="3.5" fill="#94a3b8" stroke="#1e293b" stroke-width="1.8"/>
        <ellipse cx="16" cy="9" rx="3.5" ry="2" fill="#cbd5e1" opacity="0.7"/>
    </svg>
    """,

    "brain": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="32" height="32">
        <!-- Background Soft Purple Pill -->
        <rect x="2" y="2" width="28" height="28" rx="8" fill="#eee3fc" stroke="#dfcbf9" stroke-width="1.2"/>
        <!-- Left Hemisphere -->
        <path d="M 15 8 C 12 8, 9 10, 9 13 C 7 14, 7 17, 8 19 C 7 21, 8 23, 10 24 C 12 25, 14 24, 15 23" fill="#c084fc" stroke="#233240" stroke-width="1.6" stroke-linecap="round"/>
        <!-- Right Hemisphere -->
        <path d="M 17 8 C 20 8, 23 10, 23 13 C 25 14, 25 17, 24 19 C 25 21, 24 23, 22 24 C 20 25, 18 24, 17 23" fill="#c084fc" stroke="#233240" stroke-width="1.6" stroke-linecap="round"/>
        <!-- Center Divider & Folds -->
        <path d="M 16 8 L 16 23" stroke="#233240" stroke-width="1.6" stroke-linecap="round"/>
        <path d="M 11 14 Q 14 15 15 13" stroke="#233240" stroke-width="1.4" fill="none" stroke-linecap="round"/>
        <path d="M 21 14 Q 18 15 17 13" stroke="#233240" stroke-width="1.4" fill="none" stroke-linecap="round"/>
        <path d="M 10 19 Q 13 18 15 20" stroke="#233240" stroke-width="1.4" fill="none" stroke-linecap="round"/>
        <path d="M 22 19 Q 19 18 17 20" stroke="#233240" stroke-width="1.4" fill="none" stroke-linecap="round"/>
    </svg>
    """,

    "sparkle": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="32" height="32">
        <rect x="2" y="2" width="28" height="28" rx="8" fill="#e0e8f8" stroke="#cbd9f4" stroke-width="1.2"/>
        <!-- 8-point Sunburst / Star -->
        <path d="M 16 6 L 16 26 M 6 16 L 26 16 M 9 9 L 23 23 M 23 9 L 9 23" stroke="#1d4ed8" stroke-width="2.2" stroke-linecap="round"/>
        <circle cx="16" cy="16" r="3.5" fill="#fef08a" stroke="#1e293b" stroke-width="1.5"/>
    </svg>
    """,

    "bell": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="32" height="32">
        <!-- Bell Body -->
        <path d="M 16 7 C 12 7, 10 10, 10 15 C 10 19, 8 21, 7 22 L 25 22 C 24 21, 22 19, 22 15 C 22 10, 20 7, 16 7 Z" fill="#fef3d1" stroke="#233240" stroke-width="1.8" stroke-linejoin="round"/>
        <path d="M 14 25 C 14 26.5, 15 27, 16 27 C 17 26.5, 18 26.5, 18 25" stroke="#233240" stroke-width="1.8" stroke-linecap="round"/>
        <circle cx="16" cy="5.5" r="1.5" fill="#233240"/>
        <!-- Pink Alert Badge Dot -->
        <circle cx="23" cy="8" r="3.5" fill="#f43f5e" stroke="#ffffff" stroke-width="1.5"/>
    </svg>
    """,

    "sprout": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="32" height="32">
        <rect x="2" y="2" width="28" height="28" rx="8" fill="#daf0dd" stroke="#c4e7c7" stroke-width="1.2"/>
        <!-- 3 Clover Leaves / Sprigs -->
        <path d="M 16 22 L 16 14" stroke="#233240" stroke-width="2" stroke-linecap="round"/>
        <path d="M 16 14 C 13 9, 8 12, 12 16 C 14 16, 16 14, 16 14 Z" fill="#4ade80" stroke="#233240" stroke-width="1.6"/>
        <path d="M 16 14 C 19 9, 24 12, 20 16 C 18 16, 16 14, 16 14 Z" fill="#4ade80" stroke="#233240" stroke-width="1.6"/>
        <path d="M 16 14 C 16 8, 11 6, 16 6 C 21 6, 16 8, 16 14 Z" fill="#86efac" stroke="#233240" stroke-width="1.6"/>
    </svg>
    """,

    # -----------------------------------------------------------------
    # Stepper Workflow Icons (Step 1 -> 5)
    # -----------------------------------------------------------------
    "step_cvat": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 28" width="28" height="28">
        <!-- Cloud and Download Tray -->
        <path d="M 6 15 C 4 15, 3 13, 4 11 C 5 9, 7 9, 8 9 C 9 6, 14 5, 17 8 C 19 7, 22 8, 22 11 C 24 12, 24 15, 22 15 Z" fill="#daf0dd" stroke="#233240" stroke-width="1.7" stroke-linejoin="round"/>
        <!-- Arrow down -->
        <path d="M 14 12 L 14 20 M 11 17 L 14 20 L 17 17" stroke="#166534" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <!-- Tray line -->
        <path d="M 8 23 L 20 23" stroke="#233240" stroke-width="2" stroke-linecap="round"/>
    </svg>
    """,

    "step_merge": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 28" width="28" height="28">
        <!-- Two flowing branches merging together -->
        <circle cx="8" cy="7" r="3" fill="#fef3d1" stroke="#233240" stroke-width="1.7"/>
        <circle cx="8" cy="21" r="3" fill="#fef3d1" stroke="#233240" stroke-width="1.7"/>
        <circle cx="21" cy="14" r="3.5" fill="#fde68a" stroke="#233240" stroke-width="1.7"/>
        <path d="M 11 7 C 15 7, 16 13, 17.5 14" stroke="#d97706" stroke-width="2" fill="none" stroke-linecap="round"/>
        <path d="M 11 21 C 15 21, 16 15, 17.5 14" stroke="#d97706" stroke-width="2" fill="none" stroke-linecap="round"/>
    </svg>
    """,

    "step_convert": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 28" width="28" height="28">
        <!-- Magic wand / lightning spark with pastel rose accents -->
        <path d="M 5 23 L 17 11" stroke="#233240" stroke-width="2.2" stroke-linecap="round"/>
        <!-- Wand tip glow -->
        <circle cx="18" cy="10" r="4" fill="#fde4ed" stroke="#f43f5e" stroke-width="1.5"/>
        <!-- Magic stars -->
        <path d="M 21 4 L 21 8 M 19 6 L 23 6" stroke="#e11d48" stroke-width="1.8" stroke-linecap="round"/>
        <path d="M 11 4 L 11 6 M 10 5 L 12 5" stroke="#f43f5e" stroke-width="1.5" stroke-linecap="round"/>
        <path d="M 23 15 L 23 17 M 22 16 L 24 16" stroke="#f43f5e" stroke-width="1.5" stroke-linecap="round"/>
    </svg>
    """,

    "step_split": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 28" width="28" height="28">
        <!-- 3 Partition cards (Train, Val, Test) -->
        <rect x="4" y="6" width="6" height="16" rx="2.5" fill="#bfdbfe" stroke="#1e3a8a" stroke-width="1.6"/>
        <rect x="12" y="9" width="5.5" height="13" rx="2.5" fill="#c7d2fe" stroke="#3730a3" stroke-width="1.6"/>
        <rect x="19.5" y="12" width="5" height="10" rx="2" fill="#ddd6fe" stroke="#5b21b6" stroke-width="1.6"/>
        <path d="M 7 10 L 7 14" stroke="#1e3a8a" stroke-width="1.5" stroke-linecap="round"/>
    </svg>
    """,

    "step_train": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 28" width="28" height="28">
        <!-- Rocket Body -->
        <path d="M 14 4 C 18 8, 20 13, 19 18 L 9 18 C 8 13, 10 8, 14 4 Z" fill="#ffd5be" stroke="#233240" stroke-width="1.8" stroke-linejoin="round"/>
        <!-- Window -->
        <circle cx="14" cy="12" r="2.5" fill="#67e8f9" stroke="#233240" stroke-width="1.5"/>
        <!-- Wings -->
        <path d="M 9 16 L 5 19 L 9 19 Z" fill="#f87171" stroke="#233240" stroke-width="1.5"/>
        <path d="M 19 16 L 23 19 L 19 19 Z" fill="#f87171" stroke="#233240" stroke-width="1.5"/>
        <!-- Flame -->
        <path d="M 11 19 Q 14 25 14 25 Q 14 25 17 19" fill="#facc15" stroke="#ea580c" stroke-width="1.6"/>
    </svg>
    """,

    "step_runs": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 28" width="28" height="28">
        <!-- Folder with Weights Star badge -->
        <path d="M 4 8 C 4 6.5, 5 6, 6.5 6 L 11 6 L 13 8.5 L 22 8.5 C 23.5 8.5, 24 9.5, 24 11 L 24 20 C 24 21.5, 23 22, 21.5 22 L 6.5 22 C 5 22, 4 21, 4 20 Z" fill="#fef3d1" stroke="#233240" stroke-width="1.7"/>
        <path d="M 4 11.5 L 24 11.5" stroke="#233240" stroke-width="1.5"/>
        <!-- Cute Checkmark badge -->
        <circle cx="14" cy="16.5" r="3.5" fill="#4ade80" stroke="#166534" stroke-width="1.2"/>
        <path d="M 12.5 16.5 L 13.5 17.5 L 15.5 15.5" stroke="#ffffff" stroke-width="1.4" fill="none" stroke-linecap="round"/>
    </svg>
    """,

    # -----------------------------------------------------------------
    # Navigation Arrows (Pinterest Circular Pill Style)
    # -----------------------------------------------------------------
    "arrow_left": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
        <circle cx="12" cy="12" r="10" fill="#ffffff" stroke="#233240" stroke-width="1.6"/>
        <path d="M 14 8 L 10 12 L 14 16" stroke="#233240" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    """,

    "arrow_right": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
        <circle cx="12" cy="12" r="10" fill="#ffffff" stroke="#233240" stroke-width="1.6"/>
        <path d="M 10 8 L 14 12 L 10 16" stroke="#233240" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    """,

    # -----------------------------------------------------------------
    # Action Icons: Play, Stop, Refresh, Search, Download, Folder, Key
    # -----------------------------------------------------------------
    "play": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
        <circle cx="12" cy="12" r="10" fill="#daf0dd" stroke="#233240" stroke-width="1.6"/>
        <polygon points="10,8 16,12 10,16" fill="#166534" stroke="#166534" stroke-width="1.2" stroke-linejoin="round"/>
    </svg>
    """,

    "stop": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
        <circle cx="12" cy="12" r="10" fill="#fee2e2" stroke="#233240" stroke-width="1.6"/>
        <rect x="9" y="9" width="6" height="6" rx="1.5" fill="#dc2626" stroke="#b91c1c" stroke-width="1"/>
    </svg>
    """,

    "refresh": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
        <path d="M 19 12 A 7 7 0 1 1 17 7 L 19 5 M 19 5 L 19 9 L 15 9" stroke="#233240" stroke-width="1.9" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    """,

    "search": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
        <circle cx="11" cy="11" r="6.5" fill="#f0fdf4" stroke="#233240" stroke-width="1.8"/>
        <!-- Lens highlight -->
        <path d="M 8 9 A 4 4 0 0 1 13 8" stroke="#86efac" stroke-width="1.4" fill="none" stroke-linecap="round"/>
        <path d="M 16 16 L 21 21" stroke="#233240" stroke-width="2.2" stroke-linecap="round"/>
    </svg>
    """,

    "download": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
        <path d="M 12 4 L 12 15 M 8 11 L 12 15 L 16 11" stroke="#166534" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M 5 18 C 5 19.5, 6 20, 7.5 20 L 16.5 20 C 18 20, 19 19.5, 19 18" stroke="#233240" stroke-width="1.8" fill="none" stroke-linecap="round"/>
    </svg>
    """,

    "folder": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
        <path d="M 3 6 C 3 4.5, 4 4, 5.5 4 L 9.5 4 L 11.5 6.5 L 19.5 6.5 C 21 6.5, 21.5 7.5, 21.5 9 L 21.5 18 C 21.5 19.5, 20.5 20, 19 20 L 5.5 20 C 4 20, 3 19, 3 18 Z" fill="#fef3d1" stroke="#233240" stroke-width="1.7"/>
        <path d="M 3 9.5 L 21.5 9.5" stroke="#233240" stroke-width="1.5"/>
    </svg>
    """,

    "key": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
        <circle cx="8" cy="12" r="5" fill="#fef08a" stroke="#233240" stroke-width="1.8"/>
        <circle cx="8" cy="12" r="2" fill="#ffffff" stroke="#233240" stroke-width="1.3"/>
        <path d="M 13 12 L 21 12 M 17 12 L 17 15 M 20 12 L 20 15" stroke="#233240" stroke-width="1.8" stroke-linecap="round"/>
    </svg>
    """,

    "logout": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
        <!-- Door Frame -->
        <path d="M 10 4 L 4 4 C 3 4, 2.5 4.5, 2.5 6 L 2.5 18 C 2.5 19.5, 3 20, 4 20 L 10 20" stroke="#233240" stroke-width="1.8" fill="none" stroke-linecap="round"/>
        <!-- Arrow out -->
        <path d="M 15 8 L 19 12 L 15 16 M 19 12 L 8 12" stroke="#e11d48" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    """,

    "user": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
        <circle cx="12" cy="8" r="4.5" fill="#bfdbfe" stroke="#1e293b" stroke-width="1.7"/>
        <path d="M 4 20 C 4 16, 7.5 15, 12 15 C 16.5 15, 20 16, 20 20" fill="#e0e8f8" stroke="#1e293b" stroke-width="1.7" stroke-linecap="round"/>
    </svg>
    """,

    "server": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
        <!-- Top Disk -->
        <ellipse cx="12" cy="7" rx="8" ry="3.5" fill="#e2e8f0" stroke="#233240" stroke-width="1.7"/>
        <path d="M 4 7 L 4 12 C 4 14, 20 14, 20 12 L 20 7" fill="#f1f5f9" stroke="#233240" stroke-width="1.7"/>
        <!-- Bottom Disk -->
        <path d="M 4 12 L 4 17 C 4 19, 20 19, 20 17 L 20 12" fill="#f1f5f9" stroke="#233240" stroke-width="1.7"/>
        <!-- Green indicator light -->
        <circle cx="17" cy="15" r="1.5" fill="#22c55e"/>
    </svg>
    """,

    "check": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
        <circle cx="12" cy="12" r="10" fill="#daf0dd" stroke="#233240" stroke-width="1.6"/>
        <path d="M 8 12 L 11 15 L 16 9" stroke="#166534" stroke-width="2.2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    """,

    "warning": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
        <circle cx="12" cy="12" r="10" fill="#fef3d1" stroke="#233240" stroke-width="1.6"/>
        <path d="M 12 7 L 12 13" stroke="#b45309" stroke-width="2" stroke-linecap="round"/>
        <circle cx="12" cy="16.5" r="1.2" fill="#b45309"/>
    </svg>
    """,

    # -----------------------------------------------------------------
    # Metrics & Analytics Icons (Bar Chart, Target, Clock, BBox, Polygon)
    # -----------------------------------------------------------------
    "chart": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
        <!-- Rounded vertical bars like Pinterest sleep/stress indicator -->
        <rect x="3" y="14" width="3" height="7" rx="1.5" fill="#c084fc"/>
        <rect x="7.5" y="10" width="3" height="11" rx="1.5" fill="#a855f7"/>
        <rect x="12" y="5" width="3" height="16" rx="1.5" fill="#60a5fa"/>
        <rect x="16.5" y="8" width="3" height="13" rx="1.5" fill="#3b82f6"/>
        <rect x="21" y="12" width="2" height="9" rx="1" fill="#93c5fd"/>
    </svg>
    """,

    "target": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
        <circle cx="12" cy="12" r="9.5" fill="#fef2f2" stroke="#233240" stroke-width="1.7"/>
        <circle cx="12" cy="12" r="6" fill="#fecaca" stroke="#233240" stroke-width="1.5"/>
        <circle cx="12" cy="12" r="2.5" fill="#ef4444"/>
    </svg>
    """,

    "clock": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
        <circle cx="12" cy="12" r="9.5" fill="#f0f9ff" stroke="#233240" stroke-width="1.7"/>
        <path d="M 12 7 L 12 12 L 15.5 14" stroke="#0284c7" stroke-width="2" stroke-linecap="round"/>
    </svg>
    """,

    "box": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
        <rect x="4" y="4" width="16" height="16" rx="3" fill="#e0e8f8" stroke="#1d4ed8" stroke-width="1.8" stroke-dasharray="3 2"/>
        <circle cx="4" cy="4" r="2" fill="#1d4ed8"/>
        <circle cx="20" cy="4" r="2" fill="#1d4ed8"/>
        <circle cx="4" cy="20" r="2" fill="#1d4ed8"/>
        <circle cx="20" cy="20" r="2" fill="#1d4ed8"/>
    </svg>
    """,

    "polygon": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
        <polygon points="12,3 21,9 18,20 6,20 3,9" fill="#fde4ed" stroke="#e11d48" stroke-width="1.8"/>
        <circle cx="12" cy="3" r="2" fill="#e11d48"/>
        <circle cx="21" cy="9" r="2" fill="#e11d48"/>
        <circle cx="18" cy="20" r="2" fill="#e11d48"/>
        <circle cx="6" cy="20" r="2" fill="#e11d48"/>
        <circle cx="3" cy="9" r="2" fill="#e11d48"/>
    </svg>
    """,

    "filter": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
        <path d="M 4 6 L 20 6 M 7 12 L 17 12 M 10 18 L 14 18" stroke="#233240" stroke-width="2" stroke-linecap="round"/>
    </svg>
    """,

    "hardware": """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
        <rect x="5" y="5" width="14" height="14" rx="3" fill="#ecfdf5" stroke="#059669" stroke-width="1.7"/>
        <rect x="9" y="9" width="6" height="6" rx="1.5" fill="#10b981"/>
        <path d="M 9 2 L 9 5 M 15 2 L 15 5 M 9 19 L 9 22 M 15 19 L 15 22 M 2 9 L 5 9 M 2 15 L 5 15 M 19 9 L 22 9 M 19 15 L 22 15" stroke="#059669" stroke-width="1.5" stroke-linecap="round"/>
    </svg>
    """
}


# =====================================================================
# In-Memory Cache & SVG Render Helpers
# =====================================================================

_PIXMAP_CACHE: dict[tuple[str, int, int], QPixmap] = {}
_ICON_CACHE: dict[tuple[str, int], QIcon] = {}


def get_svg_raw(name: str) -> str:
    """Returns raw SVG string by icon key name."""
    return SVG_DEFS.get(name, SVG_DEFS["sprout"]).strip()


def get_svg_pixmap(name: str, width: int = 24, height: int = 24) -> QPixmap:
    """Renders named SVG icon to a high-DPI QPixmap with caching."""
    cache_key = (name, width, height)
    if cache_key in _PIXMAP_CACHE:
        return _PIXMAP_CACHE[cache_key]

    svg_data = get_svg_raw(name)
    pix = QPixmap(width, height)
    pix.fill(Qt.transparent)

    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

    renderer = QtSvg.QSvgRenderer(QByteArray(svg_data.encode("utf-8")))
    renderer.render(painter)
    painter.end()

    _PIXMAP_CACHE[cache_key] = pix
    return pix


def get_svg_icon(name: str, size: int = 22) -> QIcon:
    """Returns a QIcon created from the SVG definition with caching."""
    cache_key = (name, size)
    if cache_key in _ICON_CACHE:
        return _ICON_CACHE[cache_key]

    pix = get_svg_pixmap(name, size, size)
    icon = QIcon(pix)
    _ICON_CACHE[cache_key] = icon
    return icon
