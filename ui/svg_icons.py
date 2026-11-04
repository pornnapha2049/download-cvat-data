"""
SVG icon library for CVAT to YOLO Studio — "Blob Duotone" style, tuned for the Navy Enterprise theme.

Style rules
- No outlines. Each icon sits on a soft organic blob in the light tone of its hue,
  shapes use the mid tone (+ a 20% deep overlay), details/lines use the deep tone.
- Blue/navy for most icons; colour only where it carries meaning:
  green = run / ok, rose = stop / leave, amber = folders / warnings, lilac = AI / magic.
- Small sizes (<= 18 px) drop the blob so the glyph stays crisp on buttons.
- inverse=True gives a white version for dark (navy / red) buttons.

API (unchanged): get_svg_icon(name, size), get_svg_pixmap(name, w, h), get_svg_raw(name)
New: inverse= flag, icon_name_for(icon) to find which icon a QIcon was built from.
"""

from PyQt5 import QtSvg
from PyQt5.QtGui import QIcon, QPixmap, QPainter
from PyQt5.QtCore import QByteArray, Qt, QRectF

SMALL_ICON_PX = 18   # at or below this size the background blob is dropped

# fmt: off
HUES = {
    "navy":  ("#e5ebf3", "#9fb6d4", "#1f3a5f"),
    "sky":   ("#e3f0f7", "#9cc9df", "#1d5c78"),
    "sage":  ("#e3f1e6", "#a5d3b0", "#2f6b3d"),
    "rose":  ("#fbe6ea", "#f0a9b6", "#b03548"),
    "amber": ("#fbf1d9", "#f1cf7d", "#8a5a08"),
    "lilac": ("#ece8f8", "#bdb2e6", "#4a3c8c"),
}

BLOB = ("M12 1.8 C17.6 1.4 22.6 5.4 22.2 11.6 C21.8 17.6 17.8 22.4 11.6 22.2 "
        "C5.6 22 1.6 17.6 1.8 11.8 C2 6 6.4 2.2 12 1.8 Z")

# (hue, [(kind, role, *geometry)])   roles: shape (mid fill), shape2 (deep fill), line (deep stroke)
ICONS = {
    "folder": ("amber", [("path", "shape", "M3.8 7.4 C3.8 6.2 4.6 5.4 5.8 5.4 L9.2 5.4 L11 7.4 L18.2 7.4 C19.4 7.4 20.2 8.2 20.2 9.4 L20.2 16.8 C20.2 18 19.4 18.8 18.2 18.8 L5.8 18.8 C4.6 18.8 3.8 18 3.8 16.8 Z"),
                         ("path", "line", "M3.8 10.4 L20.2 10.4")]),
    "download": ("navy", [("path", "line", "M12 4.6 L12 14.6 M8.2 11 L12 14.8 L15.8 11"), ("path", "line", "M5 17 L5 18 C5 19 5.8 19.6 6.8 19.6 L17.2 19.6 C18.2 19.6 19 19 19 18 L19 17")]),
    "refresh": ("navy", [("path", "line", "M18.4 12 A6.4 6.4 0 1 1 16.4 7.4 M16.8 4.4 L16.8 7.8 L13.4 7.8")]),
    "play": ("sage", [("circle", "shape", 12, 12, 8.6),
                      ("path", "shape2", "M10.2 8.6 C10.2 8 10.8 7.7 11.3 8 L15.6 11.2 C16.1 11.6 16.1 12.4 15.6 12.8 L11.3 16 C10.8 16.3 10.2 16 10.2 15.4 Z")]),
    "stop": ("rose", [("circle", "shape", 12, 12, 8.6), ("rect", "shape2", 8.8, 8.8, 6.4, 6.4, 1.8)]),
    "search": ("navy", [("circle", "shape", 10.6, 10.6, 6), ("path", "line", "M15.2 15.2 L19.6 19.6")]),
    "check": ("sage", [("circle", "shape", 12, 12, 8.6), ("path", "line", "M8 12.2 L10.9 15 L16 9.4")]),
    "warning": ("amber", [("path", "shape", "M12 4 L20.5 19 L3.5 19 Z"), ("path", "line", "M12 9.5 L12 13.4"), ("circle", "shape2", 12, 16.2, 1.2)]),
    "bell": ("navy", [("path", "shape", "M12 4.4 C8.8 4.4 7.4 7 7.4 10.4 C7.4 13.4 6.2 15 5.4 15.8 L18.6 15.8 C17.8 15 16.6 13.4 16.6 10.4 C16.6 7 15.2 4.4 12 4.4 Z"),
                      ("path", "line", "M10.2 18.4 C10.6 19.4 11.2 19.8 12 19.8 C12.8 19.8 13.4 19.4 13.8 18.4"),
                      ("circle", "accent", 17.4, 5.6, 2.6)]),
    "logout": ("rose", [("path", "line", "M10 4.5 L6 4.5 C5 4.5 4.5 5 4.5 6 L4.5 18 C4.5 19 5 19.5 6 19.5 L10 19.5"),
                        ("path", "line", "M14.4 8 L18.4 12 L14.4 16 M18.4 12 L9 12")]),
    "arrow_left": ("navy", [("path", "line", "M14.5 6.5 L9 12 L14.5 17.5")]),
    "arrow_right": ("navy", [("path", "line", "M9.5 6.5 L15 12 L9.5 17.5")]),
    "eye": ("navy", [("path", "shape", "M2.5 12 C5.4 6.6 18.6 6.6 21.5 12 C18.6 17.4 5.4 17.4 2.5 12 Z"), ("circle", "shape2", 12, 12, 3.4)]),
    "eye_off": ("navy", [("path", "shape", "M2.5 12 C5.4 6.6 18.6 6.6 21.5 12 C18.6 17.4 5.4 17.4 2.5 12 Z"), ("circle", "shape2", 12, 12, 3.2),
                         ("path", "line", "M4.5 4.5 L19.5 19.5")]),
    "filter": ("navy", [("path", "line", "M4.5 7 L19.5 7 M7.5 12 L16.5 12 M10.5 17 L13.5 17")]),
    "box": ("sky", [("rect", "shape", 5, 5, 14, 14, 3), ("circle", "shape2", 5, 5, 2), ("circle", "shape2", 19, 5, 2),
                    ("circle", "shape2", 5, 19, 2), ("circle", "shape2", 19, 19, 2)]),
    "polygon": ("rose", [("path", "shape", "M12 4 L19.6 9.4 L17 19 L7 19 L4.4 9.4 Z"), ("circle", "shape2", 12, 4, 1.9), ("circle", "shape2", 19.6, 9.4, 1.9),
                         ("circle", "shape2", 17, 19, 1.9), ("circle", "shape2", 7, 19, 1.9), ("circle", "shape2", 4.4, 9.4, 1.9)]),
    "brain": ("lilac", [("path", "shape", "M11.4 5.6 C8.4 5.6 6.8 7.4 6.8 9.4 C5.4 10.2 5.4 12.6 6.3 13.6 C5.8 15.6 7.2 18 9.6 18 C10.5 18 11.1 17.6 11.4 17.2 Z"),
                        ("path", "shape", "M12.6 5.6 C15.6 5.6 17.2 7.4 17.2 9.4 C18.6 10.2 18.6 12.6 17.7 13.6 C18.2 15.6 16.8 18 14.4 18 C13.5 18 12.9 17.6 12.6 17.2 Z"),
                        ("path", "line", "M8.6 10.6 Q10 11.2 10.6 10 M15.4 10.6 Q14 11.2 13.4 10")]),
    "sparkle": ("lilac", [("path", "shape2", "M12 4.5 L13.7 10.3 L19.5 12 L13.7 13.7 L12 19.5 L10.3 13.7 L4.5 12 L10.3 10.3 Z")]),
    "zen_stones": ("navy", [("path", "shape", "M4 18 C4 15.6 8 14.4 12 14.4 C16 14.4 20 15.6 20 18 C20 20.4 16 21.4 12 21.4 C8 21.4 4 20.4 4 18 Z"),
                            ("path", "shape", "M6.6 12 C6.6 10.4 9.2 9.4 12 9.4 C14.8 9.4 17.4 10.4 17.4 12 C17.4 13.6 14.8 14.4 12 14.4 C9.2 14.4 6.6 13.6 6.6 12 Z"),
                            ("circle", "shape2", 12, 6.6, 3)]),
    "step_cvat": ("navy", [("path", "shape", "M6.6 15.6 C4.6 15.6 3.6 13.8 4.4 12.2 C5 11 6.2 10.6 7 10.8 C7.6 7.8 11.8 6.6 14.2 9 C16.4 8.2 18.8 9.8 18.6 12.2 C20.4 12.8 20 15.6 17.8 15.6 Z"),
                           ("path", "line", "M12 11.6 L12 19.4 M9.4 17 L12 19.6 L14.6 17")]),
    "step_merge": ("amber", [("path", "line", "M8.6 7 C12 7 12.4 12 15 12 M8.6 17 C12 17 12.4 12 15 12"),
                             ("circle", "shape", 6.8, 7, 2.6), ("circle", "shape", 6.8, 17, 2.6), ("circle", "shape2", 17, 12, 3.2)]),
    "step_convert": ("lilac", [("path", "line", "M5.6 18.4 L14 10"),
                               ("path", "shape2", "M15.6 4.6 L16.5 7.5 L19.4 8.4 L16.5 9.3 L15.6 12.2 L14.7 9.3 L11.8 8.4 L14.7 7.5 Z")]),
    "step_split": ("sky", [("rect", "shape2", 5, 7, 4, 12, 2), ("rect", "shape", 10.2, 9.5, 4, 9.5, 2), ("rect", "shape", 15.4, 12, 4, 7, 2)]),
    "step_train": ("navy", [("path", "shape", "M12 3.6 C15.4 6.6 16.8 10.6 15.8 15 L8.2 15 C7.2 10.6 8.6 6.6 12 3.6 Z"),
                            ("circle", "shape2", 12, 9.4, 1.9),
                            ("path", "shape2", "M8.6 12.6 L5.8 15.8 L8.6 15.8 Z M15.4 12.6 L18.2 15.8 L15.4 15.8 Z M10.2 16.6 Q12 21 12 21 Q12 21 13.8 16.6")]),
    "step_runs": ("sage", [("path", "shape", "M4.6 8.4 C4.6 7.4 5.2 6.8 6.2 6.8 L9.4 6.8 L11 8.6 L17.8 8.6 C18.8 8.6 19.4 9.2 19.4 10.2 L19.4 16.8 C19.4 17.8 18.8 18.4 17.8 18.4 L6.2 18.4 C5.2 18.4 4.6 17.8 4.6 16.8 Z"),
                           ("path", "line", "M9.6 13.4 L11.4 15.2 L14.6 12")]),
    "step_test": ("lilac", [("circle", "shape", 11, 11, 5.6), ("path", "line", "M15.2 15.2 L19 19 M8.6 11.2 L10.4 13 L13.4 9.8")]),
    "image": ("sky", [("rect", "shape", 3.6, 4.6, 16.8, 14.8, 3.6), ("circle", "shape2", 8.4, 9, 1.7), ("path", "line", "M4 17 L9 12.4 L13 15.8 L15.8 13.2 L20 17")]),
    "video": ("rose", [("rect", "shape", 3.5, 6, 12, 12, 3), ("path", "shape2", "M15.5 10.4 L20.5 7.6 L20.5 16.4 L15.5 13.6 Z")]),
    "chart": ("navy", [("rect", "shape", 3.6, 12, 3.2, 8, 1.6), ("rect", "shape2", 8.2, 8, 3.2, 12, 1.6), ("rect", "shape", 12.8, 4.6, 3.2, 15.4, 1.6), ("rect", "shape2", 17.4, 9.6, 3.2, 10.4, 1.6)]),
}


def _el(kind, geo, attrs):
    if kind == "path":
        return f'<path d="{geo[0]}" {attrs}/>'
    if kind == "circle":
        return f'<circle cx="{geo[0]}" cy="{geo[1]}" r="{geo[2]}" {attrs}/>'
    x, y, w, h, rx = geo
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" {attrs}/>'


# fmt: on


def build_svg(name: str, inverse: bool = False, with_blob: bool = True) -> str:
    """Builds the SVG markup for an icon. Unknown names fall back to 'sparkle'."""
    hue, els = ICONS.get(name, ICONS["sparkle"])
    light, mid, deep = HUES[hue]
    if inverse:
        light = mid = deep = "#ffffff"
    out = []
    if with_blob:
        out.append(f'<path d="{BLOB}" fill="{light}" fill-opacity="{0.16 if inverse else 1}"/>')
    for kind, role, *geo in els:
        if role == "line":
            out.append(_el(kind, geo, f'fill="none" stroke="{deep}" stroke-width="1.9" '
                                      'stroke-linecap="round" stroke-linejoin="round"'))
        elif role == "shape":
            out.append(_el(kind, geo, f'fill="{mid}" fill-opacity="{0.55 if inverse else 1}"'))
            if not inverse:
                out.append(_el(kind, geo, f'fill="{deep}" fill-opacity="0.2"'))
        elif role == "accent":
            out.append(_el(kind, geo, 'fill="#e5486f" stroke="#ffffff" stroke-width="1.2"'))
        else:  # shape2
            out.append(_el(kind, geo, f'fill="{deep}"'))
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">'
            + "".join(out) + "</svg>")


# Raw SVG strings (default look) — kept for code that reads SVG_DEFS directly.
SVG_DEFS = {name: build_svg(name) for name in ICONS}

# =====================================================================
# Render helpers with caching
# =====================================================================
_PIXMAP_CACHE: dict[tuple, QPixmap] = {}
_ICON_CACHE: dict[tuple, QIcon] = {}
_ICON_NAMES: dict[int, tuple[str, int]] = {}   # QIcon.cacheKey() -> (name, size)


def get_svg_raw(name: str, inverse: bool = False, with_blob: bool = True) -> str:
    """Returns raw SVG markup for an icon."""
    return build_svg(name, inverse=inverse, with_blob=with_blob)


def get_svg_pixmap(name: str, width: int = 24, height: int = 24, inverse: bool = False) -> QPixmap:
    """Renders an icon to a crisp (2x) QPixmap. Small sizes drop the blob background."""
    with_blob = max(width, height) > SMALL_ICON_PX
    key = (name, width, height, inverse, with_blob)
    if key in _PIXMAP_CACHE:
        return _PIXMAP_CACHE[key]

    renderer = QtSvg.QSvgRenderer(QByteArray(build_svg(name, inverse, with_blob).encode("utf-8")))
    pix = QPixmap(width * 2, height * 2)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
    renderer.render(painter, QRectF(0, 0, width * 2, height * 2))
    painter.end()
    pix.setDevicePixelRatio(2.0)

    _PIXMAP_CACHE[key] = pix
    return pix


def get_svg_icon(name: str, size: int = 22, inverse: bool = False) -> QIcon:
    """Returns a cached QIcon. Use inverse=True on dark buttons."""
    key = (name, size, inverse)
    if key in _ICON_CACHE:
        return _ICON_CACHE[key]
    icon = QIcon(get_svg_pixmap(name, size, size, inverse))
    _ICON_CACHE[key] = icon
    _ICON_NAMES[icon.cacheKey()] = (name, size)
    return icon


def icon_name_for(icon: QIcon):
    """Returns (name, size) for a QIcon created by get_svg_icon, or None."""
    return _ICON_NAMES.get(icon.cacheKey())