"""
SVG icon library for CVAT to YOLO Studio — "Tinted Tiles" style (Console theme).

Style rules
- Thin, round-cap line glyphs on a 24 px grid.
- At sizes above SMALL_ICON_PX the glyph sits on a soft rounded tile in the light
  tone of its hue; the glyph itself uses the deep tone.
- Pipeline steps each have their own hue so they can be told apart at a glance:
  download = indigo, convert = violet, split = teal, train = amber, test = rose,
  auto-label = green. Generic actions use neutral grey; colour only where it means
  something (green = run / ok, rose = stop / leave, amber = warning).
- Small sizes (<= 18 px) drop the tile so the glyph stays crisp on buttons.
- inverse=True gives a white version for dark (accent / red) buttons.

API (unchanged): get_svg_icon(name, size), get_svg_pixmap(name, w, h), get_svg_raw(name),
SVG_DEFS, icon_name_for(icon), inverse= flag.
"""

from PyQt5 import QtSvg
from PyQt5.QtGui import QIcon, QPixmap, QPainter
from PyQt5.QtCore import QByteArray, Qt, QRectF

SMALL_ICON_PX = 18   # at or below this size the tile is dropped

# hue -> (tile background, glyph colour)
HUES = {
    "indigo": ("#eef0ff", "#3b4ce0"),
    "violet": ("#f4f0ff", "#6941c6"),
    "teal":   ("#e6f6f4", "#0e7c70"),
    "amber":  ("#fff6e5", "#b54708"),
    "rose":   ("#fff0f3", "#c01048"),
    "green":  ("#ecfdf3", "#067647"),
    "gray":   ("#f2f4f7", "#344054"),
}

# Hues for the six pipeline steps, in order (also used by the sidebar).
STEP_HUES = ("indigo", "violet", "teal", "amber", "rose", "green")
STEP_ICONS = ("step_cvat", "step_convert", "step_split", "step_train", "step_test", "step_autolabel")


def _c(cx, cy, r):
    """Circle as a path (QtSvg handles a single path list more predictably)."""
    return f"M{cx - r} {cy}a{r} {r} 0 1 0 {2 * r} 0a{r} {r} 0 1 0 {-2 * r} 0"


# fmt: off
# name: (hue, stroke path, optional filled path)
_DOWNLOAD = "M12 3v11M7.5 9.5L12 14l4.5-4.5M4 15v3.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V15"
_UPLOAD = "M12 15V4M7.5 8.5L12 4l4.5 4.5M4 15v3.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V15"
_FOLDER = "M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"
_EYE = "M2.5 12s3.5-6.5 9.5-6.5S21.5 12 21.5 12s-3.5 6.5-9.5 6.5S2.5 12 2.5 12z" + _c(12, 12, 3)
_BELL = "M6 16v-5a6 6 0 1 1 12 0v5l1.5 2h-15zM10 20.5a2.2 2.2 0 0 0 4 0"
_TAG = "M3 4v7l9.5 9.5a1.5 1.5 0 0 0 2 0l6-6a1.5 1.5 0 0 0 0-2L11 3H4a1 1 0 0 0-1 1zM7.5 7.5h.01"
_SPARKLE = "M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"

ICONS = {
    # --- pipeline steps ---------------------------------------------------
    "step_cvat":      ("indigo", _DOWNLOAD, None),
    "download":       ("indigo", _DOWNLOAD, None),
    "step_merge":     ("indigo", _c(6, 6, 2) + _c(6, 18, 2) + "M8 6c4.5 0 4.5 6 8.5 6M8 18c4.5 0 4.5-6 8.5-6" + _c(18.5, 12, 2.2), None),
    "step_convert":   ("violet", "M4 4h6v6H4zM14 14h6v6h-6zM10 7h5a2 2 0 0 1 2 2v3M15 10l2 2 2-2M14 17H9a2 2 0 0 1-2-2v-3M9 14l-2-2-2 2", None),
    "step_split":     ("teal",   "M3 12h5M8 12c3 0 4-6 8-6h4M8 12h12M8 12c3 0 4 6 8 6h4" + _c(5, 12, 2), None),
    "step_train":     ("amber",  "M7 7h10v10H7zM10 3v4M14 3v4M10 17v4M14 17v4M3 10h4M3 14h4M17 10h4M17 14h4M10 10h4v4h-4z", None),
    "step_runs":      ("green",  _FOLDER + "M9.5 13l2 2 3.5-3.5", None),
    "step_test":      ("rose",   _c(12, 12, 9) + _c(12, 12, 4) + "M12 1.5v4M12 18.5v4M1.5 12h4M18.5 12h4", None),
    "step_autolabel": ("green",  _TAG, None),

    # --- semantic actions ------------------------------------------------
    "play":    ("green", "M7 4.5v15l12.5-7.5z", "M7 4.5v15l12.5-7.5z"),
    "stop":    ("rose",  "M6.5 6.5h11v11h-11z", "M6.5 6.5h11v11h-11z"),
    "check":   ("green", "M5 12.5l4.5 4.5L19 7.5", None),
    "warning": ("amber", "M12 3.5L21.5 20h-19zM12 10v4.5M12 17.2h.01", None),
    "logout":  ("rose",  "M9 4H6a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h3M14 16l4-4-4-4M18 12H8", None),
    "upload":  ("green", _UPLOAD, None),

    # --- annotation types / AI -------------------------------------------
    "box":      ("indigo", "M6 6h12v12H6z" + _c(6, 6, 1.6) + _c(18, 6, 1.6) + _c(6, 18, 1.6) + _c(18, 18, 1.6), None),
    "polygon":  ("violet", "M12 3.5l8 6-3 10.5H7L4 9.5z" + _c(12, 3.5, 1.4) + _c(20, 9.5, 1.4) + _c(4, 9.5, 1.4), None),
    "brain":    ("amber",  "M9 4a3 3 0 0 0-3 3v.5A3 3 0 0 0 4 10.5a3 3 0 0 0 1 2.2A3 3 0 0 0 6 18a3 3 0 0 0 3 2h1V4zM15 4a3 3 0 0 1 3 3v.5a3 3 0 0 1 2 3a3 3 0 0 1-1 2.2A3 3 0 0 1 18 18a3 3 0 0 1-3 2h-1V4z", None),
    "sparkle":  ("violet", _SPARKLE + "M18.5 15.5l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8z", None),

    # --- neutral UI ------------------------------------------------------
    "folder":      ("gray", _FOLDER, None),
    "refresh":     ("gray", "M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7", None),
    "search":      ("gray", _c(11, 11, 7) + "M20 20l-4-4", None),
    "bell":        ("gray", _BELL, None),
    "bell_plain":  ("gray", _BELL, None),
    "arrow_left":  ("gray", "M15 6l-6 6 6 6", None),
    "arrow_right": ("gray", "M9 6l6 6-6 6", None),
    "eye":         ("gray", _EYE, None),
    "eye_off":     ("gray", _EYE + "M4 4l16 16", None),
    "filter":      ("gray", "M4 6h16M7 12h10M10 18h4", None),
    "image":       ("gray", "M3 5h18v14H3zM3 16l5-5 4 4 3-3 6 6" + _c(15.5, 9, 1.5), None),
    "video":       ("gray", "M3 6h12v12H3zM15 10l6-3.5v11L15 14", None),
    "chart":       ("gray", "M3 20h18M6 20v-7M11 20V5M16 20V10", None),
    "fullscreen":  ("gray", "M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5", None),
    "settings":    ("gray", "M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M16 4v4M10 10v4M18 16v4", None),
    "user":        ("gray", _c(12, 8, 4) + "M4 20c1.5-3.5 4.5-5 8-5s6.5 1.5 8 5", None),
    "zen_stones":  ("gray", "M4 18c0-2 3.6-3 8-3s8 1 8 3-3.6 3-8 3-8-1-8-3zM7 12c0-1.5 2.2-2.5 5-2.5s5 1 5 2.5-2.2 2.5-5 2.5-5-1-5-2.5z" + _c(12, 6, 2.5), None),
}
# fmt: on

TILE_SCALE = 0.62          # glyph size inside a tile, relative to the tile
TILE_RADIUS = 6.5          # in 24-grid units (≈ 27% of the tile)


def build_svg(name: str, inverse: bool = False, with_blob: bool = True) -> str:
    """Builds the SVG markup for an icon. Unknown names fall back to 'sparkle'.

    with_blob keeps its old name for compatibility: it now means "draw the tile".
    """
    hue, stroke_d, fill_d = ICONS.get(name, ICONS["sparkle"])
    tile, ink = HUES[hue]
    if inverse:
        tile, ink = "#ffffff", "#ffffff"

    out = []
    if with_blob:
        out.append(f'<rect x="0" y="0" width="24" height="24" rx="{TILE_RADIUS}" ry="{TILE_RADIUS}" '
                   f'fill="{tile}" fill-opacity="{0.18 if inverse else 1}"/>')
        off = 12 * (1 - TILE_SCALE)
        out.append(f'<g transform="translate({off:.2f} {off:.2f}) scale({TILE_SCALE})">')
        width = 2.3
    else:
        out.append("<g>")
        width = 1.9

    if fill_d:
        out.append(f'<path d="{fill_d}" fill="{ink}" stroke="none"/>')
    out.append(f'<path d="{stroke_d}" fill="none" stroke="{ink}" stroke-width="{width}" '
               'stroke-linecap="round" stroke-linejoin="round"/>')
    out.append("</g>")
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">'
            + "".join(out) + "</svg>")


def hue_of(name: str) -> tuple[str, str]:
    """(tile, ink) colours for an icon — handy for matching text or borders to an icon."""
    return HUES[ICONS.get(name, ICONS["sparkle"])[0]]


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


def get_svg_pixmap(name: str, width: int = 24, height: int = 24, inverse: bool = False,
                   tile: bool | None = None) -> QPixmap:
    """Renders an icon to a crisp (2x) QPixmap. Small sizes drop the tile unless tile=True."""
    with_tile = (max(width, height) > SMALL_ICON_PX) if tile is None else tile
    key = (name, width, height, inverse, with_tile)
    if key in _PIXMAP_CACHE:
        return _PIXMAP_CACHE[key]

    renderer = QtSvg.QSvgRenderer(QByteArray(build_svg(name, inverse, with_tile).encode("utf-8")))
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


def get_svg_icon(name: str, size: int = 22, inverse: bool = False, tile: bool | None = None) -> QIcon:
    """Returns a cached QIcon. Use inverse=True on dark buttons."""
    key = (name, size, inverse, tile)
    if key in _ICON_CACHE:
        return _ICON_CACHE[key]
    icon = QIcon(get_svg_pixmap(name, size, size, inverse, tile))
    _ICON_CACHE[key] = icon
    _ICON_NAMES[icon.cacheKey()] = (name, size)
    return icon


def icon_name_for(icon: QIcon):
    """Returns (name, size) for a QIcon created by get_svg_icon, or None."""
    return _ICON_NAMES.get(icon.cacheKey())