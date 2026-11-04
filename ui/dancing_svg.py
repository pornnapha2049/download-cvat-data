"""
Animated "Idea Bulb" cat mascot (sticker style) for the Navy Enterprise theme.

Design
- Bean-shaped head with chunky ears, flat navy silhouette, white facial features
- Emote symbols drawn around the head like a sticker sheet (… dots, light bulb, ?, sweat, zZ …)

States (public API kept from the old mascot, plus two new ones)
- idle         : calm "thinking" face, three dots bob one after another
- downloading / processing : looks up at an unlit bulb with a "?", bulb flickers
- training     : determined face, sweat drop, unlit bulb
- success      : bulb lights up with pulsing rays, sparkly eyes, open smile
- error  (new) : crying face with tear streams (auto-returns to idle after ~8 s)
- sleep  (new) : entered automatically after 2 minutes of idle, zZ floats up
Click the cat → happy hearts for a moment.

Also exposes get_cat_pixmap() (static face badge for cards) and AnimatedWavePill.
"""

import math
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtSvg import QSvgWidget, QSvgRenderer
from PyQt5.QtCore import QTimer, QByteArray, Qt, QRectF
from PyQt5.QtGui import QMouseEvent, QPixmap, QPainter

# ---------------------------------------------------------------------
# Colours (match ui.theme.PALETTE — Navy Enterprise)
# ---------------------------------------------------------------------
CAT_NAVY = "#1f3a5f"
CAT_LIGHT = "#c9d6e8"
WHITE = "#ffffff"
ACCENT = "#f2a7cf"       # blush / hearts
BULB_ON = "#f6e7a1"      # lit bulb
TEAR = "#a9c8f0"
ANGER = "#e5486f"

CAT_VARIANTS = {
    # name: (body, accent)
    "navy": (CAT_NAVY, ACCENT),
    "light": (CAT_LIGHT, ACCENT),
    # old names kept for compatibility with existing calls
    "teal": (CAT_NAVY, ACCENT),
    "pink": (CAT_LIGHT, ACCENT),
}

HEART = "M 12 6 C 12 3 8 0 5 3 C 0 8 12 16 12 16 C 12 16 24 8 19 3 C 16 0 12 3 12 6 Z"

# Head geometry (viewBox 150 x 150)
HX, HY, HW, HH = 70.0, 94.0, 48.0, 38.0       # head centre / half-width / half-height
FX, FY = 70.0, 92.0                           # face anchor
EX, EY = 120.0, 34.0                          # emote anchor (top-right)


# ---------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------
def _bean(cx, cy, w, h, b, ear_l=0.0, ear_r=0.0):
    ears = (f'<path d="M {cx-w*0.92} {cy-h*0.15} L {cx-w*0.8} {cy-h*1.25+ear_l} L {cx-w*0.28} {cy-h*0.78} Z" '
            f'fill="{b}" stroke="{b}" stroke-width="8" stroke-linejoin="round"/>'
            f'<path d="M {cx+w*0.92} {cy-h*0.15} L {cx+w*0.8} {cy-h*1.25+ear_r} L {cx+w*0.28} {cy-h*0.78} Z" '
            f'fill="{b}" stroke="{b}" stroke-width="8" stroke-linejoin="round"/>')
    head = (f'<path d="M {cx-w} {cy+h*0.15} C {cx-w} {cy-h*0.62}, {cx-w*0.58} {cy-h}, {cx} {cy-h} '
            f'C {cx+w*0.58} {cy-h}, {cx+w} {cy-h*0.62}, {cx+w} {cy+h*0.15} '
            f'C {cx+w} {cy+h*0.82}, {cx+w*0.6} {cy+h}, {cx} {cy+h} '
            f'C {cx-w*0.6} {cy+h}, {cx-w} {cy+h*0.82}, {cx-w} {cy+h*0.15} Z" fill="{b}"/>')
    return ears + head


def _whiskers(b):
    out = ""
    for dy, t in ((0, -4), (7, 3)):
        out += (f'<path d="M {HX-46} {HY+8+dy} L {HX-62} {HY+8+dy+t}" stroke="{b}" stroke-width="2.2" stroke-linecap="round"/>'
                f'<path d="M {HX+46} {HY+8+dy} L {HX+62} {HY+8+dy+t}" stroke="{b}" stroke-width="2.2" stroke-linecap="round"/>')
    return out


def _spark(x, y, r, c, op=1.0):
    return (f'<path d="M {x} {y-r} Q {x+r*0.18} {y-r*0.18} {x+r} {y} Q {x+r*0.18} {y+r*0.18} {x} {y+r} '
            f'Q {x-r*0.18} {y+r*0.18} {x-r} {y} Q {x-r*0.18} {y-r*0.18} {x} {y-r} Z" fill="{c}" opacity="{op:.2f}"/>')


def _heart(x, y, sc, c, op=1.0):
    return f'<g transform="translate({x-12*sc:.1f},{y-8*sc:.1f}) scale({sc:.2f})"><path d="{HEART}" fill="{c}" opacity="{op:.2f}"/></g>'


def _drop(x, y, s, b):
    return (f'<path d="M {x} {y} C {x-5*s} {y+7*s}, {x-5*s} {y+12*s}, {x} {y+12*s} C {x+5*s} {y+12*s}, {x+5*s} {y+7*s}, {x} {y} Z" '
            f'fill="{WHITE}" stroke="{b}" stroke-width="2.4" stroke-linejoin="round"/>')


def _text(x, y, t, size, b, op=1.0):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="Segoe UI, Arial" font-weight="800" '
            f'font-size="{size}" fill="{b}" opacity="{op:.2f}">{t}</text>')


def _bulb(b, lit, glow=1.0, flicker=False):
    bx, by = 126.0, 30.0
    s = ""
    if lit:
        for ang, ln in ((-60, 10), (-20, 12), (20, 12), (60, 10), (-100, 9)):
            r0, r1 = 19, 19 + ln * 0.7 * glow
            dx, dy = math.cos(math.radians(ang - 90)), math.sin(math.radians(ang - 90))
            s += (f'<path d="M {bx+dx*r0:.1f} {by+dy*r0:.1f} L {bx+dx*r1:.1f} {by+dy*r1:.1f}" '
                  f'stroke="{b}" stroke-width="3" stroke-linecap="round"/>')
    fill = BULB_ON if lit else (WHITE if not flicker else "#eef1f5")
    s += (f'<path d="M {bx} {by-16} C {bx+12} {by-16} {bx+15} {by-4} {bx+9} {by+4} C {bx+6} {by+8} {bx+6} {by+10} {bx+6} {by+12} '
          f'L {bx-6} {by+12} C {bx-6} {by+10} {bx-6} {by+8} {bx-9} {by+4} C {bx-15} {by-4} {bx-12} {by-16} {bx} {by-16} Z" '
          f'fill="{fill}" stroke="{b}" stroke-width="3" stroke-linejoin="round"/>')
    s += f'<rect x="{bx-6}" y="{by+13}" width="12" height="7" rx="2.5" fill="{b}"/>'
    return s


# ---------------------------------------------------------------------
# Faces (eyes + mouth) — relative to FX, FY
# ---------------------------------------------------------------------
def _ring(x, y, b, rx=9.0, ry=11.0, look_x=0.0, look_y=0.0, sparkle=False):
    px, py = x + look_x * 3, y + 1 + look_y * 3
    s = (f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" fill="{WHITE}"/>'
         f'<ellipse cx="{px:.1f}" cy="{py:.1f}" rx="{rx*0.55:.1f}" ry="{ry*0.6:.1f}" fill="{b}"/>'
         f'<circle cx="{px-2:.1f}" cy="{py-2.5:.1f}" r="{rx*0.22:.1f}" fill="{WHITE}"/>')
    if sparkle:
        s += _spark(px - 2, py - 3, 3.4, WHITE)
    return s


def _face(expr, b, a, blink=False, look_x=0.0, look_y=0.0):
    L, R, my = FX - 19, FX + 19, FY + 16
    if blink and expr in ("open", "busy", "working", "success"):
        return (f'<path d="M {L-8} {FY} Q {L} {FY+5} {L+8} {FY} M {R-8} {FY} Q {R} {FY+5} {R+8} {FY}" '
                f'stroke="{WHITE}" stroke-width="3.4" fill="none" stroke-linecap="round"/>'
                f'<path d="M {FX-6} {my} L {FX+6} {my}" stroke="{WHITE}" stroke-width="2.6" stroke-linecap="round"/>')
    if expr == "idle":            # thinking: flat eyes
        return (f'<path d="M {L-7} {FY} L {L+7} {FY} M {R-7} {FY} L {R+7} {FY}" stroke="{WHITE}" stroke-width="4" stroke-linecap="round"/>'
                f'<path d="M {FX-6} {my} L {FX+6} {my}" stroke="{WHITE}" stroke-width="2.6" stroke-linecap="round"/>')
    if expr == "open":            # friendly default (cards / click)
        return (_ring(L, FY, b, look_x=look_x) + _ring(R, FY, b, look_x=look_x) +
                f'<path d="M {FX-6} {my} Q {FX-3} {my+4} {FX} {my+0.5} Q {FX+3} {my+4} {FX+6} {my}" '
                f'stroke="{WHITE}" stroke-width="2.4" fill="none" stroke-linecap="round"/>')
    if expr == "busy":            # looking up at the bulb, unsure
        return (_ring(L, FY, b, 8, 10, 0.9 + look_x, -0.6 + look_y) + _ring(R, FY, b, 8, 10, 0.9 + look_x, -0.6 + look_y) +
                f'<path d="M {FX-6} {my} Q {FX} {my-4} {FX+6} {my}" stroke="{WHITE}" stroke-width="2.6" fill="none" stroke-linecap="round"/>')
    if expr == "working":         # determined brows
        return (_ring(L, FY, b, look_x=0.6) + _ring(R, FY, b, look_x=0.6) +
                f'<path d="M {L-9} {FY-15} L {L+7} {FY-10} M {R+9} {FY-15} L {R-7} {FY-10}" stroke="{WHITE}" stroke-width="2.8" stroke-linecap="round"/>'
                f'<path d="M {FX-6} {my} L {FX+6} {my}" stroke="{WHITE}" stroke-width="2.6" stroke-linecap="round"/>')
    if expr == "success":
        return (_ring(L, FY, b, look_x=0.6, sparkle=True) + _ring(R, FY, b, look_x=0.6, sparkle=True) +
                f'<path d="M {FX-8} {my-2} Q {FX} {my+10} {FX+8} {my-2} Z" fill="{WHITE}"/>'
                f'<ellipse cx="{L-10}" cy="{FY+12}" rx="5.5" ry="2.8" fill="{a}" opacity="0.85"/>'
                f'<ellipse cx="{R+10}" cy="{FY+12}" rx="5.5" ry="2.8" fill="{a}" opacity="0.85"/>')
    if expr == "happy":           # ^^ (click)
        return (f'<path d="M {L-8} {FY+3} Q {L} {FY-8} {L+8} {FY+3} M {R-8} {FY+3} Q {R} {FY-8} {R+8} {FY+3}" '
                f'stroke="{WHITE}" stroke-width="4" fill="none" stroke-linecap="round"/>'
                f'<path d="M {FX-7} {my-2} Q {FX} {my+9} {FX+7} {my-2} Z" fill="{WHITE}"/>'
                f'<ellipse cx="{L-10}" cy="{FY+12}" rx="5.5" ry="2.8" fill="{a}" opacity="0.85"/>'
                f'<ellipse cx="{R+10}" cy="{FY+12}" rx="5.5" ry="2.8" fill="{a}" opacity="0.85"/>')
    if expr == "error":           # crying
        s = _ring(L, FY, b) + _ring(R, FY, b)
        for x in (L, R):
            s += (f'<path d="M {x-3} {FY+11} q -3 6 0 12 q 3 6 0 12" stroke="{TEAR}" stroke-width="3.2" '
                  f'fill="none" stroke-linecap="round"/>')
        s += f'<path d="M {FX-6} {my+3} Q {FX} {my-4} {FX+6} {my+3}" stroke="{WHITE}" stroke-width="2.6" fill="none" stroke-linecap="round"/>'
        return s
    if expr == "sleep":
        return (f'<path d="M {L-8} {FY} Q {L} {FY+6} {L+8} {FY} M {R-8} {FY} Q {R} {FY+6} {R+8} {FY}" '
                f'stroke="{WHITE}" stroke-width="3.4" fill="none" stroke-linecap="round"/>'
                f'<ellipse cx="{FX}" cy="{my}" rx="2.8" ry="2" fill="{a}"/>')
    return _face("open", b, a)


def build_cat_svg(state: str = "idle", t: float = 0.0, *, body: str = CAT_NAVY, accent: str = ACCENT,
                  blink: bool = False, bounce: float = 0.0, size: int = 150) -> str:
    """One frame of the Idea Bulb cat. `t` is time in seconds-ish for the little animations."""
    b, a = body, accent
    emote, expr, look_x, look_y = "", state, 0.0, 0.0
    bob = math.sin(t * 2.0) * 1.6
    ear_l = ear_r = 0.0

    if state == "idle":
        for i in range(3):
            up = max(0.0, math.sin(t * 3.0 - i * 0.9)) * 4.0
            emote += f'<circle cx="{EX-22+i*13}" cy="{EY+6-i*5-up:.1f}" r="{4.5+i*0.6}" fill="{b}"/>'
        ear_l = max(0.0, math.sin(t * 1.3)) ** 10 * 4.0
    elif state == "busy":
        flicker = int(t * 6) % 7 == 0
        emote += _bulb(b, lit=False, flicker=flicker) + _text(EX - 22, EY - 2 + math.sin(t * 3) * 2, "?", 20, b)
        look_x = math.sin(t * 1.5) * 0.25
        bob = math.sin(t * 4.0) * 2.0
    elif state == "working":
        emote += _bulb(b, lit=False) + _drop(16, 50 + (t * 8) % 10, 1.0, b)
        bob = math.sin(t * 3.5) * 1.8
    elif state == "success":
        glow = 0.75 + 0.25 * math.sin(t * 6.0)
        emote += _bulb(b, lit=True, glow=glow)
        emote += _spark(26, 40, 7, b, 0.6 + 0.4 * math.sin(t * 5)) + _spark(16, 64, 4, b, 0.6 + 0.4 * math.cos(t * 5))
        bob = -abs(math.sin(t * 3.2)) * 6.0
        ear_l, ear_r = math.sin(t * 6) * 2.5, -math.sin(t * 6) * 2.5
    elif state == "happy":
        emote += _heart(EX, EY + 4 - (t * 6) % 6, 1.0, a) + _heart(EX + 16, EY + 22, 0.55, a, 0.8) + _heart(24, 44, 0.6, a, 0.8)
        bob = -abs(math.sin(t * 5.0)) * 5.0
    elif state == "error":
        emote += _drop(EX - 6, EY + 2, 1.1, b)
        bob = math.sin(t * 9.0) * 0.8
    elif state == "sleep":
        drift = (t * 6) % 12
        emote += (_text(EX - 4, EY + 20 - drift * 0.4, "z", 20, b) + _text(EX + 10, EY + 6 - drift * 0.4, "z", 15, b, 0.7)
                  + _text(EX + 21, EY - 6 - drift * 0.4, "z", 11, b, 0.5))
        bob = math.sin(t * 1.2) * 1.0

    face = _face(expr, b, a, blink=blink, look_x=look_x, look_y=look_y)
    head = _bean(HX, HY, HW, HH, b, ear_l, ear_r) + _whiskers(b) + face
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 150 150" width="{size}" height="{size}">'
            f'<ellipse cx="{HX}" cy="136" rx="{40 + min(bob, 0) * 0.8:.1f}" ry="4" fill="{b}" opacity="0.10"/>'
            f'{emote}<g transform="translate(0,{bob + bounce:.2f})">{head}</g></svg>')


def get_cat_pixmap(width: int = 56, variant: str = "navy", eye_mode: str = "open") -> QPixmap:
    """Static cat-face badge (head only) for cards and the Tip box.
    variant: 'navy' (on light backgrounds) or 'light' (on navy cards)."""
    body, accent = CAT_VARIANTS.get(variant, CAT_VARIANTS["navy"])
    expr = {"open": "open", "happy": "happy", "sleep": "sleep"}.get(eye_mode, "open")
    head = _bean(HX, HY, HW, HH, body) + _whiskers(body) + _face(expr, body if variant in ("navy", "teal") else CAT_NAVY, accent)
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="4 44 132 92">{head}</svg>'
    height = int(round(width * 92 / 132))
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pix = QPixmap(width * 2, height * 2)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing, True)
    renderer.render(painter, QRectF(0, 0, width * 2, height * 2))
    painter.end()
    pix.setDevicePixelRatio(2.0)
    return pix


class DancingSvgMascot(QWidget):
    """Animated Idea Bulb cat. Public API unchanged: set_state(), toggle_animation()."""

    STATE_IDLE = "idle"
    STATE_DOWNLOADING = "downloading"
    STATE_PROCESSING = "processing"
    STATE_TRAINING = "training"
    STATE_SUCCESS = "success"
    STATE_ERROR = "error"
    STATE_SLEEP = "sleep"

    _ALL_STATES = (STATE_IDLE, STATE_DOWNLOADING, STATE_PROCESSING, STATE_TRAINING,
                   STATE_SUCCESS, STATE_ERROR, STATE_SLEEP)
    # widget state -> drawing state
    _DRAW = {STATE_IDLE: "idle", STATE_DOWNLOADING: "busy", STATE_PROCESSING: "busy",
             STATE_TRAINING: "working", STATE_SUCCESS: "success", STATE_ERROR: "error", STATE_SLEEP: "sleep"}

    TICK_MS = 33
    SLEEP_AFTER_S = 120     # idle this long -> falls asleep
    ERROR_HOLD_S = 8        # crying face this long, then back to idle

    def __init__(self, size: int = 140, parent=None):
        super().__init__(parent)
        self.mascot_size = size
        self.state = self.STATE_IDLE
        self.frame_idx = 0
        self.state_frames = 0
        self.is_animated = True
        self.bounce_timer = 0

        self.setFixedSize(size, size)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("น้องแมวไอเดีย • คลิกเพื่อเล่นด้วยกันได้นะ 🐾")
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.svg_widget = QSvgWidget(self)
        self.svg_widget.setFixedSize(size, size)
        self.svg_widget.setStyleSheet("background: transparent;")
        layout.addWidget(self.svg_widget)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start(self.TICK_MS)
        self._render_current_frame()

    # -- interaction ------------------------------------------------------
    def mousePressEvent(self, a0: QMouseEvent):
        self.bounce_timer = 45
        if self.state == self.STATE_SLEEP:
            self.set_state(self.STATE_IDLE)
        self.state_frames = 0
        super().mousePressEvent(a0)

    def set_state(self, state: str):
        if state in self._ALL_STATES:
            if state != self.state:
                self.state_frames = 0
            self.state = state
            self._render_current_frame()

    def toggle_animation(self, enabled: bool):
        self.is_animated = enabled
        if enabled and not self.timer.isActive():
            self.timer.start(self.TICK_MS)
        elif not enabled:
            self.timer.stop()

    # -- animation --------------------------------------------------------
    def _on_tick(self):
        if not self.is_animated:
            return
        self.frame_idx += 1
        self.state_frames += 1
        if self.bounce_timer > 0:
            self.bounce_timer -= 1

        secs = self.state_frames * self.TICK_MS / 1000.0
        if self.state == self.STATE_IDLE and secs > self.SLEEP_AFTER_S:
            self.set_state(self.STATE_SLEEP)
        elif self.state == self.STATE_ERROR and secs > self.ERROR_HOLD_S:
            self.set_state(self.STATE_IDLE)
        self._render_current_frame()

    def _render_current_frame(self):
        t = self.frame_idx * self.TICK_MS / 1000.0
        draw = "happy" if self.bounce_timer > 0 else self._DRAW.get(self.state, "idle")
        blink = (self.frame_idx % 120) > 114
        svg = build_cat_svg(draw, t, blink=blink, size=self.mascot_size)
        self.svg_widget.load(QByteArray(svg.encode("utf-8")))


class AnimatedWavePill(QSvgWidget):
    """Small navy activity indicator next to the app title."""

    COLORS = ["#9fb6d4", "#6f8fb8", "#1f3a5f", "#6f8fb8", "#9fb6d4"]

    def __init__(self, width: int = 44, height: int = 20, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.phase = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(60)
        self._render()

    def _tick(self):
        self.phase += 1
        self._render()

    def _render(self):
        p = self.phase * 0.16
        bars = ""
        for i, c in enumerate(self.COLORS):
            h = 5 + 9 * abs(math.sin(p + i * 0.7))
            bars += f'<rect x="{5 + i * 7}" y="{17 - h:.1f}" width="4.5" height="{h:.1f}" rx="1.5" fill="{c}"/>'
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 44 20" width="44" height="20">'
               f'<rect x="0" y="0" width="44" height="20" rx="4" fill="#e5ebf3"/>{bars}</svg>')
        self.load(QByteArray(svg.encode("utf-8")))