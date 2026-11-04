"""
Chubby Cat Animated SVG Mascot ("Mochi Cat" • แมวอ้วนน่ารักดุ้กดิ๊ก).
Smooth 30-60 FPS dynamic SVG vector frames with wiggling fluffy tail,
twitching cat ears, blushing chubby cheeks, tapping paws, and reactive animation states:
IDLE, DOWNLOADING, PROCESSING, TRAINING, SUCCESS.
"""

import math
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtCore import QTimer, QByteArray, Qt


class DancingSvgMascot(QWidget):
    """
    Cute Chubby Cat Animated Mascot Widget (แมวอ้วนน่ารักดุ้กดิ๊ก)
    Soft warm cream fur, wiggling tail, twitching ears, and tapping paws.
    """

    STATE_IDLE = "idle"
    STATE_DOWNLOADING = "downloading"
    STATE_PROCESSING = "processing"
    STATE_TRAINING = "training"
    STATE_SUCCESS = "success"

    def __init__(self, size: int = 130, parent=None):
        super().__init__(parent)
        self.mascot_size = size
        self.state = self.STATE_IDLE
        self.frame_idx = 0
        self.is_animated = True

        self.setFixedSize(self.mascot_size, self.mascot_size)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.svg_widget = QSvgWidget(self)
        self.svg_widget.setFixedSize(self.mascot_size, self.mascot_size)
        layout.addWidget(self.svg_widget)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start(33)

        self._render_current_frame()

    def set_state(self, state: str):
        if state in (self.STATE_IDLE, self.STATE_DOWNLOADING, self.STATE_PROCESSING, self.STATE_TRAINING, self.STATE_SUCCESS):
            self.state = state

    def toggle_animation(self, enabled: bool):
        self.is_animated = enabled
        if enabled:
            if not self.timer.isActive():
                self.timer.start(33)
        else:
            self.timer.stop()

    def _on_tick(self):
        if not self.is_animated:
            return
        self.frame_idx += 1
        self._render_current_frame()

    def _render_current_frame(self):
        t = self.frame_idx * 0.08

        # Palette
        fur_color = "#fffbf0"        # Warm creamy white
        fur_stroke = "#2e1f18"       # Cozy deep brown outline
        patch_color = "#fed7aa"      # Soft pastel orange/ginger patch
        ear_pink = "#fca5a5"         # Soft rosy pink
        blush_pink = "#f87171"       # Rosy blush
        belly_color = "#fff5eb"      # Soft creamy belly
        fish_gold = "#f59e0b"        # Fish treat

        # State physics
        if self.state == self.STATE_IDLE:
            # Gentle relaxed breathing and tail swishing
            bob_y = math.sin(t * 2.2) * 3.5
            tilt_deg = math.sin(t * 1.5) * 3.0
            tail_deg = math.sin(t * 2.5) * 18.0
            ear_l_y = math.sin(t * 1.8) * 2.0
            ear_r_y = -math.sin(t * 1.8) * 2.0
            paw_l_y = math.sin(t * 2.2) * 2.0
            paw_r_y = -math.sin(t * 2.2) * 2.0
            aura_radius = 45 + 2 * math.sin(t * 2.0)
            aura_color = "#fef3c7"
            extra_decor = """
            <!-- Cute gentle sparkles -->
            <path d="M 22 36 Q 26 36 26 32 Q 26 36 30 36 Q 26 36 26 40 Q 26 36 22 36" fill="#fbbf24" opacity="0.85"/>
            <path d="M 118 42 Q 121 42 121 39 Q 121 42 124 42 Q 121 42 121 45 Q 121 42 118 42" fill="#f472b6" opacity="0.75"/>
            """
            headband = ""

        elif self.state in (self.STATE_DOWNLOADING, self.STATE_PROCESSING):
            # Fast, excited wiggling ("ดุ้กดิ๊ก") paws tapping!
            bob_y = math.sin(t * 5.0) * 5.5
            tilt_deg = math.sin(t * 4.0) * 6.0
            tail_deg = math.sin(t * 5.5) * 32.0
            ear_l_y = math.sin(t * 4.5) * 4.0
            ear_r_y = -math.sin(t * 4.5) * 4.0
            paw_l_y = 6 * math.sin(t * 6.0)
            paw_r_y = -6 * math.sin(t * 6.0)
            aura_radius = 48 + 4 * math.sin(t * 4.0)
            aura_color = "#cffafe"
            extra_decor = """
            <!-- Excited motion stars -->
            <circle cx="20" cy="48" r="3" fill="#38bdf8" opacity="0.7"/>
            <circle cx="120" cy="46" r="3.5" fill="#38bdf8" opacity="0.7"/>
            <path d="M 18 64 Q 14 62 18 60" stroke="#38bdf8" stroke-width="2" fill="none" stroke-linecap="round"/>
            <path d="M 122 64 Q 126 62 122 60" stroke="#38bdf8" stroke-width="2" fill="none" stroke-linecap="round"/>
            """
            headband = ""

        elif self.state == self.STATE_TRAINING:
            # Concentrated cat with headband and intense tail swish!
            bob_y = math.sin(t * 3.5) * 4.0
            tilt_deg = math.sin(t * 2.8) * 4.0
            tail_deg = math.sin(t * 4.0) * 25.0
            ear_l_y = math.sin(t * 3.0) * 2.5
            ear_r_y = -math.sin(t * 3.0) * 2.5
            paw_l_y = math.sin(t * 3.5) * 3.0
            paw_r_y = -math.sin(t * 3.5) * 3.0
            aura_radius = 47 + 3 * math.sin(t * 3.0)
            aura_color = "#f3e8ff"
            extra_decor = """
            <!-- Concentration energy sparks -->
            <path d="M 18 34 L 24 38 L 20 44" stroke="#a855f7" stroke-width="2" fill="none" stroke-linecap="round"/>
            <path d="M 122 34 L 116 38 L 120 44" stroke="#a855f7" stroke-width="2" fill="none" stroke-linecap="round"/>
            """
            # Red determination headband
            headband = """
            <!-- Cute Red Headband -->
            <path d="M 44 48 Q 70 42 96 48 L 96 54 Q 70 48 44 54 Z" fill="#ef4444" stroke="#2e1f18" stroke-width="1.8"/>
            <circle cx="70" cy="48" r="3" fill="#ffffff"/>
            <!-- Headband knot on side -->
            <path d="M 95 50 Q 104 46 100 56 Q 94 54 95 50" fill="#ef4444" stroke="#2e1f18" stroke-width="1.5"/>
            """

        elif self.state == self.STATE_SUCCESS:
            # Extremely happy jumping cat! Fish treat and hearts floating!
            bob_y = -abs(math.sin(t * 3.8)) * 14.0
            tilt_deg = math.sin(t * 2.5) * 7.0
            tail_deg = math.sin(t * 5.0) * 35.0
            ear_l_y = 5.0 * math.sin(t * 4.0)
            ear_r_y = -5.0 * math.sin(t * 4.0)
            paw_l_y = -8 + math.sin(t * 4.5) * 5.0
            paw_r_y = -8 - math.sin(t * 4.5) * 5.0
            aura_radius = 50 + 6 * math.sin(t * 3.5)
            aura_color = "#dcfce7"
            extra_decor = f"""
            <!-- Cute Golden Fish 🐟 and Floating Hearts 💕 -->
            <g transform="translate(14, 22) rotate(-15) scale(0.9)">
              <!-- Little Fish -->
              <ellipse cx="12" cy="12" rx="8" ry="5" fill="{fish_gold}" stroke="{fur_stroke}" stroke-width="1.5"/>
              <polygon points="18,12 25,7 25,17" fill="{fish_gold}" stroke="{fur_stroke}" stroke-width="1.5" stroke-linejoin="round"/>
              <circle cx="8" cy="11" r="1.2" fill="#ffffff"/>
            </g>
            <g transform="translate(108, 18) scale(0.9)">
              <!-- Floating Pink Heart -->
              <path d="M 12 6 C 12 3 8 0 5 3 C 0 8 12 16 12 16 C 12 16 24 8 19 3 C 16 0 12 3 12 6 Z" fill="#f43f5e" opacity="0.9"/>
            </g>
            """
            headband = ""

        # Eye animation
        blink = (self.frame_idx % 110) > 103 and (self.state != self.STATE_SUCCESS)
        eye_ry = 1.0 if blink else 4.2

        cx = 70
        cy = 70 + bob_y

        svg_code = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 140 140" width="{self.mascot_size}" height="{self.mascot_size}">
  <defs>
    <radialGradient id="catAura" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="{aura_color}" stop-opacity="0.95"/>
      <stop offset="70%" stop-color="{aura_color}" stop-opacity="0.35"/>
      <stop offset="100%" stop-color="{aura_color}" stop-opacity="0"/>
    </radialGradient>
    <filter id="catShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="3.5" stdDeviation="3" flood-color="#2e1f18" flood-opacity="0.12"/>
    </filter>
  </defs>

  <!-- Aura Glow -->
  <circle cx="{cx}" cy="{cy}" r="{aura_radius}" fill="url(#catAura)"/>

  <!-- Extra Decorative Elements -->
  {extra_decor}

  <!-- Dancing Chubby Cat Buddy -->
  <g transform="rotate({tilt_deg}, {cx}, {cy + 22})" filter="url(#catShadow)">

    <!-- Ground shadow -->
    <ellipse cx="{cx}" cy="124" rx="{30 - bob_y * 0.35}" ry="{6 - bob_y * 0.1}" fill="#2e1f18" opacity="0.12"/>

    <!-- Wiggling Fluffy Tail (behind body) -->
    <g transform="translate({cx + 25}, {cy + 16}) rotate({tail_deg})">
      <!-- Tail curve -->
      <path d="M 0 0 C 14 -2, 24 -14, 20 -26 C 17 -31, 10 -29, 13 -22 C 16 -13, 8 2, 0 6 Z"
            fill="{fur_color}" stroke="{fur_stroke}" stroke-width="2.2" stroke-linejoin="round"/>
      <!-- Soft Orange Tip on tail -->
      <path d="M 13 -22 C 16 -13, 10 -4, 4 2 C 7 -4, 15 -14, 13 -22 Z" fill="{patch_color}"/>
    </g>

    <!-- Chubby Back / Bottom Body -->
    <ellipse cx="{cx}" cy="{cy + 14}" rx="32" ry="26" fill="{fur_color}" stroke="{fur_stroke}" stroke-width="2.5"/>

    <!-- Left Foot -->
    <ellipse cx="{cx - 18}" cy="{cy + 36}" rx="8.5" ry="5.5" fill="{fur_color}" stroke="{fur_stroke}" stroke-width="2.2"/>
    <!-- Right Foot -->
    <ellipse cx="{cx + 18}" cy="{cy + 36}" rx="8.5" ry="5.5" fill="{fur_color}" stroke="{fur_stroke}" stroke-width="2.2"/>

    <!-- Cream Belly Patch -->
    <ellipse cx="{cx}" cy="{cy + 17}" rx="19" ry="16" fill="{belly_color}"/>

    <!-- Chubby Round Head -->
    <ellipse cx="{cx}" cy="{cy - 8}" rx="29" ry="24" fill="{fur_color}" stroke="{fur_stroke}" stroke-width="2.5"/>

    <!-- Ginger Patch over Right Ear -->
    <path d="M {cx + 6} {cy - 24} Q {cx + 24} {cy - 20} {cx + 27} {cy - 4} Q {cx + 18} {cy - 12} {cx + 6} {cy - 24} Z" fill="{patch_color}" opacity="0.9"/>

    <!-- Left Cat Ear (Twitching) -->
    <g transform="translate({cx - 18}, {cy - 24 + ear_l_y})">
      <polygon points="-11,8 -2,-13 9,4" fill="{fur_color}" stroke="{fur_stroke}" stroke-width="2.2" stroke-linejoin="round"/>
      <polygon points="-8,7 -2,-9 6,4" fill="{ear_pink}" opacity="0.85"/>
    </g>

    <!-- Right Cat Ear (Twitching) -->
    <g transform="translate({cx + 18}, {cy - 24 + ear_r_y})">
      <polygon points="-9,4 2,-13 11,8" fill="{fur_color}" stroke="{fur_stroke}" stroke-width="2.2" stroke-linejoin="round"/>
      <polygon points="-6,4 2,-9 8,7" fill="{ear_pink}" opacity="0.85"/>
    </g>

    <!-- Optional Headband -->
    {headband}

    <!-- Chubby Rosy Blushing Cheeks -->
    <ellipse cx="{cx - 17}" cy="{cy - 2}" rx="5.5" ry="3.2" fill="{blush_pink}" opacity="0.6"/>
    <ellipse cx="{cx + 17}" cy="{cy - 2}" rx="5.5" ry="3.2" fill="{blush_pink}" opacity="0.6"/>

    <!-- Whiskers (Left: 3 lines) -->
    <line x1="{cx - 19}" y1="{cy - 6}" x2="{cx - 38}" y2="{cy - 10}" stroke="{fur_stroke}" stroke-width="1.6" stroke-linecap="round"/>
    <line x1="{cx - 20}" y1="{cy - 2}" x2="{cx - 40}" y2="{cy - 2}" stroke="{fur_stroke}" stroke-width="1.6" stroke-linecap="round"/>
    <line x1="{cx - 19}" y1="{cy + 2}" x2="{cx - 37}" y2="{cy + 6}" stroke="{fur_stroke}" stroke-width="1.6" stroke-linecap="round"/>

    <!-- Whiskers (Right: 3 lines) -->
    <line x1="{cx + 19}" y1="{cy - 6}" x2="{cx + 38}" y2="{cy - 10}" stroke="{fur_stroke}" stroke-width="1.6" stroke-linecap="round"/>
    <line x1="{cx + 20}" y1="{cy - 2}" x2="{cx + 40}" y2="{cy - 2}" stroke="{fur_stroke}" stroke-width="1.6" stroke-linecap="round"/>
    <line x1="{cx + 19}" y1="{cy + 2}" x2="{cx + 37}" y2="{cy + 6}" stroke="{fur_stroke}" stroke-width="1.6" stroke-linecap="round"/>

    <!-- Eyes (Kawaii glossy or happy curves) -->
    {
      f'''<!-- Happy Curved Eyes -->
      <path d="M {cx - 13} {cy - 7} Q {cx - 9} {cy - 13} {cx - 5} {cy - 7}" stroke="{fur_stroke}" stroke-width="2.5" fill="none" stroke-linecap="round"/>
      <path d="M {cx + 5} {cy - 7} Q {cx + 9} {cy - 13} {cx + 13} {cy - 7}" stroke="{fur_stroke}" stroke-width="2.5" fill="none" stroke-linecap="round"/>
      ''' if self.state == self.STATE_SUCCESS else
      f'''<!-- Kawaii Round Blinking Eyes -->
      <ellipse cx="{cx - 9}" cy="{cy - 7}" rx="3.5" ry="{eye_ry}" fill="{fur_stroke}"/>
      <ellipse cx="{cx + 9}" cy="{cy - 7}" rx="3.5" ry="{eye_ry}" fill="{fur_stroke}"/>
      {f"<circle cx='{cx - 10}' cy='{cy - 8.5}' r='1.3' fill='#ffffff'/>" if not blink else ""}
      {f"<circle cx='{cx + 8}' cy='{cy - 8.5}' r='1.3' fill='#ffffff'/>" if not blink else ""}
      '''
    }

    <!-- Tiny Pink Nose -->
    <polygon points="{cx - 2.8},{cy - 2.8} {cx + 2.8},{cy - 2.8} {cx},{cy + 0.5}" fill="#f43f5e" stroke="{fur_stroke}" stroke-width="0.6"/>

    <!-- Cute Cat '3' Mouth (w-shaped smile) -->
    <path d="M {cx - 5.5} {cy + 1} Q {cx - 2.8} {cy + 4.5} {cx} {cy + 1.2} Q {cx + 2.8} {cy + 4.5} {cx + 5.5} {cy + 1}"
          stroke="{fur_stroke}" stroke-width="1.8" fill="none" stroke-linecap="round"/>

    <!-- Left Chubby Paw (Tapping / Waving) -->
    <g transform="translate({cx - 18}, {cy + 16 + paw_l_y})">
      <ellipse cx="0" cy="0" rx="6.5" ry="6" fill="{fur_color}" stroke="{fur_stroke}" stroke-width="2"/>
      <!-- Tiny Pink Paw Bean -->
      <circle cx="0" cy="0" r="2.2" fill="{ear_pink}" opacity="0.8"/>
    </g>

    <!-- Right Chubby Paw (Tapping / Waving) -->
    <g transform="translate({cx + 18}, {cy + 16 + paw_r_y})">
      <ellipse cx="0" cy="0" rx="6.5" ry="6" fill="{fur_color}" stroke="{fur_stroke}" stroke-width="2"/>
      <!-- Tiny Pink Paw Bean -->
      <circle cx="0" cy="0" r="2.2" fill="{ear_pink}" opacity="0.8"/>
    </g>

  </g>
</svg>"""

        self.svg_widget.load(QByteArray(svg_code.encode("utf-8")))


class AnimatedWavePill(QSvgWidget):
    """Aesthetic pastel activity indicator."""

    def __init__(self, width: int = 36, height: int = 18, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.phase = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(50)
        self._render()

    def _tick(self):
        self.phase += 1
        self._render()

    def _render(self):
        p = self.phase * 0.18
        h1 = 4 + 7 * abs(math.sin(p))
        h2 = 4 + 7 * abs(math.sin(p + 0.8))
        h3 = 4 + 7 * abs(math.sin(p + 1.6))
        h4 = 4 + 7 * abs(math.sin(p + 2.4))

        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 18" width="36" height="18">
          <rect x="0" y="0" width="36" height="18" rx="9" fill="#0f3747" opacity="0.12"/>
          <line x1="8" y1="{9 - h1/2}" x2="8" y2="{9 + h1/2}" stroke="#0f3747" stroke-width="2.5" stroke-linecap="round"/>
          <line x1="14" y1="{9 - h2/2}" x2="14" y2="{9 + h2/2}" stroke="#0f3747" stroke-width="2.5" stroke-linecap="round"/>
          <line x1="20" y1="{9 - h3/2}" x2="20" y2="{9 + h3/2}" stroke="#0f3747" stroke-width="2.5" stroke-linecap="round"/>
          <line x1="26" y1="{9 - h4/2}" x2="26" y2="{9 + h4/2}" stroke="#0f3747" stroke-width="2.5" stroke-linecap="round"/>
        </svg>"""
        self.load(QByteArray(svg.encode("utf-8")))
