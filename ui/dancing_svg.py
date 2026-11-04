"""
Aesthetic Dancing SVG Mascot ("Clover Byte" .☘︎ ݁˖).
Cozy Stationery & Aesthetic Sticker Theme.
Smooth 30-60 FPS dynamic SVG vector frames with swinging clover sprout,
blushing cheeks, blinking kawaii eyes, and reactive animation states:
IDLE, DOWNLOADING, PROCESSING, TRAINING, SUCCESS.
"""

import math
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtCore import QTimer, QByteArray, Qt


class DancingSvgMascot(QWidget):
    """
    Cozy Aesthetic Dancing Mascot Widget (.☘︎ ݁˖)
    Inspired by study stationery, printable stickers, warm cream & sage palette.
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

        # Aesthetic Palette
        body_color = "#fffdfa"
        body_stroke = "#3d2e24"
        clover_green = "#5e8460"
        clover_light = "#82a884"
        blush_color = "#f87171"
        honey_yellow = "#e8b056"
        apple_red = "#c85a54"

        # State physics
        if self.state == self.STATE_IDLE:
            bob_y = math.sin(t * 2.2) * 5
            tilt_deg = math.sin(t * 1.6) * 3
            clover_sway = math.sin(t * 2.8) * 12
            arm_l_deg = math.sin(t * 2.2) * 14
            arm_r_deg = -math.sin(t * 2.2) * 14
            aura_radius = 42 + 2 * math.sin(t * 2.0)
            aura_color = "#e2ece2"
            extra_elements = """
            <!-- Cute sparkle stars -->
            <path d="M 22 35 Q 26 35 26 31 Q 26 35 30 35 Q 26 35 26 39 Q 26 35 22 35" fill="#e8b056" opacity="0.85"/>
            <path d="M 116 42 Q 119 42 119 39 Q 119 42 122 42 Q 119 42 119 45 Q 119 42 116 42" fill="#5e8460" opacity="0.75"/>
            """

        elif self.state == self.STATE_DOWNLOADING:
            bob_y = math.sin(t * 3.8) * 7
            tilt_deg = math.sin(t * 3.0) * 7
            clover_sway = math.sin(t * 5.0) * 22
            arm_l_deg = 35 + math.sin(t * 4.5) * 25
            arm_r_deg = -35 - math.sin(t * 4.5) * 25
            aura_radius = 45 + 4 * math.sin(t * 4.0)
            aura_color = "#d9ebf5"

            # Stream of stationery stars & download arrow
            stream_y = (self.frame_idx * 3) % 40
            extra_elements = f"""
            <circle cx="28" cy="{25 + stream_y}" r="3" fill="#e8b056" opacity="0.8"/>
            <circle cx="112" cy="{20 + ((stream_y + 15) % 40)}" r="2.5" fill="#5e8460" opacity="0.8"/>
            <path d="M 70 {22 + (stream_y % 20)} L 70 {34 + (stream_y % 20)} M 65 {29 + (stream_y % 20)} L 70 {34 + (stream_y % 20)} L 75 {29 + (stream_y % 20)}" stroke="#5e8460" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
            """

        elif self.state == self.STATE_PROCESSING:
            bob_y = math.sin(t * 2.8) * 6
            tilt_deg = math.cos(t * 2.2) * 5
            clover_sway = math.sin(t * 3.5) * 16
            arm_l_deg = -15 + math.sin(t * 3.0) * 20
            arm_r_deg = 20 + math.cos(t * 3.0) * 20
            aura_radius = 44 + 3 * math.sin(t * 3.0)
            aura_color = "#fcedd7"

            rot = (self.frame_idx * 4) % 360
            extra_elements = f"""
            <!-- Cute rotating flower sticker -->
            <g transform="translate(112, 34) rotate({rot})">
              <circle cx="0" cy="-6" r="3" fill="#f87171" opacity="0.85"/>
              <circle cx="6" cy="0" r="3" fill="#f87171" opacity="0.85"/>
              <circle cx="0" cy="6" r="3" fill="#f87171" opacity="0.85"/>
              <circle cx="-6" cy="0" r="3" fill="#f87171" opacity="0.85"/>
              <circle cx="0" cy="0" r="3" fill="#fef08a"/>
            </g>
            """

        elif self.state == self.STATE_TRAINING:
            bob_y = math.sin(t * 4.2) * 8
            tilt_deg = math.sin(t * 3.8) * 8
            clover_sway = math.sin(t * 6.0) * 25
            arm_l_deg = math.sin(t * 5.0) * 35
            arm_r_deg = -math.sin(t * 5.0) * 35
            aura_radius = 48 + 5 * math.sin(t * 5.0)
            aura_color = "#fef2f2"

            spark_rot = (self.frame_idx * 6) % 360
            sx = 70 + math.cos(math.radians(spark_rot)) * 38
            sy = 65 + bob_y + math.sin(math.radians(spark_rot)) * 26
            extra_elements = f"""
            <!-- Red Apple & Little Flame sparkles -->
            <circle cx="{sx}" cy="{sy}" r="3.5" fill="#f87171" opacity="0.9"/>
            <path d="M 28 32 Q 32 25 36 32 Q 40 25 44 32 Q 36 42 28 32 Z" fill="#c85a54" opacity="0.8"/>
            """

        elif self.state == self.STATE_SUCCESS:
            bob_y = -abs(math.sin(t * 3.2)) * 14
            tilt_deg = math.sin(t * 2.2) * 6
            clover_sway = math.sin(t * 4.0) * 18
            arm_l_deg = 50 + math.sin(t * 4.0) * 18
            arm_r_deg = -50 - math.sin(t * 4.0) * 18
            aura_radius = 47 + 5 * math.sin(t * 3.0)
            aura_color = "#e2ece2"

            extra_elements = """
            <!-- Little Red Apple sticker 🍎 and Lucky Clover ☘︎ -->
            <g transform="translate(18, 26) scale(0.9)">
              <circle cx="10" cy="12" r="7" fill="#c85a54"/>
              <path d="M 10 5 Q 12 1 14 3" stroke="#5e8460" stroke-width="2" fill="none"/>
              <ellipse cx="8" cy="10" rx="1.5" ry="3" fill="#ffffff" opacity="0.5"/>
            </g>
            <g transform="translate(110, 24) scale(0.9)">
              <path d="M 0 6 C -4 0 -8 6 0 12 C 8 6 4 0 0 6" fill="#5e8460"/>
              <path d="M 6 0 C 0 -4 6 -8 12 0 C 6 8 0 4 6 0" fill="#6b8f6d"/>
            </g>
            """

        # Blinking cute kawaii eyes
        blink = (self.frame_idx % 115) > 108
        eye_ry = 1.0 if blink else 4.5

        cx = 70
        cy = 68 + bob_y

        svg_code = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 140 140" width="{self.mascot_size}" height="{self.mascot_size}">
  <defs>
    <radialGradient id="softAura" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="{aura_color}" stop-opacity="0.9"/>
      <stop offset="70%" stop-color="{aura_color}" stop-opacity="0.3"/>
      <stop offset="100%" stop-color="{aura_color}" stop-opacity="0"/>
    </radialGradient>
    <filter id="stickerShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="3" stdDeviation="3" flood-color="#4a3b32" flood-opacity="0.12"/>
    </filter>
  </defs>

  <!-- Soft Cozy Aura Glow -->
  <circle cx="{cx}" cy="{cy}" r="{aura_radius}" fill="url(#softAura)"/>

  <!-- Extra Decorative Stationery Elements -->
  {extra_elements}

  <!-- Dancing Aesthetic Buddy Group -->
  <g transform="rotate({tilt_deg}, {cx}, {cy + 20})" filter="url(#stickerShadow)">

    <!-- Shadow under feet -->
    <ellipse cx="{cx}" cy="122" rx="{24 - bob_y * 0.4}" ry="{5 - bob_y * 0.1}" fill="#3d2e24" opacity="0.12"/>

    <!-- Left Arm -->
    <g transform="translate({cx - 24}, {cy + 14}) rotate({arm_l_deg})">
      <ellipse cx="-2" cy="10" rx="5" ry="9" fill="{body_color}" stroke="{body_stroke}" stroke-width="2"/>
    </g>

    <!-- Right Arm -->
    <g transform="translate({cx + 24}, {cy + 14}) rotate({arm_r_deg})">
      <ellipse cx="2" cy="10" rx="5" ry="9" fill="{body_color}" stroke="{body_stroke}" stroke-width="2"/>
    </g>

    <!-- Round Cute Body / Head (Chibi egg-shaped buddy) -->
    <ellipse cx="{cx}" cy="{cy}" rx="27" ry="29" fill="{body_color}" stroke="{body_stroke}" stroke-width="2.5"/>

    <!-- Soft Cream Bell Shape / Belly Patch -->
    <ellipse cx="{cx}" cy="{cy + 11}" rx="16" ry="12" fill="#f7f1e5" opacity="0.75"/>

    <!-- Lucky Four-Leaf Clover Sprout on Top (.☘︎ ݁˖) -->
    <g transform="translate({cx}, {cy - 29}) rotate({clover_sway})">
      <!-- Curved Green Stem -->
      <path d="M 0 0 Q -2 -9 0 -16" stroke="{clover_green}" stroke-width="2.5" fill="none" stroke-linecap="round"/>
      
      <!-- 4 Clover Leaf Petals -->
      <!-- Top Leaf -->
      <path d="M 0 -16 C -4 -22 4 -22 0 -16" fill="{clover_green}" stroke="{body_stroke}" stroke-width="0.8"/>
      <!-- Right Leaf -->
      <path d="M 0 -16 C 6 -20 6 -12 0 -16" fill="{clover_light}" stroke="{body_stroke}" stroke-width="0.8"/>
      <!-- Bottom Leaf -->
      <path d="M 0 -16 C 4 -10 -4 -10 0 -16" fill="{clover_green}" stroke="{body_stroke}" stroke-width="0.8"/>
      <!-- Left Leaf -->
      <path d="M 0 -16 C -6 -12 -6 -20 0 -16" fill="{clover_light}" stroke="{body_stroke}" stroke-width="0.8"/>
      <!-- Clover Center Knot -->
      <circle cx="0" cy="-16" r="1.5" fill="{honey_yellow}"/>
    </g>

    <!-- Cute Rosy Blushing Cheeks -->
    <ellipse cx="{cx - 15}" cy="{cy + 4}" rx="4.5" ry="2.5" fill="{blush_color}" opacity="0.6"/>
    <ellipse cx="{cx + 15}" cy="{cy + 4}" rx="4.5" ry="2.5" fill="{blush_color}" opacity="0.6"/>

    <!-- Kawaii Glossy Eyes -->
    <ellipse cx="{cx - 9}" cy="{cy}" rx="3.2" ry="{eye_ry}" fill="{body_stroke}"/>
    <ellipse cx="{cx + 9}" cy="{cy}" rx="3.2" ry="{eye_ry}" fill="{body_stroke}"/>

    <!-- Eye Highlights -->
    {"<circle cx='" + str(cx - 10) + "' cy='" + str(cy - 1.5) + "' r='1.2' fill='#ffffff'/>" if not blink else ""}
    {"<circle cx='" + str(cx + 8) + "' cy='" + str(cy - 1.5) + "' r='1.2' fill='#ffffff'/>" if not blink else ""}

    <!-- Sweet Happy Smile -->
    {"<path d='M " + str(cx - 4) + " " + str(cy + 6) + " Q " + str(cx) + " " + str(cy + 10) + " " + str(cx + 4) + " " + str(cy + 6) + "' stroke='" + body_stroke + "' stroke-width='2' fill='none' stroke-linecap='round'/>" if not blink else "<line x1='" + str(cx - 3) + "' y1='" + str(cy + 7) + "' x2='" + str(cx + 3) + "' y2='" + str(cy + 7) + "' stroke='" + body_stroke + "' stroke-width='2' stroke-linecap='round'/>"}

  </g>
</svg>"""

        self.svg_widget.load(QByteArray(svg_code.encode("utf-8")))


class AnimatedWavePill(QSvgWidget):
    """Aesthetic pastel activity indicator."""

    def __init__(self, width: int = 42, height: int = 22, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.frame = 0
        self.active = True

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(50)

    def _tick(self):
        if not self.active:
            return
        self.frame += 1
        f = self.frame
        h1 = 6 + math.sin(f * 0.3) * 5
        h2 = 8 + math.cos(f * 0.3) * 6
        h3 = 7 + math.sin(f * 0.3 + 1) * 5
        h4 = 5 + math.cos(f * 0.3 + 2) * 4

        svg = f"""<svg width="42" height="22" viewBox="0 0 42 22" xmlns="http://www.w3.org/2000/svg">
          <line x1="8" y1="{11 - h1/2}" x2="8" y2="{11 + h1/2}" stroke="#5e8460" stroke-width="3.5" stroke-linecap="round"/>
          <line x1="17" y1="{11 - h2/2}" x2="17" y2="{11 + h2/2}" stroke="#e8b056" stroke-width="3.5" stroke-linecap="round"/>
          <line x1="26" y1="{11 - h3/2}" x2="26" y2="{11 + h3/2}" stroke="#c85a54" stroke-width="3.5" stroke-linecap="round"/>
          <line x1="35" y1="{11 - h4/2}" x2="35" y2="{11 + h4/2}" stroke="#7e99ab" stroke-width="3.5" stroke-linecap="round"/>
        </svg>"""
        self.load(QByteArray(svg.encode("utf-8")))
