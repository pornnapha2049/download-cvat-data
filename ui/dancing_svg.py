"""
Animated Dancing SVG Widget ("Byte" - The AI Robot Mascot).
Generates silky-smooth dynamic SVG vector frames at 30-60 FPS using QTimer.
Reactive states: IDLE, DOWNLOADING, PROCESSING, TRAINING, SUCCESS.
"""

import math
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtCore import QTimer, QByteArray, Qt, pyqtProperty


class DancingSvgMascot(QWidget):
    """
    Animated dynamic SVG mascot widget with reactive dancing behaviors.
    """

    STATE_IDLE = "idle"
    STATE_DOWNLOADING = "downloading"
    STATE_PROCESSING = "processing"
    STATE_TRAINING = "training"
    STATE_SUCCESS = "success"

    def __init__(self, size: int = 140, parent=None):
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

        # 30 FPS animation timer (~33ms)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start(33)

        self._render_current_frame()

    def set_state(self, state: str):
        """Change animation state: idle, downloading, processing, training, success."""
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
        t = self.frame_idx * 0.08  # time factor

        # Physics variables
        if self.state == self.STATE_IDLE:
            speed = 1.0
            bob_y = math.sin(t * 2.2) * 6
            tilt_deg = math.sin(t * 1.5) * 4
            arm_l_deg = math.sin(t * 2.5) * 12
            arm_r_deg = -math.sin(t * 2.5) * 12
            antenna_glow = 0.6 + 0.4 * math.sin(t * 3.0)
            aura_radius = 42 + 2 * math.sin(t * 2.0)
            theme_color = "#6366f1"  # Indigo
            glow_color = "#818cf8"
            eye_color = "#38bdf8"    # Sky blue
            extra_elements = ""

        elif self.state == self.STATE_DOWNLOADING:
            speed = 2.0
            t_fast = t * speed
            bob_y = math.sin(t_fast * 3.0) * 8
            tilt_deg = math.sin(t_fast * 2.5) * 8
            arm_l_deg = 30 + math.sin(t_fast * 4.0) * 25
            arm_r_deg = -30 - math.sin(t_fast * 4.0) * 25
            antenna_glow = 0.8 + 0.2 * math.sin(t_fast * 6.0)
            aura_radius = 45 + 5 * math.sin(t_fast * 4.0)
            theme_color = "#06b6d4"  # Cyan
            glow_color = "#67e8f9"
            eye_color = "#22d3ee"

            # Animated downward data arrows / rings
            stream_y = (self.frame_idx * 4) % 60
            extra_elements = f"""
            <g opacity="0.85">
              <circle cx="70" cy="{30 + (stream_y % 50)}" r="3" fill="#38bdf8" />
              <circle cx="50" cy="{25 + ((stream_y + 20) % 50)}" r="2.5" fill="#67e8f9" />
              <circle cx="90" cy="{35 + ((stream_y + 35) % 50)}" r="2.5" fill="#38bdf8" />
              <line x1="70" y1="{30 + (stream_y % 50)}" x2="70" y2="{40 + (stream_y % 50)}" stroke="#38bdf8" stroke-width="2" stroke-linecap="round"/>
            </g>
            """

        elif self.state == self.STATE_PROCESSING:
            speed = 1.6
            t_proc = t * speed
            bob_y = math.sin(t_proc * 2.8) * 7
            tilt_deg = math.cos(t_proc * 2.0) * 6
            arm_l_deg = -15 + math.sin(t_proc * 3.0) * 20
            arm_r_deg = 20 + math.cos(t_proc * 3.0) * 20
            antenna_glow = 0.7 + 0.3 * math.sin(t_proc * 5.0)
            aura_radius = 44 + 4 * math.sin(t_proc * 3.0)
            theme_color = "#8b5cf6"  # Purple
            glow_color = "#c084fc"
            eye_color = "#a855f7"

            # Rotating gear/matrix particles
            gear_rot = (self.frame_idx * 5) % 360
            extra_elements = f"""
            <g transform="translate(110, 35) rotate({gear_rot})">
              <circle cx="0" cy="0" r="10" fill="none" stroke="#c084fc" stroke-width="2" stroke-dasharray="4,3"/>
            </g>
            <g transform="translate(30, 95) rotate({-gear_rot})">
              <circle cx="0" cy="0" r="8" fill="none" stroke="#a855f7" stroke-width="2" stroke-dasharray="3,3"/>
            </g>
            """

        elif self.state == self.STATE_TRAINING:
            speed = 2.4
            t_train = t * speed
            bob_y = math.sin(t_train * 4.0) * 9
            tilt_deg = math.sin(t_train * 3.5) * 10
            arm_l_deg = math.sin(t_train * 5.0) * 35
            arm_r_deg = -math.sin(t_train * 5.0) * 35
            antenna_glow = 0.9 + 0.1 * math.sin(t_train * 8.0)
            aura_radius = 48 + 7 * math.sin(t_train * 6.0)
            theme_color = "#f59e0b"  # Amber / Fire
            glow_color = "#fbbf24"
            eye_color = "#fef08a"

            # Lightning / Energy sparks around head
            spark_angle = (self.frame_idx * 9) % 360
            sx = 70 + math.cos(math.radians(spark_angle)) * 36
            sy = 65 + bob_y + math.sin(math.radians(spark_angle)) * 26
            extra_elements = f"""
            <circle cx="{sx}" cy="{sy}" r="3.5" fill="#fef08a" opacity="0.9"/>
            <path d="M 60 {25 + bob_y} L 66 {18 + bob_y} L 63 {30 + bob_y} L 72 {22 + bob_y}" stroke="#fbbf24" stroke-width="2" fill="none" stroke-linecap="round"/>
            """

        elif self.state == self.STATE_SUCCESS:
            speed = 1.8
            t_succ = t * speed
            # Jump animation
            bob_y = -abs(math.sin(t_succ * 3.0)) * 14
            tilt_deg = math.sin(t_succ * 2.0) * 7
            arm_l_deg = 50 + math.sin(t_succ * 4.0) * 20
            arm_r_deg = -50 - math.sin(t_succ * 4.0) * 20
            antenna_glow = 1.0
            aura_radius = 46 + 6 * math.sin(t_succ * 3.0)
            theme_color = "#10b981"  # Emerald
            glow_color = "#34d399"
            eye_color = "#6ee7b7"

            # Stars celebration
            extra_elements = f"""
            <polygon points="25,25 28,32 35,32 29,37 31,44 25,39 19,44 21,37 15,32 22,32" fill="#fbbf24" transform="scale(0.7) translate(15, 10)"/>
            <polygon points="120,30 122,35 127,35 123,38 124,43 120,40 116,43 117,38 113,35 118,35" fill="#34d399" transform="scale(0.8) translate(15, 10)"/>
            """

        # Blinking eye effect (every ~3.5 seconds)
        blink_cycle = (self.frame_idx % 110)
        is_blinking = blink_cycle > 104
        eye_ry = 1.2 if is_blinking else 5.5

        # Robot coordinate anchors
        head_cx = 70
        head_cy = 65 + bob_y

        svg_code = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 140 140" width="{self.mascot_size}" height="{self.mascot_size}">
  <defs>
    <radialGradient id="auraGlow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="{glow_color}" stop-opacity="0.45"/>
      <stop offset="60%" stop-color="{theme_color}" stop-opacity="0.18"/>
      <stop offset="100%" stop-color="{theme_color}" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="bodyGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#334155"/>
      <stop offset="50%" stop-color="#1e293b"/>
      <stop offset="100%" stop-color="#0f172a"/>
    </linearGradient>
    <linearGradient id="visorGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#090d16"/>
      <stop offset="100%" stop-color="#111827"/>
    </linearGradient>
    <filter id="neonBlur" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>

  <!-- Pulsing Energy Aura -->
  <circle cx="{head_cx}" cy="{head_cy}" r="{aura_radius}" fill="url(#auraGlow)"/>

  <!-- Extra Dynamic State Elements -->
  {extra_elements}

  <!-- Dancing Robot Group with tilt and bob -->
  <g transform="rotate({tilt_deg}, {head_cx}, {head_cy + 25})">
    
    <!-- Shadow underneath -->
    <ellipse cx="{head_cx}" cy="126" rx="{26 - bob_y * 0.5}" ry="{6 - bob_y * 0.15}" fill="#000000" opacity="0.35"/>

    <!-- Left Arm -->
    <g transform="translate({head_cx - 28}, {head_cy + 18}) rotate({arm_l_deg})">
      <rect x="-4" y="0" width="8" height="22" rx="4" fill="#475569" stroke="#64748b" stroke-width="1.5"/>
      <circle cx="0" cy="22" r="5" fill="{theme_color}"/>
    </g>

    <!-- Right Arm -->
    <g transform="translate({head_cx + 28}, {head_cy + 18}) rotate({arm_r_deg})">
      <rect x="-4" y="0" width="8" height="22" rx="4" fill="#475569" stroke="#64748b" stroke-width="1.5"/>
      <circle cx="0" cy="22" r="5" fill="{theme_color}"/>
    </g>

    <!-- Torso -->
    <rect x="{head_cx - 18}" y="{head_cy + 16}" width="36" height="28" rx="8" fill="url(#bodyGrad)" stroke="#475569" stroke-width="2"/>
    
    <!-- Chest Core Reactor -->
    <circle cx="{head_cx}" cy="{head_cy + 30}" r="6" fill="{theme_color}" filter="url(#neonBlur)"/>
    <circle cx="{head_cx}" cy="{head_cy + 30}" r="3" fill="#ffffff"/>

    <!-- Antenna -->
    <line x1="{head_cx}" y1="{head_cy - 22}" x2="{head_cx}" y2="{head_cy - 34}" stroke="#64748b" stroke-width="3" stroke-linecap="round"/>
    <circle cx="{head_cx}" cy="{head_cy - 36}" r="6" fill="{glow_color}" opacity="{antenna_glow}" filter="url(#neonBlur)"/>
    <circle cx="{head_cx}" cy="{head_cy - 36}" r="3" fill="#ffffff"/>

    <!-- Head -->
    <rect x="{head_cx - 27}" y="{head_cy - 22}" width="54" height="40" rx="12" fill="url(#bodyGrad)" stroke="{theme_color}" stroke-width="2.5"/>

    <!-- Cute Ear Bolts -->
    <rect x="{head_cx - 31}" y="{head_cy - 12}" width="5" height="18" rx="2" fill="#475569"/>
    <rect x="{head_cx + 26}" y="{head_cy - 12}" width="5" height="18" rx="2" fill="#475569"/>

    <!-- Visor Face Plate -->
    <rect x="{head_cx - 21}" y="{head_cy - 14}" width="42" height="24" rx="6" fill="url(#visorGrad)" stroke="#1e293b" stroke-width="1.5"/>

    <!-- Glowing LED Eyes -->
    <ellipse cx="{head_cx - 10}" cy="{head_cy - 3}" rx="4.5" ry="{eye_ry}" fill="{eye_color}" filter="url(#neonBlur)"/>
    <ellipse cx="{head_cx + 10}" cy="{head_cy - 3}" rx="4.5" ry="{eye_ry}" fill="{eye_color}" filter="url(#neonBlur)"/>
    
    <!-- Pupil shines -->
    {"<circle cx='" + str(head_cx - 8.5) + "' cy='" + str(head_cy - 4.5) + "' r='1.5' fill='#ffffff'/>" if not is_blinking else ""}
    {"<circle cx='" + str(head_cx + 11.5) + "' cy='" + str(head_cy - 4.5) + "' r='1.5' fill='#ffffff'/>" if not is_blinking else ""}

    <!-- Cheerful Smile LED -->
    {"<path d='M " + str(head_cx - 6) + " " + str(head_cy + 5) + " Q " + str(head_cx) + " " + str(head_cy + 9) + " " + str(head_cx + 6) + " " + str(head_cy + 5) + "' stroke='" + eye_color + "' stroke-width='1.5' fill='none' stroke-linecap='round'/>" if not is_blinking else ""}

  </g>
</svg>"""

        self.svg_widget.load(QByteArray(svg_code.encode("utf-8")))


class AnimatedWavePill(QSvgWidget):
    """Mini pulsing activity indicator for status bars and buttons."""

    def __init__(self, width: int = 40, height: int = 20, parent=None):
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

        svg = f"""<svg width="40" height="20" viewBox="0 0 40 20" xmlns="http://www.w3.org/2000/svg">
          <line x1="8" y1="{10 - h1/2}" x2="8" y2="{10 + h1/2}" stroke="#10b981" stroke-width="3" stroke-linecap="round"/>
          <line x1="16" y1="{10 - h2/2}" x2="16" y2="{10 + h2/2}" stroke="#06b6d4" stroke-width="3" stroke-linecap="round"/>
          <line x1="24" y1="{10 - h3/2}" x2="24" y2="{10 + h3/2}" stroke="#8b5cf6" stroke-width="3" stroke-linecap="round"/>
          <line x1="32" y1="{10 - h4/2}" x2="32" y2="{10 + h4/2}" stroke="#f59e0b" stroke-width="3" stroke-linecap="round"/>
        </svg>"""
        self.load(QByteArray(svg.encode("utf-8")))
