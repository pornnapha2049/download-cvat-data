#!/bin/bash
set -e

# Default settings
RESOLUTION=${RESOLUTION:-"1600x950"}
WEB_GUI=${WEB_GUI:-"true"}
VNC_PORT=${VNC_PORT:-5900}
WEB_PORT=${WEB_PORT:-6080}

echo "============================================="
echo " Starting CVAT to YOLO Studio in Docker "
echo "============================================="
echo " Mode: WEB_GUI=${WEB_GUI}"

if [ "$WEB_GUI" = "true" ]; then
    echo " Display resolution: ${RESOLUTION}"
    export DISPLAY=:99

    # Clean up old lock files if container restarted
    rm -f /tmp/.X99-lock /tmp/.X11-unix/X99

    echo " Starting virtual X server (Xvfb)..."
    Xvfb :99 -screen 0 ${RESOLUTION}x24 -ac +extension GLX +render -noreset &
    XVFB_PID=$!
    sleep 1

    echo " Starting window manager (Fluxbox)..."
    fluxbox &
    FLUXBOX_PID=$!
    sleep 1

    echo " Starting VNC server (x11vnc)..."
    x11vnc -display :99 -forever -shared -nopw -rfbport ${VNC_PORT} -bg -quiet

    echo " Starting noVNC web interface on port ${WEB_PORT}..."
    websockify --web /usr/share/novnc ${WEB_PORT} localhost:${VNC_PORT} &
    WEBSOCK_PID=$!

    echo "=========================================================="
    echo " 🚀 Studio is ready! Open in your browser:"
    echo " 👉 http://localhost:${WEB_PORT}"
    echo "=========================================================="
else
    echo " Running with host DISPLAY=${DISPLAY}"
fi

# Ensure stray OpenCV Qt plugins never hijack PyQt5's xcb plugin
rm -rf /usr/local/lib/python*/site-packages/cv2/qt 2>/dev/null || true

# Run the Python application or passed command
if [ $# -gt 0 ]; then
    exec "$@"
else
    exec python app.py
fi
