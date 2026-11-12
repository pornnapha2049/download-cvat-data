FROM python:3.10-slim

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    QT_X11_NO_MITSHM=1 \
    DISPLAY=:99 \
    RESOLUTION=1600x950 \
    WEB_GUI=true \
    QT_QPA_PLATFORM_PLUGIN_PATH=/usr/local/lib/python3.10/site-packages/PyQt5/Qt5/plugins

# Install system dependencies:
# 1. Qt5 & GUI runtime libraries
# 2. X11, OpenGL, FFmpeg
# 3. Thai, CJK, Emoji fonts for proper multi-language UI rendering
# 4. Xvfb, Fluxbox, x11vnc, noVNC for browser-based GUI access
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    libxkbcommon-x11-0 \
    libxcb1 \
    libx11-xcb1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xinput0 \
    libfontconfig1 \
    libdbus-1-3 \
    fonts-dejavu-core \
    fonts-noto-cjk \
    fonts-noto-color-emoji \
    fonts-thai-tlwg \
    xvfb \
    x11vnc \
    fluxbox \
    novnc \
    websockify \
    net-tools \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Ensure noVNC defaults to auto-connecting with auto-scaling to browser window
RUN if [ -d /usr/share/novnc ]; then \
        echo "<!DOCTYPE html><html><head><meta http-equiv=\"refresh\" content=\"0; url='vnc.html?autoconnect=true&resize=scale'\"><title>CVAT to YOLO Studio</title></head><body><p>Connecting to Studio...</p></body></html>" > /usr/share/novnc/index.html; \
    fi

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY core/ ./core/
COPY ui/ ./ui/
COPY app.py .
COPY docker-entrypoint.sh .

# Convert line endings, set executable permissions, and create mount directories
RUN dos2unix /app/docker-entrypoint.sh && \
    chmod +x /app/docker-entrypoint.sh && \
    mkdir -p /app/data /app/models /app/runs

# Expose noVNC web port (6080) and raw VNC port (5900)
EXPOSE 6080 5900

ENTRYPOINT ["/app/docker-entrypoint.sh"]
