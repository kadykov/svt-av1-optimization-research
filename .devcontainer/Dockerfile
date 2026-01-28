FROM mcr.microsoft.com/devcontainers/python:2-3.14-trixie

# Install Just and basic tools
RUN apt-get update && \
    apt-get install -y just wget xz-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install static FFmpeg build with libvmaf support
# Using John Van Sickle's builds which include libvmaf
RUN cd /tmp && \
    wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && \
    tar xf ffmpeg-release-amd64-static.tar.xz && \
    cd ffmpeg-*-amd64-static && \
    cp ffmpeg ffprobe /usr/local/bin/ && \
    chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe && \
    cd / && rm -rf /tmp/ffmpeg-*

# Verify FFmpeg has libvmaf
RUN ffmpeg -filters 2>&1 | grep libvmaf || (echo "ERROR: libvmaf not found in FFmpeg" && exit 1)

