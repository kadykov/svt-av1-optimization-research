FROM mcr.microsoft.com/devcontainers/python:2-3.14-trixie

# Install Just and basic tools
RUN apt-get update && \
    apt-get install -y just wget xz-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install static FFmpeg build with both libsvtav1 and libvmaf support
# Using BtbN/FFmpeg-Builds which includes comprehensive codec/filter support
# See: https://github.com/BtbN/FFmpeg-Builds
RUN cd /tmp && \
    wget -q https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz && \
    tar xf ffmpeg-master-latest-linux64-gpl.tar.xz && \
    cd ffmpeg-master-latest-linux64-gpl && \
    cp bin/ffmpeg bin/ffprobe /usr/local/bin/ && \
    chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe && \
    cd / && rm -rf /tmp/ffmpeg-*

# Verify FFmpeg has both libsvtav1 (encoder) and libvmaf (filter)
RUN ffmpeg -encoders 2>&1 | grep -q libsvtav1 || (echo "ERROR: libsvtav1 encoder not found in FFmpeg" && exit 1)
RUN ffmpeg -filters 2>&1 | grep -q libvmaf || (echo "ERROR: libvmaf filter not found in FFmpeg" && exit 1)
