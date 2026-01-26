FROM mcr.microsoft.com/devcontainers/python:2-3.14-trixie

# Install FFmpeg and Just
RUN apt-get update && \
    apt-get install -y ffmpeg just && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
