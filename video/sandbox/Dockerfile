# Agent image for the local Video-Index v1.1 replica.
# Tools match the official protocol: shell, ffmpeg/ffprobe, Python with OpenCV, Pillow, NumPy, faster-whisper.
# Whisper models (large-v3, small) are NOT baked in; they are mounted read-only at /opt/models/faster-whisper.
FROM node:22-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg python3 python3-pip python3-venv ca-certificates bash coreutils procps file bc \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/eval
RUN /opt/eval/bin/pip install --retries 10 --timeout 120 numpy pillow
RUN /opt/eval/bin/pip install --retries 10 --timeout 120 opencv-python-headless
RUN /opt/eval/bin/pip install --retries 10 --timeout 120 faster-whisper && rm -rf /root/.cache/pip

# Harness: Claude Code at the exact version used for the official runs.
# The native Claude Code binary is vendored per architecture (the npm postinstall download was unreliable):
#   curl -LO https://registry.npmjs.org/@anthropic-ai/claude-code-linux-x64/-/claude-code-linux-x64-2.1.300.tgz
#   (or -linux-arm64-) into vendor/
ARG TARGETARCH
COPY vendor/ /tmp/vendor/
RUN A=$([ "$TARGETARCH" = "arm64" ] && echo arm64 || echo x64) && mkdir -p /opt/claude \
    && tar xzf /tmp/vendor/*claude-code-linux-${A}-2.1.300.tgz -C /opt/claude && rm -rf /tmp/vendor \
    && ln -s /opt/claude/package/claude /usr/local/bin/claude && chmod +x /opt/claude/package/claude && claude --version

RUN useradd -m -u 1001 agent && mkdir -p /work && chown agent /work
USER agent
ENV PATH=/opt/eval/bin:/usr/local/bin:/usr/bin:/bin
WORKDIR /work
