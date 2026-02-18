# Dockerfile.slim - Minimal RTL Development Environment
# Includes only: Icarus Verilog, Cocotb, and essential Python tools

FROM ubuntu:22.04

LABEL maintainer="ML-Audio Project"
LABEL description="Minimal RTL environment for Cocotb testing"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials (minimal)
    build-essential \
    git \
    make \
    g++ \
    flex \
    bison \
    openssh-client \
    # Python
    python3 \
    python3-pip \
    python3-dev \
    # EDA Tools
    iverilog \
    yosys \
    # Synthesis visualization
    nodejs \
    npm \
    librsvg2-bin \
    # Audio I/O (for soundfile package)
    libsndfile1 \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install netlistsvg for synthesis netlist visualization
RUN npm install -g netlistsvg

# Install minimal Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir \
    cocotb==1.9.2 \
    cocotb-test \
    pytest \
    gitpython \
    # For reference model and audio processing
    numpy \
    scipy \
    soundfile

# Install ML dependencies (torch, torchaudio, pyyaml, tqdm, etc.)
COPY requirements-ml.txt /tmp/requirements-ml.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements-ml.txt && \
    rm /tmp/requirements-ml.txt

# Create non-root user for development
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y --no-install-recommends sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Environment setup
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace:/workspace/util:$PYTHONPATH

# Switch to non-root user
USER $USERNAME

CMD ["/bin/bash"]
