# Dockerfile for ML-Audio RTL Development Environment
# Includes: Icarus Verilog, Verilator, Yosys, Cocotb, and Python tools

FROM ubuntu:22.04

LABEL maintainer="ML-Audio Project"
LABEL description="RTL development environment for ASIC design with Cocotb testing"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

# Install system dependencies and EDA tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    git \
    make \
    cmake \
    autoconf \
    g++ \
    flex \
    bison \
    # Python
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    # EDA Tools
    iverilog \
    verilator \
    yosys \
    gtkwave \
    # Utilities
    tree \
    vim \
    nano \
    wget \
    curl \
    ca-certificates \
    graphviz \
    # For waveform export
    imagemagick \
    # Audio dependencies
    libsndfile1 \
    portaudio19-dev \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install Python dependencies (lightweight packages only)
# ML/Audio packages are installed via postCreateCommand to keep image small
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir \
    # Cocotb and testing
    cocotb==1.9.2 \
    cocotb-test \
    pytest \
    pytest-xdist \
    pytest-json-report \
    pytest-timeout \
    # Utilities
    gitpython

# Create non-root user for development (matches VSCode Dev Container pattern)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Set Python to not buffer output (useful for seeing logs in real-time)
ENV PYTHONUNBUFFERED=1

# Add workspace to PYTHONPATH for imports
ENV PYTHONPATH=/workspace:/workspace/util:$PYTHONPATH

# Verify installations
RUN echo "=== Tool Versions ===" && \
    iverilog -V | head -1 && \
    verilator --version | head -1 && \
    yosys -V && \
    python3 --version && \
    pip3 list | grep cocotb

# Switch to non-root user
USER $USERNAME

# Default command
CMD ["/bin/bash"]