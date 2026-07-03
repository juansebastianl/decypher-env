FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

# C++ toolchain and common native-extension build deps
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        gdb \
        valgrind \
        ccache \
        libomp-dev \
        libeigen3-dev \
        libboost-all-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Keep compiled artifacts out of the bind mount when possible
ENV CCACHE_DIR=/home/vscode/.ccache
ENV CC="ccache gcc"
ENV CXX="ccache g++"
