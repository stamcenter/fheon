# Multi-stage Dockerfile to build OpenFHE and FHEON

# ==============================================================================
# Builder Stage
# ==============================================================================
FROM ubuntu:24.04 AS builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies needed for compiling OpenFHE and FHEON
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libomp-dev \
    libomp5 \
    ca-certificates \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Clone and build OpenFHE (using instructions from appendix.md)
# Disable unittests, examples, and benchmarks to speed up build time
WORKDIR /usr/src
RUN git clone --branch v1.4.2 https://github.com/openfheorg/openfhe-development.git && \
    cd openfhe-development && \

    mkdir build && \
    cd build && \
    cmake -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF .. && \
    make -j$(nproc) && \
    make install && \
    cd /usr/src && \
    rm -rf openfhe-development

# Set up FHEON build
WORKDIR /usr/src
RUN git clone https://github.com/stamcenter/fheon.git /app
WORKDIR /app

# Build FHEON in SINGLE_INPUT mode as described in appendix.md
RUN mkdir -p build && \
    cd build && \
    cmake -DMODE=SINGLE_INPUT -DSINGLE_MODEL=ALL .. && \
    make -j$(nproc)

# ==============================================================================
# Runtime Stage
# ==============================================================================
FROM ubuntu:24.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libomp5 \
    libgomp1 \
    python3 \
    time \
    && rm -rf /var/lib/apt/lists/*

# Copy OpenFHE installed libraries and configuration from builder
COPY --from=builder /usr/local /usr/local

# Update library cache so that OpenFHE libraries can be resolved
RUN ldconfig

# Copy compiled binaries and necessary folders (weights, images, results, etc.)
WORKDIR /app
COPY --from=builder /app /app

# Set WORKDIR to /app/build so executing binaries with `./LeNet5` works
# and their relative paths (like `./../weights/...`) resolve correctly.
WORKDIR /app/build

# Default command starts an interactive bash shell
CMD ["/bin/bash"]
