# Stage 1: Build
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build tools and Python headers
# - build-essential: g++, make
# - cmake: build system
# - wget + unzip: to download LibTorch
# - python3-dev: Python headers pybind11 needs to compile against
# - python3-pip: to install gymnasium
# - swig: required by gymnasium's Box2D physics (LunarLander)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    unzip \
    python3 \
    python3-pip \
    python3-dev \
    swig \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# gymnasium[box2d] covers LunarLander-v3
# gymnasium[classic-control] covers CartPole and MountainCar
RUN pip3 install "gymnasium[box2d,classic-control]"

# Download and unzip LibTorch (CPU build)
# To use CUDA instead, swap this URL for the CUDA build from:
#   https://pytorch.org/get-started/locally/
# and add --gpus all to your docker run command.
RUN wget -q \
    https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip \
    -O libtorch.zip \
    && unzip -q libtorch.zip \
    && rm libtorch.zip

WORKDIR /app
COPY CMakeLists.txt .
COPY config.json .
COPY src/ ./src/

RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DTorch_DIR=/libtorch/share/cmake/Torch \
    && cmake --build build --config Release -j$(nproc)

# Stage 2: runtime
FROM ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgomp1 \
    swig \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install "gymnasium[box2d,classic-control]"

COPY --from=builder /libtorch /libtorch

COPY --from=builder /app/build/cxx_ppo_agent /app/cxx_ppo_agent
COPY --from=builder /app/config.json /app/config.json

ENV LD_LIBRARY_PATH=/libtorch/lib:$LD_LIBRARY_PATH

WORKDIR /app

CMD ["./cxx_ppo_agent"]