# OpenFace 2.x — simple, no-frills build that just works
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# 1) System deps + wget (needed by download_models.sh)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates build-essential python3-pip \
    libopencv-dev libboost-all-dev libopenblas-dev liblapacke-dev gfortran \
    libtbb-dev libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev \
    libx11-dev libpng-dev libjpeg-dev \
 && rm -rf /var/lib/apt/lists/*

# 2) Modern CMake + Ninja (small, reliable)
RUN pip3 install --no-cache-dir "cmake>=3.26" ninja

# 3) Build & install dlib >= 19.13 (no GUI, no CUDA)
RUN git clone --depth=1 https://github.com/davisking/dlib.git /opt/dlib && \
    cmake -S /opt/dlib -B /opt/dlib/build \
      -D CMAKE_BUILD_TYPE=Release \
      -D DLIB_NO_GUI_SUPPORT=ON \
      -D DLIB_USE_CUDA=OFF && \
    cmake --build /opt/dlib/build -j"$(nproc)" && \
    cmake --install /opt/dlib/build

# 4) Get OpenFace, download models, build, expose tools on PATH
WORKDIR /opt
RUN git clone --depth=1 https://github.com/TadasBaltrusaitis/OpenFace.git
WORKDIR /opt/OpenFace
RUN bash ./download_models.sh \
 && cmake -S . -B build -D CMAKE_BUILD_TYPE=Release -G Ninja \
      -D dlib_DIR=/usr/local/lib/cmake/dlib \
 && cmake --build build -j"$(nproc)" \
 && ln -s /opt/OpenFace/build/bin/* /usr/local/bin/ \
 # Place patch_experts where binaries commonly look (./model/patch_experts)
 && mkdir -p /opt/OpenFace/model/patch_experts \FG2021-BoLD
 && cp lib/local/LandmarkDetector/model/patch_experts/*.dat /opt/OpenFace/model/patch_experts/

WORKDIR /opt/OpenFace
CMD ["/bin/bash"]