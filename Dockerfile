FROM ubuntu:24.04 AS builder

RUN apt-get update && apt-get install -y \
    cmake ninja-build git curl zip unzip tar \
    g++ pkg-config \
    libhdf5-dev \
    libopenblas-dev \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install vcpkg
RUN git clone https://github.com/microsoft/vcpkg /vcpkg && \
    /vcpkg/bootstrap-vcpkg.sh -disableMetrics

ENV VCPKG_ROOT=/vcpkg

WORKDIR /app
COPY vcpkg.json .
COPY CMakeLists.txt .
COPY CMakePresets.json .
COPY src/ src/
COPY main.cpp .

RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=/vcpkg/scripts/buildsystems/vcpkg.cmake && \
    cmake --build build --parallel

FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    libhdf5-dev libopenblas-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/build/main .

RUN mkdir -p /app/data /app/results

ENTRYPOINT ["./main"]