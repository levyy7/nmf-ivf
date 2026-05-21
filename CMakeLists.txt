cmake_minimum_required(VERSION 3.20)
project(nmf-ivf)

set(CMAKE_CXX_STANDARD 20)

add_executable(main
        src/main.cpp
        src/index/nmf_index.h
        src/index/nmf_index.cpp
        src/nmf/base.h
        src/nmf/base.cpp
        src/nmf/mu_nmf.h
        src/nmf/mu_nmf.cpp
        src/nmf/hals_nmf.h
        src/nmf/hals_nmf.cpp
        src/utils/hdf5_loader.h
        src/experiments/nmf_evaluation.h
)

target_include_directories(main PRIVATE src)

find_package(OpenMP REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(HDF5 CONFIG REQUIRED)
find_package(OpenBLAS CONFIG REQUIRED)
find_package(CLI11 CONFIG REQUIRED)

target_link_libraries(main PRIVATE
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
        hdf5::hdf5-static
        OpenBLAS::OpenBLAS
        CLI11::CLI11
)

# ── Compiler flags ───────────────────────────────────────────────────
target_compile_options(main PRIVATE
        -O3                 # full optimisation
        -march=native       # enable AVX2/FMA/AVX512 for the current CPU
        -ffast-math         # allows SIMD reordering of FP ops (safe for NMF)
)

# ── Tell Eigen to use OpenBLAS for large dense matmuls ───────────────
# Without this Eigen uses its own implementation even if OpenBLAS is linked.
# With this, H*H.T, W^T*X etc. go through OpenBLAS dgemm/dsyrk.
target_compile_definitions(main PRIVATE
        EIGEN_USE_BLAS
)