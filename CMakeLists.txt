cmake_minimum_required(VERSION 3.20)
project(nmf-ivf)

set(CMAKE_CXX_STANDARD 20)

# ── Shared source files (used by both executables) ───────────────────
set(INDEX_SOURCES
        src/index/nmf_index.cpp
        src/index/nmf/base.cpp
        src/index/nmf/mu_nmf.cpp
        src/index/nmf/hals_nmf.cpp
        src/index/backend/base.cpp
        src/index/backend/naive.cpp
        src/index/backend/adaptive.cpp
)


add_executable(main
        main.cpp
        src/run.h
        ${INDEX_SOURCES}
)

add_executable(recall_benchmark
        src/experiments/recall_analysis.cpp
        ${INDEX_SOURCES}
)

add_executable(index_coverage_benchmark
        src/experiments/index_coverage_analysis.cpp
        ${INDEX_SOURCES}
)

add_executable(sampling_recall_analysis
        src/experiments/sampling_recall_analysis.cpp
        src/run.h
        ${INDEX_SOURCES}
)

add_executable(k_sweep
        src/experiments/k_sweep.cpp
        ${INDEX_SOURCES}
)

find_package(OpenMP REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(HDF5 CONFIG REQUIRED)
find_package(OpenBLAS CONFIG REQUIRED)
find_package(CLI11 CONFIG REQUIRED)

foreach (TARGET main recall_benchmark index_coverage_benchmark sampling_recall_analysis k_sweep)
    target_include_directories(${TARGET} PRIVATE src)

    target_link_libraries(${TARGET} PRIVATE
            Eigen3::Eigen
            OpenMP::OpenMP_CXX
            hdf5::hdf5-static
            OpenBLAS::OpenBLAS
            CLI11::CLI11
    )

    target_compile_options(${TARGET} PRIVATE
            -O3
            -march=native
            -ffast-math
    )

    target_compile_definitions(${TARGET} PRIVATE
            EIGEN_USE_BLAS
    )
endforeach ()