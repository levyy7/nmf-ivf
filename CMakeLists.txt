cmake_minimum_required(VERSION 3.20)
project(chnsw)

set(CMAKE_CXX_STANDARD 20)

add_executable(main src/main.cc)

target_include_directories(main PRIVATE
        src/utils
)

find_package(Eigen3 CONFIG REQUIRED)
find_package(hdf5 CONFIG REQUIRED)

target_link_libraries(main PRIVATE Eigen3::Eigen)
target_link_libraries(main PRIVATE hdf5::hdf5-static)