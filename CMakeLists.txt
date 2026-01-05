cmake_minimum_required(VERSION 2.8.12)
project(shock_tube)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenMP REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIR}
    include
)

add_compile_options(-std=c++17)
add_executable(bgk_2d ./bgk_2d.cpp)
target_link_libraries(bgk_2d OpenMP::OpenMP_CXX)

add_executable(euler_2d euler_2d.cpp)
