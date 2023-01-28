cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

project(pathtracer
    VERSION 1.0
    DESCRIPTION "Simple Pathtracer"
    LANGUAGES CXX)

add_executable(pathtracer
    main.cc
    src/camera.cc
    include/variantcontainer.h)

target_include_directories(pathtracer PRIVATE ${DPCPP_HOME}/llvm/build/install/include)
find_package(RapidObj REQUIRED HINTS $ENV{HOME}/local/cmake)
target_compile_options(pathtracer PRIVATE -fsycl -std=c++17 -I./ -Wall -Wextra -g)
target_link_libraries(pathtracer PRIVATE rapidobj::rapidobj)