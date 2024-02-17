cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

project(pathtracer
    VERSION 1.0
    DESCRIPTION "Simple Pathtracer"
    LANGUAGES CXX)

add_executable(pathtracer
    main.cc
    src/utils.cc
    src/camera.cc
    src/object.cc
    src/material.cc
    src/objects/plane.cc
    src/objects/sphere.cc)

target_include_directories(pathtracer PRIVATE ${DPCPP_HOME}/llvm/build/install/include ${HOME}/local/include/)
target_compile_options(pathtracer PRIVATE -fsycl -std=c++20 -I./ -Wall -Wextra -g)