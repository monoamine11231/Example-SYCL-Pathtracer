cmake_minimum_required(VERSION 3.22)

project(pathtracer
    VERSION 1.0
    DESCRIPTION "Simple Pathtracer"
    LANGUAGES CXX)

add_executable(pathtracer
    main.cpp
    include/VariantContainer.hpp)


find_package(RapidObj REQUIRED HINTS $ENV{HOME}/local/cmake)
target_compile_options(pathtracer PRIVATE -fsycl -std=c++17 -Isrc/ -Wall -Wextra -g)
target_link_libraries(pathtracer PRIVATE rapidobj::rapidobj)