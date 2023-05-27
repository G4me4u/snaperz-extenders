cmake_minimum_required(VERSION 3.12)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

project (extender)
add_compile_definitions("_DEBUG=$<CONFIG:Debug>")
add_executable(extender
    src/main.cpp
)
