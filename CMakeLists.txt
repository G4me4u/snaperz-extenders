cmake_minimum_required(VERSION 3.12)

set (CMAKE_CXX_STANDARD 14)

project (extender)
add_compile_definitions("_DEBUG=$<CONFIG:Debug>")
add_executable(extender
    src/main.cpp
)
