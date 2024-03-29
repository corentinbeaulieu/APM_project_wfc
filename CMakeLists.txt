cmake_minimum_required(VERSION 3.17)
project(WaveFunctionCollapse VERSION 0.1 LANGUAGES C)
cmake_policy(SET CMP0009 NEW) # Do not follow symlinks with GLOB_RECURSE

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

STRING(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}" insource)
GET_FILENAME_COMPONENT(PARENTDIR ${CMAKE_SOURCE_DIR} PATH)
STRING(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${PARENTDIR}" insourcesubdir)
if(insource OR insourcesubdir)
    MESSAGE(FATAL_ERROR "${CMAKE_PROJECT_NAME} requires an out of source build")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
else()
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON) # Pass -fPIC
set(CMAKE_COLOR_MAKEFILE            ON)
set(CMAKE_COLOR_DIAGNOSTICS         ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS   ON) # Always to that...
set(THREADS_PREFER_PTHREAD_FLAG     ON) # Pthread ftw

file(GLOB_RECURSE WaveFunctionCollapse_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")
file(GLOB_RECURSE WaveFunctionCollapse_INC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h")

add_executable(wfc ${WaveFunctionCollapse_SRC} ${WaveFunctionCollapse_INC})
target_include_directories(wfc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
target_link_libraries(wfc  PRIVATE -pthread -fopenmp -lm)
target_compile_options(wfc PRIVATE -std=c17 -march=native -g -O3 -funroll-loops -pthread -fopenmp)
target_compile_options(wfc PRIVATE
    -Wall -Wextra -pedantic -Werror
    -Wshadow
    -Wcast-align -Wconversion -Wno-sign-conversion -Wdouble-promotion
    -Wunused-variable
    -Wmisleading-indentation
    -Wnull-dereference
    -Wformat=2
)

if(${CMAKE_C_COMPILER} STREQUAL "clang")
    target_compile_options(wfc PRIVATE
        -Weverything
        -Wno-c99-compat
        -Wno-vla
        -Wno-disabled-macro-expansion
        -Wno-padded
        -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60
        # -Wno-unsafe-buffer-usage
        -Wno-declaration-after-statement
    )
elseif(${CMAKE_C_COMPILER} STREQUAL "GNU")
    target_compile_options(wfc PRIVATE
        -Wno-switch
        -Wno-unsafe-buffer-usage
    )
endif()
