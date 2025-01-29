cmake_minimum_required(VERSION 3.10)
project(xielu LANGUAGES CXX)

set(LIB_INSTALL_DIR "lib" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install libraries")
set(BIN_INSTALL_DIR "bin" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install DLL/binaries")
set(INCLUDE_INSTALL_DIR "include" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install headers")

find_package(Python COMPONENTS Interpreter Development REQUIRED)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE BOOL "" FORCE)
else()
    message(FATAL_ERROR "Could not find CUDA Compiler. Make sure $CUDA_HOME is set.")
endif()

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
    RESULT_VARIABLE TORCH_CMAKE_PATH_RESULT
    OUTPUT_VARIABLE TORCH_CMAKE_PATH_OUTPUT
    ERROR_VARIABLE TORCH_CMAKE_PATH_ERROR
)

if (NOT ${TORCH_CMAKE_PATH_RESULT} EQUAL 0)
message(FATAL_ERROR "failed to find your pytorch installation, error: ${TORCH_CMAKE_PATH_ERROR}\n")
endif()

string(STRIP ${TORCH_CMAKE_PATH_OUTPUT} TORCH_CMAKE_PATH_OUTPUT)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${TORCH_CMAKE_PATH_OUTPUT}")

message(STATUS "TORCH_CMAKE_PATH_OUTPUT: ${TORCH_CMAKE_PATH_OUTPUT}")

find_package(Torch 2.0 REQUIRED)

add_library(xielu SHARED
    "common/utils.cuh"

    "cpp/include/xielu.hpp"
    "cpp/src/xielu.cpp"

    "cuda/include/xielu_impl.hpp"
    "cuda/src/xielu_impl.cu"
)

target_include_directories(xielu PRIVATE
    ${CMAKE_SOURCE_DIR}/cuda/include
    ${CMAKE_SOURCE_DIR}/cpp/include
    ${CMAKE_SOURCE_DIR}/common
    ${Python_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(xielu PRIVATE "${TORCH_LIBRARIES}")

target_compile_features(xielu PRIVATE cxx_std_17)

install(TARGETS xielu
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/${LIB_INSTALL_DIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${BIN_INSTALL_DIR}
)
