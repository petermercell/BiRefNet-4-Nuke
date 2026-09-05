cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# ============================================================================
# CRITICAL: Set MSVC runtime library policy BEFORE project()
# ============================================================================
cmake_policy(SET CMP0091 NEW)

project(TRTBiRefNet VERSION 1.0.0 LANGUAGES CXX)

# ============================================================================
# Windows-only guard
# ============================================================================
if(NOT WIN32)
    message(FATAL_ERROR "This CMakeLists.txt is for Windows only")
endif()

if(MSVC)
    if(MSVC_VERSION LESS 1930)
        message(FATAL_ERROR "Visual Studio 2022 or later required (MSVC 19.30+)")
    endif()
    message(STATUS "Using MSVC version ${MSVC_VERSION}")
else()
    message(FATAL_ERROR "This build requires MSVC (Visual Studio)")
endif()

# ============================================================================
# C++ Standard
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ============================================================================
# CRITICAL: Static runtime — no VC++ Redistributable needed
# ============================================================================
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# ============================================================================
# Windows Definitions & Compiler Flags
# ============================================================================
add_definitions(
    -DWIN32
    -D_WINDOWS
    -DNDEBUG
    -DNOMINMAX
    -D_CRT_SECURE_NO_WARNINGS
    -D_SCL_SECURE_NO_WARNINGS
    -D_USE_MATH_DEFINES
    -DWIN32_LEAN_AND_MEAN
)

add_compile_options(
    /W3 /O2 /Ob2 /EHsc /FS /nologo /MP
    /wd4996 /wd4244 /wd4267 /wd4305 /wd4251 /wd4018 /wd4514 /wd4710 /wd4711 /wd4668
)

# ============================================================================
# CUDA
# ============================================================================
find_package(CUDAToolkit QUIET)

if(NOT CUDAToolkit_FOUND)
    # Manual fallback
    set(CUDA_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1"
        CACHE PATH "CUDA installation root")
    find_program(NVCC_EXECUTABLE nvcc PATHS "${CUDA_ROOT}/bin" NO_DEFAULT_PATH)
    if(NOT NVCC_EXECUTABLE)
        message(FATAL_ERROR "CUDA not found. Set CUDA_ROOT or install CUDA 13.1.")
    endif()
    set(CUDA_INCLUDE_DIR "${CUDA_ROOT}/include")
    set(CUDA_LIBRARY_DIR "${CUDA_ROOT}/lib/x64")
    message(STATUS "CUDA found manually: ${CUDA_ROOT}")
else()
    message(STATUS "CUDA ${CUDAToolkit_VERSION} found via CUDAToolkit")
endif()

# ============================================================================
# TensorRT
# ============================================================================
set(TENSORRT_ROOT "C:/TensorRT-10.15.1.29" CACHE PATH "TensorRT installation root")

if(NOT EXISTS "${TENSORRT_ROOT}/include/NvInfer.h")
    message(FATAL_ERROR "TensorRT headers not found at ${TENSORRT_ROOT}/include — check TENSORRT_ROOT")
endif()

find_library(NVINFER_LIBRARY
    NAMES nvinfer_10 nvinfer
    PATHS "${TENSORRT_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)
message(STATUS "TensorRT found: ${TENSORRT_ROOT}")
message(STATUS "  nvinfer: ${NVINFER_LIBRARY}")

# ============================================================================
# Nuke NDK
# ============================================================================
if(NOT DEFINED NUKE_VERSION)
    set(NUKE_VERSION "17.1v1")
endif()

if(NOT DEFINED NUKE_ROOT)
    set(NUKE_ROOT "C:/Program Files/Nuke${NUKE_VERSION}")
endif()

set(NUKE_INCLUDE_DIR "${NUKE_ROOT}/include")

if(NOT EXISTS "${NUKE_INCLUDE_DIR}/DDImage/Iop.h")
    message(FATAL_ERROR "Nuke headers not found at ${NUKE_INCLUDE_DIR} — check NUKE_ROOT or NUKE_VERSION")
endif()

if(NOT EXISTS "${NUKE_ROOT}/DDImage.lib")
    message(FATAL_ERROR "DDImage.lib not found in ${NUKE_ROOT}")
endif()

message(STATUS "Nuke ${NUKE_VERSION}: ${NUKE_ROOT}")

# ============================================================================
# Build Plugin
# ============================================================================
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(SOURCES TRTBiRefNet.cpp)

add_library(TRTBiRefNet SHARED ${SOURCES})

# Static runtime for this target
set_property(TARGET TRTBiRefNet PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# ---- Includes ----
target_include_directories(TRTBiRefNet PRIVATE
    ${NUKE_INCLUDE_DIR}
    ${TENSORRT_ROOT}/include
)

if(CUDAToolkit_FOUND)
    target_include_directories(TRTBiRefNet PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
else()
    target_include_directories(TRTBiRefNet PRIVATE ${CUDA_INCLUDE_DIR})
endif()

# ---- Links ----
target_link_libraries(TRTBiRefNet PRIVATE
    "${NUKE_ROOT}/DDImage.lib"
    ${NVINFER_LIBRARY}
)

if(CUDAToolkit_FOUND)
    if(TARGET CUDA::cudart_static)
        target_link_libraries(TRTBiRefNet PRIVATE CUDA::cudart_static)
        message(STATUS "Using static CUDA runtime")
    else()
        target_link_libraries(TRTBiRefNet PRIVATE CUDA::cudart)
        message(STATUS "Using dynamic CUDA runtime")
    endif()
else()
    set(CUDART_LIB "${CUDA_LIBRARY_DIR}/cudart_static.lib")
    if(NOT EXISTS "${CUDART_LIB}")
        set(CUDART_LIB "${CUDA_LIBRARY_DIR}/cudart.lib")
    endif()
    target_link_libraries(TRTBiRefNet PRIVATE "${CUDART_LIB}")
endif()

# ---- Output properties ----
set_target_properties(TRTBiRefNet PROPERTIES
    OUTPUT_NAME "TRTBiRefNet"
    SUFFIX ".dll"
    PREFIX ""
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/lib
)

# ============================================================================
# Installation
# ============================================================================
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "$ENV{USERPROFILE}/.nuke"
        CACHE PATH "Install path" FORCE)
endif()

install(TARGETS TRTBiRefNet
    RUNTIME DESTINATION .
    LIBRARY DESTINATION .
)

# ============================================================================
# Build Summary
# ============================================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "TRTBiRefNet Windows Build Configuration")
message(STATUS "========================================")
message(STATUS "  Nuke:          ${NUKE_ROOT}")
message(STATUS "  TensorRT:      ${TENSORRT_ROOT}")
if(CUDAToolkit_FOUND)
message(STATUS "  CUDA:          ${CUDAToolkit_VERSION} (CUDAToolkit)")
else()
message(STATUS "  CUDA:          ${CUDA_ROOT} (manual)")
endif()
message(STATUS "  Build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  Runtime:       STATIC (/MT) - no VC++ Redist needed")
message(STATUS "  Output:        TRTBiRefNet.dll")
message(STATUS "  Install:       ${CMAKE_INSTALL_PREFIX}")
message(STATUS "========================================")
message(STATUS "")