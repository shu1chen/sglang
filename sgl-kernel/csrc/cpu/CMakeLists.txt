cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(sgl_kernel)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE}
            -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_PY_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS ${TORCH_PY_PREFIX})
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PY_PREFIX}/Torch)
find_package(Torch REQUIRED)

include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INSTALL_PREFIX}/include
    ${Python_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Platform-specific library directory
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(PLAT_LIB_DIR "/usr/lib/x86_64-linux-gnu")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(PLAT_LIB_DIR "/usr/lib/aarch64-linux-gnu")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le|ppc64")
    set(PLAT_LIB_DIR "/usr/lib/powerpc64le-linux-gnu")
else()
    set(PLAT_LIB_DIR "/usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu")
endif()
link_directories(${PLAT_LIB_DIR})

# ========== oneDNN Configuration ==========
# Priority order:
# 1. Custom path from DNNL_ROOT environment variable
# 2. Conda environment
# 3. System installation
# 4. CMake find_package

set(DNNL_FOUND FALSE)

# Option 1: Check for custom oneDNN installation via DNNL_ROOT
if(DEFINED ENV{DNNL_ROOT})
    set(DNNL_ROOT "$ENV{DNNL_ROOT}")
    message(STATUS "Looking for oneDNN in custom path: ${DNNL_ROOT}")
    
    # Add to CMAKE_PREFIX_PATH for find_package
    list(APPEND CMAKE_PREFIX_PATH "${DNNL_ROOT}")
    
    # Try to find oneDNN package config
    find_package(dnnl CONFIG QUIET PATHS "${DNNL_ROOT}" NO_DEFAULT_PATH)
    
    if(dnnl_FOUND)
        message(STATUS "Found oneDNN package config at: ${DNNL_ROOT}")
        set(DNNL_FOUND TRUE)
        set(DNNL_INCLUDE_DIRS "${DNNL_ROOT}/include")
        
        # Try to find the actual library file
        find_library(DNNL_LIBRARY 
            NAMES dnnl libdnnl
            PATHS "${DNNL_ROOT}/lib" "${DNNL_ROOT}/lib64"
            NO_DEFAULT_PATH
        )
        
        if(DNNL_LIBRARY)
            set(DNNL_LIBRARIES "${DNNL_LIBRARY}")
            message(STATUS "oneDNN library: ${DNNL_LIBRARY}")
        else()
            # Fallback to target name and add link directory
            set(DNNL_LIB_DIR "${DNNL_ROOT}/lib")
            link_directories(${DNNL_LIB_DIR})
            set(DNNL_LIBRARIES dnnl)
            message(STATUS "oneDNN library directory: ${DNNL_LIB_DIR}")
        endif()
    else()
        # Fallback: manually set paths for custom build
        set(DNNL_INCLUDE_DIR "${DNNL_ROOT}/include")
        set(DNNL_LIB_DIR "${DNNL_ROOT}/lib")
        
        if(EXISTS "${DNNL_INCLUDE_DIR}/oneapi/dnnl/dnnl.hpp")
            find_library(DNNL_LIB dnnl HINTS "${DNNL_LIB_DIR}" NO_DEFAULT_PATH)
            if(DNNL_LIB)
                message(STATUS "Found custom oneDNN library: ${DNNL_LIB}")
                message(STATUS "oneDNN include directory: ${DNNL_INCLUDE_DIR}")
                set(DNNL_FOUND TRUE)
                set(DNNL_INCLUDE_DIRS "${DNNL_INCLUDE_DIR}")
                set(DNNL_LIBRARIES "${DNNL_LIB}")
                include_directories(${DNNL_INCLUDE_DIRS})
                link_directories(${DNNL_LIB_DIR})
            else()
                message(WARNING "DNNL_ROOT is set but library not found in ${DNNL_LIB_DIR}")
            endif()
        else()
            message(WARNING "DNNL_ROOT is set but headers not found in ${DNNL_INCLUDE_DIR}")
        endif()
    endif()
endif()

# Conda library path support
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_LIB_DIR "$ENV{CONDA_PREFIX}/lib")
    message(STATUS "Using Conda lib dir: ${CONDA_LIB_DIR}")
    link_directories(${CONDA_LIB_DIR})
    set(CONDA_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include")
    include_directories(${CONDA_INCLUDE_DIR})

    # Look for libnuma in Conda's lib directory
    find_library(NUMA_LIB numa HINTS "${CONDA_LIB_DIR}")
    if(NUMA_LIB)
        message(STATUS "Found libnuma: ${NUMA_LIB}")
    else()
        message(FATAL_ERROR "libnuma not found in Conda environment at ${CONDA_LIB_DIR}\n"
                            "Please install it using: conda install libnuma numactl\n")
    endif()
    
    # Option 2: Look for oneDNN in Conda (if not already found)
    if(NOT DNNL_FOUND)
        find_library(DNNL_LIB dnnl HINTS "${CONDA_LIB_DIR}")
        if(DNNL_LIB)
            message(STATUS "Found oneDNN in Conda: ${DNNL_LIB}")
            set(DNNL_FOUND TRUE)
            set(DNNL_INCLUDE_DIRS "${CONDA_INCLUDE_DIR}")
            set(DNNL_LIBRARIES "${DNNL_LIB}")
        else()
            message(WARNING "oneDNN not found in Conda environment at ${CONDA_LIB_DIR}\n"
                            "Please install it using: conda install onednn\n")
        endif()
    endif()
else()
    if(DEFINED ENV{VIRTUAL_ENV})
        set(VENV_LIB_DIR "$ENV{VIRTUAL_ENV}/lib")
        message(STATUS "Using venv lib dir: ${VENV_LIB_DIR}")
        link_directories(${VENV_LIB_DIR})
        set(VENV_INCLUDE_DIR "$ENV{VIRTUAL_ENV}/include")
        include_directories(${VENV_INCLUDE_DIR})
    endif()
    
    # Look for libnuma in system env paths
    find_library(NUMA_LIB numa)
    if(NUMA_LIB)
        message(STATUS "Found libnuma: ${NUMA_LIB}")
    else()
        message(FATAL_ERROR "libnuma not found in system environment\n"
                            "Please install it using: apt-get install libnuma numactl\n")
    endif()
    
    # Option 3: Look for oneDNN in system env paths (if not already found)
    if(NOT DNNL_FOUND)
        find_package(dnnl CONFIG QUIET)
        if(dnnl_FOUND)
            message(STATUS "Found oneDNN via find_package")
            set(DNNL_FOUND TRUE)
            set(DNNL_LIBRARIES dnnl)
        else()
            find_library(DNNL_LIB dnnl)
            if(DNNL_LIB)
                message(STATUS "Found oneDNN in system: ${DNNL_LIB}")
                set(DNNL_FOUND TRUE)
                set(DNNL_LIBRARIES "${DNNL_LIB}")
            else()
                message(WARNING "oneDNN not found in system environment\n"
                                "To use oneDNN optimizations, please:\n"
                                "1. Set DNNL_ROOT=/path/to/onednn/install, or\n"
                                "2. Install via: apt-get install libdnnl-dev, or\n"
                                "3. Build from source (see instructions)\n")
            endif()
        endif()
    endif()
endif()

# Summary
if(DNNL_FOUND)
    message(STATUS "========================================")
    message(STATUS "oneDNN Configuration:")
    message(STATUS "  Found: YES")
    if(DNNL_INCLUDE_DIRS)
        message(STATUS "  Include dirs: ${DNNL_INCLUDE_DIRS}")
    endif()
    message(STATUS "  Libraries: ${DNNL_LIBRARIES}")
    message(STATUS "========================================")
else()
    message(STATUS "========================================")
    message(STATUS "oneDNN Configuration:")
    message(STATUS "  Found: NO")
    message(STATUS "  Building without oneDNN optimizations")
    message(STATUS "========================================")
endif()

file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

if(NOT DEFINED ENV{SGLANG_CPU_FP8_CVT_FTZ})
    set(ENV{SGLANG_CPU_FP8_CVT_FTZ} "1")
endif()

if("$ENV{SGLANG_CPU_FP8_CVT_FTZ}" STREQUAL "1")
    message(STATUS "Enabling macro: SGLANG_CPU_FP8_CVT_FTZ")
    add_compile_definitions(SGLANG_CPU_FP8_CVT_FTZ)
endif()

# Check for SGLANG_USE_ONEDNN_MOE environment variable to enable oneDNN MoE implementation
# Value 0 or unset: use original brgemm/tinygemm implementation
# Value 1: use oneDNN with sequential expert loop
# Value 2: use oneDNN with batched matmul (all experts in one call)
if(DEFINED ENV{SGLANG_USE_ONEDNN_MOE})
    set(ONEDNN_MOE_VALUE "$ENV{SGLANG_USE_ONEDNN_MOE}")
    if(ONEDNN_MOE_VALUE STREQUAL "1" OR ONEDNN_MOE_VALUE STREQUAL "2")
        if(DNNL_FOUND)
            message(STATUS "Enabling macro: SGLANG_USE_ONEDNN_MOE=${ONEDNN_MOE_VALUE} (using oneDNN for MoE INT8)")
            add_compile_definitions(SGLANG_USE_ONEDNN_MOE=${ONEDNN_MOE_VALUE})
        else()
            message(WARNING "SGLANG_USE_ONEDNN_MOE is set but oneDNN was not found. Using default implementation.")
        endif()
    endif()
endif()

add_compile_options(
    -O3
    -Wno-unknown-pragmas
    -march=native
    -fopenmp
)

Python_add_library(common_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})
target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES} ${NUMA_LIB})

# Link oneDNN if found
if(DNNL_FOUND)
    target_link_libraries(common_ops PRIVATE ${DNNL_LIBRARIES})
    if(DNNL_INCLUDE_DIRS)
        target_include_directories(common_ops PRIVATE ${DNNL_INCLUDE_DIRS})
    endif()
    message(STATUS "Linking oneDNN to common_ops")
    
    # Set RPATH for runtime library search (important for custom installations)
    if(DEFINED ENV{DNNL_ROOT})
        set_target_properties(common_ops PROPERTIES
            BUILD_RPATH "$ENV{DNNL_ROOT}/lib"
            INSTALL_RPATH "$ENV{DNNL_ROOT}/lib"
        )
        message(STATUS "Setting RPATH to: $ENV{DNNL_ROOT}/lib")
    endif()
endif()

target_include_directories(common_ops PRIVATE ${TORCH_INCLUDE_DIRS})

install(TARGETS common_ops
    LIBRARY DESTINATION sgl_kernel
)
