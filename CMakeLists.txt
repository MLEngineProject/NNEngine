cmake_minimum_required(VERSION 3.18)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

project(NN_ENGINE LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT MSVC)
    if(APPLE)
        add_compile_options(-O3 -ffast-math)
    else()
        add_compile_options(-O3 -march=native -ffast-math)
    endif()
endif()

include(FetchContent)

find_package(OpenMP)

FetchContent_Declare(
    eigen
    URL "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"
    SOURCE_SUBDIR ""
)
FetchContent_MakeAvailable(eigen)

FetchContent_Declare(
    pybind11
    URL "https://github.com/pybind/pybind11/archive/refs/tags/v2.13.1.tar.gz"
)
FetchContent_MakeAvailable(pybind11)

if(MSVC)
    message(WARNING "MSVC detected: Skipping OpenBLAS fetch. Falling back to Eigen internal math for Windows compatibility.")
else()
    set(NOFORTRAN 1 CACHE BOOL "Build without Fortran" FORCE)
    set(USE_OPENMP 1 CACHE BOOL "Use OpenMP in OpenBLAS" FORCE)
    set(BUILD_WITHOUT_LAPACK 1 CACHE BOOL "We only need BLAS for Eigen" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static OpenBLAS" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "Disable OpenBLAS tests" FORCE)

    if(APPLE AND (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64" OR CMAKE_OSX_ARCHITECTURES MATCHES "arm64|aarch64"))
        set(TARGET "VORTEX" CACHE STRING "OpenBLAS Apple Silicon Target" FORCE)
        message(STATUS "Apple Silicon detected: Forcing OpenBLAS target to VORTEX")
    endif()

    message(STATUS "Fetching and configuring OpenBLAS from source...")
    FetchContent_Declare(
        OpenBLAS
        URL "https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.26/OpenBLAS-0.3.26.tar.gz"
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(OpenBLAS)
endif()

pybind11_add_module(nn_core NO_EXTRAS
    src/binding.cpp 
    src/core/Model.cpp
    src/layers/DenseLayer.cpp
)

target_include_directories(nn_core SYSTEM PRIVATE include ${eigen_SOURCE_DIR})

if(NOT MSVC)
    target_link_libraries(nn_core PRIVATE pybind11::module openblas)
    target_compile_definitions(nn_core PRIVATE EIGEN_USE_BLAS)
else()
    target_link_libraries(nn_core PRIVATE pybind11::module)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(nn_core PRIVATE OpenMP::OpenMP_CXX)
    
    if(MSVC)
        target_compile_options(nn_core PRIVATE /openmp:experimental)
    endif()
    
    message(STATUS "OpenMP found: Multi-threading enabled in NNEngine.")
else()
    message(WARNING "OpenMP NOT found: Falling back to single-threaded NNEngine.")
endif()

install(TARGETS nn_core DESTINATION .)