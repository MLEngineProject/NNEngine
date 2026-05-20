cmake_minimum_required(VERSION 3.18)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

project(NN_ENGINE LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT MSVC)
    add_compile_options(-O3 -march=native)
endif()

include(FetchContent)

FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
endif()

FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.1.tar.gz
)
FetchContent_MakeAvailable(pybind11)

find_package(OpenMP REQUIRED)

find_package(BLAS)
find_package(LAPACK)

if(BLAS_FOUND)
    target_link_libraries(nn_core PRIVATE ${BLAS_LIBRARIES})
    target_compile_definitions(nn_core PRIVATE EIGEN_USE_BLAS)
endif()

pybind11_add_module(nn_core NO_EXTRAS
    src/binding.cpp 
    src/parametric/LogisticNeuron.cpp
    src/parametric/DenseLayer.cpp
    src/parametric/ReLULayer.cpp
)

target_include_directories(nn_core SYSTEM PRIVATE 
    include 
    ${eigen_SOURCE_DIR}
)

target_link_libraries(nn_core PRIVATE pybind11::module OpenMP::OpenMP_CXX)

target_compile_definitions(nn_core PRIVATE EIGEN_USE_THREADS)

install(TARGETS nn_core DESTINATION .)