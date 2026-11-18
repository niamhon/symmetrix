cmake_minimum_required(VERSION 3.20)
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/../VERSION SYMMETRIX_VERSION)
project(LibSymmetrix VERSION ${SYMMETRIX_VERSION})

message(STATUS "Symmetrix: Beginning to process CMakeLists.txt.")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ----- dependencies -----

# sphericart fetches gpu-lite at configure time; use the vendored submodule so builds work offline.
set(FETCHCONTENT_SOURCE_DIR_GPULITE ${CMAKE_CURRENT_SOURCE_DIR}/external/gpu-lite CACHE PATH "")
if(SYMMETRIX_SPHERICART_CUDA)
    set(SPHERICART_ENABLE_CUDA ON CACHE BOOL "" FORCE)
endif()
add_subdirectory(external/sphericart)
option(SYMMETRIX_KOKKOS "Enable Kokkos mode in Symmetrix." ON)
if(SYMMETRIX_KOKKOS)
    if(TARGET Kokkos::kokkos)
        message(STATUS "Symmetrix: Found existing Kokkos target.")
        set(BUILD_SHARED_LIBS OFF)
    else()
        add_subdirectory(external/kokkos)
    endif()
    add_subdirectory(external/kokkos-kernels)
endif()

# ----- libsymmetrix -----

set(SYMMETRIX_SOURCE_FILES
    source/cubic_spline.cpp
    source/cubic_spline_set.cpp
    source/mace.cpp
    source/mace_lsc.cpp
    source/mace_lsc_cuda_parameters.cpp
    source/mace_polar.cpp
    source/multilayer_perceptron.cpp
    source/multivariate_polynomial.cpp
    source/nequip.cpp
    source/spherical_harmonic.cpp
    source/tools.cpp
    source/zbl.cpp)

if(SYMMETRIX_KOKKOS)
    list(APPEND SYMMETRIX_SOURCE_FILES
        source/cubic_spline_kokkos.cpp
        source/cubic_spline_set_kokkos.cpp
        source/mace_kokkos.cpp
        source/multilayer_perceptron_kokkos.cpp
        source/multivariate_polynomial_kokkos.cpp
        source/nequip_kokkos.cpp
        source/radial_function_set_kokkos.cpp
        source/tools_kokkos.cpp
        source/zbl_kokkos.cpp)
endif()

add_library(symmetrix ${SYMMETRIX_SOURCE_FILES})
# Public graph headers use std::span; consumers must compile as C++20 too.
target_compile_features(symmetrix PUBLIC cxx_std_20)

# Optional standalone one-process LSC mesh backend; ordinary builds need no FFTW.
option(SYMMETRIX_LSC_MESH "Build CPU LSC mesh energy with FFTW3" OFF)
if(SYMMETRIX_LSC_MESH)
    find_path(LSC_FFTW3_INCLUDE_DIR fftw3.h REQUIRED)
    find_library(LSC_FFTW3_LIBRARY NAMES fftw3 REQUIRED)
    target_sources(symmetrix PRIVATE source/mace_lsc_mesh.cpp)
    target_include_directories(symmetrix PRIVATE ${LSC_FFTW3_INCLUDE_DIR})
    target_link_libraries(symmetrix PRIVATE ${LSC_FFTW3_LIBRARY})
else()
    target_sources(symmetrix PRIVATE source/mace_lsc_mesh_stub.cpp)
endif()

option(SYMMETRIX_NATIVE_ARCH "Tune host C++ code for the build CPU" ON)

# TODO: generalize
target_compile_options(symmetrix
    PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${SYMMETRIX_NATIVE_ARCH}>>:-march=native>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:NVHPC>>>:-ffast-math>)

target_include_directories(symmetrix
    PUBLIC  source
    PRIVATE external/json/single_include
            external/sphericart/sphericart/include)

target_compile_definitions(symmetrix PUBLIC SYMMETRIX_VERSION="${PROJECT_VERSION}")

target_link_libraries(symmetrix
    PRIVATE sphericart)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(symmetrix PRIVATE OpenMP::OpenMP_CXX)
endif()
if(SYMMETRIX_KOKKOS)
    target_link_libraries(symmetrix
        PRIVATE Kokkos::kokkos
                Kokkos::kokkoskernels)
    if(Kokkos_ENABLE_CUDA)
        target_compile_options(symmetrix PRIVATE $<$<CONFIG:Release>:--use_fast_math>)
        find_package(CUDAToolkit REQUIRED)
        target_sources(symmetrix PRIVATE source/mace_polar_cuda.cpp source/mace_lsc_cuda.cpp source/mace_lsc_cuda_mesh.cpp source/mace_lsc_cuda_partition_mesh.cpp)
        set_source_files_properties(source/mace_polar_cuda.cpp source/mace_lsc_cuda.cpp source/mace_lsc_cuda_mesh.cpp source/mace_lsc_cuda_partition_mesh.cpp PROPERTIES COMPILE_OPTIONS --expt-relaxed-constexpr)
        target_link_libraries(symmetrix PRIVATE CUDA::cublas CUDA::cufft)
    endif()
endif()

# A CPU stub keeps the API/host parameter tests usable without CUDA dependencies.
if(NOT (SYMMETRIX_KOKKOS AND Kokkos_ENABLE_CUDA))
    target_sources(symmetrix PRIVATE source/mace_lsc_cuda_stub.cpp source/mace_lsc_cuda_mesh_stub.cpp)
endif()


# ----- options -----

option(SYMMETRIX_SPHERICART_CUDA OFF)
if (SYMMETRIX_SPHERICART_CUDA)
    message(STATUS "Symmetrix: Will use CUDA version of sphericart.")
    target_compile_definitions(symmetrix PRIVATE SYMMETRIX_SPHERICART_CUDA)
endif()
option(SYMMETRIX_SPHERICART_SYCL OFF)
if (SYMMETRIX_SPHERICART_SYCL)
    message(STATUS "Symmetrix: Will use SYCL version of sphericart.")
    target_compile_definitions(symmetrix PRIVATE SYMMETRIX_SPHERICART_SYCL)
endif()
