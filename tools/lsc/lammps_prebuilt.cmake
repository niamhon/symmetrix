# Development integration: reuse the validated native build while compiling
# LAMMPS with its normal host compiler. The standard pair_symmetrix/install.sh
# also installs the LSC style and builds libsymmetrix from source as usual.
if(NOT EXISTS "${LSC_SYMMETRIX_LIBRARY}" OR NOT EXISTS "${LSC_SPHERICART_LIBRARY}")
  message(FATAL_ERROR "Build the Symmetrix CPU/CUDA development library first")
endif()
add_library(symmetrix_lsc_backend STATIC IMPORTED)
set_target_properties(symmetrix_lsc_backend PROPERTIES
  IMPORTED_LOCATION "${LSC_SYMMETRIX_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${LSC_SYMMETRIX_ROOT}/libsymmetrix/source")
# These development libraries include the optional host mesh evaluator.
find_library(LSC_FFTW3_LIBRARY NAMES fftw3 REQUIRED)
find_package(OpenMP REQUIRED)
target_link_libraries(lammps PRIVATE symmetrix_lsc_backend "${LSC_SPHERICART_LIBRARY}" "${LSC_FFTW3_LIBRARY}" OpenMP::OpenMP_CXX)
# Symmetrix's public graph API requires C++20 (std::span).
set_target_properties(lammps PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
if(LSC_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(lammps PRIVATE SYMMETRIX_CUDA)
  target_link_libraries(lammps PRIVATE CUDA::cudart CUDA::cublas CUDA::cufft)
endif()

# Polar-style distributed grid halos and pencil remaps, double-precision FFT.
if(PKG_KSPACE)
  target_compile_definitions(lammps PRIVATE SYMMETRIX_KSPACE)
endif()
target_include_directories(lammps PRIVATE "${LSC_SYMMETRIX_ROOT}/pair_symmetrix")
target_sources(lammps PRIVATE "${LSC_SYMMETRIX_ROOT}/pair_symmetrix/pair_symmetrix_mace_lsc_mesh.cpp")
