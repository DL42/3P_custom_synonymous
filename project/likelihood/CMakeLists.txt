cmake_minimum_required(VERSION 2.8)
find_package(CUDA)
find_package(PythonLibs 3.6 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS} ../../3P/)
cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)
set(CUDA_NVCC_FLAGS "-std=c++11 -O3")
LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

cuda_add_library(mse_gpu SHARED
  ../../3P/_internal/spectrum.cu
  mse_gpu.cpp
  mse_gpu.cu)

target_link_libraries(mse_gpu
  ${PYTHON_LIBRARIES}
  ${CUDA_LIBRARIES})

set_target_properties(mse_gpu PROPERTIES PREFIX "" SUFFIX ".so")
