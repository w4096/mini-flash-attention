cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.0/bin/nvcc")

project(cute LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_ARCHITECTURES native)

project(mini-flash-attention CXX CUDA)

# cutlass
set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rd/cutlass")
include_directories(${CUTLASS_DIR}/include)
include_directories(${CUTLASS_DIR}/tools/util/include)

include_directories(/usr/local/cuda/include/cccl/)

set(TORCH_CUDA_ARCH_LIST 12.0)

set(Torch_DIR ~/Downloads/libtorch-shared-with-deps-2.9.1+cu130/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

#find_package(Python3 REQUIRED Interpreter Development)
#include_directories(${Python3_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
#
#link_libraries(${Python3_LIBRARIES})
link_libraries(${TORCH_LIBRARIES})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/csrc")
FILE(GLOB_RECURSE SRC_FILES "csrc/*/*.cuh" "csrc/*/*.cpp" "src/*/*.h" "csrc/*/*.cu")
#LIST(REMOVE_ITEM SRC_FILES "csrc/api.cpp")
#add_definitions(-DMFA_DEV=0)
add_executable(flash ${SRC_FILES})
