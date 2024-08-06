cmake_minimum_required(VERSION 3.18) # 3.18 is required for CUDAToolkit
project(xtree LANGUAGES CXX CUDA) # Specify CUDA as a required language

# Set C++ Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_50,code=sm_50
    )

# FIND LIBRARIES
find_package(OpenCV QUIET REQUIRED)
find_package(CUDAToolkit QUIET REQUIRED) # Use CUDAToolkit instead of CUDA

# Find CUDA libraries
find_library(CUDA_LIBRARIES NAMES cudart HINTS ${CUDAToolkit_LIBRARY_DIR})

# Specify include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS} # Use CUDAToolkit_INCLUDE_DIRS instead of CUDA_INCLUDE_DIRS
)

# Target the only executable
add_executable(
    ${PROJECT_NAME}
    main.cu
)

add_executable(hello_world_cpu examples/hello_world_cpu.cu)
target_link_libraries(hello_world_cpu ${CUDA_LIBRARIES})

add_executable(hello_world examples/hello_world.cu)
target_link_libraries(hello_world ${CUDA_LIBRARIES})

add_executable(matrix_multiplication examples/matrix_multiplication.cu)
target_link_libraries(matrix_multiplication ${CUDA_LIBRARIES})

add_executable(vector_sum examples/vector_sum.cu)
target_link_libraries(vector_sum ${CUDA_LIBRARIES})

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(
    ${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)