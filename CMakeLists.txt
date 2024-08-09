cmake_minimum_required(VERSION 3.18) # Requerido para CUDAToolkit
project(xtree LANGUAGES CXX CUDA) # Definir el nombre del proyecto y los lenguajes

# Establecer el estándar de C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Búsqueda de bibliotecas y paquetes
find_package(OpenCV QUIET REQUIRED)
find_package(CUDAToolkit QUIET REQUIRED)

# Encontrar las bibliotecas de CUDA
find_library(CUDA_LIBRARIES NAMES cudart HINTS ${CUDAToolkit_LIBRARY_DIR})

# Directorios de inclusión globales
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Opciones de NVCC
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_50,code=sm_50
)

# Incluir subdirectorios
add_subdirectory(src)
add_subdirectory(examples)
