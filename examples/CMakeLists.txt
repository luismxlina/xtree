# Define los ejecutables de ejemplos
add_executable(hello_world_cpu hello_world_cpu.cu)
add_executable(hello_world hello_world.cu)
add_executable(matrix_multiplication matrix_multiplication.cu)
add_executable(vector_sum vector_sum.cu)
add_executable(sobel_filter sobel_filter.cu)
add_executable(kernel kernel.cu)

# Vincula las bibliotecas a los ejecutables de ejemplo
target_link_libraries(hello_world_cpu ${CUDA_LIBRARIES})
target_link_libraries(hello_world ${CUDA_LIBRARIES})
target_link_libraries(matrix_multiplication ${CUDA_LIBRARIES})
target_link_libraries(vector_sum ${CUDA_LIBRARIES})
target_link_libraries(sobel_filter ${OpenCV_LIBS} ${CUDA_LIBRARIES})
target_link_libraries(kernel ${CUDA_LIBRARIES})
