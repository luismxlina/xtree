# Define el ejecutable principal usando PROJECT_NAME
add_executable(${PROJECT_NAME} main.cu)

# Vincula las bibliotecas al ejecutable xtree
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${CUDA_LIBRARIES})
