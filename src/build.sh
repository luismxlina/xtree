#!/bin/bash

# Compilar kernels.cu
nvcc -c -o kernels.o kernels.cu

# Compilar main.cu y enlazar con kernels.o y OpenCV
nvcc -o image_processor main.cu kernels.o $(pkg-config --cflags --libs opencv4) -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -std=c++11
