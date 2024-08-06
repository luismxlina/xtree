#include <cuda_runtime.h>

// Kernel para contar los valores RGB de cada píxel
__global__ void count_colors(const unsigned char* input, int width, int height, int* red_count, int* green_count, int* blue_count) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;  // Asumiendo una imagen RGB
        atomicAdd(red_count, input[idx]);
        atomicAdd(green_count, input[idx + 1]);
        atomicAdd(blue_count, input[idx + 2]);
    }
}
