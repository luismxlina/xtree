#include <stdio.h>

__global__ void helloFromGPU(void) {
    printf("¡Hola, mundo desde la GPU!\n");
}

int main(void) {
    printf("¡Hola, mundo desde la CPU!\n");

    helloFromGPU<<<1, 4>>>();  // Llama a kernel con 1 bloque y 4 hilos
    cudaDeviceReset();         // Espera a que todos los hilos terminen

    return 0;
}

// #include <cuda_runtime.h>

// #include <iostream>

// // Kernel de CUDA que ejecuta el código en la GPU
// __global__ void holaMundoKernel() {
//     printf("Hola Mundo desde el hilo %d\n", threadIdx.x);
// }

// int main() {
//     // Llamar al kernel con 1 bloque y 5 hilos
//     holaMundoKernel<<<1, 5>>>();

//     // Esperar a que todos los threads terminen
//     cudaDeviceSynchronize();

//     // Devolver el control al host y verificar errores de lanzamiento
//     cudaError_t error = cudaGetLastError();
//     if (error != cudaSuccess) {
//         std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
//     }

//     return 0;
// }
