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
