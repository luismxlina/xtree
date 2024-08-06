#include <stdio.h>

__global__ void matMul(int n, float *A, float *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;
    for (int i = 0; i < n; i++)
        sum += A[row * n + i] * B[i * n + col];

    C[row * n + col] = sum;
}

int main(void) {
    int N = 1 << 10;
    float *A, *B, *C;

    cudaMallocManaged(&A, N * N * sizeof(float));
    cudaMallocManaged(&B, N * N * sizeof(float));
    cudaMallocManaged(&C, N * N * sizeof(float));

    for (int i = 0; i < N * N; i++) {
        A[i] = rand() / (float)RAND_MAX;
        B[i] = rand() / (float)RAND_MAX;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    matMul<<<numBlocks, threadsPerBlock>>>(N, A, B, C);

    cudaDeviceSynchronize();

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
