#include <opencv2/cudaarithm.hpp>
#include <opencv2/opencv.hpp>

__global__ void extractColorKernel(uchar3* img, int width, int height, int* redCount, int* greenCount, int* blueCount) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        uchar3 pixel = img[idx];

        atomicAdd(redCount, pixel.x);
        atomicAdd(greenCount, pixel.y);
        atomicAdd(blueCount, pixel.z);
    }
}

void extractColors(cv::cuda::GpuMat& d_image, int& red, int& green, int& blue) {
    int *d_red, *d_green, *d_blue;
    int h_red = 0, h_green = 0, h_blue = 0;

    cudaMalloc(&d_red, sizeof(int));
    cudaMalloc(&d_green, sizeof(int));
    cudaMalloc(&d_blue, sizeof(int));

    cudaMemcpy(d_red, &h_red, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_green, &h_green, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blue, &h_blue, sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((d_image.cols + block.x - 1) / block.x, (d_image.rows + block.y - 1) / block.y);

    extractColorKernel<<<grid, block>>>(d_image.ptr<uchar3>(), d_image.cols, d_image.rows, d_red, d_green, d_blue);

    cudaMemcpy(&h_red, d_red, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_green, d_green, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_blue, d_blue, sizeof(int), cudaMemcpyDeviceToHost);

    red = h_red;
    green = h_green;
    blue = h_blue;

    cudaFree(d_red);
    cudaFree(d_green);
    cudaFree(d_blue);
}
