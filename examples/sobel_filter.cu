#include <cuda_runtime.h>

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Definición del kernel de CUDA para aplicar el filtro de Sobel
__global__ void sobelFilterKernel(unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int Gx =
            -1 * input[(y - 1) * width + (x - 1)] + 1 * input[(y - 1) * width + (x + 1)] +
            -2 * input[(y)*width + (x - 1)] + 2 * input[(y)*width + (x + 1)] +
            -1 * input[(y + 1) * width + (x - 1)] + 1 * input[(y + 1) * width + (x + 1)];

        int Gy =
            -1 * input[(y - 1) * width + (x - 1)] + 1 * input[(y + 1) * width + (x - 1)] +
            -2 * input[(y - 1) * width + (x)] + 2 * input[(y + 1) * width + (x)] +
            -1 * input[(y - 1) * width + (x + 1)] + 1 * input[(y + 1) * width + (x + 1)];

        int magnitude = sqrtf(Gx * Gx + Gy * Gy);

        output[y * width + x] = magnitude > 255 ? 255 : magnitude;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        cout << "Uso: " << argv[0] << " <imagen>" << endl;
        return -1;
    }
    // Cargar la imagen en escala de grises
    string imagePath = argv[1];
    Mat image = imread(imagePath, IMREAD_GRAYSCALE);
    if (image.empty()) {
        cout << "Error al cargar la imagen." << endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    // Crear la imagen de salida
    Mat output = Mat::zeros(height, width, CV_8UC1);

    // Reservar memoria en el dispositivo (GPU)
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, width * height * sizeof(unsigned char));
    cudaMalloc(&d_output, width * height * sizeof(unsigned char));

    // Copiar datos de la imagen de la CPU a la GPU
    cudaMemcpy(d_input, image.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Definir la estructura de bloques e hilos
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Ejecutar el kernel
    sobelFilterKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height);

    // Copiar el resultado de vuelta a la CPU
    cudaMemcpy(output.data, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Guardar la imagen de salida
    imwrite("output.jpg", output);

    // Liberar la memoria en la GPU
    cudaFree(d_input);
    cudaFree(d_output);

    cout << "Filtro de Sobel aplicado y guardado como output.jpg." << endl;

    return 0;
}
