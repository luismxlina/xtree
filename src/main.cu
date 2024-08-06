// #include <opencv2/cudaimgproc.hpp>
// #include <opencv2/cudawarping.hpp>
// #include <opencv2/opencv.hpp>

// __global__ void calcHistogram(unsigned char* img, int width, int height, int* histogram) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;

//     if (x < width && y < height) {
//         int color = img[y * width + x];
//         atomicAdd(&histogram[color], 1);
//     }
// }

// int main() {
//     cv::Mat img = cv::imread("arbol.jpg", cv::IMREAD_COLOR);
//     cv::cuda::GpuMat d_img;
//     d_img.upload(img);

//     // Convertir la imagen a escala de grises
//     cv::cuda::GpuMat d_gray;
//     cv::cuda::cvtColor(d_img, d_gray, cv::COLOR_BGR2GRAY);

//     // Crear un histograma en la memoria del dispositivo
//     int* d_histogram;
//     cudaMalloc(&d_histogram, 256 * sizeof(int));
//     cudaMemset(d_histogram, 0, 256 * sizeof(int));

//     // Calcular el histograma
//     dim3 blockSize(16, 16);
//     dim3 gridSize((d_gray.cols + blockSize.x - 1) / blockSize.x, (d_gray.rows + blockSize.y - 1) / blockSize.y);
//     calcHistogram<<<gridSize, blockSize>>>(d_gray.ptr<unsigned char>(), d_gray.cols, d_gray.rows, d_histogram);

//     // Descargar el histograma a la memoria del host
//     int histogram[256];
//     cudaMemcpy(histogram, d_histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);

//     // Liberar la memoria del dispositivo
//     cudaFree(d_histogram);

//     // Ahora el histograma contiene el número de píxeles de cada color en la imagen

//     return 0;
// }

#include <cuda_runtime.h>

#include <iostream>
#include <opencv.hpp>
#include <vector>

// Declaración del kernel
__global__ void count_colors(const unsigned char* input, int width, int height, int* red_count, int* green_count, int* blue_count);

// Función para cargar imágenes usando OpenCV
std::vector<cv::Mat> load_images(const std::vector<std::string>& image_paths) {
    std::vector<cv::Mat> images;
    for (const auto& path : image_paths) {
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (!img.empty()) {
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);  // Convertir de BGR a RGB
            images.push_back(img);
        } else {
            std::cerr << "Error: could not load image at " << path << std::endl;
        }
    }
    return images;
}

int main() {
    // Rutas de las imágenes
    std::vector<std::string> image_paths = {
        "arbol1.jpg",
        "urban_tree.jpg",
    };

    // Cargar las imágenes
    std::vector<cv::Mat> images = load_images(image_paths);

    if (images.empty()) {
        std::cerr << "Error: no images loaded" << std::endl;
        return -1;
    }

    // Procesar la primera imagen
    cv::Mat img = images[0];
    int width = img.cols;
    int height = img.rows;

    size_t img_size = width * height * 3 * sizeof(unsigned char);

    // Reservar memoria en la GPU
    unsigned char* d_input;
    int *d_red_count, *d_green_count, *d_blue_count;
    int h_red_count = 0, h_green_count = 0, h_blue_count = 0;

    cudaMalloc((void**)&d_input, img_size);
    cudaMalloc((void**)&d_red_count, sizeof(int));
    cudaMalloc((void**)&d_green_count, sizeof(int));
    cudaMalloc((void**)&d_blue_count, sizeof(int));

    // Copiar la imagen a la memoria de la GPU
    cudaMemcpy(d_input, img.data, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_red_count, &h_red_count, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_green_count, &h_green_count, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blue_count, &h_blue_count, sizeof(int), cudaMemcpyHostToDevice);

    // Configurar el grid y los bloques
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Ejecutar el kernel
    count_colors<<<numBlocks, threadsPerBlock>>>(d_input, width, height, d_red_count, d_green_count, d_blue_count);

    // Copiar el resultado de vuelta a la CPU
    cudaMemcpy(&h_red_count, d_red_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_green_count, d_green_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_blue_count, d_blue_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Calcular porcentajes
    int total_pixels = width * height;
    float red_percentage = (float)h_red_count / (total_pixels * 255) * 100;
    float green_percentage = (float)h_green_count / (total_pixels * 255) * 100;
    float blue_percentage = (float)h_blue_count / (total_pixels * 255) * 100;

    // Mostrar resultados
    std::cout << "Red Percentage: " << red_percentage << "%" << std::endl;
    std::cout << "Green Percentage: " << green_percentage << "%" << std::endl;
    std::cout << "Blue Percentage: " << blue_percentage << "%" << std::endl;

    // Liberar memoria en la GPU
    cudaFree(d_input);
    cudaFree(d_red_count);
    cudaFree(d_green_count);
    cudaFree(d_blue_count);

    return 0;
}
