#include <iostream>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/opencv.hpp>

#include "image_processing.cu"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image-path>" << std::endl;
        return -1;
    }

    std::string imagePath = argv[1];
    cv::Mat image = cv::imread(imagePath);

    if (image.empty()) {
        std::cerr << "Error: Image not loaded!" << std::endl;
        return -1;
    }

    cv::cuda::GpuMat d_image;
    d_image.upload(image);

    int red = 0, green = 0, blue = 0;
    extractColors(d_image, red, green, blue);

    std::cout << "Red: " << red << std::endl;
    std::cout << "Green: " << green << std::endl;
    std::cout << "Blue: " << blue << std::endl;

    return 0;
}
