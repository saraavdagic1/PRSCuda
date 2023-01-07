#include <string>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

class Sobel {
private:

public:
    Sobel() {};
    ~Sobel() {};
   
    void sobel_grad_x(unsigned char* d_grayImage, char* d_gradImage_x, size_t numRows, size_t numCols, const int filterWidth, const float* d_filter);

    void sobel_grad_y(unsigned char* d_grayImage, char* d_gradImage_y, size_t numRows, size_t numCols, const int filterWidth, const float* d_filter);

    void sobel_grad_calc(char* d_gradImage_x, char* d_gradImage_y, unsigned char* d_gradImage_val, float* d_gradImage_dir, size_t numRows, size_t numCols);

  
};
