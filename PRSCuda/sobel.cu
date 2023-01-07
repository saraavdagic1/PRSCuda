#include <iostream>
#include "sobel.h"



__global__
void sobel_grad_x_ker(unsigned char* const inputChannel,
    char* const outputChannel,
    int numRows, int numCols,
    const float* const filter, const int filterWidth)
{

    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y);
    const int m = p.y * numCols + p.x;
    if (p.x >= numCols || p.y >= numRows)
        return;

    float color = 0.0f;

    for (int f_y = 0; f_y < filterWidth; f_y++) {
        for (int f_x = 0; f_x < filterWidth; f_x++) {

            int c_x = p.x + f_x - filterWidth / 2;
            int c_y = p.y + f_y - filterWidth / 2;
            c_x = min(max(c_x, 0), numCols - 1);
            c_y = min(max(c_y, 0), numRows - 1);
            float filter_value = filter[f_y * filterWidth + f_x];
            color += filter_value * static_cast<float>(inputChannel[c_y * numCols + c_x]);

        }
    }

    outputChannel[m] = color;
}
//////////////////////////////////////////////////////////////////////

void Sobel::sobel_grad_x(unsigned char* d_grayImage, char* d_gradImage_x,
    size_t numRows, size_t numCols, const int filterWidth, const float* d_filter)
{
    const dim3 blockSize(32, 32);

    const dim3 gridSize(numCols / blockSize.x + 1, numRows / blockSize.y + 1);
    sobel_grad_x_ker << <gridSize, blockSize >> > (
        d_grayImage,
        d_gradImage_x,
        numRows,
        numCols,
        d_filter,
        filterWidth);
    cudaDeviceSynchronize();

}


///////////////////////////////////////////////////////////////////////


__global__
void sobel_grad_y_ker(unsigned char* const inputChannel,
    char* const outputChannel,
    int numRows, int numCols,
    const float* const filter, const int filterWidth)
{

    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y);
    const int m = p.y * numCols + p.x;
    if (p.x >= numCols || p.y >= numRows)
        return;

    float color = 0.0f;

    for (int f_y = 0; f_y < filterWidth; f_y++) {
        for (int f_x = 0; f_x < filterWidth; f_x++) {

            int c_x = p.x + f_x - filterWidth / 2;
            int c_y = p.y + f_y - filterWidth / 2;
            c_x = min(max(c_x, 0), numCols - 1);
            c_y = min(max(c_y, 0), numRows - 1);
            float filter_value = filter[f_y * filterWidth + f_x];
            color += filter_value * static_cast<float>(inputChannel[c_y * numCols + c_x]);


        }
    }

    outputChannel[m] = color;
}
//////////////////////////////////////////////////////////////////////

void Sobel::sobel_grad_y(unsigned char* d_grayImage, char* d_gradImage_y,
    size_t numRows, size_t numCols, const int filterWidth, const float* d_filter)
{
    const dim3 blockSize(32, 32);

    const dim3 gridSize(numCols / blockSize.x + 1, numRows / blockSize.y + 1);
    sobel_grad_y_ker << <gridSize, blockSize >> > (
        d_grayImage,
        d_gradImage_y,
        numRows,
        numCols,
        d_filter,
        filterWidth);
    cudaDeviceSynchronize();

}

///////////////////////////////////////////////////////////////////

__global__
void sobel_grad_calc_ker(char* const inputGrad_x,
    char* const inputGrad_y,
    unsigned char* const outputGrad_val,
    float* const outputGrad_dir,
    int numRows, int numCols)
{

    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y);
    const int m = p.y * numCols + p.x;
    if (p.x >= numCols || p.y >= numRows)
        return;
    // calculate gradient values at each pixel
    outputGrad_val[m] = sqrt(static_cast<float>(inputGrad_x[m]) * static_cast<float>(inputGrad_x[m]) + static_cast<float>(inputGrad_y[m]) * static_cast<float>(inputGrad_y[m]));

    // caculate gradient directions at each pixel
    outputGrad_dir[m] = atan((static_cast<float>(inputGrad_y[m])) / (static_cast<float>(inputGrad_x[m])));




}

//////////////////////////////////////////////////////////////////////

void Sobel::sobel_grad_calc(char* d_gradImage_x, char* d_gradImage_y, unsigned char* d_gradImage_val, float* d_gradImage_dir, size_t numRows, size_t numCols)
{
    const dim3 blockSize(32, 32);

    const dim3 gridSize(numCols / blockSize.x + 1, numRows / blockSize.y + 1);
    sobel_grad_calc_ker << <gridSize, blockSize >> > (
        d_gradImage_x,
        d_gradImage_y,
        d_gradImage_val,
        d_gradImage_dir,
        numRows,
        numCols);
    cudaDeviceSynchronize();
}


