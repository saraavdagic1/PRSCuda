#include <iostream>
#include "utils.h"
#include "canny.h"
#include "timer.h"
#include "sobel.h"

#define CV_RGB2GRAY cv::COLOR_BGRA2GRAY
#define CV_LOAD_IMAGE_COLOR cv::IMREAD_COLOR
#define CV_BGR2RGBA cv::COLOR_BGR2RGBA
#define CV_WINDOW_AUTOSIZE cv::WINDOW_AUTOSIZE

int main(int argc, char** argv) {

    // create object instances
    Utils utils;
    Utils Sutils;
    Canny canny;
    Sobel sobel;

    cv::Mat cvGrayOutput;
    cv::Mat cvBlurOutput;
    cv::Mat cvCannyOutput;

    cv::Mat cvsobelOutput;

    // create ptr to memory on host and device
    uchar4* h_rgbaImage, * d_rgbaImage;
    unsigned char* h_grayImage, * d_grayImage, * s_grayImage, * hs_grayImage; //

    unsigned char* h_blurredImage, * d_blurredImage;
    char* h_gradImage_x, * d_gradImage_x, * s_gradImage_x, * hs_gradImage_x; //
    char* h_gradImage_y, * d_gradImage_y, * s_gradImage_y, * hs_gradImage_y; //
    unsigned char* h_gradImage_val, * d_gradImage_val, * s_gradImage_val, * hs_gradImage_val; //
    unsigned char* h_gradImage_thin, * d_gradImage_thin;
    unsigned char* h_gradImage_thresh, * d_gradImage_thresh;
    unsigned char* h_gradImage_hyster, * d_gradImage_hyster;
    float* h_gradImage_dir, * d_gradImage_dir, * s_gradImage_dir, * hs_gradImage_dir; //
    float* h_filter_gauss, * d_filter_gauss;
    float* h_filter_grad_x, * d_filter_grad_x, * s_filter_grad_x, * hs_filter_grad_x; //
    float* h_filter_grad_y, * d_filter_grad_y, * s_filter_grad_y, * hs_filter_grad_y; //
    int filterWidth_gauss;
    int filterWidth_grad;
 
    utils.loadImageC("C:/Users/DT User4/source/repos/PRSCuda/PRSCuda/tumor2.jpeg");
    Sutils.loadImageS("C:/Users/DT User4/source/repos/PRSCuda/PRSCuda/tumor2.jpeg"); // 
    int numRows = utils.getImage().rows;
    int numCols = utils.getImage().cols;

    printf("image rows = %d \n", numRows);
    printf("image cols = %d \n", numCols);

    // allocate memory on device - canny
    utils.memAllocC(&h_rgbaImage, &h_grayImage, &d_rgbaImage, &d_grayImage,
        &h_blurredImage, &d_blurredImage, &h_gradImage_x, &d_gradImage_x,
        &h_gradImage_y, &d_gradImage_y, &h_gradImage_val, &d_gradImage_val,
        &h_gradImage_thin, &d_gradImage_thin, &h_gradImage_thresh, &d_gradImage_thresh,
        &h_gradImage_hyster, &d_gradImage_hyster, &h_gradImage_dir, &d_gradImage_dir);

    // allocate memory on device -sobel
    Sutils.memAllocS(&hs_grayImage, &s_grayImage,
        &hs_gradImage_x, &s_gradImage_x,
        &hs_gradImage_y, &s_gradImage_y,
        &hs_gradImage_val, &s_gradImage_val,
       &hs_gradImage_dir, &s_gradImage_dir);

    // create filters
    utils.createGaussianFilter(&h_filter_gauss, &d_filter_gauss, &filterWidth_gauss);
    utils.createGradientFilter_x(&h_filter_grad_x, &d_filter_grad_x, &filterWidth_grad);
    utils.createGradientFilter_y(&h_filter_grad_y, &d_filter_grad_y, &filterWidth_grad);

    Sutils.createGradientFilter_x(&hs_filter_grad_x, &s_filter_grad_x, &filterWidth_grad); //
    Sutils.createGradientFilter_y(&hs_filter_grad_y, &s_filter_grad_y, &filterWidth_grad); //
    /////////////////////////////////////////////////////////////// 
    // GPU rgba to gray
    GpuTimer timer;

    timer.Start();

    canny.rgbaToGrayscale(h_rgbaImage, d_rgbaImage, d_grayImage, numRows, numCols);

    timer.Stop();

    cudaDeviceSynchronize();
    cudaMemcpy(h_grayImage, d_grayImage, sizeof(unsigned char) * utils.getNumPixels(utils.getImage()), cudaMemcpyDeviceToHost);

    printf("GPU rgbaToGray elapsed time is: %f msecs.\n", timer.Elapsed());

    /////////////////////////////////////////////////////////////////
    // OpenCV rgba to gray

    timer.Start();

    cv::cvtColor(utils.getImage(), cvGrayOutput, CV_RGB2GRAY);

    timer.Stop();

    //printf("OpenCV rgbaToGray elapsed time is: %f msecs.\n", timer.Elapsed());

     
    /////////////////////////////////////////////////////////////////
     //GPU gaussian blur  -- shared memory
    timer.Start();
    canny.gaussianBlurShared(d_grayImage, d_blurredImage, numRows,  numCols, filterWidth_gauss, d_filter_gauss );
    timer.Stop();
    cudaDeviceSynchronize();
    cudaMemcpy(h_blurredImage, d_blurredImage, sizeof(unsigned char) * utils.getNumPixels(utils.getImage()), cudaMemcpyDeviceToHost);

    printf("GPU gaussian blur --shared-- elapsed time is: %f msecs.\n", timer.Elapsed());
    
    /////////////////////////////////////////////////////////////////
    // GPU gaussian blur  -- global memory 
    /*
    timer.Start();

    canny.gaussianBlur(d_grayImage, d_blurredImage, numRows, numCols, filterWidth_gauss, d_filter_gauss);

    timer.Stop();

    cudaDeviceSynchronize();
    cudaMemcpy(h_blurredImage, d_blurredImage, sizeof(unsigned char) * utils.getNumPixels(utils.getImage()), cudaMemcpyDeviceToHost);

    printf("GPU gaussian blur elapsed time is: %f msecs.\n", timer.Elapsed());
    */
    ///////////////////////////////////////////////////////////////// 
    // OpenCV gaussian blur 

    timer.Start();
    cv::GaussianBlur(utils.getImageGray(), cvBlurOutput, cv::Size(3, 3), 0.2, 0.2);
    timer.Stop();

    //printf("OpenCV gaussian blur elapsed time is: %f msecs.\n", timer.Elapsed());


    /////////////////////////////////////////////////////////////////
    // GPU gradiant_x canny

    timer.Start();
    canny.grad_x(d_grayImage, d_gradImage_x, numRows, numCols, filterWidth_grad, d_filter_grad_x);
    timer.Stop();

    cudaDeviceSynchronize();
    cudaMemcpy(h_gradImage_x, d_gradImage_x, sizeof(unsigned char) * utils.getNumPixels(utils.getImage()), cudaMemcpyDeviceToHost);

    printf("GPU grad_x elapsed time is: %f msecs.\n", timer.Elapsed());

    /////////////////////////////////////////////////////////////////
    // GPU gradiant_y canny

    timer.Start();
    canny.grad_y(d_grayImage, d_gradImage_y, numRows, numCols, filterWidth_grad, d_filter_grad_y);
    timer.Stop();

    cudaDeviceSynchronize();
    cudaMemcpy(h_gradImage_y, d_gradImage_y, sizeof(unsigned char) * utils.getNumPixels(utils.getImage()), cudaMemcpyDeviceToHost);

    printf("GPU grad_y elapsed time is: %f msecs.\n", timer.Elapsed());

    ///////////////////////////////////////////////////////////////////
    // GPU gradiant_val, gradiant_dir

    timer.Start();

    canny.grad_calc(d_gradImage_x, d_gradImage_y, d_gradImage_val, d_gradImage_dir, numRows, numCols);
    canny.nonMax_suppress(d_gradImage_val, d_gradImage_thin, d_gradImage_dir, numRows, numCols);
    canny.threshold(d_gradImage_thin, d_gradImage_thresh, numRows, numCols);
    canny.hysteresis(d_gradImage_thin, d_gradImage_thresh, d_gradImage_hyster, numRows, numCols);

    timer.Stop();

    printf("GPU nonMax_suppress and Hysteresis elapsed time is: %f msecs.\n", timer.Elapsed());

    cudaDeviceSynchronize();
    cudaMemcpy(h_gradImage_val, d_gradImage_val, sizeof(unsigned char) * utils.getNumPixels(utils.getImage()), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gradImage_thin, d_gradImage_thin, sizeof(unsigned char) * utils.getNumPixels(utils.getImage()), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gradImage_thresh, d_gradImage_thresh, sizeof(unsigned char) * utils.getNumPixels(utils.getImage()), cudaMemcpyDeviceToHost);
    timer.Start();
    cudaMemcpy(h_gradImage_hyster, d_gradImage_hyster, sizeof(unsigned char) * utils.getNumPixels(utils.getImage()), cudaMemcpyDeviceToHost);
    timer.Stop();
    cudaMemcpy(h_gradImage_dir, d_gradImage_dir, sizeof(unsigned char) * utils.getNumPixels(utils.getImage()), cudaMemcpyDeviceToHost);

    printf("GPU Hysteresis memory copy elapsed time is: %f msecs.\n", timer.Elapsed());
    ///////////////////////////////////////////////////////////////////

    timer.Start();
    cv::Canny(utils.getImageGray(), cvCannyOutput, 50, 100, 3);
    timer.Stop();

    printf("OpenCV canny edge detection elapsed time is: %f msecs. \n", timer.Elapsed());
    // write images

    cv::imwrite("blurred.jpg", utils.getImageBlurred());
    cv::imwrite("gradient.jpg", utils.getImageGrad_val());
    cv::imwrite("thin.jpg", utils.getImageGrad_thin());
    cv::imwrite("High_threshold.jpg", utils.getImageGrad_thresh());
    cv::imwrite("final_Output_GPU.jpg", utils.getImageGrad_hyster());
    
    //promjena
    cv::imwrite("final_Output_CV.jpg", cvCannyOutput);
    //promjena
    // 
    // display images
    utils.displayImage(utils.getImage());
    cv::waitKey(0);

    utils.displayImage(utils.getImageGray());
    cv::waitKey(0);

    utils.displayImage(cvGrayOutput);
    cv::waitKey(0);

    //utils.displayImage(utils.getImageBlurred());
    //cv::waitKey(0);

    //utils.displayImage(cvBlurOutput);
    //cv::waitKey(0);

    //utils.displayImage(utils.getImageGrad_x());
    //cv::waitKey(0);

    //utils.displayImage(utils.getImageGrad_y());
    //cv::waitKey(0);

    utils.displayImage(utils.getImageGrad_val());
    cv::waitKey(0);

    //utils.displayImage(utils.getImageGrad_thin());
    //cv::waitKey(0);

    utils.displayImage(utils.getImageGrad_thresh());
    cv::waitKey(0);

    utils.displayImage(utils.getImageGrad_hyster());
    cv::waitKey(0);

    //utils.displayImage(cvCannyOutput);
    //cv::waitKey(0);



    /////////////////////////////////////////////////////////////////
    // GPU gradiant_x sobel

    timer.Start();
    sobel.sobel_grad_x(s_grayImage, s_gradImage_x, numRows, numCols, filterWidth_grad, s_filter_grad_x);
    timer.Stop();

    cudaDeviceSynchronize();
    cudaMemcpy(hs_gradImage_x, s_gradImage_x, sizeof(unsigned char)* Sutils.getNumPixels(Sutils.getImage()), cudaMemcpyDeviceToHost);

    printf("GPU grad_x elapsed time for sobel is: %f msecs.\n", timer.Elapsed());

    /////////////////////////////////////////////////////////////////
    // GPU gradiant_y sobel

    timer.Start();
    sobel.sobel_grad_y(s_grayImage, s_gradImage_y, numRows, numCols, filterWidth_grad, s_filter_grad_y);
    timer.Stop();

    cudaDeviceSynchronize();
    cudaMemcpy(hs_gradImage_y, s_gradImage_y, sizeof(unsigned char)* Sutils.getNumPixels(utils.getImage()), cudaMemcpyDeviceToHost);

    printf("GPU grad_y elapsed for sobel time is: %f msecs.\n", timer.Elapsed());

    ///////////////////////////////////////////////////////////////////
    // GPU gradiant_val, gradiant_dir

    timer.Start();

    sobel.sobel_grad_calc(s_gradImage_x, s_gradImage_y, s_gradImage_val, s_gradImage_dir, numRows, numCols);
    //canny.nonMax_suppress(d_gradImage_val, d_gradImage_thin, d_gradImage_dir, numRows, numCols);
    //canny.threshold(d_gradImage_thin, d_gradImage_thresh, numRows, numCols);
    //canny.hysteresis(d_gradImage_thin, d_gradImage_thresh, d_gradImage_hyster, numRows, numCols);

    timer.Stop();

    //printf("GPU nonMax_suppress and Hysteresis elapsed time for sobel is: %f msecs.\n", timer.Elapsed());

    cudaDeviceSynchronize();
    cudaMemcpy(hs_gradImage_val, s_gradImage_val, sizeof(unsigned char)* Sutils.getNumPixels(Sutils.getImage()), cudaMemcpyDeviceToHost);
    //cudaMemcpy(hs_gradImage_thin, s_gradImage_thin, sizeof(unsigned char)* utils.getNumPixels(utils.getImage()), cudaMemcpyDeviceToHost);
    //cudaMemcpy(hs_gradImage_thresh, d_gradImage_thresh, sizeof(unsigned char)* utils.getNumPixels(utils.getImage()), cudaMemcpyDeviceToHost);
   // timer.Start();
   // cudaMemcpy(h_gradImage_hyster, d_gradImage_hyster, sizeof(unsigned char)* utils.getNumPixels(utils.getImage()), cudaMemcpyDeviceToHost);
   // timer.Stop();
    cudaMemcpy(hs_gradImage_dir, s_gradImage_dir, sizeof(unsigned char)* Sutils.getNumPixels(Sutils.getImage()), cudaMemcpyDeviceToHost);

  //  printf("GPU Hysteresis memory copy elapsed time for sobel is: %f msecs.\n", timer.Elapsed());
    ///////////////////////////////////////////////////////////////////

    timer.Start();
    cv::Sobel(Sutils.getImageGray(), cvCannyOutput, 50, 100, 3);
    timer.Stop();

    printf("OpenCV sobel edge detection elapsed time is: %f msecs. \n", timer.Elapsed());
    // write images

   // cv::imwrite("blurred.jpg", utils.getImageBlurred());
    cv::imwrite("sobelgradient.jpg", Sutils.getImageGrad_val());
  //  cv::imwrite("thin.jpg", utils.getImageGrad_thin());
  //  cv::imwrite("High_threshold.jpg", utils.getImageGrad_thresh());
    cv::imwrite("final_Output_GPU_sobel.jpg", Sutils.getImageGrad_hyster());

    //promjena
   // cv::imwrite("final_Output_CV.jpg", cvCannyOutput);
    //promjena
    // 
    // display images
   // utils.displayImage(utils.getImage());
   // cv::waitKey(0);

  //  utils.displayImage(utils.getImageGray());
  //  cv::waitKey(0);

  //  utils.displayImage(cvGrayOutput);
  //  cv::waitKey(0);

    //utils.displayImage(utils.getImageBlurred());
    //cv::waitKey(0);

    //utils.displayImage(cvBlurOutput);
    //cv::waitKey(0);

    //utils.displayImage(utils.getImageGrad_x());
    //cv::waitKey(0);

    //utils.displayImage(utils.getImageGrad_y());
    //cv::waitKey(0);

    Sutils.displayImage(Sutils.getImageGrad_val());
    cv::waitKey(0);

    //utils.displayImage(utils.getImageGrad_thin());
    //cv::waitKey(0);

  //  utils.displayImage(utils.getImageGrad_thresh());
  //  cv::waitKey(0);

   // utils.displayImage(utils.getImageGrad_hyster());
 //   cv::waitKey(0);

    //utils.displayImage(cvCannyOutput);
    //cv::waitKey(0);

    // free memory
   // cudaFree(h_rgbaImage);
  //  cudaFree(d_rgbaImage);
    cudaFree(hs_grayImage);
    cudaFree(s_grayImage);
   // cudaFree(h_blurredImage);
  //  cudaFree(d_blurredImage);
  //  cudaFree(h_filter_gauss);
  //  cudaFree(d_filter_gauss);
    cudaFree(hs_gradImage_x);
    cudaFree(s_gradImage_x);
    cudaFree(hs_filter_grad_x);
    cudaFree(s_filter_grad_x);
    cudaFree(hs_gradImage_y);
    cudaFree(s_gradImage_y);
    cudaFree(hs_filter_grad_y);
    cudaFree(s_filter_grad_y);
    cudaFree(hs_gradImage_val);
    cudaFree(s_gradImage_val);
//    cudaFree(h_gradImage_thin);
//    cudaFree(d_gradImage_thin);
 //   cudaFree(h_gradImage_thresh);
 //   cudaFree(d_gradImage_thresh);
//    cudaFree(h_gradImage_hyster);
 //   cudaFree(d_gradImage_hyster);
    cudaFree(hs_gradImage_dir);
    cudaFree(s_gradImage_dir);
    
    // free memory
    cudaFree(h_rgbaImage);
    cudaFree(d_rgbaImage);
    cudaFree(h_grayImage);
    cudaFree(d_grayImage);
    cudaFree(h_blurredImage);
    cudaFree(d_blurredImage);
    cudaFree(h_filter_gauss);
    cudaFree(d_filter_gauss);
    cudaFree(h_gradImage_x);
    cudaFree(d_gradImage_x);
    cudaFree(h_filter_grad_x);
    cudaFree(d_filter_grad_x);
    cudaFree(h_gradImage_y);
    cudaFree(d_gradImage_y);
    cudaFree(h_filter_grad_y);
    cudaFree(d_filter_grad_y);
    cudaFree(h_gradImage_val);
    cudaFree(d_gradImage_val);
    cudaFree(h_gradImage_thin);
    cudaFree(d_gradImage_thin);
    cudaFree(h_gradImage_thresh);
    cudaFree(d_gradImage_thresh);
    cudaFree(h_gradImage_hyster);
    cudaFree(d_gradImage_hyster);
    cudaFree(h_gradImage_dir);
    cudaFree(d_gradImage_dir);
    return 0;
}