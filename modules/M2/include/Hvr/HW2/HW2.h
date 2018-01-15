#ifndef MODULES_M2_INCLUDE_HVR_HW2_HW2_H_
#define MODULES_M2_INCLUDE_HVR_HW2_HW2_H_

#include <cuda_runtime.h>
#include <string>

#include "opencv2/opencv.hpp"

M2_DLL
size_t numRows();

M2_DLL
size_t numCols();

M2_DLL
void preProcess(uchar4 **h_inputImageRGBA,
                uchar4 **h_outputImageRGBA,
                uchar4 **d_inputImageRGBA,
                uchar4 **d_outputImageRGBA,
                unsigned char **d_redBlurred,
                unsigned char **d_greenBlurred,
                unsigned char **d_blueBlurred,
                float **h_filter,
                int *filterWidth,
                const std::string &filename);

M2_DLL
void postProcess(const std::string &output_file, uchar4 *data_ptr);

M2_DLL void cleanUp(void);

M2_DLL
void generateReferenceImage(std::string input_file,
                            std::string reference_file,
                            int kernel_size);

cv::Mat imageInputRGBA;
cv::Mat imageOutputRGBA;

uchar4 *d_inputImageRGBA__;
uchar4 *d_outputImageRGBA__;

float *h_filter__;

#endif  // MODULES_M2_INCLUDE_HVR_HW2_HW2_H_
