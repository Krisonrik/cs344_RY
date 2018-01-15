#ifndef MODULES_M1_INCLUDE_HVR_HW1_HW1_H_
#define MODULES_M1_INCLUDE_HVR_HW1_HW1_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>

M1_DLL
size_t numRows();

M1_DLL
size_t numCols();

M1_DLL
void preProcess(uchar4** inputImage,
                unsigned char** greyImage,
                uchar4** d_rgbaImage,
                unsigned char** d_greyImage,
                const std::string& filename);

M1_DLL
void postProcess(const std::string& output_file, unsigned char* data_ptr);

M1_DLL
void cleanup();

M1_DLL
void generateReferenceImage(std::string input_filename,
                            std::string output_filename);

#endif  // MODULES_M1_INCLUDE_HVR_HW1_HW1_H_
