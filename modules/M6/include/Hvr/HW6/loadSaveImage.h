#ifndef MODULES_M6_INCLUDE_HVR_HW6_LOADSAVEIMAGE_H_
#define MODULES_M6_INCLUDE_HVR_HW6_LOADSAVEIMAGE_H_

#include <cuda_runtime.h>  // for uchar4
#include <string>

M6_DLL
void loadImageHDR(const std::string &filename,
                  float **imagePtr,
                  size_t *numRows,
                  size_t *numCols);

M6_DLL
void loadImageRGBA(const std::string &filename,
                   uchar4 **imagePtr,
                   size_t *numRows,
                   size_t *numCols);

M6_DLL
void loadImageGrey(const std::string &filename,
                   unsigned char **imagePtr,
                   size_t *numRows,
                   size_t *numCols);
M6_DLL
void saveImageRGBA(const uchar4 *const image,
                   const size_t numRows,
                   const size_t numCols,
                   const std::string &output_file);

M6_DLL
void saveImageHDR(const float *const image,
                  const size_t numRows,
                  const size_t numCols,
                  const std::string &output_file);

#endif  // MODULES_M6_INCLUDE_HVR_HW6_LOADSAVEIMAGE_H_
