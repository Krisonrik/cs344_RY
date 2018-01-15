#ifndef MODULES_M2_INCLUDE_HVR_HW2_REFERENCE_CALC_H_
#define MODULES_M2_INCLUDE_HVR_HW2_REFERENCE_CALC_H_

#include <cuda_runtime.h>

M2_DLL
void referenceCalculation(const uchar4* const rgbaImage,
                          uchar4* const outputImage,
                          size_t numRows,
                          size_t numCols,
                          const float* const filter,
                          const int filterWidth);

#endif  // MODULES_M2_INCLUDE_HVR_HW2_REFERENCE_CALC_H_
