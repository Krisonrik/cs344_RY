#ifndef MODULES_M6_INCLUDE_HVR_HW6_REFERENCE_CALC_H_
#define MODULES_M6_INCLUDE_HVR_HW6_REFERENCE_CALC_H_

#include <cuda_runtime.h>

M6_DLL
void reference_calc(const uchar4* const h_sourceImg,
                    const size_t numRowsSource,
                    const size_t numColsSource,
                    const uchar4* const h_destImg,
                    uchar4* const h_blendedImg);

#endif  // MODULES_M6_INCLUDE_HVR_HW6_REFERENCE_CALC_H_
