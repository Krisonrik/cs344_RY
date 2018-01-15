#ifndef REFERENCE_H__
#define REFERENCE_H__

#include <cuda_runtime.h>

M1_DLL
void referenceCalculation(const uchar4 *const rgbaImage,
                          unsigned char *const greyImage,
                          size_t numRows,
                          size_t numCols);

#endif
