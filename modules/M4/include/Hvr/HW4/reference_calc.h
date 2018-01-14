#ifndef MODULES_M4_INCLUDE_HVR_HW4_REFERENCE_CALC_H_
#define MODULES_M4_INCLUDE_HVR_HW4_REFERENCE_CALC_H_

// A simple un-optimized reference radix sort calculation
// Only deals with power-of-2 radices

#include "thrust/host_vector.h"
M4_DLL
void reference_calculation(thrust::host_vector<unsigned int> &inputVals,
                           thrust::host_vector<unsigned int> &inputPos,
                           thrust::host_vector<unsigned int> &outputVals,
                           thrust::host_vector<unsigned int> &outputPos,
                           const size_t numElems);
#endif  // MODULES_M4_INCLUDE_HVR_HW4_REFERENCE_CALC_H_
