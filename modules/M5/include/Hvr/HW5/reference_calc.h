#ifndef MODULES_M5_INCLUDE_HVR_HW5_REFERENCE_CALC_H_
#define MODULES_M5_INCLUDE_HVR_HW5_REFERENCE_CALC_H_

// Reference Histogram calculation
M5_DLL
void reference_calculation(const unsigned int* const vals,
                           unsigned int* const histo,
                           const size_t numBins,
                           const size_t numElems);

#endif  // MODULES_M5_INCLUDE_HVR_HW5_REFERENCE_CALC_H_
