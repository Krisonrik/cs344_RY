#ifndef MODULES_M2_INCLUDE_HVR_HW2_COMPARE_H_
#define MODULES_M2_INCLUDE_HVR_HW2_COMPARE_H_

#include <string>

M2_DLL
void compareImages(std::string reference_filename,
                   std::string test_filename,
                   bool useEpsCheck,
                   double perPixelError,
                   double globalError);

#endif  // MODULES_M2_INCLUDE_HVR_HW2_COMPARE_H_
