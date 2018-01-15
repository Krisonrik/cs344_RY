#ifndef MODULES_M6_INCLUDE_HVR_HW6_COMPARE_H_
#define MODULES_M6_INCLUDE_HVR_HW6_COMPARE_H_

#include <string>

M6_DLL
void compareImages(std::string reference_filename,
                   std::string test_filename,
                   bool useEpsCheck,
                   double perPixelError,
                   double globalError);

#endif  // MODULES_M6_INCLUDE_HVR_HW6_COMPARE_H_
