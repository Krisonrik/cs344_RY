// Copyright 2015 Jason Juang

#ifndef MODULES_SAMPLE_INCLUDE_HVR_SAMPLE_SAMPLE_H_
#define MODULES_SAMPLE_INCLUDE_HVR_SAMPLE_SAMPLE_H_

HVR_WINDOWS_DISABLE_ALL_WARNING
#include "opencv2/opencv.hpp"
HVR_WINDOWS_ENABLE_ALL_WARNING

namespace hvr
{
/**
  * @brief      Minimal class
  *
  * @ingroup    Sample
  * @author     Jason Juang
  * @attention  This is for internal use only, DO NOT distribute the code
  */
class HVR_SAMPLE_DLL Sample
{
 public:
  Sample();
  ~Sample();

  /**
   * @brief      Prints a mysterious message
   */
  void PrintSomething() const;

  /**
   * @brief      Adds two integers together
   *
   * @param[in]  a     first number to add
   * @param[in]  b     second number to add
   *
   * @return     sum of a and b
   */
  int AddTwoInteger(const int a, const int b) const;

  /**
   * @brief         Turns an image to plain white
   *
   * @param[in,out] img   image that is going to become white
   */
  void TurnImageToWhite(cv::Mat &img) const;
};

}  // namespace hvr

#endif  // MODULES_SAMPLE_INCLUDE_HVR_SAMPLE_SAMPLE_H_
