// Copyright 2015 Jason Juang

#ifndef MODULES_CUDASAMPLE_INCLUDE_HVR_CUDASAMPLE_CUDASAMPLE_CUH_
#define MODULES_CUDASAMPLE_INCLUDE_HVR_CUDASAMPLE_CUDASAMPLE_CUH_

// HVR_WINDOWS_DISABLE_ALL_WARNING
#include "opencv2/opencv.hpp"
// HVR_WINDOWS_ENABLE_ALL_WARNING

namespace hvr
{
/**
 * @brief      Minimal cuda class
 *
 * @ingroup    CUDASample
 * @author     Jason Juang
 * @attention  This is for internal use only, DO NOT distribute the code
 */
class HVR_CUDASAMPLE_DLL CUDASample
{
 public:
  CUDASample();

  ~CUDASample();

  /**
   * @brief         Turn an image to white
   *
   * @param[in,out] img   input image
   */

  void SetImageToWhite(cv::Mat &img) const;
};

}  // namespace hvr

#endif  // MODULES_CUDASAMPLE_INCLUDE_HVR_CUDASAMPLE_CUDASAMPLE_CUH_
