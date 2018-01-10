// Copyright 2015 Jason Juang

#ifndef MODULES_CUDASAMPLE_INCLUDE_HVR_CUDASAMPLE_CUDACONFIG_H_
#define MODULES_CUDASAMPLE_INCLUDE_HVR_CUDASAMPLE_CUDACONFIG_H_

namespace hvr
{
/**
  * @brief      CUDA configurations
  *
  * @author     Jason Juang
  * @attention  This is for internal use only, DO NOT distribute the code
  */
struct CUDAConfig
{
  // ID to identify GPU
  int id;

  unsigned int h, w;
  unsigned int h_a, w_a;

  // Cuda kernel defines
  unsigned int blk_h   = 16;
  unsigned int blk_w   = 32;
  unsigned int shift_h = 4;
  unsigned int shift_w = 5;

  /**
   * @brief      This will align 'n' so that it is a multiple of 'a'
   *
   * @param[in]  n     input
   * @param[in]  a     number to align to
   *
   * @return     aligned value
   */
  int align(const int n, const int a) const;
};

}  // namespace hvr

#endif  // MODULES_CUDASAMPLE_INCLUDE_HVR_CUDASAMPLE_CUDACONFIG_H_
