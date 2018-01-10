// Copyright 2015 Jason Juang

#include "Hvr/CUDASample/CUDASample.cuh"

// HVR_WINDOWS_DISABLE_ALL_WARNING
#include "opencv2/opencv.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
// HVR_WINDOWS_ENABLE_ALL_WARNING

#include "Hvr/CUDASample/CUDAConfig.h"

__global__ static void CUDAKernelSetImageToWhite(int *data)
{
  const int x  = blockIdx.x * blockDim.x + threadIdx.x;
  const int y  = blockIdx.y * blockDim.y + threadIdx.y;
  const int mx = gridDim.x * blockDim.x;

  data[y * mx + x] = 255;
}

namespace hvr
{
CUDASample::CUDASample()
{
}

CUDASample::~CUDASample()
{
}

void CUDASample::SetImageToWhite(cv::Mat &img) const
{
  if (img.empty()) return;

  const int h = img.rows;
  const int w = img.cols;

  CUDAConfig cudaconfig;
  cudaconfig.h_a = cudaconfig.align(h, cudaconfig.blk_h);
  cudaconfig.w_a = cudaconfig.align(w, cudaconfig.blk_w);

  const int h_a = cudaconfig.h_a;
  const int w_a = cudaconfig.w_a;

  int *img_gpu;
  cudaMalloc(reinterpret_cast<void **>(&img_gpu), h_a * w_a * sizeof(int));

  std::vector<int> img_vec(h_a * w_a, 0);

  for (int i = 0; i < h; i++)
    for (int j = 0; j < w; j++)
    {
      img_vec[i * w_a + j] = img.at<uchar>(i, j);
    }

  cudaMemcpy(
      img_gpu, img_vec.data(), h_a * w_a * sizeof(int), cudaMemcpyHostToDevice);

  dim3 blks((w_a >> cudaconfig.shift_w), (h_a >> cudaconfig.shift_h));
  dim3 threads(cudaconfig.blk_w, cudaconfig.blk_h);

  CUDAKernelSetImageToWhite<<<blks, threads>>>(img_gpu);

  cudaMemcpy(
      img_vec.data(), img_gpu, h_a * w_a * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < h; i++)
    for (int j = 0; j < w; j++)
    {
      img.at<uchar>(i, j) = img_vec[i * w_a + j];
    }

  cudaFree(img_gpu);
}

}  // namespace hvr
