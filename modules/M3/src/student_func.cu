/* Udacity Homework 3
HDR Tone-mapping

Background HDR
==============

A High Dynamic Range (HDR) image contains a wider variation of intensity
and color than is allowed by the RGB format with 1 byte per channel that we
have used in the previous assignment.

To store this extra information we use single precision floating point for
each channel.  This allows for an extremely wide range of intensity values.

In the image for this assignment, the inside of church with light coming in
through stained glass windows, the raw input floating point values for the
channels range from 0 to 275.  But the mean is .41 and 98% of the values are
less than 3!  This means that certain areas (the windows) are extremely bright
compared to everywhere else.  If we linearly map this [0-275] range into the
[0-255] range that we have been using then most values will be mapped to zero!
The only thing we will be able to see are the very brightest areas - the
windows - everything else will appear pitch black.

The problem is that although we have cameras capable of recording the wide
range of intensity that exists in the real world our monitors are not capable
of displaying them.  Our eyes are also quite capable of observing a much wider
range of intensities than our image formats / monitors are capable of
displaying.

Tone-mapping is a process that transforms the intensities in the image so that
the brightest values aren't nearly so far away from the mean.  That way when
we transform the values into [0-255] we can actually see the entire image.
There are many ways to perform this process and it is as much an art as a
science - there is no single "right" answer.  In this homework we will
implement one possible technique.

Background Chrominance-Luminance
================================

The RGB space that we have been using to represent images can be thought of as
one possible set of axes spanning a three dimensional space of color.  We
sometimes choose other axes to represent this space because they make certain
operations more convenient.

Another possible way of representing a color image is to separate the color
information (chromaticity) from the brightness information.  There are
multiple different methods for doing this - a common one during the analog
television days was known as Chrominance-Luminance or YUV.

We choose to represent the image in this way so that we can remap only the
intensity channel and then recombine the new intensity values with the color
information to form the final image.

Old TV signals used to be transmitted in this way so that black & white
televisions could display the luminance channel while color televisions would
display all three of the channels.


Tone-mapping
============

In this assignment we are going to transform the luminance channel (actually
the log of the luminance, but this is unimportant for the parts of the
algorithm that you will be implementing) by compressing its range to [0, 1].
To do this we need the cumulative distribution of the luminance values.

Example
-------

input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
min / max / range: 0 / 9 / 9

histo with 3 bins: [4 7 3]

cdf : [4 11 14]


Your task is to calculate this cumulative distribution by following these
steps.

*/

#include <algorithm>
#include <memory>
#include <vector>
#include "hvr/HW3/utils.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>

// Kernel for calculating minimum and maximum value of given array (output array
// from reduction of 1/4 the size of the original array)
M3_DLL
__global__ void findMinMax(const float* const d_logLuminance,
                           float* d_logLum_Min,
                           float* d_logLum_Max,
                           float* d_min_Lums,
                           float* d_max_Lums,
                           int pxCount)
{
  const int absId = blockDim.x * blockIdx.x + threadIdx.x;
  const int tId   = threadIdx.x;
  float d_min     = d_logLuminance[0];
  float d_max     = d_logLuminance[0];
  for (int i = 0; i < 4; i++)
  {
    if (absId < pxCount)
    {
      d_min = fminf(d_min, d_logLuminance[absId + i * blockDim.x * gridDim.x]);
      d_max = fmaxf(d_max, d_logLuminance[absId + i * blockDim.x * gridDim.x]);
    }
  }
  d_logLum_Min[absId] = d_min;
  d_logLum_Max[absId] = d_max;

  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (tId < s)
    {
      d_logLum_Min[absId] = fminf(d_logLum_Min[absId], d_logLum_Min[absId + s]);
      d_logLum_Max[absId] = fmaxf(d_logLum_Max[absId], d_logLum_Max[absId + s]);
    }
    __syncthreads();
  }

  if (tId == 0)
  {
    d_min_Lums[blockIdx.x] = d_logLum_Min[absId];
    d_max_Lums[blockIdx.x] = d_logLum_Max[absId];
  }
}

// Kernel for calculating the histogram
M3_DLL
__global__ void calcHisto(unsigned int* d_histo,
                          const float* const d_logLuminance,
                          float min_logLum,
                          float lumRange,
                          const size_t numBins,
                          const unsigned int pxCount)
{
  const int tId = threadIdx.x;
  extern __shared__ unsigned int sh_bin[];
  int bin       = 0;
  int loopCount = (pxCount + blockDim.x - 1) / blockDim.x;

  // use each block as a single bin and process through the entire array to get
  // a single bin value for each block
  for (int i = 0; i < loopCount; i++)
  {
    if ((tId + i * blockDim.x) < pxCount)
    {
      if (blockIdx.x ==
          int((d_logLuminance[tId + i * blockDim.x] - min_logLum) / lumRange *
              numBins))
      {
        bin++;
      }
    }
    __syncthreads();
  }
  sh_bin[tId] = bin;

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (tId < s)
    {
      sh_bin[tId] = sh_bin[tId] + sh_bin[tId + s];
    }
    __syncthreads();
  }
  if (tId == 0)
  {
    d_histo[blockIdx.x] = sh_bin[0];
  }
}

// Kernel for calculating the cdf from given histogram
M3_DLL
__global__ void calcCDF(unsigned int* const d_cdf,
                        unsigned int* d_histo,
                        const size_t numBins,
                        const int cycles)
{
  const int absId = blockDim.x * blockIdx.x + threadIdx.x;
  const int tId   = threadIdx.x;
  extern __shared__ unsigned int prevSum[];
  prevSum[tId] = 0;

  // calculate the total sum of the values before current block
  for (int i = 0; i < blockIdx.x; i++)
  {
    prevSum[tId] += d_histo[tId + i * blockDim.x];
  }
  __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (tId < s)
    {
      prevSum[tId] += prevSum[tId + s];
    }
    __syncthreads();
  }

  // calculate the inclusive scan with Hillis Steele scan
  int j = 1;
  for (int i = 0; i < cycles; i++)
  {
    if ((tId - j) >= 0)
    {
      d_histo[absId] += d_histo[absId - j];
    }
    j = j * 2;
    __syncthreads();
  }

  d_histo[absId] += prevSum[0];

  __syncthreads();

  // copy the result into a excludive scan array.
  d_cdf[0] = 0;
  // this piece of code is error prone to gpu code creating race conditions.
  // if (absId > 0) {
  //    d_cdf[absId] = d_histo[absId - 1];
  //}
  if (absId < (blockDim.x * gridDim.x - 1))
  {
    d_cdf[absId + 1] = d_histo[absId];
  }
}

M3_DLL
void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float& min_logLum,
                                  float& max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  // TODO
  /*Here are the steps you need to implement
  1) find the minimum and maximum value in the input logLuminance channel
  store in min_logLum and max_logLum
  2) subtract them to find the range
  3) generate a histogram of all the values in the logLuminance channel using
  the formula: bin = (lum[i] - lumMin) / lumRange * numBins
  4) Perform an exclusive scan (prefix sum) on the histogram to get
  the cumulative distribution of luminance values (this should go in the
  incoming d_cdf pointer which already has been allocated for you)       */
  const unsigned int pxCount = numRows * numCols;
  const int pxSize           = pxCount * sizeof(float);
  const int blockSize        = 512;
  const int gridSize         = (pxCount + blockSize * 4 - 1) / (blockSize * 4);

  float *d_min_Lums, *d_max_Lums, *d_logLum_Min, *d_logLum_Max;

  // std::unique_ptr<float[]> h_min_Lums(new
  // float[gridSize]);//std::make_unique<float[]>(gridSize);

  checkCudaErrors(cudaMalloc((void**)&d_min_Lums, gridSize * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&d_max_Lums, gridSize * sizeof(float)));
  checkCudaErrors(
      cudaMalloc((void**)&d_logLum_Min, pxCount * sizeof(float) / 4));
  checkCudaErrors(
      cudaMalloc((void**)&d_logLum_Max, pxCount * sizeof(float) / 4));

  // printf("blockSize is %i\n", blockSize);
  // printf("gridSize is %i\n", gridSize);
  // printf("pxCount is %i\n", pxCount);
  // printf("pxSize is %i\n", pxSize);
  // printf("numBins is %i\n", numBins);

  findMinMax<<<gridSize, blockSize>>>(d_logLuminance,
                                      d_logLum_Min,
                                      d_logLum_Max,
                                      d_min_Lums,
                                      d_max_Lums,
                                      pxCount);
  cudaDeviceSynchronize();

  float min_Lums = 10;
  float max_Lums = -10;

  float* h_min_Lums = new float[gridSize];
  float* h_max_Lums = new float[gridSize];

  // copy back arrays of minimum and maximum values for further reduction on cpu
  // to a single minimum and maximum
  checkCudaErrors(cudaMemcpy(h_min_Lums,
                             d_min_Lums,
                             gridSize * sizeof(float),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_max_Lums,
                             d_max_Lums,
                             gridSize * sizeof(float),
                             cudaMemcpyDeviceToHost));

  for (int i = 0; i < gridSize; i++)
  {
    min_Lums = std::min<float>(h_min_Lums[i], min_Lums);
    max_Lums = std::max<float>(h_max_Lums[i], max_Lums);
  }

  min_logLum = min_Lums;
  max_logLum = max_Lums;

  float lumRange = max_logLum - min_logLum;

  // printf("min is %f\n", min_logLum);
  // printf("max is %f\n", max_logLum);
  // printf("range is %f\n", lumRange);

  // unsigned int * h_histo = new unsigned int[numBins];
  // unsigned int * h_cdf = new unsigned int[numBins];

  // for (int i = 0; i < numBins; i++) {
  //    h_histo[i] = 0;
  //    h_cdf[i] = 0;
  //}
  unsigned int* d_histo;

  // std::vector<unsigned int> binCollect = {};
  unsigned int sharedMemSize = blockSize * sizeof(unsigned int);

  checkCudaErrors(cudaMalloc((void**)&d_histo, numBins * sizeof(unsigned int)));

  calcHisto<<<numBins, blockSize, sharedMemSize>>>(
      d_histo, d_logLuminance, min_logLum, lumRange, numBins, pxCount);
  cudaDeviceSynchronize();

  // checkCudaErrors(cudaMemcpy(h_histo, d_histo, numBins * sizeof(unsigned
  // int), cudaMemcpyDeviceToHost));
  //
  // for (int i = 1; i < numBins; i++) {
  //    h_cdf[i] = h_cdf[i - 1] + h_histo[i - 1];
  //    printf("cdf_cpu at %i is %i.\n", i, h_cdf[i]);

  //    printf("Histo at %i is %i.\n", i, h_histo[i]);
  //}

  // this section of code replace the log2 function that is too expensive.
  // calculate how many loops the Hillis Steele scan is required for given array
  // size
  int cycles = 0;
  for (int i = numBins; i != 0; i >>= 1)
  {
    cycles++;
  }
  cycles -= 1;
  int modCheck = 1 << cycles;
  if (numBins > modCheck)
  {
    cycles += 1;
  }

  // same as above but much more expensive
  // int cycles = int(log2(float(numBins)));
  // if (int(log2(float(numBins))*10) > int(log2(float(numBins))*10)) {
  //    cycles = int(log2(float(numBins)))+1;
  //}

  int cdfGridSize = numBins / blockSize;

  calcCDF<<<cdfGridSize, blockSize, sharedMemSize>>>(
      d_cdf, d_histo, numBins, cycles);
  cudaDeviceSynchronize();

  // unsigned int * h_cdf = new unsigned int[numBins];
  // checkCudaErrors(cudaMemcpy(h_cdf, d_cdf, numBins * sizeof(unsigned int),
  // cudaMemcpyDeviceToHost));

  // for (int i = 0; i < numBins; i++) {
  //    printf("cdf_gpu at %i is %i.\n", i, h_cdf[i]);
  //}

  // delete[] h_min_lums;
  // delete[] h_max_lums;
  // delete[] h_histo;
  // delete[] h_cdf;
}
