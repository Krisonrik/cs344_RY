#ifndef GPU_TIMER_H__
#define GPU_TIMER_H__

#include <cuda_runtime.h>

struct GpuTimer
{
  cudaEvent_t start;
  cudaEvent_t stop;

  M3_DLL
  GpuTimer()
  {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  M3_DLL
  ~GpuTimer()
  {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  M3_DLL
  void Start()
  {
    cudaEventRecord(start, 0);
  }

  M3_DLL
  void Stop()
  {
    cudaEventRecord(stop, 0);
  }

  M3_DLL
  float Elapsed()
  {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

#endif /* GPU_TIMER_H__ */
