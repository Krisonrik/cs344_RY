// Copyright 2015 Jason Juang

#include "Hvr/CUDASample/CUDAConfig.h"

namespace hvr
{
int CUDAConfig::align(const int n, const int a) const
{
  const int r = (n & (a - 1));

  if (r == 0)
  {
    return n;
  }
  else
  {
    return n - r + a;
  }
}

}  // namespace hvr
