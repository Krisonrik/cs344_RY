// Copyright 2015 Jason Juang

#include "Hvr/Sample/Sample.h"

#include <iostream>

HVR_WINDOWS_DISABLE_ALL_WARNING
#include "opencv2/opencv.hpp"
HVR_WINDOWS_ENABLE_ALL_WARNING

namespace hvr
{
Sample::Sample()
{
}

Sample::~Sample()
{
}

void Sample::PrintSomething() const
{
  std::cout << "you passed the test!" << std::endl;
}

int Sample::AddTwoInteger(const int a, const int b) const
{
  return a + b;
}

void Sample::TurnImageToWhite(cv::Mat &img) const
{
  const int h = img.rows;
  const int w = img.cols;

  for (int i = 0; i < h; i++)
    for (int j = 0; j < w; j++)
      for (int c = 0; c < 3; c++)
      {
        img.at<cv::Vec3b>(i, j).val[c] = 255;
      }
}

}  // namespace hvr
