// Copyright 2015 Jason Juang

#include <iostream>

HVR_WINDOWS_DISABLE_ALL_WARNING
#include "opencv2/opencv.hpp"
HVR_WINDOWS_ENABLE_ALL_WARNING

#include "Hvr/CUDASample/CUDASample.cuh"

int main()
{
  cv::Mat black = cv::Mat(100, 100, CV_8UC1, cv::Scalar(0));

  hvr::CUDASample cudasample;

  cudasample.SetImageToWhite(black);

  for (int i = 0; i < 100; i++)
    for (int j = 0; j < 100; j++)
    {
      if (black.at<uchar>(i, j) == 255)
      {
        continue;
      }
      else
      {
        std::cout << "something is wrong..." << std::endl;
        return -1;
      }
    }

  std::cout << "you passed the test!" << std::endl;

  return 0;
}
