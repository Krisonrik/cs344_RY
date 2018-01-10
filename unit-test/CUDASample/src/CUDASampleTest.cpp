// Copyright 2015 Jason Juang

#include "CUDASampleTest.h"

HVR_WINDOWS_DISABLE_ALL_WARNING
#include "opencv2/opencv.hpp"
HVR_WINDOWS_ENABLE_ALL_WARNING

#include "Hvr/CUDASample/CUDASample.cuh"

CUDASampleTest::CUDASampleTest()
{
}

CUDASampleTest::~CUDASampleTest()
{
}

void CUDASampleTest::SetUp()
{
}

void CUDASampleTest::TearDown()
{
}

TEST_F(CUDASampleTest, TurnImageToWhiteTestGood)
{
  bool correct = true;

  cv::Mat black = cv::Mat(100, 100, CV_8UC1, cv::Scalar(0));

  hvr::CUDASample cudasample;

  cudasample.SetImageToWhite(black);

  const int h = black.rows;
  const int w = black.cols;

  for (int i = 0; i < h; i++)
    for (int j = 0; j < w; j++)
    {
      if (black.at<uchar>(i, j) != 255) correct = false;
    }

  ASSERT_EQ(correct, true);
}
