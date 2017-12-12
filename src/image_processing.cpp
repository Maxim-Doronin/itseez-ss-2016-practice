#include "image_processing.hpp"

#include <cstddef>

using namespace std;
using namespace cv;

Mat ImageProcessorImpl::CvtColor(const cv::Mat &src, const cv::Rect &roi) {
	Mat src_cpy;
	src.copyTo(src_cpy);
	Mat src_cpy_roi = src_cpy(roi);
	Mat dst1c;
	cvtColor(src_cpy_roi, dst1c, COLOR_BGR2GRAY);

	vector<Mat> channels;
	channels.push_back(dst1c);
	channels.push_back(dst1c);
	channels.push_back(dst1c);

	Mat dst3c;
	merge(channels, dst3c);
	dst3c.copyTo(src_cpy_roi);
	return src_cpy;
}

Mat ImageProcessorImpl::Filter(const cv::Mat &src, const cv::Rect &roi,
							const int kSize) {
	Mat src_cpy = src;
	Mat src_cpy_roi = src_cpy(roi);
	medianBlur(src_cpy_roi, src_cpy_roi, kSize);
	
	return src_cpy;
}

Mat ImageProcessorImpl::DetectEdges(const cv::Mat &src, const cv::Rect &roi,
							const int filterSize, const int lowThreshold, 
							const int ratio, const int kernelSize) {
	Mat src_roi = src(roi);
	Mat src_gray_roi;
  if (src_gray_roi.type() != CV_8UC1) {
    cvtColor(src_roi, src_gray_roi, COLOR_BGR2GRAY);
  } else {
    src_gray_roi = src_roi;
  }

	Mat src_blurred;
	blur(src_gray_roi, src_blurred, Size(filterSize, filterSize));

	Mat detected_edges;
	Canny(src_blurred, detected_edges, lowThreshold, lowThreshold * ratio, kernelSize);

	Mat dst;
	src.copyTo(dst);
	Mat dst_roi = dst(roi);
	dst_roi = Scalar::all(0);
	
	src_roi.copyTo(dst_roi, detected_edges);

	return dst;
}

Mat ImageProcessorImpl::Pixelize(const cv::Mat &src, const cv::Rect &roi,
							const int kDivs) {
	Mat src_cpy;
	src.copyTo(src_cpy);
	Mat src_cpy_roi = src_cpy(roi);

	int block_size_x = roi.width / kDivs;
	int block_size_y = roi.height / kDivs;
	
  for (int i = 0; i < kDivs; i++) {
    for (int j = 0; j < kDivs; j++) {
      Mat src_roi_block = src_cpy_roi(Rect(i*block_size_x, j*block_size_y, block_size_x, block_size_y));
      blur(src_roi_block, src_roi_block, Size(block_size_x, block_size_y));
    }
  }

	return src_cpy;
}

Mat ImageProcessorImpl::DistanceTransform(const cv::Mat &src, 
    const int distLowThreshold, const int distHighThreshold, 
    const int distanceType, const int distMaskSize) {
  Mat src_cpy;
  src.copyTo(src_cpy);
  cvtColor(src_cpy, src_cpy, COLOR_BGR2GRAY);

  const int edgesFilterSize = 5;
  const int edgesLowThreshold = 200;
  const int edgesRatio = 3;
  const int edgesKernelSize = 5;
  Mat src_cpy_canny = src_cpy;
  src_cpy_canny = this->DetectEdges(src_cpy,
    Rect(0, 0, src_cpy.cols, src_cpy.rows), edgesFilterSize,
    edgesLowThreshold, edgesRatio, edgesKernelSize);

  Mat thresholded;
  threshold(src_cpy_canny, thresholded, distLowThreshold, distHighThreshold,
    CV_THRESH_BINARY_INV);
  Mat distTransformed;
  distanceTransform(thresholded, distTransformed, distanceType, distMaskSize);
  distTransformed.convertTo(distTransformed, CV_8UC1);
 
  return distTransformed;
}

Mat ImageProcessorImpl::AverageFilter(const Mat &src) {
  Mat src_cpy;
  src.copyTo(src_cpy);

  Mat integralImage;
  integral(src_cpy, integralImage);
  integralImage.convertTo(integralImage, CV_32S);
  
  Mat distTransformedImage;
  const int distLowThreshold = 0;
  const int distHighThreshold = 255;
  const int distanceType = CV_DIST_L2;
  const int distMaskSize = 3;
  distTransformedImage = this->DistanceTransform(src_cpy, distLowThreshold, 
    distHighThreshold, distanceType, distMaskSize);

  Mat dst;
  src.copyTo(dst);
  dst.convertTo(dst, CV_8UC3);
  int rows = distTransformedImage.rows;
  int cols = distTransformedImage.cols;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int dist = (int)distTransformedImage.at<uint8_t>(i, j);
      if (dist % 2 != 1) {
        dist++;
      }
      int halfDist = dist / 2;

      if ((i - halfDist < 0) || (i + halfDist + 1 > rows - 1) ||
        (j - halfDist < 0) || (j + halfDist + 1 > cols - 1)) {
        continue;
      }

      if (dist > 0) {
        dst.at<cv::Vec3b>(i, j) = (
          integralImage.at<Vec3i>(i - halfDist, j - halfDist) -
          integralImage.at<Vec3i>(i - halfDist, j + halfDist + 1) -
          integralImage.at<Vec3i>(i + halfDist + 1, j - halfDist) +
          integralImage.at<Vec3i>(i + halfDist + 1, j + halfDist + 1)) 
          / (dist * dist);
      }
    }
  }

  return dst;
}

float ReLU(float x) {
  if (x < 0) {
    return 0;
  }
  return x;
}

Mat ImageProcessorImpl::Convolution(const cv::Mat &src) {
  Mat src_cpy;
  src.copyTo(src_cpy);
  cvtColor(src_cpy, src_cpy, COLOR_RGB2GRAY);
  Mat convolution(src_cpy.size(), CV_8UC1);

  const int w = 3;
  const int h = 3;
  int conv_matrix[w][h] = { { 0, -1, 0 },{ -1, 5, -1 },{ 0, -1, 0} };
  
  int rows = src_cpy.rows;
  int cols = src_cpy.cols;
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      float result = 0;
      for (int i = -w/2; i <= w/2; i++) {
        for (int j = -h/2; j <= h/2; j++) {
          int i_index = row + i;
          int j_index = col + j;

          i_index = i_index < 0 ? 0 : i_index;
          i_index = i_index >= src_cpy.rows ? src_cpy.rows - 1 : i_index;
          j_index = j_index < 0 ? 0 : j_index;
          j_index = j_index >= src_cpy.cols ? src_cpy.cols - 1 : j_index;

          result += conv_matrix[i + 1][j + 1] * src_cpy.at<uint8_t>(i_index, j_index);
        }
      }
      convolution.at<uint8_t>(row, col) = ReLU(result);
    }
  }
  return convolution;
}