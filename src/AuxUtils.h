#ifndef AUXUTILS_H
#define AUXUTILS_H
#include <opencv2/opencv.hpp>
#include <vector>

void showPyr(const std::vector<cv::Mat> pyr,int nOctaves,int nOctaveLayers=3+3);

void writePyr(const std::vector<cv::Mat> pyr,int nOctaves,int nOctaveLayers,
	const char *dir,bool isStretch);

void writePyrValue(const std::vector<cv::Mat> pyr,int nOctaves,int nOctaveLayers,
	const char *dir);

//»­Ô²
void DrawCirlcle(cv::Mat& image,const std::vector<cv::KeyPoint>& keypoints,
	const cv::Scalar& color=cv::Scalar::all(-1));

//»­½»²æÊ®×Ö
void DrawCross(cv::Mat& image,const std::vector<cv::KeyPoint>& keypoints,
	const cv::Scalar& color=cv::Scalar::all(-1));

#endif