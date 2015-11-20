#ifndef AUXUTILS_H
#define AUXUTILS_H
#include <opencv2/opencv.hpp>
#include <vector>

void showPyr(const std::vector<cv::Mat> pyr,int nOctaves,int nOctaveLayers=3+3);

void writePyr(const std::vector<cv::Mat> pyr,int nOctaves,int nOctaveLayers,
	const char *dir,bool isStretch);

void writePyrValue(const std::vector<cv::Mat> pyr,int nOctaves,int nOctaveLayers,
	const char *dir);

//画圆
void DrawCirlcle(cv::Mat& image,const std::vector<cv::KeyPoint>& keypoints,
	const cv::Scalar& color=cv::Scalar::all(-1));

//画交叉十字
void DrawCross(cv::Mat& image,const std::vector<cv::KeyPoint>& keypoints,
	const cv::Scalar& color=cv::Scalar::all(-1));

//测试角度计算
void TestAngle();

//比较两个特征点集是否完全相等
void CompareKeypoints(const std::vector<cv::KeyPoint> kepoints1,const std::vector<cv::KeyPoint> kepoints2);

//比较两个特征描述集是否完全相等
bool CompareDescriptors(const cv::Mat descr1,const cv::Mat descr2);

#endif