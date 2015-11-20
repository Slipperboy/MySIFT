#ifndef MYSIFT_H
#define MYSIFT_H
#include <opencv2/opencv.hpp>
#include <vector>
namespace mysift
{
// default number of sampled intervals per octave
const int SIFT_INTVLS = 3;

// default sigma for initial gaussian smoothing
const float SIFT_SIGMA = 1.6f;

// default threshold on keypoint contrast |D(x)|
const float SIFT_CONTR_THR = 0.04f;

// default threshold on keypoint ratio of principle curvatures
const float SIFT_CURV_THR = 10.f;

// double image size before pyramid construction?
const bool SIFT_IMG_DBL = true;

// default width of descriptor histogram array
const int SIFT_DESCR_WIDTH = 4;

// default number of bins per histogram in descriptor array
const int SIFT_DESCR_HIST_BINS = 8;

// assumed gaussian blur for input image
const float SIFT_INIT_SIGMA = 0.5f;

// width of border in which to ignore keypoints
const int SIFT_IMG_BORDER = 5;

// maximum steps of keypoint interpolation before failure
const int SIFT_MAX_INTERP_STEPS = 5;

// default number of bins in histogram for orientation assignment
const int SIFT_ORI_HIST_BINS = 36;

// determines gaussian sigma for orientation assignment
const float SIFT_ORI_SIG_FCTR = 1.5f;

// determines the radius of the region used in orientation assignment
const float SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR;

// orientation magnitude relative to max that results in new feature
const float SIFT_ORI_PEAK_RATIO = 0.8f;

// determines the size of a single descriptor orientation histogram
const float SIFT_DESCR_SCL_FCTR = 3.f;

// threshold on magnitude of elements of descriptor vector
const float SIFT_DESCR_MAG_THR = 0.2f;

// factor used to convert floating-point descriptor to unsigned char
const float SIFT_INT_DESCR_FCTR = 512.f;

//创建金字塔的第一层图像
void CreateInitialImage(const cv::Mat &src,cv::Mat &dst, bool doubleImageSize,float sigma=SIFT_SIGMA);

void BuildGaussianPyramid( const cv::Mat& base, std::vector<cv::Mat>& pyr,
	int nOctaves,int nOctaveLayers=3 ,double sigma=SIFT_SIGMA);

void BuildDoGPyramid(std::vector<cv::Mat>& pyr,std::vector<cv::Mat>& dogpyr
	,int nOctaves,int nOctaveLayers=3);

void FindSpaceScaleExtrema(std::vector<cv::Mat>& dogpyr,std::vector<cv::Mat>& guasspyr,
	std::vector<cv::KeyPoint>& keypoints,int nOctaves,int nOctaveLayers=SIFT_INTVLS,
	float contrastThreshold=SIFT_CONTR_THR,float edgeThreshold=SIFT_CURV_THR);

void FindSpaceScaleExtrema(std::vector<cv::Mat>& dogpyr,std::vector<cv::KeyPoint>& initialKeypoints,
	std::vector<cv::KeyPoint>& interpKeypoints,std::vector<cv::KeyPoint>& finalKeypoints,int nOctaves,
	int nOctaveLayers=SIFT_INTVLS,float contrastThreshold=SIFT_CONTR_THR,float edgeThreshold=SIFT_CURV_THR);

void AdjustByInitialImage(std::vector<cv::KeyPoint>& keypoints,int firstOctave);

void UnpackKeypoint(const cv::KeyPoint& kpt,int &oct,int &layer,float &scale);

void CalcDescritors(const std::vector<cv::Mat>& guasspyr,const std::vector<cv::KeyPoint>& keypoints,
	cv::Mat descriptors,int firstOctave,int nOcatveLayers=SIFT_INTVLS);

void MySIFT(const cv::Mat img,std::vector<cv::KeyPoint>& keypoints,cv::Mat descriptors,
	int nOctaveLayers=SIFT_INTVLS,double contrastThreshold=SIFT_CONTR_THR,
	float edgeThreshold=SIFT_CURV_THR,float sigma=SIFT_SIGMA );

//OpenCV源码
void createInitialImageCV( const cv::Mat& img,cv::Mat &dst, bool doubleImageSize, float sigma=SIFT_SIGMA );
void buildDoGPyramidCV( const std::vector<cv::Mat>& gpyr, std::vector<cv::Mat>& dogpyr,
int nOctaves,int nOctaveLayers=SIFT_INTVLS);
void buildGaussianPyramidCV( const cv::Mat& base, std::vector<cv::Mat>& pyr, int nOctaves
	,int nOctaveLayers=SIFT_INTVLS,float sigma=SIFT_SIGMA);

void findScaleSpaceExtremaCV( const std::vector<cv::Mat>& gauss_pyr, const std::vector<cv::Mat>& dog_pyr,
	std::vector<cv::KeyPoint>& keypoints,int nOctaves,int nOctaveLayers=SIFT_INTVLS,
	float contrastThreshold=SIFT_CONTR_THR, float edgeThreshold=SIFT_CURV_THR, float sigma=SIFT_SIGMA );

void siftCV(cv::InputArray _image, cv::InputArray _mask,std::vector<cv::KeyPoint>& keypoints,
	cv::OutputArray _descriptors,
	int nfeatures,int nOctaveLayers=SIFT_INTVLS,float contrastThreshold=SIFT_CONTR_THR,
	float edgeThreshold=SIFT_CURV_THR,float sigma=SIFT_SIGMA );
}

#endif