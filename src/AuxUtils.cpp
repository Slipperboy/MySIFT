#include "AuxUtils.h"
#include <sstream>
#include <fstream>
#include <algorithm>
using namespace std;
using namespace cv;

void showPyr(const std::vector<cv::Mat> pyr,int nOctaves,int nOctaveLayers)
{
	ostringstream oss;
	Mat mat;
	for (int o=0;o<nOctaves;o++)
	{
		for (int s=0;s<nOctaveLayers;s++)
		{
			oss.clear();
			oss.str("");
			oss<<"Octave"<<o<<",Layer"<<s;
			pyr.at(o*nOctaveLayers+s).convertTo(mat,1./255);
			imshow(oss.str(),mat);
		}
	}
	
	
	
	
	waitKey(0);
}

void LinearStretch(Mat &src,Mat &dst,double minVal,double maxVal)
{
	uchar data;
	uchar result;
	for (int x=0;x<src.cols;x++)
	{
		for (int y=0;y<src.rows;y++)
		{
			data=src.at<uchar>(y,x);
			if (data>maxVal)
			{
				result=255;
			}
			else if (data<minVal)
			{
				result=0;
			}
			else
			{
				result=(data-minVal)/(maxVal-minVal)*255;
			}
			dst.at<uchar>(y,x)=result;
		}
	}
}

void HistogramAccumlateMinMax(const Mat &mat,double *minVal,double *maxVal)
{
	double p[1024],p1[1024],num[1024];

	memset(p,0,sizeof(p));
	memset(p1,0,sizeof(p1));
	memset(num,0,sizeof(num));

	int height=mat.rows;
	int width=mat.cols;
	long wMulh = height * width;

	//statistics
	for(int x=0;x<width;x++)
	{
		for(int y=0;y<height;y++){
			uchar v=mat.at<uchar>(y,x);
			num[v]++;
		}
	}

	//calculate probability
	for(int i=0;i<1024;i++)
	{
		p[i]=num[i]/wMulh;
	}

	int min=0,max=0;
	double minProb=0.0,maxProb=0.0;
	while(min<1024&&minProb<0.02)
	{
		minProb+=p[min];
		min++;
	}
	do 
	{
		maxProb+=p[max];
		max++;
	} while (max<1024&&maxProb<0.98);

	*minVal=min;
	*maxVal=max;
}

void writePyr(const std::vector<cv::Mat> pyr,int nOctaves,int nOctaveLayers,
	const char *dir,bool isStretch)
{
	ostringstream oss;
	Mat mat;
	for (int o=0;o<nOctaves;o++)
	{
		for (int s=0;s<nOctaveLayers;s++)
		{
			oss.clear();
			oss.str("");
			oss<<dir<<"\\Octave"<<o<<",Layer"<<s<<".jpg";;
			pyr.at(o*nOctaveLayers+s).convertTo(mat,CV_8UC1);
			if (isStretch)
			{
				double min,max;
				HistogramAccumlateMinMax(mat,&min,&max);
				LinearStretch(mat,mat,min,max);
			}
			imwrite(oss.str(),mat);
		}
	}
}

//把金字塔图像的像素值保存，对图像不做任何操作
void writePyrValue(const std::vector<cv::Mat> pyr,int nOctaves,int nOctaveLayers,
	const char *dir)
{
	ostringstream oss;
	
	
	for (int o=0;o<nOctaves;o++)
	{
		for (int s=0;s<nOctaveLayers;s++)
		{
			oss.clear();
			oss.str("");
			oss<<dir<<"\\Octave"<<o<<"Layer"<<s<<".txt";
			ofstream fout(oss.str().c_str());
			const Mat &mat=pyr.at(o*nOctaveLayers+s);
			for (int r=0;r<mat.rows;r++)
			{
				for (int c=0;c<mat.cols;c++)
				{
					float val=mat.at<float>(r,c);
					fout<<val<<" ";
				}
				fout<<"\n";
			}
			fout.close();
		}
	}
}

const int draw_shift_bits = 4;
const int draw_multiplier = 1 << draw_shift_bits;

//画圆
void DrawCirlcle(cv::Mat& image,const vector<KeyPoint>& keypoints,const Scalar& color)
{
	RNG &rng=theRNG();
	bool isRandColor=color==Scalar::all(-1);
	vector<KeyPoint>::const_iterator cite=keypoints.begin();
	for (;cite!=keypoints.end();cite++)
	{
		Scalar _color=isRandColor?Scalar(rng(256), rng(256), rng(256)):color;
		Point center( cvRound((*cite).pt.x * draw_multiplier), cvRound((*cite).pt.y * draw_multiplier) );
		int radius = cvRound((*cite).size/2 * draw_multiplier); 
		circle(image,center,radius,_color,1,CV_AA,draw_shift_bits);
	}
}

//画交叉十字
void DrawCross(cv::Mat& image,const vector<KeyPoint>& keypoints,const Scalar& color)
{
	int halfLength=4;
	RNG &rng=theRNG();
	bool isRandColor=color==Scalar::all(-1);
	vector<KeyPoint>::const_iterator cite=keypoints.begin();
	for (;cite!=keypoints.end();cite++)
	{
		Scalar _color=isRandColor?Scalar(rng(256), rng(256), rng(256)):color;
		Point right(cvRound(((*cite).pt.x+halfLength) * draw_multiplier), cvRound((*cite).pt.y * draw_multiplier));
		Point left(cvRound(((*cite).pt.x-halfLength) * draw_multiplier), cvRound((*cite).pt.y * draw_multiplier));
		Point up(cvRound((*cite).pt.x * draw_multiplier), cvRound(((*cite).pt.y+halfLength) * draw_multiplier));
		Point down(cvRound((*cite).pt.x * draw_multiplier), cvRound(((*cite).pt.y-halfLength) * draw_multiplier));
		line(image,up,down,_color,1,CV_AA,draw_shift_bits);
		line(image,left,right,_color,1,CV_AA,draw_shift_bits);
	}
}

void TestAngle()
{
	float a=fastAtan2(1.,std::sqrt(3.));
	/*float srcAngleRad = p.angle*(float)CV_PI/180.f;
	Point orient( cvRound(cos(srcAngleRad)*radius ),
		cvRound(sin(srcAngleRad)*radius )
		);
	line( img, center, center+orient, color, 1, CV_AA, draw_shift_bits );*/
	//图像坐标系与笛卡尔坐标系是与x轴对称的
	float srcAngleRad = (360.f-a)*(float)CV_PI/180.f;
	float x=cos(srcAngleRad);
	float y=sin(srcAngleRad);
}

bool CmpKpt(const KeyPoint& kp1,const KeyPoint& kp2)
{
	if( std::abs(kp1.pt.x -kp2.pt.x)<FLT_EPSILON &&std::abs(kp1.pt.y -kp2.pt.y)<FLT_EPSILON&&
		std::abs(kp1.size -kp2.size)<FLT_EPSILON&&std::abs(kp1.angle-kp2.angle)<FLT_EPSILON&&
		std::abs(kp1.response-kp2.response)<FLT_EPSILON&&kp1.octave == kp2.octave)
		return true;
	return false;
}

void CompareKeypoints(const std::vector<cv::KeyPoint> kepoints1,const std::vector<cv::KeyPoint> kepoints2)
{
	if (kepoints1.size()!=kepoints2.size())
	{
		std::cout<<"特征点个数不一致";
		return;
	}
	vector<KeyPoint>::iterator ite;
	bool isEq=equal(kepoints1.begin(),kepoints1.end(),kepoints2.begin(),CmpKpt);
}

bool DescrEqual(const float *descr1,const float *descr2,int n)
{
	for (int i=0;i<n;i++)
	{
		if (std::abs(descr1[i]-descr2[i])>=FLT_EPSILON)
			return false;
	}
	return true;
}

//比较两个特征描述集是否完全相等
bool CompareDescriptors(const cv::Mat descr1,const cv::Mat descr2)
{
	int r=descr1.rows;
	for (int i=0;i<r;r++)
	{
		if (!DescrEqual(descr1.ptr<float>(i),descr2.ptr<float>(i),128))
			return false;
	}
	return true;
}
