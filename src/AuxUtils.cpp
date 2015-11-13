#include "AuxUtils.h"
#include <sstream>
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