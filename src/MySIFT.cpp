#include "MySIFT.h"
using std::vector;
using namespace cv;
namespace mysift
{

void CreateInitialImage(const Mat &src,Mat &dst, bool doubleImageSize,double sigma)
{
	Mat gray,gray32F,gray32Fdbl;
	if (src.channels()==1)
		src.copyTo(gray);
	else if (src.channels()==3||src.channels()==4)
		cvtColor(src,gray,COLOR_BGR2GRAY);
	//ת��Ϊ32λ����ͼ��
	gray.convertTo(gray32F,CV_32FC1);
	if (doubleImageSize)
	{
		double sigma_dif=sqrt(sigma*sigma-SIFT_INIT_SIGMA*SIFT_INIT_SIGMA*4);
		resize(gray32F,gray32Fdbl,Size(gray32F.cols*2,gray32F.rows*2),0,0,INTER_LINEAR);
		GaussianBlur(gray32Fdbl,dst,Size(),sigma_dif,sigma_dif);
	}
	else
	{
		double sigma_dif=sqrt(sigma*sigma-SIFT_INIT_SIGMA*SIFT_INIT_SIGMA);
		GaussianBlur(gray32F,dst,Size(),sigma_dif,sigma_dif);
	}
	//����ͼ�����ʾЧ��**********************************************************
	//double sigma_dif=sqrt(sigma*sigma-SIFT_INIT_SIGMA*SIFT_INIT_SIGMA*4);
	//resize(gray32F,gray32Fdbl,Size(gray32F.cols*2,gray32F.rows*2),0,0,INTER_LINEAR);
	//GaussianBlur(gray32Fdbl,gray32Fdbl,Size(),sigma_dif,sigma_dif);
	//gray32Fdbl.convertTo(gray32Fdbl,1.0/255);

	//sigma_dif=sqrt(sigma*sigma-SIFT_INIT_SIGMA*SIFT_INIT_SIGMA);
	//GaussianBlur(gray32F,gray32F,Size(),sigma_dif,sigma_dif);
	//gray32F.convertTo(gray32F,1.0/255);

	//imshow("gray",gray);

	//imshow("gray32F sigma=1.52",gray32F);
	//imshow("gray32fdbl sigma=1.25",gray32Fdbl);
	//waitKey(0);
	//����ͼ�����ʾЧ��**********************************************************
}

void BuildGaussianPyramid( const Mat& base, vector<Mat>& pyr,int nOctaves,int nOctaveLayers ,double sigma)
{
	vector<double> sigOtc(nOctaveLayers+3);
	//��˹����������=����*ÿ�����
	pyr.resize(nOctaves*(nOctaveLayers+3));

	//����ÿ���˹ģ����sigma����
	sigOtc.at(0)=sigma;
	double k=pow(2.,1./nOctaveLayers);
	for (int i=1;i<nOctaveLayers+3;i++)
	{
		double sig_pre=sigma*pow(k,i-1);
		double sig_total=sig_pre*k;
		//��Ϊ�õ����ʼ��ͼ��������һ���߶ȣ������޷�ֱ�����ɹ̶��ĳ߶ȣ����Դ�߶���С�߶�����
		sigOtc.at(i)=sqrt(sig_total*sig_total-sig_pre*sig_pre);
	}
	//���ɸ�˹������
	for (int o=0;o<nOctaves;o++)
	{
		for (int s=0;s<nOctaveLayers+3;s++)
		{
			Mat &dst=pyr.at(o*(nOctaveLayers+3)+s);
			if (s==0&&o==0)
				base.copyTo(dst);
			else if (s==0)
			{
				//�����ÿ��ĵ�һ��ͼ��ͽ���һ��ĵ����ڶ���ͼ����С����
				const Mat &src=pyr.at(o*(nOctaveLayers+3)-2);
				resize(src,dst,Size(src.cols/2,src.rows/2));
			}
			else
			{
				//����ͼ������һ��ͼ��ͨ����˹ģ���õ�
				const Mat &src=pyr.at(o*(nOctaveLayers+3)+s-1);
				GaussianBlur(src,dst,Size(),sigOtc.at(s));
			}
		}
	}
}

//������˹��ֽ�����
void BuildDoGPyramid(vector<cv::Mat>& pyr,vector<cv::Mat>& dogpyr,int nOctaves,int nOctaveLayers)
{
	dogpyr.resize(nOctaves*(nOctaveLayers+2));

	for (int o=0;o<nOctaves;o++)
	{
		for (int s=0;s<nOctaveLayers+2;s++)
		{
			Mat &dst=dogpyr.at(o*(nOctaveLayers+2)+s);
			const Mat& src1=pyr.at(o*(nOctaveLayers+3)+s);
			const Mat& src2=pyr.at(o*(nOctaveLayers+3)+s+1);
			subtract(src2,src1,dst,noArray(),CV_32FC1);
		}
	}
}

//�ж��Ƿ��ǳ߶ȿռ�ļ�ֵ��
bool IsExtrema(const std::vector<cv::Mat>& dogpyr,int idx,int r,int c)
{
	float val=dogpyr.at(idx).at<float>(r,c);
	int i,j,k;
	//�ж��Ƿ����ֵ
	if (val>0)
	{
		for (i=-1;i<=1;i++)
		{
			for (j=-1;j<=1;j++)
			{
				for(k=-1;k<=1;k++)
				{
					if (val<dogpyr.at(idx+k).at<float>(r+i,c+j))
						return false;
				}
			}
		}
	}
	else
	{
		//�ж��Ƿ���Сֵ
		for (i=-1;i<=1;i++)
		{
			for (j=-1;j<=1;j++)
			{
				for(k=-1;k<=1;k++)
				{
					if (val>dogpyr.at(idx+k).at<float>(r+i,c+j))
						return false;
				}
			}
		}
	}
	return true;
	
}

//�ҵ��߶ȿռ�ļ�ֵ��
void FindSpaceScaleExtrema(std::vector<cv::Mat>& dogpyr,int nOctaves,int nOctaveLayers,
	float contrastThreshold,float edgeThreshold)
{
	//1.���ҵ������ļ�ֵ��
	int o,s,idx,octRows,octCols,r,c;
	float value;
	for (o=0;o<nOctaves;o++)
	{
		for (s=1;s<=nOctaveLayers;s++)
		{
			idx=o*(nOctaveLayers+2)+s;
			
			octRows=dogpyr.at(o*nOctaveLayers).rows;
			octCols=dogpyr.at(o*nOctaveLayers).cols;
			for (r=SIFT_IMG_BORDER;r<octRows-SIFT_IMG_BORDER;r++)
			{
				for (c=SIFT_IMG_BORDER;c<octCols-SIFT_IMG_BORDER;c++)
				{
					value=dogpyr.at(idx).at<float>(r,c);
					if (IsExtrema(dogpyr,idx,r,c)&&value>0)
					{
					}
				}
			}
		}
	}
}

}