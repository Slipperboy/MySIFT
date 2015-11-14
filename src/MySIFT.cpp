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
	//转化为32位浮点图像
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
	//测试图像的显示效果**********************************************************
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
	//测试图像的显示效果**********************************************************
}

void BuildGaussianPyramid( const Mat& base, vector<Mat>& pyr,int nOctaves,int nOctaveLayers ,double sigma)
{
	vector<double> sigOtc(nOctaveLayers+3);
	//高斯金字塔个数=组数*每组层数
	pyr.resize(nOctaves*(nOctaveLayers+3));

	//计算每组高斯模糊的sigma参数
	sigOtc.at(0)=sigma;
	double k=pow(2.,1./nOctaveLayers);
	for (int i=1;i<nOctaveLayers+3;i++)
	{
		double sig_pre=sigma*pow(k,i-1);
		double sig_total=sig_pre*k;
		//因为得到的最开始的图像总是有一定尺度，所以无法直接生成固定的尺度，所以大尺度由小尺度生成
		sigOtc.at(i)=sqrt(sig_total*sig_total-sig_pre*sig_pre);
	}
	//生成高斯金字塔
	for (int o=0;o<nOctaves;o++)
	{
		for (int s=0;s<nOctaveLayers+3;s++)
		{
			Mat &dst=pyr.at(o*(nOctaveLayers+3)+s);
			if (s==0&&o==0)
				base.copyTo(dst);
			else if (s==0)
			{
				//如果是每组的第一层图像就将上一组的倒数第二层图像缩小两倍
				const Mat &src=pyr.at(o*(nOctaveLayers+3)-2);
				resize(src,dst,Size(src.cols/2,src.rows/2));
			}
			else
			{
				//本层图像由上一层图像通过高斯模糊得到
				const Mat &src=pyr.at(o*(nOctaveLayers+3)+s-1);
				GaussianBlur(src,dst,Size(),sigOtc.at(s));
			}
		}
	}
}

//构建高斯差分金字塔
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

//判断是否是尺度空间的极值点
bool IsExtrema(const std::vector<cv::Mat>& dogpyr,int idx,int r,int c)
{
	float val=dogpyr.at(idx).at<float>(r,c);
	int i,j,k;
	//判断是否最大值
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
		//判断是否最小值
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

//通过泰勒展开式拟合出真正的极值点，并消除边界响应
bool AdjustLocalExtrema(const vector<Mat>& dog_pyr, KeyPoint& kpt, int octv,
	int& layer, int& r, int& c, int nOctaveLayers,
	float contrastThreshold, float edgeThreshold, float sigma)
{
	//1.找到真正的极值点
	const float image_scale=1.f/255;
	const float deriv_scale=0.5f*image_scale;
	const float second_deriv_scale=image_scale;
	const float corss_deriv_scale=0.25f*image_scale;
	int i;
	float xc,xr,xi,contr;
	for (i=0;i<SIFT_MAX_INTERP_STEPS;i++)
	{
		//layer经过调整后位置可能会变
		const Mat &curr=dog_pyr.at(octv*(nOctaveLayers+2)+layer);
		const Mat &prev=dog_pyr.at(octv*(nOctaveLayers+2)+layer-1);
		const Mat &next=dog_pyr.at(octv*(nOctaveLayers+2)+layer+1);

		//{ dI/dx, dI/dy, dI/ds }^T
		Vec3f dD((curr.at<float>(r,c+1)-curr.at<float>(r,c+1))*deriv_scale,
			(curr.at<float>(r+1,c)-curr.at<float>(r-1,c))*deriv_scale,
			(next.at<float>(r,c)-prev.at<float>(r,c))*deriv_scale);
		/*	/ Ixx  Ixy  Ixs \ <BR>
			| Ixy  Iyy  Iys | <BR>
			\ Ixs  Iys  Iss /     */

		float dxx=(curr.at<float>(r,c+1)-curr.at<float>(r,c-1)-2*curr.at<float>(r,c))*second_deriv_scale;
		float dyy=(curr.at<float>(r+1,c)-curr.at<float>(r-1,c)-2*curr.at<float>(r,c))*second_deriv_scale;
		float dss=(next.at<float>(r,c)-prev.at<float>(r,c)-2*curr.at<float>(r,c))*second_deriv_scale;
		float dxy=(curr.at<float>(r+1,c+1)+curr.at<float>(r-1,c-1)
			-curr.at<float>(r-1,c+1)-curr.at<float>(r-1,c+1))*corss_deriv_scale;
		float dxs=(next.at<float>(r,c+1)+prev.at<float>(r,c-1)
			-next.at<float>(r,c-1)-prev.at<float>(r,c+1))*corss_deriv_scale;
		float dys=(next.at<float>(r+1,c)+prev.at<float>(r-1,c)
			-next.at<float>(r-1,c)-prev.at<float>(r+1,c))*corss_deriv_scale;
		Matx33f H(dxx,dxy,dxs,dxy,dyy,dys,dxs,dys,dss);
		//H*X=dD
		Vec3f X=H.solve(dD,DECOMP_LU);
		xc=-X[0];
		xr=-X[1];
		xi=-X[2];
		//找到
		if (std::abs(xc)<0.5f&&std::abs(xr)<0.5f&&std::abs(xi)<0.5f)
			break;
		//如果特别大，就返回失败
		if (std::abs(xc)>float(INT_MAX)||std::abs(xc)>float(INT_MAX)||std::abs(xc)>float(INT_MAX))
			return false;
		//否则继续迭代
		c+=cvRound(xc);
		r+=cvRound(xr);
		layer+=cvRound(xi);

		if (layer<1||layer>nOctaveLayers||c<SIFT_IMG_BORDER||c>=curr.cols-SIFT_IMG_BORDER
			||r<SIFT_IMG_BORDER||r>=curr.rows-SIFT_IMG_BORDER)
			return false;
	}
	if(i>=SIFT_MAX_INTERP_STEPS)
		return false;

	//2.计算极值，根据thr=constras/nOctaveLabeys消除不稳定极值点
	const Mat &curr=dog_pyr.at(octv*(nOctaveLayers+2)+layer);
	const Mat &prev=dog_pyr.at(octv*(nOctaveLayers+2)+layer-1);
	const Mat &next=dog_pyr.at(octv*(nOctaveLayers+2)+layer+1);

	//是否和Vec3f类型一致？
	Matx31f dD((curr.at<float>(r,c+1)-curr.at<float>(r,c+1))*deriv_scale,
		(curr.at<float>(r+1,c)-curr.at<float>(r-1,c))*deriv_scale,
		(next.at<float>(r,c)-prev.at<float>(r,c))*deriv_scale);

	float t= dD.dot(dD);
	contr=curr.at<float>(r,c)*image_scale+0.5f*t;
	if (std::abs(contr)*nOctaveLayers<contrastThreshold)
		return false;
	//3.消除边缘响应

	/*	| Dxx  Dxy | 
		| Dxy  Dyy |     */
	float dxx=(curr.at<float>(r,c+1)-curr.at<float>(r,c-1)-2*curr.at<float>(r,c))*second_deriv_scale;
	float dyy=(curr.at<float>(r+1,c)-curr.at<float>(r-1,c)-2*curr.at<float>(r,c))*second_deriv_scale;
	float dxy=(curr.at<float>(r+1,c+1)+curr.at<float>(r-1,c-1)
		-curr.at<float>(r-1,c+1)-curr.at<float>(r-1,c+1))*corss_deriv_scale;

	float tr=dxx+dyy;
	float det=dxx*dyy-dxy*dxy;

	//required: tr/det < (edge_thresh+1)^2/edge_thresh
	if (det <= 0||tr*edgeThreshold>=det*(edgeThreshold+1)*(edgeThreshold+1))
		return false;

	kpt.pt.x = (c + xc) * (1 << octv);
	kpt.pt.y = (r + xr) * (1 << octv);
	//下面这段代码的意义？
	kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5)*255) << 16);
	//参考
	/*if( firstOctave < 0 )
		for( size_t i = 0; i < keypoints.size(); i++ )
		{
			KeyPoint& kpt = keypoints[i];
			float scale = 1.f/(float)(1 << -firstOctave);
			kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
			kpt.pt *= scale;
			kpt.size *= scale;
		}*/

	kpt.size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv)*2;
}

//找到尺度空间的极值点
void FindSpaceScaleExtrema(std::vector<cv::Mat>& dogpyr,int nOctaves,int nOctaveLayers,
	float contrastThreshold,float edgeThreshold)
{
	//1.先找到初步的极值点
	int o,s,idx,octRows,octCols,r,c;
	float value;
	int threshold=0.5f*contrastThreshold/nOctaveLayers*255;
	KeyPoint initialKpt;
	KeyPoint interpKpt;
	KeyPoint finalKpt;
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
					if (IsExtrema(dogpyr,idx,r,c)&&value>threshold)
					{
						/*initialKpt.pt.x=c;
						initialKpt.pt.y=r;
						initialKpt.octave=*/
						int r1=r,c1=c,layer=s;
						AdjustLocalExtrema(dogpyr,interpKpt,o,layer,r1,c1,
							nOctaveLayers,SIFT_CONTR_THR,SIFT_CURV_THR,1.6f);
					}
				}
			}
		}
	}
}

}