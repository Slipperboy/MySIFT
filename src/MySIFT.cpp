#include "MySIFT.h"
using std::vector;
using namespace cv;
namespace mysift
{

const float image_scale=1.f/255*1.f;
const float deriv_scale=0.5f*image_scale;
const float second_deriv_scale=image_scale;
const float cross_deriv_scale=0.25f*image_scale;

//消除边界响应
bool EliminateEdegResponse(const vector<Mat>& dog_pyr, int octv,int nOctaveLayers,
	int r,int c,int layer,float edgeThreshold);

float CaclulateContrast(const vector<Mat>& dog_pyr, int idx,
	int r,int c,int layer,float xc,float xr,float xi);

float calcOrientationHist( const Mat& img, Point pt, int radius,
	float sigma, float* hist, int n );

void CreateInitialImage(const Mat &src,Mat &dst, bool doubleImageSize,float sigma)
{
	Mat gray,gray32F,gray32Fdbl;
	if (src.channels()==1)
		src.copyTo(gray);
	else if (src.channels()==3||src.channels()==4)
		cvtColor(src,gray,COLOR_BGR2GRAY);
	//转化为32位浮点图像
	gray.convertTo(gray32F,CV_32FC1,1.f,0);
	if (doubleImageSize)
	{
		double sigma_dif=sqrt(sigma*sigma-SIFT_INIT_SIGMA*SIFT_INIT_SIGMA*4);
		resize(gray32F,gray32Fdbl,Size(gray.cols*2,gray.rows*2),0,0,INTER_LINEAR);
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
		double sig_pre=sigma*pow(k,double(i-1));
		double sig_total=sig_pre*k;
		//因为得到的最开始的图像总是有一定尺度，所以无法直接生成固定的尺度，所以大尺度由小尺度生成
		sigOtc.at(i)=std::sqrt(sig_total*sig_total-sig_pre*sig_pre);
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
				//int idx1=o*(nOctaveLayers+3)-2;  这段代码错误
				int idx2=(o-1)*(nOctaveLayers+3)+3;
				const Mat &src=pyr.at(idx2);
				resize(src,dst,Size(src.cols/2,src.rows/2),0.0,0.0,INTER_NEAREST);
			}
			else
			{
				//本层图像由上一层图像通过高斯模糊得到
				const Mat &src=pyr.at(o*(nOctaveLayers+3)+s-1);
				//高斯模糊是5个参数，别忘了加上两个sigma
				GaussianBlur(src,dst,Size(),sigOtc.at(s),sigOtc.at(s));
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

//对泰勒展开式求导
void deriv3D(const vector<Mat>& dog_pyr,Vec3f &X,int idx,int layer, int r, int c)
{
	//layer经过调整后位置可能会变
	const Mat &curr=dog_pyr.at(idx);
	const Mat &prev=dog_pyr.at(idx-1);
	const Mat &next=dog_pyr.at(idx+1);

	
	//{ dI/dx, dI/dy, dI/ds }^T
	//求导要仔细
	Vec3f dD((curr.at<float>(r,c+1)-curr.at<float>(r,c-1))*deriv_scale,
		(curr.at<float>(r+1,c)-curr.at<float>(r-1,c))*deriv_scale,
		(next.at<float>(r,c)-prev.at<float>(r,c))*deriv_scale);
	/*	/ Ixx  Ixy  Ixs \ <BR>
		| Ixy  Iyy  Iys | <BR>
		\ Ixs  Iys  Iss /     */

	float dxx=(curr.at<float>(r,c+1)+curr.at<float>(r,c-1)-2*curr.at<float>(r,c))*second_deriv_scale;
	float dyy=(curr.at<float>(r+1,c)+curr.at<float>(r-1,c)-2*curr.at<float>(r,c))*second_deriv_scale;
	float dss=(next.at<float>(r,c)+prev.at<float>(r,c)-2*curr.at<float>(r,c))*second_deriv_scale;
	float dxy=(curr.at<float>(r+1,c+1)+curr.at<float>(r-1,c-1)
		-curr.at<float>(r-1,c+1)-curr.at<float>(r+1,c-1))*cross_deriv_scale;
	float dxs=(next.at<float>(r,c+1)+prev.at<float>(r,c-1)
		-next.at<float>(r,c-1)-prev.at<float>(r,c+1))*cross_deriv_scale;
	float dys=(next.at<float>(r+1,c)+prev.at<float>(r-1,c)
		-next.at<float>(r-1,c)-prev.at<float>(r+1,c))*cross_deriv_scale;
	Matx33f H(dxx,dxy,dxs,dxy,dyy,dys,dxs,dys,dss);
	X=H.solve(dD,DECOMP_LU);

	//OpenCV
	/*const Mat &img=dog_pyr.at(idx);
	Vec3f dD((img.at<float>(r, c+1) - img.at<float>(r, c-1))*deriv_scale,
	(img.at<float>(r+1, c) - img.at<float>(r-1, c))*deriv_scale,
	(next.at<float>(r, c) - prev.at<float>(r, c))*deriv_scale);

	float v2 = (float)img.at<float>(r, c)*2;
	float dxx = (img.at<float>(r, c+1) + img.at<float>(r, c-1) - v2)*second_deriv_scale;
	float dyy = (img.at<float>(r+1, c) + img.at<float>(r-1, c) - v2)*second_deriv_scale;
	float dss = (next.at<float>(r, c) + prev.at<float>(r, c) - v2)*second_deriv_scale;
	float dxy = (img.at<float>(r+1, c+1) - img.at<float>(r+1, c-1) -
	img.at<float>(r-1, c+1) + img.at<float>(r-1, c-1))*cross_deriv_scale;
	float dxs = (next.at<float>(r, c+1) - next.at<float>(r, c-1) -
	prev.at<float>(r, c+1) + prev.at<float>(r, c-1))*cross_deriv_scale;
	float dys = (next.at<float>(r+1, c) - next.at<float>(r-1, c) -
	prev.at<float>(r+1, c) + prev.at<float>(r-1, c))*cross_deriv_scale;

	Matx33f H(dxx, dxy, dxs,
	dxy, dyy, dys,
	dxs, dys, dss);

	X = H.solve(dD, DECOMP_LU);*/
}

//通过泰勒展开式拟合出真正的极值点，并消除边界响应
bool AdjustLocalExtrema(const vector<Mat>& dog_pyr, KeyPoint& kpt, int octv,
	int& layer, int& r, int& c, int nOctaveLayers,
	float contrastThreshold, float edgeThreshold, float sigma)
{
	//1.找到真正的极值点
	
	int i;
	float xc,xr,xi,contr;
	for (i=0;i<SIFT_MAX_INTERP_STEPS;i++)
	{
		//H*X=dD
		Vec3f X;
		int idx=octv*(nOctaveLayers+2)+layer;
		deriv3D(dog_pyr,X,idx,layer,r,c);
		xc=-X[0];
		xr=-X[1];
		xi=-X[2];
		//找到
		if (std::abs(xc)<0.5f&&std::abs(xr)<0.5f&&std::abs(xi)<0.5f)
			break;
		//如果特别大，就返回失败
		if (std::abs(xc)>float(INT_MAX/3)||std::abs(xr)>float(INT_MAX/3)
			||std::abs(xi)>float(INT_MAX/3))
			return false;
		//否则继续迭代
		c+=cvRound(xc);
		r+=cvRound(xr);
		layer+=cvRound(xi);

		if (layer<1||layer>nOctaveLayers||c<SIFT_IMG_BORDER||c>=dog_pyr.at(idx).cols-SIFT_IMG_BORDER
			||r<SIFT_IMG_BORDER||r>=dog_pyr.at(idx).rows-SIFT_IMG_BORDER)
			return false;
	}
	if(i>=SIFT_MAX_INTERP_STEPS)
		return false;

	//2.计算极值，根据thr=constras/nOctaveLabeys消除不稳定极值点
	int idx=octv*(nOctaveLayers+2)+layer;
	contr=CaclulateContrast(dog_pyr,idx,r,c,layer,xc,xr,xi);
	if (std::abs(contr)*nOctaveLayers<contrastThreshold)
		return false;
	//3.消除边缘响应
	if (!EliminateEdegResponse(dog_pyr,octv,nOctaveLayers,r,c,layer,edgeThreshold))
		return false;

	kpt.pt.x = (c + xc) * (1 << octv);
	kpt.pt.y = (r + xr) * (1 << octv);
	//下面这段代码的意义？
	//kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5)*255) << 16);
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
	//直径
	kpt.size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv)*2;
	kpt.response=std::abs(contr);
	return true;
}

//计算极值点3*sigma的领域的加权梯度方向直方图，并返回直方图的最大幅值
//其中sigma=1.5*oct_sigma,oct_sigma是每组内相对于组内第一层影像的尺度
void CacOriHist(const Mat& guass_pyr,int r,int c,int radius
	,float sigma,float *hist,int bins,float &maxMagni)
{
	//领域内总的像素个数
	int len=(2*radius+1)*(2*radius+1);
	int i,j,k;
	//在栈上开辟空间，加快处理速度
	//超过fixed_size=4096/sizeof(_Tp)+8+padding会在堆上重新分配内存
	//padding=(int)((16 + sizeof(_Tp) - 1)/sizeof(_Tp))
	AutoBuffer<float> buf(4*len+bins+4);
	float *X=buf,*Y=X+len,*Mag=X,*Ori=Y+len,*W=Ori+len,*tmpHist=W+len+2;
	//float expf_scale=-0.5f*sigma*sigma; 错误 sigma为分母
	float expf_scale=-1.f/(2.f*sigma*sigma);

	for (i=0;i<bins;i++)
		tmpHist[i]=0.f;
	//对领域的每个点求dx,dy以及权重
	for (i=-radius,k=0;i<=radius;i++)
	{
		int y=r+i;
		if (y<=0||y>=guass_pyr.rows-1)
			continue;
		for (j=-radius;j<=radius;j++)
		{
			int x=c+j;
			if (x<=0||x>=guass_pyr.cols-1)
				continue;
			float dx=guass_pyr.at<float>(y,x+1)-guass_pyr.at<float>(y,x-1);
			float dy=guass_pyr.at<float>(y-1,x)-guass_pyr.at<float>(y+1,x);
			X[k]=dx;
			Y[k]=dy;
			W[k]=(i*i+j*j)*expf_scale;
			k++;
		}
	}
	//由于是以up-down方向求梯度的，所以得到的角度是以顺时针与x轴的夹角
	exp(W,W,k);
	fastAtan2(Y,X,Ori,k,true);
	magnitude(X,Y,Mag,k);

	//计算梯度角度直方图
	for (i=0;i<k;i++)
	{
		int bin=cvRound(Ori[i]/360.f*bins);
		//0-5,5-14,...355-0
		bin=bin>=bins?0:bin;
		tmpHist[bin]+=Mag[i]*W[i];
	}

	//平滑，一维模板[1/16,4/16,6/16,4/16,1/16]
	tmpHist[-2]=tmpHist[bins-2];
	tmpHist[-1]=tmpHist[bins-1];
	tmpHist[bins]=tmpHist[0];
	tmpHist[bins+1]=tmpHist[1];
	for (i=0;i<bins;i++)
	{
		hist[i]=(tmpHist[i-2]+4*tmpHist[i-1]+6*tmpHist[i]+4*tmpHist[i+1]+tmpHist[i+2])/16.f;
	}
	//求梯度角度直方图的最大值
	maxMagni=hist[0];
	for (i=1;i<bins;i++)
		maxMagni=std::max(maxMagni,hist[i]);
}

//找到尺度空间的极值点
void FindSpaceScaleExtrema(vector<Mat>& dogpyr,vector<Mat>& guasspyr,vector<KeyPoint>& keypoints,
	int nOctaves,int nOctaveLayers,float contrastThreshold,float edgeThreshold)
{
	//1.先找到初步的极值点
	int o,s,idx,octRows,octCols,r,c;
	const int bins=SIFT_ORI_HIST_BINS;
	float value;
	int threshold=cvFloor(0.5f*contrastThreshold/nOctaveLayers*255);
	KeyPoint kpt;
	keypoints.clear();
	for (o=0;o<nOctaves;o++)
	{
		for (s=1;s<=nOctaveLayers;s++)
		{
			idx=o*(nOctaveLayers+2)+s;
			
			//获得每组第一层金字塔的行数和列数
			octRows=dogpyr.at(o*(nOctaveLayers+2)).rows;
			octCols=dogpyr.at(o*(nOctaveLayers+2)).cols;
			for (r=SIFT_IMG_BORDER;r<octRows-SIFT_IMG_BORDER;r++)
			{
				for (c=SIFT_IMG_BORDER;c<octCols-SIFT_IMG_BORDER;c++)
				{
					value=dogpyr.at(idx).at<float>(r,c);
					if (IsExtrema(dogpyr,idx,r,c)&&std::abs(value)>threshold)
					{
						int r1=r,c1=c,layer=s;
						if (!AdjustLocalExtrema(dogpyr,kpt,o,layer,r1,c1,
							nOctaveLayers,SIFT_CONTR_THR,SIFT_CURV_THR,1.6f))
							continue;
						
						int idx2=o*(nOctaveLayers+3)+layer;
						float oct_scale=kpt.size*.5f/(1<<o);
						
						float hist[bins];
						float maxMagni/*=calcOrientationHist(guasspyr.at(idx2),
							Point(c1, r1),
							cvRound(SIFT_ORI_RADIUS * oct_scale),
							SIFT_ORI_SIG_FCTR * oct_scale,
							hist, bins)*/;
						CacOriHist(guasspyr.at(idx2),r1,c1,
							cvRound(SIFT_ORI_RADIUS*oct_scale),SIFT_ORI_SIG_FCTR*oct_scale,
							hist,bins,maxMagni);
						float mag_thr=(float)(maxMagni*SIFT_ORI_PEAK_RATIO);
						//计算特征点真正的角度
						for (int i=0;i<bins;i++)
						{
							int left=i-1>=0?i-1:bins-1;
							int right=i+1<=bins-1?i+1:0;
							if (hist[i]>hist[left]&&hist[i]>hist[right]&&hist[i]>mag_thr)
							{
								//内插出真正的角度
								float bin = i+0.5f*(hist[left]-hist[right])/(hist[left]-2*hist[i]+hist[right]);
								bin = bin < 0 ? bins + bin : bin >= bins ? bin - bins : bin;
								kpt.angle = 360.f - (float)((360.f/bins) * bin);
								if(std::abs(kpt.angle - 360.f) < FLT_EPSILON)
									kpt.angle = 0.f;
								//std::cout<<"找到特征点,o="<<o<<",s="<<s<<"r="<<r<<",c"<<c<<std::endl;
								keypoints.push_back(kpt);
							}
						}
					}
				}
			}
		}
	}
}

float CaclulateContrast(const vector<Mat>& dog_pyr, int idx,
	int r,int c,int layer,float xc,float xr,float xi)
{
	//计算极值，根据thr=constras/nOctaveLabeys消除不稳定极值点
	const Mat &curr=dog_pyr.at(idx);
	const Mat &prev=dog_pyr.at(idx-1);
	const Mat &next=dog_pyr.at(idx+1);

	//是否和Vec3f类型一致？
	Matx31f dD((curr.at<float>(r,c+1)-curr.at<float>(r,c-1))*deriv_scale,
		(curr.at<float>(r+1,c)-curr.at<float>(r-1,c))*deriv_scale,
		(next.at<float>(r,c)-prev.at<float>(r,c))*deriv_scale);

	float t= dD.dot(Matx31f(xc, xr, xi));
	return curr.at<float>(r,c)*image_scale+0.5f*t;

	//openCV
	/*const Mat &img=dog_pyr.at(idx);
	Matx31f dD((img.at<float>(r, c+1) - img.at<float>(r, c-1))*deriv_scale,
		(img.at<float>(r+1, c) - img.at<float>(r-1, c))*deriv_scale,
		(next.at<float>(r, c) - prev.at<float>(r, c))*deriv_scale);
	float t = dD.dot(Matx31f(xc, xr, xi));

	return img.at<float>(r, c)*image_scale + t * 0.5f;*/

}

//找到真正的极值点
bool InterpExtrema(const vector<Mat>& dog_pyr, KeyPoint& realKpt, int octv, 
	int& layer, int& r, int& c,int nOctaveLayers,float contrastThreshold,float sigma)
{

	int i;
	float xc,xr,xi,contr;
	for (i=0;i<SIFT_MAX_INTERP_STEPS;i++)
	{
		//H*X=dD
		int idx=octv*(nOctaveLayers+2)+layer;
		Vec3f X;
		deriv3D(dog_pyr,X,idx,layer,r,c);
		xc=-X[0];
		xr=-X[1];
		xi=-X[2];
		//找到
		if (std::abs(xc)<0.5f&&std::abs(xr)<0.5f&&std::abs(xi)<0.5f)
			break;
		//如果特别大，就返回失败
		if (std::abs(xc)>float(INT_MAX/3)||std::abs(xr)>float(INT_MAX/3)
			||std::abs(xi)>float(INT_MAX/3))
			return false;
		//否则继续迭代
		c+=cvRound(xc);
		r+=cvRound(xr);
		layer+=cvRound(xi);
		if (layer<1||layer>nOctaveLayers||c<SIFT_IMG_BORDER||c>=dog_pyr.at(idx).cols-SIFT_IMG_BORDER
			||r<SIFT_IMG_BORDER||r>=dog_pyr.at(idx).rows-SIFT_IMG_BORDER)
			return false;
	}
	if(i>=SIFT_MAX_INTERP_STEPS)
		return false;

	int idx=octv*(nOctaveLayers+2)+layer;
	const Mat &curr=dog_pyr.at(idx);
	const Mat &prev=dog_pyr.at(idx-1);
	const Mat &next=dog_pyr.at(idx+1);

	//是否和Vec3f类型一致？
	Matx31f dD((curr.at<float>(r,c+1)-curr.at<float>(r,c-1))*deriv_scale,
		(curr.at<float>(r+1,c)-curr.at<float>(r-1,c))*deriv_scale,
		(next.at<float>(r,c)-prev.at<float>(r,c))*deriv_scale);

	float t= dD.dot(Matx31f(xc, xr, xi));
	float contr2=curr.at<float>(r,c)*image_scale+0.5f*t;

	contr=CaclulateContrast(dog_pyr,idx,r,c,layer,xc,xr,xi);
	if (std::abs(contr)*nOctaveLayers<contrastThreshold)
		return false;
	realKpt.pt.x=(c+xc)*(1<<octv);
	realKpt.pt.y=(r+xr)*(1<<octv);
	realKpt.size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv)*2;
	realKpt.response=std::abs(contr);
	return true;
}

//消除边界响应
bool EliminateEdegResponse(const vector<Mat>& dog_pyr, int octv,int nOctaveLayers,
	int r,int c,int layer,float edgeThreshold)
{
	const Mat &curr=dog_pyr.at(octv*(nOctaveLayers+2)+layer);
	const Mat &prev=dog_pyr.at(octv*(nOctaveLayers+2)+layer-1);
	const Mat &next=dog_pyr.at(octv*(nOctaveLayers+2)+layer+1);

	//消除边缘响应

	/*	| Dxx  Dxy | 
		| Dxy  Dyy |     */
	float dxx=(curr.at<float>(r,c+1)+curr.at<float>(r,c-1)-2*curr.at<float>(r,c))*second_deriv_scale;
	float dyy=(curr.at<float>(r+1,c)+curr.at<float>(r-1,c)-2*curr.at<float>(r,c))*second_deriv_scale;
	float dxy=(curr.at<float>(r+1,c+1)+curr.at<float>(r-1,c-1)
		-curr.at<float>(r-1,c+1)-curr.at<float>(r+1,c-1))*cross_deriv_scale;

	float tr=dxx+dyy;
	float det=dxx*dyy-dxy*dxy;

	//required: tr*tr/det < (edge_thresh+1)^2/edge_thresh
	//注意：是tr^2
	if (det <= 0||tr*tr*edgeThreshold>=det*(edgeThreshold+1)*(edgeThreshold+1))
		return false;
	
	//OpenCV
	// principal curvatures are computed using the trace and det of Hessian
	/*const Mat &img=dog_pyr.at(octv*(nOctaveLayers+2)+layer);
	float v2 = img.at<float>(r, c)*2.f;
	float dxx = (img.at<float>(r, c+1) + img.at<float>(r, c-1) - v2)*second_deriv_scale;
	float dyy = (img.at<float>(r+1, c) + img.at<float>(r-1, c) - v2)*second_deriv_scale;
	float dxy = (img.at<float>(r+1, c+1) - img.at<float>(r+1, c-1) -
		img.at<float>(r-1, c+1) + img.at<float>(r-1, c-1)) * cross_deriv_scale;
	float tr = dxx + dyy;
	float det = dxx * dyy - dxy * dxy;

	if( det <= 0 || tr*tr*edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1)*det )
		return false;*/

	return true;
}

void FindSpaceScaleExtrema(std::vector<cv::Mat>& dogpyr,std::vector<cv::KeyPoint>& initialKeypoints,
	std::vector<cv::KeyPoint>& interpKeypoints,std::vector<cv::KeyPoint>& finalKeypoints,int nOctaves,
	int nOctaveLayers,float contrastThreshold,float edgeThreshold)
{
	//1.先找到初步的极值点
	int o,s,idx,octRows,octCols,r,c;
	float value;
	int threshold=cvFloor(0.5f*contrastThreshold/nOctaveLayers*255);
	KeyPoint initialKpt;
	KeyPoint interpKpt;
	for (o=0;o<nOctaves;o++)
	{
		for (s=1;s<=nOctaveLayers;s++)
		{
			idx=o*(nOctaveLayers+2)+s;
			
			octRows=dogpyr.at(o*(nOctaveLayers+2)).rows;
			octCols=dogpyr.at(o*(nOctaveLayers+2)).cols;
			for (r=SIFT_IMG_BORDER;r<octRows-SIFT_IMG_BORDER;r++)
			{
				for (c=SIFT_IMG_BORDER;c<octCols-SIFT_IMG_BORDER;c++)
				{
					value=dogpyr.at(idx).at<float>(r,c);
					if (IsExtrema(dogpyr,idx,r,c)&&std::abs(value)>threshold)
					{
						//初始极值点
						int r1=r,c1=c,layer=s;
						initialKpt.pt.x=c*(1<<o);
						initialKpt.pt.y=r*(1<<o);
						initialKpt.size=1.f*powf(2.f, layer/ nOctaveLayers)*(1 << o)*2;
						initialKeypoints.push_back(initialKpt);
						if (!InterpExtrema(dogpyr,interpKpt,o,layer,r1,c1,
							nOctaveLayers,SIFT_CONTR_THR,1.6f))
							continue;
						interpKeypoints.push_back(interpKpt);
						if (!EliminateEdegResponse(dogpyr,o,nOctaveLayers,r1,c1,layer,SIFT_CURV_THR))
							continue;
						finalKeypoints.push_back(interpKpt);
					}
				}
			}
		}
	}
}

//根据初始影像是否扩大了两倍将关键点适应到输入图像上，同时过滤掉重复的点
void AdjustByInitialImage(std::vector<cv::KeyPoint>& keypoints,int firstOctave)
{
	KeyPointsFilter::removeDuplicated(keypoints);
	if( firstOctave < 0 )
		for( size_t i = 0; i < keypoints.size(); i++ )
		{
			KeyPoint& kpt = keypoints[i];
			float scale = 1.f/(float)(1 << -firstOctave);
			kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
			kpt.pt *= scale;
			kpt.size *= scale;
		}
}


//OpenCV源码

void createInitialImageCV( const Mat& img,Mat &dst, bool doubleImageSize, float sigma )
{
	Mat gray, gray_fpt;
	if( img.channels() == 3 || img.channels() == 4 )
		cvtColor(img, gray, COLOR_BGR2GRAY);
	else
		img.copyTo(gray);
	gray.convertTo(gray_fpt, DataType<float>::type, 1.f, 0);

	float sig_diff;

	if( doubleImageSize )
	{
		sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01f) );
		Mat dbl;
		resize(gray_fpt, dbl, Size(gray.cols*2, gray.rows*2), 0, 0, INTER_LINEAR);
		GaussianBlur(dbl, dst, Size(), sig_diff, sig_diff);
	}
	else
	{
		sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f) );
		GaussianBlur(gray_fpt, dst, Size(), sig_diff, sig_diff);
	}
}


void buildGaussianPyramidCV( const Mat& base, vector<Mat>& pyr, int nOctaves,int nOctaveLayers,float sigma) 
{
	vector<double> sig(nOctaveLayers + 3);
	pyr.resize(nOctaves*(nOctaveLayers + 3));

	// precompute Gaussian sigmas using the following formula:
	//  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
	sig[0] = sigma;
	double k = pow( 2., 1. / nOctaveLayers );
	for( int i = 1; i < nOctaveLayers + 3; i++ )
	{
		double sig_prev = pow(k, (double)(i-1))*sigma;
		double sig_total = sig_prev*k;
		sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
	}

	for( int o = 0; o < nOctaves; o++ )
	{
		for( int i = 0; i < nOctaveLayers + 3; i++ )
		{
			Mat& dst = pyr[o*(nOctaveLayers + 3) + i];
			if( o == 0  &&  i == 0 )
				dst = base;
			// base of new octave is halved image from end of previous octave
			else if( i == 0 )
			{
				const Mat& src = pyr[(o-1)*(nOctaveLayers + 3) + nOctaveLayers];
				resize(src, dst, Size(src.cols/2, src.rows/2),
					0, 0, INTER_NEAREST);
			}
			else
			{
				const Mat& src = pyr[o*(nOctaveLayers + 3) + i-1];
				GaussianBlur(src, dst, Size(), sig[i], sig[i]);
			}
		}
	}
}


void buildDoGPyramidCV( const vector<Mat>& gpyr, vector<Mat>& dogpyr,int nOctaves,int nOctaveLayers)
{
	dogpyr.resize( nOctaves*(nOctaveLayers + 2) );

	for( int o = 0; o < nOctaves; o++ )
	{
		for( int i = 0; i < nOctaveLayers + 2; i++ )
		{
			const Mat& src1 = gpyr[o*(nOctaveLayers + 3) + i];
			const Mat& src2 = gpyr[o*(nOctaveLayers + 3) + i + 1];
			Mat& dst = dogpyr[o*(nOctaveLayers + 2) + i];
			subtract(src2, src1, dst, noArray(), DataType<float>::type);
		}
	}
}

//
// Interpolates a scale-space extremum's location and scale to subpixel
// accuracy to form an image feature. Rejects features with low contrast.
// Based on Section 4 of Lowe's paper.
bool adjustLocalExtremaCV( const vector<Mat>& dog_pyr, KeyPoint& kpt, int octv,
	int& layer, int& r, int& c, int nOctaveLayers,
	float contrastThreshold, float edgeThreshold, float sigma )
{
	const float img_scale = 1.f/(255*1.f);
	const float deriv_scale = img_scale*0.5f;
	const float second_deriv_scale = img_scale;
	const float cross_deriv_scale = img_scale*0.25f;

	float xi=0, xr=0, xc=0, contr=0;
	int i = 0;

	for( ; i < SIFT_MAX_INTERP_STEPS; i++ )
	{
		int idx = octv*(nOctaveLayers+2) + layer;
		const Mat& img = dog_pyr[idx];
		const Mat& prev = dog_pyr[idx-1];
		const Mat& next = dog_pyr[idx+1];

		Vec3f dD((img.at<float>(r, c+1) - img.at<float>(r, c-1))*deriv_scale,
			(img.at<float>(r+1, c) - img.at<float>(r-1, c))*deriv_scale,
			(next.at<float>(r, c) - prev.at<float>(r, c))*deriv_scale);

		float v2 = (float)img.at<float>(r, c)*2;
		float dxx = (img.at<float>(r, c+1) + img.at<float>(r, c-1) - v2)*second_deriv_scale;
		float dyy = (img.at<float>(r+1, c) + img.at<float>(r-1, c) - v2)*second_deriv_scale;
		float dss = (next.at<float>(r, c) + prev.at<float>(r, c) - v2)*second_deriv_scale;
		float dxy = (img.at<float>(r+1, c+1) - img.at<float>(r+1, c-1) -
			img.at<float>(r-1, c+1) + img.at<float>(r-1, c-1))*cross_deriv_scale;
		float dxs = (next.at<float>(r, c+1) - next.at<float>(r, c-1) -
			prev.at<float>(r, c+1) + prev.at<float>(r, c-1))*cross_deriv_scale;
		float dys = (next.at<float>(r+1, c) - next.at<float>(r-1, c) -
			prev.at<float>(r+1, c) + prev.at<float>(r-1, c))*cross_deriv_scale;

		Matx33f H(dxx, dxy, dxs,
			dxy, dyy, dys,
			dxs, dys, dss);

		Vec3f X = H.solve(dD, DECOMP_LU);

		xi = -X[2];
		xr = -X[1];
		xc = -X[0];

		if( std::abs(xi) < 0.5f && std::abs(xr) < 0.5f && std::abs(xc) < 0.5f )
			break;

		if( std::abs(xi) > (float)(INT_MAX/3) ||
			std::abs(xr) > (float)(INT_MAX/3) ||
			std::abs(xc) > (float)(INT_MAX/3) )
			return false;

		c += cvRound(xc);
		r += cvRound(xr);
		layer += cvRound(xi);

		if( layer < 1 || layer > nOctaveLayers ||
			c < SIFT_IMG_BORDER || c >= img.cols - SIFT_IMG_BORDER  ||
			r < SIFT_IMG_BORDER || r >= img.rows - SIFT_IMG_BORDER )
			return false;
	}

	// ensure convergence of interpolation
	if( i >= SIFT_MAX_INTERP_STEPS )
		return false;

	{
		int idx = octv*(nOctaveLayers+2) + layer;
		const Mat& img = dog_pyr[idx];
		const Mat& prev = dog_pyr[idx-1];
		const Mat& next = dog_pyr[idx+1];
		Matx31f dD((img.at<float>(r, c+1) - img.at<float>(r, c-1))*deriv_scale,
			(img.at<float>(r+1, c) - img.at<float>(r-1, c))*deriv_scale,
			(next.at<float>(r, c) - prev.at<float>(r, c))*deriv_scale);
		float t = dD.dot(Matx31f(xc, xr, xi));

		contr = img.at<float>(r, c)*img_scale + t * 0.5f;
		if( std::abs( contr ) * nOctaveLayers < contrastThreshold )
			return false;

		// principal curvatures are computed using the trace and det of Hessian
		float v2 = img.at<float>(r, c)*2.f;
		float dxx = (img.at<float>(r, c+1) + img.at<float>(r, c-1) - v2)*second_deriv_scale;
		float dyy = (img.at<float>(r+1, c) + img.at<float>(r-1, c) - v2)*second_deriv_scale;
		float dxy = (img.at<float>(r+1, c+1) - img.at<float>(r+1, c-1) -
			img.at<float>(r-1, c+1) + img.at<float>(r-1, c-1)) * cross_deriv_scale;
		float tr = dxx + dyy;
		float det = dxx * dyy - dxy * dxy;

		if( det <= 0 || tr*tr*edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1)*det )
			return false;
	}

	kpt.pt.x = (c + xc) * (1 << octv);
	kpt.pt.y = (r + xr) * (1 << octv);
	kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5)*255) << 16);
	kpt.size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv)*2;
	kpt.response = std::abs(contr);

	return true;
}


// Computes a gradient orientation histogram at a specified pixel
float calcOrientationHist( const Mat& img, Point pt, int radius,
	float sigma, float* hist, int n )
{
	int i, j, k, len = (radius*2+1)*(radius*2+1);

	float expf_scale = -1.f/(2.f * sigma * sigma);
	AutoBuffer<float> buf(len*4 + n+4);
	float *X = buf, *Y = X + len, *Mag = X, *Ori = Y + len, *W = Ori + len;
	float* temphist = W + len + 2;

	for( i = 0; i < n; i++ )
		temphist[i] = 0.f;

	for( i = -radius, k = 0; i <= radius; i++ )
	{
		int y = pt.y + i;
		if( y <= 0 || y >= img.rows - 1 )
			continue;
		for( j = -radius; j <= radius; j++ )
		{
			int x = pt.x + j;
			if( x <= 0 || x >= img.cols - 1 )
				continue;

			float dx = (float)(img.at<float>(y, x+1) - img.at<float>(y, x-1));
			float dy = (float)(img.at<float>(y-1, x) - img.at<float>(y+1, x));

			X[k] = dx; Y[k] = dy; W[k] = (i*i + j*j)*expf_scale;
			k++;
		}
	}

	len = k;

	// compute gradient values, orientations and the weights over the pixel neighborhood
	exp(W, W, len);
	fastAtan2(Y, X, Ori, len, true);
	magnitude(X, Y, Mag, len);

	for( k = 0; k < len; k++ )
	{
		int bin = cvRound((n/360.f)*Ori[k]);
		if( bin >= n )
			bin -= n;
		if( bin < 0 )
			bin += n;
		temphist[bin] += W[k]*Mag[k];
	}

	// smooth the histogram
	temphist[-1] = temphist[n-1];
	temphist[-2] = temphist[n-2];
	temphist[n] = temphist[0];
	temphist[n+1] = temphist[1];
	for( i = 0; i < n; i++ )
	{
		hist[i] = (temphist[i-2] + temphist[i+2])*(1.f/16.f) +
			(temphist[i-1] + temphist[i+1])*(4.f/16.f) +
			temphist[i]*(6.f/16.f);
	}

	float maxval = hist[0];
	for( i = 1; i < n; i++ )
		maxval = std::max(maxval, hist[i]);

	return maxval;
}

//
// Detects features at extrema in DoG scale space.  Bad features are discarded
// based on contrast and ratio of principal curvatures.
void findScaleSpaceExtremaCV( const vector<Mat>& gauss_pyr, const vector<Mat>& dog_pyr,
	vector<KeyPoint>& keypoints,int nOctaves,int nOctaveLayers,
	float contrastThreshold, float edgeThreshold, float sigma)
{
	int threshold = cvFloor(0.5 * contrastThreshold / nOctaveLayers * 255 );
	const int n = SIFT_ORI_HIST_BINS;
	float hist[n];
	KeyPoint kpt;

	keypoints.clear();

	for( int o = 0; o < nOctaves; o++ )
		for( int i = 1; i <= nOctaveLayers; i++ )
		{
			int idx = o*(nOctaveLayers+2)+i;
			const Mat& img = dog_pyr[idx];
			const Mat& prev = dog_pyr[idx-1];
			const Mat& next = dog_pyr[idx+1];
			int step = (int)img.step1();
			int rows = img.rows, cols = img.cols;

			for( int r = SIFT_IMG_BORDER; r < rows-SIFT_IMG_BORDER; r++)
			{
				const float* currptr = img.ptr<float>(r);
				const float* prevptr = prev.ptr<float>(r);
				const float* nextptr = next.ptr<float>(r);

				for( int c = SIFT_IMG_BORDER; c < cols-SIFT_IMG_BORDER; c++)
				{
					float val = currptr[c];

					// find local extrema with pixel accuracy
					if( std::abs(val) > threshold &&
						((val > 0 && val >= currptr[c-1] && val >= currptr[c+1] &&
						val >= currptr[c-step-1] && val >= currptr[c-step] && val >= currptr[c-step+1] &&
						val >= currptr[c+step-1] && val >= currptr[c+step] && val >= currptr[c+step+1] &&
						val >= nextptr[c] && val >= nextptr[c-1] && val >= nextptr[c+1] &&
						val >= nextptr[c-step-1] && val >= nextptr[c-step] && val >= nextptr[c-step+1] &&
						val >= nextptr[c+step-1] && val >= nextptr[c+step] && val >= nextptr[c+step+1] &&
						val >= prevptr[c] && val >= prevptr[c-1] && val >= prevptr[c+1] &&
						val >= prevptr[c-step-1] && val >= prevptr[c-step] && val >= prevptr[c-step+1] &&
						val >= prevptr[c+step-1] && val >= prevptr[c+step] && val >= prevptr[c+step+1]) ||
						(val < 0 && val <= currptr[c-1] && val <= currptr[c+1] &&
						val <= currptr[c-step-1] && val <= currptr[c-step] && val <= currptr[c-step+1] &&
						val <= currptr[c+step-1] && val <= currptr[c+step] && val <= currptr[c+step+1] &&
						val <= nextptr[c] && val <= nextptr[c-1] && val <= nextptr[c+1] &&
						val <= nextptr[c-step-1] && val <= nextptr[c-step] && val <= nextptr[c-step+1] &&
						val <= nextptr[c+step-1] && val <= nextptr[c+step] && val <= nextptr[c+step+1] &&
						val <= prevptr[c] && val <= prevptr[c-1] && val <= prevptr[c+1] &&
						val <= prevptr[c-step-1] && val <= prevptr[c-step] && val <= prevptr[c-step+1] &&
						val <= prevptr[c+step-1] && val <= prevptr[c+step] && val <= prevptr[c+step+1])))
					{
						int r1 = r, c1 = c, layer = i;
						if( !adjustLocalExtremaCV(dog_pyr, kpt, o, layer, r1, c1,
							nOctaveLayers, (float)contrastThreshold,
							(float)edgeThreshold, (float)sigma) )
							continue;
						/*if( !AdjustLocalExtrema(dog_pyr, kpt, o, layer, r1, c1,
							nOctaveLayers, (float)contrastThreshold,
							(float)edgeThreshold, (float)sigma) )
							continue;*/
						float scl_octv = kpt.size*0.5f/(1 << o);
						float omax = calcOrientationHist(gauss_pyr[o*(nOctaveLayers+3) + layer],
							Point(c1, r1),
							cvRound(SIFT_ORI_RADIUS * scl_octv),
							SIFT_ORI_SIG_FCTR * scl_octv,
							hist, n);
						float mag_thr = (float)(omax * SIFT_ORI_PEAK_RATIO);
						for( int j = 0; j < n; j++ )
						{
							int l = j > 0 ? j - 1 : n - 1;
							int r2 = j < n-1 ? j + 1 : 0;

							if( hist[j] > hist[l]  &&  hist[j] > hist[r2]  &&  hist[j] >= mag_thr )
							{
								float bin = j + 0.5f * (hist[l]-hist[r2]) / (hist[l] - 2*hist[j] + hist[r2]);
								bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
								kpt.angle = 360.f - (float)((360.f/n) * bin);
								if(std::abs(kpt.angle - 360.f) < FLT_EPSILON)
									kpt.angle = 0.f;
								keypoints.push_back(kpt);
							}
						}
					}
				}
			}
		}
}


}