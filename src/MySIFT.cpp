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
						initialKpt.pt.x=(float)c*(1<<o);
						initialKpt.pt.y=(float)r*(1<<o);
						initialKpt.size=1.f*powf(2.f, (float)layer/ nOctaveLayers)*(1 << o)*2;
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
			//kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
			int oct1 = (kpt.octave & ~255)+ ((kpt.octave + firstOctave) & 255);
			int oct2 = (kpt.octave & ~255)|((kpt.octave + firstOctave) & 255);
			if (oct1!=oct2)
			{
				std::cout<<"位运算错误";
			}
			kpt.octave = oct2;
			kpt.pt *= scale;
			kpt.size *= scale;
		}
}

void UnpackKeypoint(const cv::KeyPoint& kpt,int &octave,int &layer,float &scale)
{
	/*kpt.pt.x = (c + xc) * (1 << octv);
	kpt.pt.y = (r + xr) * (1 << octv);
	--------|--------|--------
	   xi     layer     octv
	kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5)*255) << 16);
	kpt.size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv)*2;
	kpt.response=std::abs(contr);*/

	octave = kpt.octave & 255;
	layer = (kpt.octave >> 8) & 255;
	octave = octave < 128 ? octave : (-128 | octave);
	scale = octave >= 0 ? 1.f/(1 << octave) : (float)(1 << -octave);

	//std::cout<<"octave:"<<octave<<",layer:"<<layer<<",scale:"<<scale<<std::endl;
}

void CaclSIFTDescriptor(const Mat &img,int r,int c,float ori,float scl_octv,int d,int bins,float *dst)
{
	float hist_width=SIFT_DESCR_SCL_FCTR*scl_octv;
	//(3*sigma*(d+1)*(1.414/2))
	int radius=cvRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
	radius = std::min(radius, (int) sqrt((double) img.cols*img.cols + img.rows*img.rows));
	float cos_t=cosf(ori*((float)CV_PI/180.f));
	float sin_t=sinf(ori*((float)CV_PI/180.f));
	int len=(2*radius+1)*(2*radius+1);
	int histlen=(d+2)*(d+2)*(bins+2);
	float exp_scale=-1.f/(2.f*d*0.5f*d*0.5f);
	float bins_per_rad=bins/360.f;
	AutoBuffer<float> buf(len*6+histlen);
	float *X=buf,*Y= X+len,*Mag=X,*Ori=Y+len,*W=Ori+len,*RBin=W+len,*CBin=RBin+len;
	float *tmp_hist=CBin+len;

	int i,j,k;
	int r1,c1;
	float r_rot,c_rot,rbin,cbin;
	for (i=0;i<histlen;i++)
	{
		tmp_hist[i]=0.f;
	}

	for (i=-radius,k=0;i<=radius;i++)
	{
		for (j=-radius;j<=radius;j++)
		{
			//坐标系方向和笛卡尔坐标系一致
			c_rot=j*cos_t-i*sin_t;
			r_rot=j*sin_t+i*cos_t;
			rbin= 1.f/hist_width*r_rot+0.5f*d-0.5f;
			cbin= 1.f/hist_width*c_rot+0.5f*d-0.5f;
			r1=r+i;
			c1=c+j;
			//判断两个条件(如果该点经过旋转后在区域内，且该点不在图像的边缘，就计算这个点的梯度)
			//1.该点经过坐标转换后在(2d+1)*(2d+1)的区域内
			//2.该点不应该在图像边缘，否则无法计算梯度
			if (rbin>-1&&rbin<d&&cbin>-1&&cbin<d&&
				r1>0&&r1<img.rows-1&&c1>0&&c1<img.cols-1)
			{
				float dx=img.at<float>(r1,c1+1)-img.at<float>(r1,c1-1);
				float dy=img.at<float>(r1-1,c1)-img.at<float>(r1+1,c1);
				X[k]=dx;
				Y[k]=dy;
				W[i]=(r_rot*r_rot+c_rot*c_rot)*exp_scale;
				//存储以特征点方向为X轴的(2d+1)*(2d+1)区域块的坐标
				RBin[k]=rbin;
				CBin[k]=cbin;
				k++;
			}
		}
	}

	exp(W,W,k);
	fastAtan2(Y,X,Ori,k,true);
	magnitude(X,Y,Mag,k);

	len=k;
	//计算灰度梯度直方图
	for (i=0;i<len;i++)
	{
		float rbin = RBin[i], cbin = CBin[i];
		float obin = (Ori[i] - ori)*bins_per_rad;
		float mag = Mag[i]*W[i];

		int r0 = cvFloor( rbin );
		int c0 = cvFloor( cbin );
		int o0 = cvFloor( obin );
		rbin -= r0;
		cbin -= c0;
		obin -= o0;

		if( o0 < 0 )
			o0 += bins;
		if( o0 >= bins )
			o0 -= bins;

		// histogram update using tri-linear interpolation
		float v_r1 = mag*rbin, v_r0 = mag - v_r1;
		float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
		float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
		float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
		float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
		float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
		float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

		int idx = ((r0+1)*(d+2) + c0+1)*(bins+2) + o0;
		tmp_hist[idx] += v_rco000;
		tmp_hist[idx+1] += v_rco001;
		tmp_hist[idx+(bins+2)] += v_rco010;
		tmp_hist[idx+(bins+3)] += v_rco011;
		tmp_hist[idx+(d+2)*(bins+2)] += v_rco100;
		tmp_hist[idx+(d+2)*(bins+2)+1] += v_rco101;
		tmp_hist[idx+(d+3)*(bins+2)] += v_rco110;
		tmp_hist[idx+(d+3)*(bins+2)+1] += v_rco111;
	}

	// finalize histogram, since the orientation histograms are circular
	for( i = 0; i < d; i++ )
		for( j = 0; j < d; j++ )
		{
			int idx = ((i+1)*(d+2) + (j+1))*(bins+2);
			tmp_hist[idx] += tmp_hist[idx+bins];
			tmp_hist[idx+1] += tmp_hist[idx+bins+1];
			if (tmp_hist[idx+bins+1]>FLT_EPSILON)
			{
				std::cout<<tmp_hist[idx+bins+1]<<std::endl;
			}
			for( k = 0; k < bins; k++ )
				dst[(i*d + j)*bins + k] = tmp_hist[idx+k];
		}

	// copy histogram to the descriptor,
	// apply hysteresis thresholding
	// and scale the result, so that it can be easily converted
	// to byte array
	float nrm2 = 0;
	len = d*d*bins;
	for( k = 0; k < len; k++ )
		nrm2 += dst[k]*dst[k];
	float thr = std::sqrt(nrm2)*SIFT_DESCR_MAG_THR;
	for( i = 0, nrm2 = 0; i < k; i++ )
	{
		float val = std::min(dst[i], thr);
		dst[i] = val;
		nrm2 += val*val;
	}
	nrm2 = SIFT_INT_DESCR_FCTR/std::max(std::sqrt(nrm2), FLT_EPSILON);

	//为了更好的匹配，转化成uchar类型
	for( k = 0; k < len; k++ )
	{
		dst[k] = saturate_cast<uchar>(dst[k]*nrm2);
	}
}

//计算特征点的描述符
void CalcDescritors(const std::vector<cv::Mat>& guasspyr,const std::vector<cv::KeyPoint>& keypoints,
	cv::Mat descriptors,int firstOctave,int nOcatveLayers)
{
	//由于为了方便在图像上显示，一般特征点之前会根据初始图像有没有扩大来做一些调整
	//所以需要根据调整后的特征点来重新计算组和层
	int d = SIFT_DESCR_WIDTH, bins = SIFT_DESCR_HIST_BINS;
	int descritorSize = d*d*bins;
	descriptors.create((int)keypoints.size(),descritorSize,CV_32FC1);
	int octv,layer,r,c;
	float scale,scl_octv;
	
	for(size_t i=0;i<keypoints.size();i++)
	{
		const KeyPoint &kpt= keypoints.at(i);
		UnpackKeypoint(kpt,octv,layer,scale);
		scl_octv=kpt.size*scale*0.5f;
		r=cvRound(kpt.pt.y*scale);
		c=cvRound(kpt.pt.x*scale);
		const Mat &img=guasspyr.at((octv-firstOctave)*(nOcatveLayers+3)+layer);
		float angle = 360.f - kpt.angle;
		if(std::abs(angle - 360.f) < FLT_EPSILON)
			angle = 0.f;
		CaclSIFTDescriptor(img,r,c,angle,scl_octv,d,bins,descriptors.ptr<float>((int)i));
		//std::cout<<i<<std::endl;
	}

}

void MySIFT(const cv::Mat img,std::vector<cv::KeyPoint>& keypoints,cv::Mat descriptors,
	int nOctaveLayers,double contrastThreshold,float edgeThreshold,float sigma )
{
	Mat dst;
	vector<Mat> gaussianPyr,dogPyr;
	mysift::CreateInitialImage(img,dst,SIFT_IMG_DBL);
	int firstOctave=SIFT_IMG_DBL?-1:0;
	int nOctaves=(int)(log((double)std::min(img.cols,img.rows))/log(2.)-2-firstOctave);
	mysift::BuildGaussianPyramid(dst,gaussianPyr,nOctaves);
	mysift::BuildDoGPyramid(gaussianPyr,dogPyr,nOctaves);
	//showPyr(gaussianPyr,nOctaves,6);
	//writePyr(gaussianPyr,nOctaves,6,"C:\\Users\\Dell\\Desktop\\论文\\影像匹配研究\\sift图像结果\\gaussian");
	//writePyr(dogPyr,nOctaves,5,"C:\\Users\\Dell\\Desktop\\论文\\影像匹配研究\\sift图像结果\\dog_nostretch",false);
	//writePyrValue(dogPyr,nOctaves,5,"C:\\Users\\Dell\\Desktop\\论文\\影像匹配研究\\sift图像结果\\dog_value");
	mysift::FindSpaceScaleExtrema(dogPyr,gaussianPyr,keypoints,nOctaves);
	mysift::AdjustByInitialImage(keypoints,firstOctave);
	mysift::CalcDescritors(gaussianPyr,keypoints,descriptors,firstOctave);


	//显示所有阶段的特征点
	/*vector<KeyPoint> keypoints,interpKeypoints,finalKeypoints;
	mysift::FindSpaceScaleExtrema(dogPyr,keypoints,interpKeypoints,finalKeypoints,nOctaves);
	mysift::AdjustByInitialImage(keypoints,firstOctave);
	mysift::AdjustByInitialImage(interpKeypoints,firstOctave);
	mysift::AdjustByInitialImage(finalKeypoints,firstOctave);*/

	//Mat imgInital,imgInterp,imgFinal;
	//src.copyTo(imgInital);
	//src.copyTo(imgInterp);
	//src.copyTo(imgFinal);
	//
	//DrawCirlcle(imgInital,keypoints);
	//DrawCirlcle(imgInterp,interpKeypoints);
	//DrawCirlcle(imgFinal,finalKeypoints);

	//imshow("initial",imgInital);
	//imshow("interp",imgInterp);
	//imshow("imgFinal",imgFinal);
	//waitKey(0);
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

inline void unpackOctaveCV(const KeyPoint& kpt, int& octave, int& layer, float& scale)
{
	octave = kpt.octave & 255;
	layer = (kpt.octave >> 8) & 255;
	octave = octave < 128 ? octave : (-128 | octave);
	scale = octave >= 0 ? 1.f/(1 << octave) : (float)(1 << -octave);
}

void calcSIFTDescriptorCV( const Mat& img, Point2f ptf, float ori, float scl,
	int d, int n, float* dst )
{
	Point pt(cvRound(ptf.x), cvRound(ptf.y));
	float cos_t = cosf(ori*(float)(CV_PI/180));
	float sin_t = sinf(ori*(float)(CV_PI/180));
	float bins_per_rad = n / 360.f;
	float exp_scale = -1.f/(d * d * 0.5f);
	float hist_width = SIFT_DESCR_SCL_FCTR * scl;
	int radius = cvRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
	// Clip the radius to the diagonal of the image to avoid autobuffer too large exception
	radius = std::min(radius, (int) sqrt((double) img.cols*img.cols + img.rows*img.rows));
	cos_t /= hist_width;
	sin_t /= hist_width;

	int i, j, k, len = (radius*2+1)*(radius*2+1), histlen = (d+2)*(d+2)*(n+2);
	int rows = img.rows, cols = img.cols;

	AutoBuffer<float> buf(len*6 + histlen);
	float *X = buf, *Y = X + len, *Mag = Y, *Ori = Mag + len, *W = Ori + len;
	float *RBin = W + len, *CBin = RBin + len, *hist = CBin + len;

	for( i = 0; i < d+2; i++ )
	{
		for( j = 0; j < d+2; j++ )
			for( k = 0; k < n+2; k++ )
				hist[(i*(d+2) + j)*(n+2) + k] = 0.;
	}

	for( i = -radius, k = 0; i <= radius; i++ )
		for( j = -radius; j <= radius; j++ )
		{
			// Calculate sample's histogram array coords rotated relative to ori.
			// Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
			// r_rot = 1.5) have full weight placed in row 1 after interpolation.
			float c_rot = j * cos_t - i * sin_t;
			float r_rot = j * sin_t + i * cos_t;
			float rbin = r_rot + d/2 - 0.5f;
			float cbin = c_rot + d/2 - 0.5f;
			int r = pt.y + i, c = pt.x + j;

			if( rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
				r > 0 && r < rows - 1 && c > 0 && c < cols - 1 )
			{
				float dx = (float)(img.at<float>(r, c+1) - img.at<float>(r, c-1));
				float dy = (float)(img.at<float>(r-1, c) - img.at<float>(r+1, c));
				X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
				W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;
				k++;
			}
		}

		len = k;
		fastAtan2(Y, X, Ori, len, true);
		magnitude(X, Y, Mag, len);
		exp(W, W, len);

		for( k = 0; k < len; k++ )
		{
			float rbin = RBin[k], cbin = CBin[k];
			float obin = (Ori[k] - ori)*bins_per_rad;
			float mag = Mag[k]*W[k];

			int r0 = cvFloor( rbin );
			int c0 = cvFloor( cbin );
			int o0 = cvFloor( obin );
			rbin -= r0;
			cbin -= c0;
			obin -= o0;

			if( o0 < 0 )
				o0 += n;
			if( o0 >= n )
				o0 -= n;

			// histogram update using tri-linear interpolation
			float v_r1 = mag*rbin, v_r0 = mag - v_r1;
			float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
			float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
			float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
			float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
			float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
			float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

			int idx = ((r0+1)*(d+2) + c0+1)*(n+2) + o0;
			hist[idx] += v_rco000;
			hist[idx+1] += v_rco001;
			hist[idx+(n+2)] += v_rco010;
			hist[idx+(n+3)] += v_rco011;
			hist[idx+(d+2)*(n+2)] += v_rco100;
			hist[idx+(d+2)*(n+2)+1] += v_rco101;
			hist[idx+(d+3)*(n+2)] += v_rco110;
			hist[idx+(d+3)*(n+2)+1] += v_rco111;
		}

		// finalize histogram, since the orientation histograms are circular
		for( i = 0; i < d; i++ )
			for( j = 0; j < d; j++ )
			{
				int idx = ((i+1)*(d+2) + (j+1))*(n+2);
				hist[idx] += hist[idx+n];
				hist[idx+1] += hist[idx+n+1];
				for( k = 0; k < n; k++ )
					dst[(i*d + j)*n + k] = hist[idx+k];
			}
			// copy histogram to the descriptor,
			// apply hysteresis thresholding
			// and scale the result, so that it can be easily converted
			// to byte array
			float nrm2 = 0;
			len = d*d*n;
			for( k = 0; k < len; k++ )
				nrm2 += dst[k]*dst[k];
			float thr = std::sqrt(nrm2)*SIFT_DESCR_MAG_THR;
			for( i = 0, nrm2 = 0; i < k; i++ )
			{
				float val = std::min(dst[i], thr);
				dst[i] = val;
				nrm2 += val*val;
			}
			nrm2 = SIFT_INT_DESCR_FCTR/std::max(std::sqrt(nrm2), FLT_EPSILON);

#if 1
			for( k = 0; k < len; k++ )
			{
				dst[k] = saturate_cast<uchar>(dst[k]*nrm2);
			}
#else
			float nrm1 = 0;
			for( k = 0; k < len; k++ )
			{
				dst[k] *= nrm2;
				nrm1 += dst[k];
			}
			nrm1 = 1.f/std::max(nrm1, FLT_EPSILON);
			for( k = 0; k < len; k++ )
			{
				dst[k] = std::sqrt(dst[k] * nrm1);//saturate_cast<uchar>(std::sqrt(dst[k] * nrm1)*SIFT_INT_DESCR_FCTR);
			}
#endif
}

void calcDescriptorsCV(const vector<Mat>& gpyr, const vector<KeyPoint>& keypoints,
	Mat& descriptors,int nOctaveLayers, int firstOctave )
{
	int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;

	for( size_t i = 0; i < keypoints.size(); i++ )
	{
		KeyPoint kpt = keypoints[i];
		int octave, layer;
		float scale;
		unpackOctaveCV(kpt, octave, layer, scale);
		CV_Assert(octave >= firstOctave && layer <= nOctaveLayers+2);
		float size=kpt.size*scale;
		Point2f ptf(kpt.pt.x*scale, kpt.pt.y*scale);
		const Mat& img = gpyr[(octave - firstOctave)*(nOctaveLayers + 3) + layer];

		float angle = 360.f - kpt.angle;
		if(std::abs(angle - 360.f) < FLT_EPSILON)
			angle = 0.f;
		calcSIFTDescriptorCV(img, ptf, angle, size*0.5f, d, n, descriptors.ptr<float>((int)i));
	
	}
}

inline int descriptorSize()
{
	return SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS;
}

void siftCV(InputArray _image, InputArray _mask,vector<KeyPoint>& keypoints,OutputArray _descriptors,
	int nfeatures,int nOctaveLayers,float contrastThreshold, float edgeThreshold,float sigma )
{
	int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;
	Mat image = _image.getMat(), mask = _mask.getMat();

	if( image.empty() || image.depth() != CV_8U )
		CV_Error( CV_StsBadArg, "image is empty or has incorrect depth (!=CV_8U)" );

	if( !mask.empty() && mask.type() != CV_8UC1 )
		CV_Error( CV_StsBadArg, "mask has incorrect type (!=CV_8UC1)" );

	Mat base;
	createInitialImageCV(image, base,firstOctave < 0, sigma);
	vector<Mat> gpyr, dogpyr;
	int nOctaves = actualNOctaves > 0 ? actualNOctaves : cvRound(log( (double)std::min( base.cols, base.rows ) ) / log(2.) - 2) - firstOctave;

	//double t, tf = getTickFrequency();
	//t = (double)getTickCount();
	buildGaussianPyramidCV(base, gpyr, nOctaves,nOctaveLayers,sigma);
	buildDoGPyramidCV(gpyr, dogpyr,nOctaves,nOctaveLayers);

	//t = (double)getTickCount() - t;
	//printf("pyramid construction time: %g\n", t*1000./tf);


	//t = (double)getTickCount();
	findScaleSpaceExtremaCV(gpyr, dogpyr, keypoints,nOctaves,nOctaveLayers,
		contrastThreshold,edgeThreshold,sigma);
	KeyPointsFilter::removeDuplicated( keypoints );

	if( nfeatures > 0 )
		KeyPointsFilter::retainBest(keypoints, nfeatures);
	//t = (double)getTickCount() - t;
	//printf("keypoint detection time: %g\n", t*1000./tf);

	if( firstOctave < 0 )
		for( size_t i = 0; i < keypoints.size(); i++ )
		{
			KeyPoint& kpt = keypoints[i];
			float scale = 1.f/(float)(1 << -firstOctave);
			kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
			kpt.pt *= scale;
			kpt.size *= scale;
		}

		if( !mask.empty() )
			KeyPointsFilter::runByPixelsMask( keypoints, mask );
	

	if( _descriptors.needed() )
	{
		//t = (double)getTickCount();
		int dsize = descriptorSize();
		_descriptors.create((int)keypoints.size(), dsize, CV_32F);
		Mat descriptors = _descriptors.getMat();

		calcDescriptorsCV(gpyr, keypoints, descriptors, nOctaveLayers, firstOctave);
		//t = (double)getTickCount() - t;
		//printf("descriptor extraction time: %g\n", t*1000./tf);
	}
}

}