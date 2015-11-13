#include "MySIFT.h"
#include "AuxUtils.h"
using namespace cv;
using std::vector;

int main(int argc,char** argv)
{
	const char *filename="C:\\Users\\Dell\\Desktop\\assets\\lena.jpg";
	Mat src=imread(filename);
	Mat dst;
	vector<Mat> gaussianPyr,dogPyr;
	mysift::CreateInitialImage(src,dst,mysift::SIFT_IMG_DBL);
	int firstOctave=mysift::SIFT_IMG_DBL?-1:0;
	int nOctaves=log((double)std::min(src.cols,src.rows))/log(2.)-2-firstOctave;
	mysift::BuildGaussianPyramid(dst,gaussianPyr,nOctaves);
	mysift::BuildDoGPyramid(gaussianPyr,dogPyr,nOctaves);
	//showPyr(gaussianPyr,nOctaves,6);
	//writePyr(gaussianPyr,nOctaves,6,"C:\\Users\\Dell\\Desktop\\论文\\影像匹配研究\\sift图像结果\\gaussian");
	//writePyr(dogPyr,nOctaves,5,"C:\\Users\\Dell\\Desktop\\论文\\影像匹配研究\\sift图像结果\\dog_nostretch",false);
}