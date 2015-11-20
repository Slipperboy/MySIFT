#include "MySIFT.h"
#include "AuxUtils.h"
using namespace cv;
using std::vector;

int main(int argc,char** argv)
{
	//TestAngle();
	const char *filename="C:\\Users\\Dell\\Desktop\\assets\\lena.jpg";
	Mat src=imread(filename);
	Mat descriptors1,descriptors2;
	vector<KeyPoint> keypoints1,keypoints2;
	mysift::MySIFT(src,keypoints1,descriptors1);
	mysift::siftCV(src,noArray(),keypoints2,descriptors2,0);
	//CompareKeypoints(keypoints1,keypoints2);
	//CompareDescriptors(descriptors1,descriptors2);
}