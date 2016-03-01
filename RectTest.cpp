/*
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;
IplImage* zoomed = NULL;
IplImage* img = NULL;
IplImage* imgOutput = NULL;
IplImage* im1 = NULL;
IplImage* im2 = NULL;
int main(int argc, const char** argv)
{
	int i = 10;
	img = cvLoadImage("C:/Users/Nitin/Documents/Visual Studio 2013/files/P0010.bmp");
	if (!img)
	{
		cout << "Image cannot be loaded..!!" << endl;
		return -1;
	}
	zoomed = cvCloneImage(img);
	//Rect roi(50, 112, 242, 242);
	Rect roi(108, 177, 138, 54);
	int interpolation_type = CV_INTER_LINEAR;
	cvSetImageROI(img, roi);
	cvResize(img, img, interpolation_type);
	cvShowImage("eyes", img);
	cvResetImageROI(img);
	cvShowImage("Original Image", img);
	waitKey(0);
	destroyAllWindows();
	return 0;
}
*/