/*
#include<opencv/cv.h>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include<opencv/cxcore.h>


int main(int argc, char* argv[]) {

	int c = 1;
	IplImage* img = 0;
	char buffer[1000];
	CvCapture* cv_cap = cvCaptureFromFile("E:/Users/nkotw/Documents/Visual Studio 2013/files/Quadruped.flv");
	//CvCapture* cv_cap = cvCaptureFromCAM(0);
	cvNamedWindow("Video", CV_WINDOW_AUTOSIZE);
	while (1) {

		img = cvQueryFrame(cv_cap);
		cvShowImage("Video", img);
		sprintf(buffer, "E:/Users/nkotw/Documents/Visual Studio 2013/files/Frames/image%u.jpg", c);
		cvSaveImage(buffer, img);
		c++;
		if (cvWaitKey(100) == 27) break;
	}

	cvDestroyWindow("Video");
	return 0;
}
*/