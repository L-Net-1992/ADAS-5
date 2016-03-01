/*
//#include "stdafx.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

int main(int argc, char** argv)
{
	//create 2 empty windows
	namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
	namedWindow("Smoothed Image", CV_WINDOW_AUTOSIZE);

	// Load an image from file
	//Mat src = imread("C:/Users/Nitin/Documents/Visual Studio 2013/files/FakeEye.jpg", 1);
	Mat src = imread("C:/Users/Nitin/Documents/Visual Studio 2013/files/2.0.jpg", 1);
	//Mat src = imread("C:/Users/Nitin/Documents/Visual Studio 2013/files/eye11.jpg", 1);
	//show the loaded image
	imshow("Original Image", src);
	
	Mat dst;
	Mat sdt;
	Mat tds;
	Mat dts;
	char zBuffer[35];

	for (int i = 1; i < 31; i = i + 2)
	{
		//copy the text to the "zBuffer"
		_snprintf_s(zBuffer, 35, "Kernel Size : %d x %d", i, i);


		//1) Homogeneous blur
		//smooth the image in the "src" and save it to "dst"
		//		blur(src, dst, Size(i, i));

		//put the text in the "zBuffer" to the "dst" image
		//		putText(dst, zBuffer, Point(src.cols / 4, src.rows / 8), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255));



		//2)Gaussian blur
		//smooth the image using Gaussian kernel in the "src" and save it to "dst"
		GaussianBlur(src, dst, Size(i, i), 0, 0);
		imshow("gaussianed", dst);
		cvtColor(dst, sdt, CV_BGR2GRAY);
		imshow("Grayed", sdt);
		threshold(sdt, tds, 100, 255,CV_THRESH_OTSU);
		imshow("Thresholded", tds);

		//put the text in the "zBuffer" to the "dst" image
		putText(dst, zBuffer, Point(src.cols / 4, src.rows / 8), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255), 2);

		//3)Median Blur
		//smooth the image using Median kernel in the "src" and save it to "dst"
		//		medianBlur(src, dst, i);

		//put the text in the "zBuffer" to the "dst" image
		//		putText(dst, zBuffer, Point(src.cols / 4, src.rows / 8), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255), 2);

		//4)Bilateral filter
		//smooth the image using Bilateral filter in the "src" and save it to "dst"
		//		bilateralFilter(src, dst, i, i, i);

		//put the text in the "zBuffer" to the "dst" image
		//		putText(dst, zBuffer, Point(src.cols / 4, src.rows / 8), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255), 2);




		//show the blurred image with the text
		imshow("Smoothed Image", dst);

		//wait for 20 seconds
		int c = waitKey(20000);

		//if the "esc" key is pressed during the wait, return
		if (c == 27)
		{
			return 0;
		}
	}

	//make the "dst" image, black
	dst = Mat::zeros(src.size(), src.type());

	//copy the text to the "zBuffer"
	_snprintf_s(zBuffer, 35, "Press Any Key to Exit");

	//put the text in the "zBuffer" to the "dst" image
	putText(dst, zBuffer, Point(src.cols / 4, src.rows / 2), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255));

	//show the black image with the text
	imshow("Smoothed Image", dst);

	//wait for a key press infinitely
	waitKey(0);

	return 0;

}
*/