/*
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;



char window_name[30] = "HSV Segmentation";
Mat src,input;

static void onMouse(int event, int x, int y, int f, void*){
	Mat image = src.clone();
	Vec3b rgb = image.at<Vec3b>(y, x);
	int B = rgb.val[0];
	int G = rgb.val[1];
	int R = rgb.val[2];

	Mat HSV;
	Mat RGB = image(Rect(x, y, 1, 1));
	cvtColor(RGB, HSV, CV_BGR2HSV);

	Vec3b hsv = HSV.at<Vec3b>(0, 0);
	int H = hsv.val[0];
	int S = hsv.val[1];
	int V = hsv.val[2];

	char name[30];
	sprintf(name, "B=%d", B);
	putText(image, name, Point(150, 40), FONT_HERSHEY_SIMPLEX, .7, Scalar(0, 255, 0), 2, 8, false);

	sprintf(name, "G=%d", G);
	putText(image, name, Point(150, 80), FONT_HERSHEY_SIMPLEX, .7, Scalar(0, 255, 0), 2, 8, false);

	sprintf(name, "R=%d", R);
	putText(image, name, Point(150, 120), FONT_HERSHEY_SIMPLEX, .7, Scalar(0, 255, 0), 2, 8, false);

	sprintf(name, "H=%d", H);
	putText(image, name, Point(25, 40), FONT_HERSHEY_SIMPLEX, .7, Scalar(0, 255, 0), 2, 8, false);

	sprintf(name, "S=%d", S);
	putText(image, name, Point(25, 80), FONT_HERSHEY_SIMPLEX, .7, Scalar(0, 255, 0), 2, 8, false);

	sprintf(name, "V=%d", V);
	putText(image, name, Point(25, 120), FONT_HERSHEY_SIMPLEX, .7, Scalar(0, 255, 0), 2, 8, false);

	sprintf(name, "X=%d", x);
	putText(image, name, Point(25, 300), FONT_HERSHEY_SIMPLEX, .7, Scalar(0, 0, 255), 2, 8, false);

	sprintf(name, "Y=%d", y);
	putText(image, name, Point(25, 340), FONT_HERSHEY_SIMPLEX, .7, Scalar(0, 0, 255), 2, 8, false);

	//imwrite("hsv.jpg",image);
	imshow(window_name, image);
}



int main(){
//	src = imread("C:/Users/Nitin/Documents/Visual Studio 2013/files/Yawn/3.1.jpg", 1);
	input = imread("C:/Users/Nitin/Documents/Visual Studio 2013/files/Yawn/3.1.jpg", 1);
//	imshow(window_name, src);
	cvtColor(input, input, CV_RGB2HSV);

	double newval[3];
	newval[0] = 0;
	newval[1] = 0;
	newval[2] = 0;
	int p;
//	for (int i = 0; i < 3; i++)
//	{
		for (int y = 0; y < input.rows; y++)
		{
			for (int x = 0; x < input.cols; x++)
			{
				//  Bmean = newval[0]   Gmean = newval[1]     Bmean = newval[2]  
				//newval[i] += input.at<Vec3b>(y, x)[i];
//				cout << (int)input.at<Vec3b>(y, x)[1] << endl;
				p = input.at<Vec3b>(y, x)[0];
				if (p > 100)
					input.at<Vec3b>(y, x)[2] = 255;
				else
					input.at<Vec3b>(y, x)[2] = 0;

			}

		}
//	}
		Mat splt[3];
		split(input, splt);


		imshow("input2", splt[2]);
//	setMouseCallback(window_name, onMouse, 0);
	waitKey(300000000);
}
*/