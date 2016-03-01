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


int main(int argc, char** argv)
{
	Mat input = imread("C:/Users/Nitin/Documents/Visual Studio 2013/files/test1.png", 1);
	Mat splt[3];
	split(input, splt);
	Mat R = splt[0];
	Mat G = splt[1];
	Mat B = splt[2];
	waitKey(10);
	imshow("R", R);
	waitKey(10);
	imshow("G", G);
	waitKey(10);
	imshow("B", B);
	waitKey(10);
	waitKey(200000000);
	return 0;

}
*/