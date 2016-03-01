/*
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace cv;
using namespace std;

void help()
{
	cout << "\nThis program demonstrates line finding with the Hough transform.\n"
		"Usage:\n"
		"./houghlines <image_name>, Default is pic1.jpg\n" << endl;
}

int main(int argc, char** argv)
{
	Mat src = imread("C:/Users/Nitin/Documents/Visual Studio 2013/files/3.23.jpg", 0);
	if (src.empty())
	{
		help();
		cout << "cannot open image" << endl;
		return -1;
	}
	Mat dst, cdst;
//	Mat gray;
//	namedWindow("src1", CV_WINDOW_AUTOSIZE);
//	int thresh = 234;
//	cvtColor(~src, gray, CV_BGR2GRAY);
//	createTrackbar("thresh","src1", &thresh, 255);
//	threshold(gray, gray, thresh, 255, THRESH_BINARY);
//	morphologyEx(gray, gray, 3, getStructuringElement(MORPH_ELLIPSE, Size(6, 6)));
//	imshow("grayy", gray);
//	cdst = gray;
	
	Canny(src, dst, 10, 200, 3);
	cvtColor(dst, cdst, CV_GRAY2BGR);

#if 0
	vector<Vec2f> lines;
	HoughLines(dst, lines, 1, CV_PI / 180, 100, 0, 0);

	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA);
	}
#else
	vector<Vec4i> lines;
	HoughLinesP(dst, lines, 1, CV_PI / 180, 50, 50, 10);
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, CV_AA);
	}
#endif
	imshow("source", src);
	imshow("detected lines", cdst);

	waitKey();

	return 0;
}*
*/