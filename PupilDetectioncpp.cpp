/*
#include<opencv2\opencv.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<stdio.h>
#include<math.h>
#include<iostream>

using namespace std;
using namespace cv;

int main()
{
	int thresh = 234;
	CvCapture* capture;
	capture = cvCaptureFromFile("C:/Users/Nitin/Documents/Visual Studio 2013/files/Pupil.mp4");
	if (capture)
		while (1)
	{
//		CvCapture* capture;
		Mat src, gray;
		src = cvQueryFrame(capture);
		Mat element = getStructuringElement(MORPH_ELLIPSE, Size(6, 6));
		namedWindow("src", CV_WINDOW_AUTOSIZE);
		namedWindow("src1", CV_WINDOW_AUTOSIZE);

//		src = imread("C:/Users/Nitin/Documents/Visual Studio 2013/files/eyes6O.jpg");
		
		--

		imshow("old", src);
		for (int y = 0; y < src.rows; y++)
		{
		for (int x = 0; x < src.cols; x++)
		{
		for (int c = 0; c < 3; c++)
		{
		src.at<Vec3b>(y, x)[c] += 3;
		}
		}
		}
		imshow("new", src);
		
		--

		if (src.empty())
			return -1;
		cvtColor(~src, gray, CV_BGR2GRAY);
		createTrackbar("thresh", "src1", &thresh, 255);
		threshold(gray, gray, thresh, 255, THRESH_BINARY);
		imshow("gray", gray);

		morphologyEx(gray, gray, 3, element);
		imshow("gray2", gray);
//		imwrite("C:/Users/Nitin/Documents/Visual Studio 2013/files/2.0.jpg", gray);
		vector<std::vector<cv::Point> > contours;
		findContours(gray.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

		drawContours(gray, contours, -1, (255, 255, 255), -1);

		imshow("gray1", gray);

		for (int i = 0; i < contours.size(); i++)
		{
			double area = contourArea(contours[i]);          // Blob area
			Rect rect = boundingRect(contours[i]);           // Bounding box
			int radius = rect.width / 2;                     // Approximate radius

			// Look for round shaped blob
			if (area >= 30 &&
				abs(1 - ((double)rect.width / (double)rect.height)) <= 0.5 &&
				abs(1 - (area / (CV_PI * pow(radius, 2)))) <= 0.3)
			{
				circle(src, Point(rect.x + radius, rect.y + radius), radius, CV_RGB(255, 0, 0), 2);
			}
		}
		imshow("src", src);
		if (char(waitKey(10)) == 27)
			break;
	}
	return 0;
}*/