/*
#include <windows.h>
#include <winuser.h>
#include <iostream>
#include <conio.h>
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include <stdio.h>

using namespace cv;
using namespace std;

int xmin = 1920, ymin = 1080, xmax = 0, ymax = 0, in2 = 0;
int lab = 0;

Mat skin_segmentation(Mat input)
{
	Mat imgHSV, imgGray, imgThresholded1, outThresh;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	Mat element1 = getStructuringElement(MORPH_ELLIPSE, Size(4, 4));
	Mat element2 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));

	cvtColor(input, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
	cvtColor(input, imgGray, COLOR_BGR2GRAY); //Convert the captured frame from BGR to Gray

	inRange(imgHSV, Scalar(0, 33, 31), Scalar(15, 171, 195), imgThresholded1);

	threshold(imgThresholded1, outThresh, 100, 255, THRESH_BINARY);

	morphologyEx(imgThresholded1, imgThresholded1, 2, element);

	morphologyEx(imgThresholded1, imgThresholded1, 3, element1);
	return imgThresholded1;
}

vector<Point>  contour(Mat input)
{
	Mat threshold_output;
	int in2;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	threshold(input, threshold_output, 60, 255, THRESH_BINARY);

	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<vector<Point> > contours_poly(contours.size());

	//cout << contours.size() << endl;
	in2 = contours.size();
	int imax = 0;
	double pqPre = 0;
	for (int i = 0; i< contours.size(); i++)
	{
		double pq = contourArea(contours[i], 0);
		if (pq > 10000 && pq < 300000)
		{
			if (pq > pqPre)
			{
				approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
				pqPre = pq;
				imax = i;
			}
		}

	}
	return contours_poly[imax];
}
Mat roi_cont(Mat frame, vector<Point> cont)
{
	for (int i = 0; i < cont.size(); i++)
	{
		if (cont[i].x < xmin)
			xmin = cont[i].x;

		if (cont[i].y < ymin)
			ymin = cont[i].y;

		if (cont[i].x > xmax)
			xmax = cont[i].x;

		if (cont[i].y > ymax)
			ymax = cont[i].y;
	}
	Rect eye_roi;
	if (lab == 0)
	{
		eye_roi.x = (abs(xmin) - 20);
		eye_roi.y = abs(ymin) - 40;
		eye_roi.width = abs(xmax - xmin) + 50;
		eye_roi.height = abs(ymax - ymin);
	}
	else if (lab == 1)
	{
		eye_roi.x = abs(xmin);
		eye_roi.y = abs(ymin) + abs(ymax - ymin) / 2;
		eye_roi.width = abs(xmax - xmin);
		eye_roi.height = (abs(ymax - ymin) / 2);
	}
	else if (lab == 2)
	{
		eye_roi.x = abs(xmin);
		eye_roi.y = abs(ymin) + abs(ymax - ymin) / 4;
		eye_roi.width = 2 * abs(xmax - xmin) / 3;
		eye_roi.height = abs(ymax - ymin) / 4;
	}
	//Rect eye_roi(abs(xmin)-20, abs(ymin) - 40, abs(xmax - xmin) + 50, abs(ymax - ymin) + 40);
	//else
	//Rect  eye_roi(abs(xmin), abs(ymin) + abs(ymax - ymin)/2, abs(xmax - xmin), abs(ymax - ymin) / 2);

	IplImage* frame_clone = new IplImage(frame);
	cvSetImageROI(frame_clone, eye_roi);
	cvResize(frame_clone, frame_clone, CV_INTER_LINEAR);
	Mat frame3(frame_clone);

	return frame3;
}

vector<vector<Point> >  contour1(Mat input)
{
	vector<vector<Point> > cont_m;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat threshold_output;
	int in2;
	threshold(input, threshold_output, 60, 255, THRESH_BINARY);

	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<vector<Point> > contours_poly(contours.size());

	//cout << contours.size() << endl;
	in2 = contours.size();
	int imax = 0;
	double pqPre = 0;
	for (int i = 0; i< contours.size(); i++)
	{
		double pq = contourArea(contours[i], 0);
		cout << "Area = " << pq<<endl;
		if (pq > 1200 && pq < 6000)
		{
				approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
				cont_m.push_back(contours_poly[i]);
		}

	}
	return cont_m;
}

int main(int argc, char** argv)
{
	Mat imgOriginal,src,face,mouth;
	CvCapture* capture;
	capture = cvCaptureFromAVI("C:/Users/Nitin/Documents/Visual Studio 2013/files/Yawn/Yawning.avi");
	while (true)
	{
		imgOriginal = cvQueryFrame(capture);
		src = skin_segmentation(imgOriginal);
		vector<Point> cont2;
		cont2 = contour(src);
		face = roi_cont(imgOriginal, cont2);
		imshow("face", face);
		lab = 1;
		mouth = roi_cont(imgOriginal, cont2);

		//		imgOriginal = imread("C:/Users/Nitin/Documents/Visual Studio 2013/files/Yawn/9.png", 1);
		if (!imread) //if not success, break loop
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}
		Mat imgHSV, imgYCrCb, imgGray;
		cvtColor(mouth, imgHSV, COLOR_BGR2HSV);
		Mat imgThresholded1, imgThresholded2, outThresh;
		inRange(imgHSV, Scalar(0, 0, 0), Scalar(130, 90, 21), imgThresholded1);
		threshold(imgThresholded1, outThresh, 100, 255, THRESH_BINARY);
		erode(imgThresholded1, imgThresholded1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(imgThresholded1, imgThresholded1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(imgThresholded1, imgThresholded1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(imgThresholded1, imgThresholded1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

		imshow("Thresholded Image", imgThresholded1);
		vector<vector<Point>> contm = contour1(imgThresholded1);
		cout << contm.size() << endl;
		waitKey(100);
	}
	system("pause");
	return 0;
}
*/