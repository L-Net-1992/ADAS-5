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

// Function Headers

void detectAndDisplay(Mat frame);

// Global variables 
//int x1, y1, w1, h1;

IplImage* face = NULL;
IplImage* eye1 = NULL;
IplImage* eye2 = NULL;
IplImage* eyesa = NULL;


String face_cascade_name = "E:/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml";
String eyes_cascade_name = "E:/opencv/sources/data/haarcascades/haarcascade_eye.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

// @function main

Mat rotate(Mat src, double angle)
{
	Mat dst;
	Point2f pt(src.cols / 2., src.rows / 2.);
	Mat r = getRotationMatrix2D(pt, angle, 1.0);
	warpAffine(src, dst, r, Size(src.cols, src.rows));
	return dst;
}


int main(int argc, const char** argv)
{
	CvCapture* capture;
	Mat frame;

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)){ printf("--(!)Error loading\n"); return -1; };
	frame = imread("C:/Users/Nitin/Documents/Visual Studio 2013/files/2.jpg", CV_LOAD_IMAGE_COLOR);
	//-- 2. Read the video stream
	//capture = cvCaptureFromCAM(0);
	//capture = cvCaptureFromCAM("C:/Users/Nitin/Documents/Visual Studio 2013/files/P0011.bmp");
	int k = 1;
	if (1)
	{
		while (k==1)
		{
			frame = imread("C:/Users/Nitin/Documents/Visual Studio 2013/files/2.jpg", CV_LOAD_IMAGE_COLOR);
			face = cvLoadImage("C:/Users/Nitin/Documents/Visual Studio 2013/files/2.jpg");
			eyesa= cvLoadImage("C:/Users/Nitin/Documents/Visual Studio 2013/files/2.jpg");
			//-- 3. Apply the classifier to the frame
			if (!frame.empty())
			{
				detectAndDisplay(frame);
			}
			else
			{
				printf(" --(!) No captured frame -- Break!"); break;
			}

			int c = waitKey(30000);
			if ((char)c == 'c') { break; }
			//k = 2;
		}
		
	}
	return 0;
}

// @function detectAndDisplay

void detectAndDisplay

Mat frame)
{

	int interpolation_type = CV_INTER_LINEAR;

	std::vector<Rect> faces;
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	//imshow("frame2",frame_gray);
	equalizeHist(frame_gray, frame_gray);
	//imshow("frame3",frame_gray);
	//-- Detect faces

	--

	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	//face_cascade.detectMultiScale(frame_gray, faces);
	cout << "faces.size() =  " << faces.size() << endl;
	//x1 = faces[0];
	cout << faces[0].x << " ,, " << faces[0].y << " ,, " << faces[0].width << " ,, " << faces[0].height << endl;
	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		cout << "x = " << faces[i].x << " ;;y = " << faces[i].y;
		cout << endl << "width = " << faces[i].width*0.5 << " ;; height = " << faces[i].height*0.5 << endl;

		Rect roi(faces[i].x, faces[i].y, faces[i].width, faces[i].height);

		cvSetImageROI(face, roi);
		cvResize(face, face, interpolation_type);
		cvShowImage("Face", face);
		cvResetImageROI(face);

		ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		Mat faceROI = frame_gray(faces[i]);

		--
		
		std::vector<Rect> eyes;
		Mat faceROI = imread("C:/Users/Nitin/Documents/Visual Studio 2013/files/2.jpg", CV_LOAD_IMAGE_COLOR);
		//-- In each face, detect eyes
		Mat framenew;
		//	framenew = rotate(faceROI, 20);

		

		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		//cout << "eyes.size() =  " << eyes.size() << endl;
		//		cout << endl << eyes[0].x << " ,, " << eyes[0].y << " ,, " << ((eyes[0].width) + ((eyes[1].x) - (eyes[0].x))) << " ,, " << eyes[1].height <<eyes[1].width<<" ,, "<<eyes[1].x<< endl;
		//Rect roi2(((faces[0].x) + (eyes[0].x)), ((faces[0].y) + (eyes[1].y)), ((eyes[1].width) + (((faces[0].x) + (eyes[1].x)) - ((faces[0].x) + (eyes[0].x)))), eyes[1].height);
		//		cout << ((faces[0].x) + (eyes[0].x)) << " ,,,, " << ((faces[0].y) + (eyes[1].y)) << " ,,,, " << ((eyes[1].width) + (((faces[0].x) + (eyes[1].x)) - ((faces[0].x) + (eyes[0].x)))) << " ,,,, " << eyes[1].height << endl;
		for (size_t j = 0; j < eyes.size(); j++)
		{
			//	Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
			Point center(eyes[j].x + eyes[j].width*0.5, eyes[j].y + eyes[j].height*0.5);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			circle(frame, center, radius, Scalar(255, 0, 0), 4, 8, 0);
		}

		//		cvSetImageROI(eyesa, roi2);
		cvResize(eyesa, eyesa, interpolation_type);
		cvShowImage("eyes", eyesa);
		cvSaveImage("C:/Users/Nitin/Documents/Visual Studio 2013/files/eyesb.bmp", eyesa);
		cvResetImageROI(eyesa);

		//	}
		imshow(window_name, frame);
	}
*/