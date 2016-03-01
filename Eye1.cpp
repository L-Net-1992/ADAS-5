/*
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include <iostream>
#include <stdio.h>
#include <Windows.h>
#include <Tlhelp32.h>
#include <fstream>
#include <sstream>
#include <time.h>
using namespace std;
using namespace cv;
CvHaarClassifierCascade *cascade;
CvMemStorage *storage;
HWND hWindow;
void mouseLeftClick(const int x, const int y);
void detectFaces(IplImage *img);

int main(int argc, char** argv)
{
	CvCapture *capture;
	IplImage *frame;
	int key = waitKey(10);
	char *filename = "E:/opencv/sources/data/haarcascades/haarcascade_eye.xml";

	cascade = (CvHaarClassifierCascade*)cvLoad(filename, 0, 0, 0);
	storage = cvCreateMemStorage(0);
	capture = cvCaptureFromCAM(0);

	//assert(cascade && storage && capture);

	cvNamedWindow("video", 1);

	while ((char)key != 'q') {
		frame = cvQueryFrame(capture);

		if (!frame) {
			fprintf(stderr, "Cannot query frame!\n");
			break;
		}

		cvFlip(frame, frame, 2);
		frame->origin = 0;

		detectFaces(frame);

		key = cvWaitKey(25);
	}

	cvReleaseCapture(&capture);
	cvDestroyWindow("video");
	cvReleaseHaarClassifierCascade(&cascade);
	cvReleaseMemStorage(&storage);

	return 0;
}

void detectFaces(IplImage *img)
{
	int i;

	CvSeq *faces = cvHaarDetectObjects(img, cascade, storage, 1.15, 3, 0, cvSize(40, 20));

	for (i = 0; i < (faces ? faces->total : 0); i++)
	{
		CvRect *r = (CvRect*)cvGetSeqElem(faces, i);
		cvRectangle(img, cvPoint(r->x, r->y), cvPoint(r->x + r->width, r->y + r->height), CV_RGB(255, 0, 0), 1, 8, 0);
		//hWindow = FindWindow(
			//NULL,
			//"video");
		mouseLeftClick(2.1*r->x, 1.8*r->y);
		Sleep(500);
		//mouse(2.1*r->x, 1.8*r->y);
	}

	cvShowImage("video", img);
}


void mouseLeftClick(const int x, const int y)
{
	// get the window position
	RECT rect;
	GetWindowRect(hWindow, &rect);

	// calculate scale factor
	const double XSCALEFACTOR = 65535 / (GetSystemMetrics(SM_CXSCREEN) - 1);
	const double YSCALEFACTOR = 65535 / (GetSystemMetrics(SM_CYSCREEN) - 1);

	// get current position
	POINT cursorPos;
	GetCursorPos(&cursorPos);
	double cx = cursorPos.x * XSCALEFACTOR;
	double cy = cursorPos.y * YSCALEFACTOR;

	// calculate target position relative to application
	double nx = (x + rect.left) * XSCALEFACTOR;
	double ny = (y + rect.top) * YSCALEFACTOR;

	INPUT Input = { 0 };
	Input.type = INPUT_MOUSE;

	Input.mi.dx = (LONG)nx;
	Input.mi.dy = (LONG)ny;

	// set move cursor directly and left click
	Input.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_LEFTUP;

	SendInput(1, &Input, sizeof(INPUT));

}
*/