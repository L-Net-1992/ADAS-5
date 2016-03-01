/*
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <fstream>
#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define PI 3.14159265

using namespace std;
using namespace cv;

//              GLOBAL VARIABLES

char rot1[40], rot2[40], yawn[40];
int xmin = 1920, ymin = 1080, xmax = 0, ymax = 0, in2 = 0;
int lab, m;
Point eye1, eye2, rotp(20, 20);
double angle1, angle2;
Mat threshold_output, src2, frame3, imgHSV, imgGray, imgThresholded1, outThresh, frame, dst, frame_gray, frame_edge, inImage, out, frame2, frame4;
vector<Point> cont3, cont, cont1;
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
String face_cascade_name = "C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml";
String eyes_cascade_name = "C:/opencv/sources/data/haarcascades/haarcascade_eye.xml";
String nose_cascade_name = "C:/opencv/sources/data/haarcascades/haarcascade_mcs_nose.xml";
String mouth_cascade_name = "C:/opencv/sources/data/haarcascades/haarcascade_mcs_mouth.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier nose_cascade;
CascadeClassifier mouth_cascade;
RNG rng(12345);

//              FUNCTION

void dump(const Mat &mat, const char* fname)
{
	ofstream filestream;
	filestream.open(fname);
	filestream << mat << endl << endl;
	filestream.close();
}
// Used to avoid noise in the image.

void applyClosing(cv::Mat &binaryImage, int element_radius = 2) {
	int element_type = cv::MORPH_ELLIPSE;

	// The structuring element used for dilation and erosion.
	Mat element = cv::getStructuringElement(element_type,
		Size(2 * element_radius + 1, 2 * element_radius + 1),
		Point(element_radius, element_radius));

	dump(element, "element.data");

	cv::dilate(binaryImage, binaryImage,
		element,
		Point(-1, -1),
		2
		);

	cv::erode(binaryImage, binaryImage,
		element,
		// Position of the anchor within the structuring element.
		// The default value -1,-1 means that the anchor is at the element center
		Point(-1, -1),
		// Iterations: the number of times this operation is applied
		2
		);
}

void applyGaussian(cv::Mat &input, cv::Mat &output) {
	double sigma = 1.5;
	cv::Mat gaussKernel = cv::getGaussianKernel(9, sigma, CV_32F);
	cv::GaussianBlur(input, output, cv::Size(3, 3), 1.5);
}

Mat rotate(Mat src3, double angle)
{
	Mat dst;
	Point2f pt(src3.cols / 2., src3.rows / 2.);
	Mat r = getRotationMatrix2D(pt, angle, 1.0);
	warpAffine(src3, dst, r, Size(src3.cols, src3.rows));
	return dst;
}

Mat skin_segmentation(Mat input)
{
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

Mat detectAndDisplayEye(Mat frame)
{
	Mat pupil;
	lab = 0;

	src2 = skin_segmentation(frame);

	cont = contour(src2);

	frame2 = roi_cont(frame, cont);

	std::vector<Rect> eyes;

	eyes_cascade.detectMultiScale(frame2, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	for (size_t j = 0; j < eyes.size(); j++)
	{

		Point center1(eyes[j].x + eyes[j].width*0.5, eyes[j].y + eyes[j].height*0.5);
		int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
		if (j == 1)
		{
			eye2 = center1;
			circle(frame2, eye2, radius, Scalar(0, 255, 0), 4, 8, 0);
			if (eye2.x < eye1.x)
				swap(eye1, eye2);
		}
		else if (j == 0)
		{
			eye1 = center1;
			circle(frame2, eye1, radius, Scalar(0, 255, 0), 4, 8, 0);
		}

	}

	if (abs(eye1.y - eye2.y) > 10)
	{
		m = 1;

		if ((eye1.y - eye2.y) < 0)
		{
			angle1 = atan((double)(eye2.y - eye1.y) / (double)(eye2.x - eye1.x)) * 180 / PI;
			frame2 = rotate(frame2, angle1);
			frame4 = frame2;
			sprintf(rot1, "Rotated anticlockwise by %f degrees.", (float)angle1);
			putText(frame2, rot1, rotp, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2, 8, 0);
		}
		else
		{
			angle2 = atan((double)(eye1.y - eye2.y) / (double)(eye2.x - eye1.x)) * 180 / PI;
			frame2 = rotate(frame2, 360 - angle2);
			frame4 = frame2;
			sprintf(rot1, "Rotated clockwise by %f degrees.", (float)(360 - angle2));
			putText(frame2, rot1, rotp, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2, 8, 0);
		}
	}
	src2 = skin_segmentation(frame2);
	cont = contour(src2);
	//imshow("frame2", frame2);
	//imshow("src2", src2);
	return frame2;
}

Mat detectAndDisplayPupil(Mat frame)
{
	lab = 2;
	Mat src, gray;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(6, 6));

	src2 = skin_segmentation(frame);

	cont = contour(src2);

	src = roi_cont(frame, cont);
	//imshow("pupil",src);
	//namedWindow("gray", CV_WINDOW_AUTOSIZE);

	src = imread("D:/DRIVER DROWSINESS/images/eye_image4.jpg");
	cvtColor(~src, gray, CV_BGR2GRAY);
	//imshow("src_inv", ~src);
	//createTrackbar("thresh", "gray", &thresh, 255);
	threshold(gray, gray, 234, 255, THRESH_BINARY);
	//imshow("gray", gray);
	morphologyEx(gray, gray, 3, element);
	//imshow("gray2", gray);
	vector<std::vector<cv::Point> > contours;
	findContours(gray.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	drawContours(gray, contours, -1, (255, 255, 255), -1);

	//imshow("gray1", gray);

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
	return src;
	//imshow("src", src);
}

Mat detectAndDisplayFace(Mat frame)
{
	src2 = skin_segmentation(frame);

	cont = contour(src2);

	int interpolation_type = CV_INTER_LINEAR;

	std::vector<Rect> faces;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
	}
	return frame;
}

Mat detectAndDisplayNose(Mat frame)
{
	lab = 1;
	Mat frame2;

	src2 = skin_segmentation(frame);

	cont = contour(src2);

	frame2 = roi_cont(frame, cont);

	int interpolation_type = CV_INTER_LINEAR;

	std::vector<Rect> nose;

	cvtColor(frame2, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	nose_cascade.detectMultiScale(frame_gray, nose, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	for (size_t i = 0; i < nose.size(); i++)
	{
		Point center(nose[i].x + nose[i].width*0.5, nose[i].y + nose[i].height*0.5);
		ellipse(frame2, center, Size(nose[i].width*0.5, nose[i].height*0.5), 0, 0, 360, Scalar(255, 0, 0), 4, 8, 0);
	}
	return frame2;
}

Mat detectAndDisplayMouth(Mat frame)
{
	lab = 1;
	Mat frame2;

	src2 = skin_segmentation(frame);

	cont = contour(src2);

	frame2 = roi_cont(frame, cont);

	int interpolation_type = CV_INTER_LINEAR;

	std::vector<Rect> mouth;

	cvtColor(frame2, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	mouth_cascade.detectMultiScale(frame_gray, mouth, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	for (size_t i = 0; i < mouth.size(); i++)
	{
		Point center(mouth[i].x + mouth[i].width*0.5, mouth[i].y + mouth[i].height*0.5);
		if ((eye1.x + 50 < center.x) && (center.x < eye2.x - 50))
		{
			ellipse(frame2, center, Size(mouth[i].width*0.5, mouth[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
			cout << mouth[i].height << endl;
			if ((double)mouth[i].height > 85)
			{
				sprintf(yawn, "The person is yawning");
				putText(frame2, yawn, rotp, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 2, 8, 0);
			}
		}
	}
	return frame2;
}

void otsu()
{

	src2 = skin_segmentation(frame);

	vector<Point> cont2 = contour(src2);

	inImage = roi_cont(frame, cont2);
	Mat gray, edge, draw;
	cvtColor(inImage, gray, CV_BGR2GRAY);

	Canny(gray, edge, 30, 255, 3);

	edge.convertTo(draw, CV_8U);
	//imshow("draw", draw);

	Mat grayImage;

	--

	CONVERTING HSV TO GRAY
	Mat imHSV;
	cvtColor(im, imHSV, CV_BGR2HSV);
	imshow("HSV", imHSV);

	//cvtColor(imHSV, imHSV, CV_BGR2GRAY);
	Mat hsv_channels[3];
	cv::split(imHSV, hsv_channels);
	imshow("HSV to gray", hsv_channels[2]);
	
	--

	cvtColor(inImage, grayImage, CV_BGR2GRAY);
	//equalizeHist(grayImage, grayImage);
	//imshow("grayImage", grayImage);

	applyGaussian(grayImage, grayImage);

	Mat binaryImage;
	threshold(grayImage, binaryImage
		, 0    // the value doesn't matter for Otsu thresholding
		, 255  // we could choose any non-zero value. 255 (white) makes it easy to see the binary image
		, CV_THRESH_OTSU + CV_THRESH_BINARY_INV);

	//imshow("Threshed", binaryImage);

	applyClosing(binaryImage, 2);

	cont1 = contour(binaryImage);
	//cout << "cont1" << cont1 << endl;
}

Mat intensity_variation(Mat input)
{
	double val[3], newval[3], scale[3], avg;
	newval[0] = 0;
	newval[1] = 0;
	newval[2] = 0;
	for (int i = 0; i < 3; i++)
	{
		for (int y = 0; y < input.rows; y++)
		{
			for (int x = 0; x < input.cols; x++)
			{
				//  Bmean = newval[0]   Gmean = newval[1]     Bmean = newval[2]  
				newval[i] += input.at<Vec3b>(y, x)[i];
				//cout << newval[i] << endl;

			}

		}
		newval[i] /= ((input.rows)*(input.cols));
		//cout << newval[i] << endl;

	}

	avg = (newval[0] + newval[1] + newval[2]) / 3;
	//cout <<"avg = " << avg << endl;

	for (int i = 0; i < 3; i++)
	{
		scale[i] = (avg / newval[i]);
		//cout << "scale = " << scale[i] << endl;
	}
	for (int j = 0; j < 3; j++)
	{
		for (int m = 0; m < input.rows; m++)
		{
			for (int n = 0; n < input.cols; n++)
			{
				input.at<Vec3b>(m, n)[j] *= scale[j];
				//cout << (int)input.at<Vec3b>(m, n)[j] << endl;
			}
		}
	}

	return input;
}

//                MAIN
int main()
{
	if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)){ printf("--(!)Error loading\n"); return -1; };
	if (!nose_cascade.load(nose_cascade_name)){ printf("--(!)Error loading\n"); return -1; };
	if (!mouth_cascade.load(mouth_cascade_name)){ printf("--(!)Error loading\n"); return -1; };
	//frame3 = imread("D:/DRIVER DROWSINESS/images/3.jpg", CV_LOAD_IMAGE_COLOR);
	if (1)
	{
		while (1)
		{
			frame = imread("D:/DRIVER DROWSINESS/images/3.3.jpg", CV_LOAD_IMAGE_COLOR);
			if (!frame.empty())
			{
				otsu();
				//cout << frame.rows << endl;
				//cout << frame.cols << endl;
				//cout << (int)frame.at<Vec3b>(0, 255)[0] << endl;
				//resize(frame, frame,Size(100,100));
				imshow("frame", frame);
				//cout << frame.rows << endl;
				out = intensity_variation(frame);
				imshow("output", out);
				Mat eye = detectAndDisplayEye(out);
				if (m == 1)
				{
					out = intensity_variation(frame4);
					//imshow("frame4",frame4);
					//imshow("new", out);
				}
				Mat nose = detectAndDisplayNose(out);
				Mat mouth = detectAndDisplayMouth(out);

				imshow("face", eye);
				imshow("mouth", mouth);
				Mat pupil = detectAndDisplayPupil(out);
				imshow("pupil", pupil);
				//Mat eye = detectAndDisplayMouth(out);
				//imshow("eye",eye);
				//src2 = skin_segmentation(out);
				//imshow("src2",src2);
				//frame = rotate(frame,30);
				//detectAndDisplayFace(frame);
				//detectAndDisplayEye(frame);
			}
			else
			{
				printf(" --(!) No captured frame -- Break!");
				break;
			}

			int c = waitKey(30000);
			if ((char)c == 'c') { break; }
		}

	}
	return 0;
}







*/