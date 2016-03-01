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
#define IMAGE_PATH "C:/Users/Nitin/Documents/Visual Studio 2013/files/Face/vdo2.avi"

using namespace std;
using namespace cv;

////       GLOBAL VARIABLES         ////

int xmin = 1920, ymin = 1080, xmax = 0, ymax = 0, in2 = 0, ymin1 = 1080, ymax1 = 0, lab = 0, m1 = 0, m2 = 0;
Point center1_m1(0, 0), center2_m1(0, 0), center1_m2(0, 0), center2_m2(0, 0);
vector<Point> cont, eye_loc, eye_loc_m1;
vector<Mat> eyebrows_roi, eyes_detect;
int eye_loc_m3[4] = { 0, 0, 0, 0 };
int mo, f = 0, f1 = 0, f2 = 0, f3 = 0;
int y_count = 0, alert_count = 0, both_eyes = 0;
int display = 1, display2 = 1;
int yawn[10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, alert[10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
Point eye1, eye2;
Mat threshold_output, imgHSV, imgGray, imgThresholded1, outThresh, temp,source;
vector<vector<Point>> contours, contm;
vector<Vec4i> hierarchy;

////        FUNCTION DECLARATIONS     ////


//             EYEBROWS             //
Mat_<uchar> CRTransform(const Mat& image)
{
	Mat_<Vec3b> _image = image;
	Mat_<uchar> CR_image(image.size());
	for (int i = 0; i < image.rows; ++i)
	{
		for (int j = 0; j < image.cols; ++j)
			CR_image.at<uchar>(i, j) = (255 - _image(i, j)[2]);
	}
	return CR_image;
}

Mat_<uchar> exponentialTransform(const Mat_<uchar>& image)
{
	vector<int> exponential_transform(256, 0);
	for (int i = 0; i < 256; ++i)
		exponential_transform[i] = round(exp((i * log(255)) / 255));

	Mat_<uchar> image_exp(image.size());
	for (int i = 0; i < image.rows; ++i)
	{
		for (int j = 0; j < image.cols; ++j)
			image_exp.at<uchar>(i, j) = exponential_transform[image.at<uchar>(i, j)];
	}
	return image_exp;
}

pair<double, double> returnImageStats(const Mat_<uchar>& image)
{
	double mean = 0.0, std_dev = 0.0;
	int total_pixels = (image.rows * image.cols);

	int intensity_sum = 0;
	for (int i = 0; i < image.rows; ++i)
	{
		for (int j = 0; j < image.cols; ++j)
			intensity_sum += image.at<uchar>(i, j);
	}
	mean = (double)intensity_sum / total_pixels;

	int sum_sq = 0;
	for (int i = 0; i < image.rows; ++i)
	{
		for (int j = 0; j < image.cols; ++j)
			sum_sq += ((image.at<uchar>(i, j) - mean) * (image.at<uchar>(i, j) - mean));
	}
	std_dev = sqrt((double)sum_sq / total_pixels);

	return make_pair(mean, std_dev);
}

Mat_<uchar> binaryThresholding(const Mat_<uchar>& image, const pair<double, double>& stats)
{
	Mat_<uchar> image_binary(image.size());

	double Z = 1.0;
	double threshold = stats.first + (Z * stats.second);
	for (int i = 0; i < image.rows; ++i)
	{
		for (int j = 0; j < image.cols; ++j)
		{
			if (image.at<uchar>(i, j) >= threshold + numeric_limits<double>::epsilon())
				image_binary.at<uchar>(i, j) = 255;
			else
				image_binary.at<uchar>(i, j) = 0;
		}
	}
	return image_binary;
}

int returnLargestContourIndex(vector<vector<Point> > contours)
{
	int max_contour_size = 0;
	int max_contour_idx = -1;
	for (int i = 0; i < contours.size(); ++i)
	{
		if (contours[i].size() > max_contour_size)
		{
			max_contour_size = contours[i].size();
			max_contour_idx = i;
		}
	}
	return max_contour_idx;
}

void Eyebrow_ROI(Mat input)
{
	//	imshow("put", input);
	int q = 0;
	int m;
	if (eye_loc.size() + eye_loc_m1.size() == 8)
		m = 2;
	else if (eye_loc.size() + eye_loc_m1.size() == 6 || eye_loc.size() + eye_loc_m1.size() == 4)
		m = 1;
	for (unsigned int j = 0; j < (eye_loc.size() + eye_loc_m1.size()) / 2; j += m)
	{
		//cout << "here1" << endl;
		if (q < eye_loc_m1.size() / 2)
		{
			if (m == 1 && j == 1)
				j++;
			Rect_<int> e1(eye_loc_m1[j].x, eye_loc_m1[j].y, eye_loc_m1[j + 1].x, eye_loc_m1[j + 1].y);
			int eyebrow_bbox_x = e1.x;
			int eyebrow_bbox_y = (e1.y - e1.height * 0.4);

			int eyebrow_bbox_height = (e1.height * 3) / 5;
			int eyebrow_bbox_width = round((double)e1.width * 0.9);

			eyebrows_roi.push_back(input(Rect(eyebrow_bbox_x, eyebrow_bbox_y, eyebrow_bbox_width, eyebrow_bbox_height)));
			//rectangle(input, Point(eyebrow_bbox_x, eyebrow_bbox_y), Point(eyebrow_bbox_x + eyebrow_bbox_width, eyebrow_bbox_y + eyebrow_bbox_height), Scalar(255, 0, 0), 1, 4);

		}
		if (q < eye_loc.size() / 2)
		{

			if (m == 1 && j == 1)
				j++;

			Rect_<int> e2(eye_loc[j].x, eye_loc[j].y, eye_loc[j + 1].x, eye_loc[j + 1].y);

			int eyebrow_bbox_x = e2.x;
			int eyebrow_bbox_y;
			int eyebrow_bbox_height = (e2.height);
			int eyebrow_bbox_width = round((double)e2.width * 1);
			if (e2.y - e2.height*1.25>0)
			{
				eyebrow_bbox_y = (e2.y - e2.height*1.25);
				eyebrows_roi.push_back(input(Rect(eyebrow_bbox_x, eyebrow_bbox_y, eyebrow_bbox_width, eyebrow_bbox_height)));
			}
			else if ((e2.y - e2.height*0.8) > 0)
			{
				eyebrow_bbox_y = (e2.y - e2.height*0.8);
				eyebrows_roi.push_back(input(Rect(eyebrow_bbox_x, eyebrow_bbox_y, eyebrow_bbox_width, eyebrow_bbox_height)));
			}

		}
		q++;

	}

}


//               GENERAL           //
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

vector<Point>  contour(Mat input)//
{
	vector<Point> null_vec;
	threshold(input, threshold_output, 60, 255, THRESH_BINARY);
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	vector<vector<Point> > contours_poly(contours.size());


	in2 = contours.size();
	if (in2 == 0)
	{
		return null_vec;
	}
	int imax = 0;
	double pqPre = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		double pq = contourArea(contours[i], 0);
		if (pq > 10000 && pq < 300000 && (lab == 0 || lab == 2))
		{
			if (pq > pqPre)
			{
				approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
				pqPre = pq;
				imax = i;
			}
		}
		else if (pq > 2000 && pq < 5000 && lab == 1)
		{
			if (pq > pqPre)
			{
				//cout << "area = " << pq << endl;
				approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
				pqPre = pq;
				imax = i;
			}

		}
	}

	//cout << "Emptiness = " <<contours_poly.empty() << endl;

	return contours_poly[imax];
}

Mat roi_cont(Mat frame, vector<Point> cont)
{
	Rect face_roi;
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
	if (lab == 0)
	{
		face_roi.x = abs(xmin);
		face_roi.y = abs(ymin);
		face_roi.width = abs(xmax - xmin) + 10;
		face_roi.height = (abs(ymax - ymin) + 20);
	}
	else if (lab == 1 || lab == 2)
	{
		face_roi.x = abs(xmin);
		face_roi.y = abs(ymin) + (abs(ymax - ymin) * 0.55);
		face_roi.width = abs(xmax - xmin);
		face_roi.height = (abs(ymax - ymin) *0.5);
		mo = face_roi.y;
	}
	IplImage* frame_clone = new IplImage(frame);
	cvSetImageROI(frame_clone, face_roi);
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
		//cout << "Area = " << pq << endl;
		if (pq > 1500 && pq < 6000)
		{
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			cont_m.push_back(contours_poly[i]);
		}

	}
	return cont_m;
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
				newval[i] += input.at<Vec3b>(y, x)[i];
			}

		}
		newval[i] /= ((input.rows)*(input.cols));
	}

	avg = (newval[0] + newval[1] + newval[2]) / 3;

	for (int i = 0; i < 3; i++)
	{
		scale[i] = (avg / newval[i]);
	}
	for (int j = 0; j < 3; j++)
	{
		for (int m = 0; m < input.rows; m++)
		{
			for (int n = 0; n < input.cols; n++)
			{
				input.at<Vec3b>(m, n)[j] *= scale[j];
			}
		}
	}

	return input;
}

void imgplus(Point imgp, Mat inp, int plus, Point3d color)
{
	int m=0, n=0, o=0;
	m = (int)color.x;
	n = (int)color.y;
	o = (int)color.z;
	temp = Mat::zeros(inp.size(), CV_8UC3);

	int j = imgp.y;
	int t = imgp.x;
	
		for (int a = imgp.x; a < (imgp.x + plus); a++)
		{
			temp.at<Vec3b>(j, a)[0] = m;
			temp.at<Vec3b>(j, a)[1] = n;
			temp.at<Vec3b>(j, a)[2] = o;
		}
		for (int a = imgp.x; a > (imgp.x - plus); a--)
		{
			temp.at<Vec3b>(j, a)[0] = m;
			temp.at<Vec3b>(j, a)[1] = n;
			temp.at<Vec3b>(j, a)[2] = o;
		}
		
		for (int b = imgp.y; b < (imgp.y + plus); b++)
		{
			temp.at<Vec3b>(b, t)[0] = m;
			temp.at<Vec3b>(b, t)[1] = n;
			temp.at<Vec3b>(b, t)[2] = o;
		}
		for (int b = imgp.y; b > (imgp.y - plus); b--)
		{
			temp.at<Vec3b>(b, t)[0] = m;
			temp.at<Vec3b>(b, t)[1] = n;
			temp.at<Vec3b>(b, t)[2] = o;
		}
		//imshow("temp", temp);
}


//          PUPIL DETECTION        //

void pupil_detection(Mat input)
{
	source = input.clone();
	int thresh = 223;
	Mat gray;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(6, 6));


	cvtColor(~source, gray, CV_BGR2GRAY);
	threshold(gray, gray, thresh, 255, THRESH_BINARY);
	morphologyEx(gray, gray, 3, element);
	vector<std::vector<cv::Point> > contours;
	findContours(gray.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	//imshow("gray", gray);
	drawContours(gray, contours, -1, (255, 255, 255), -1);
	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);          // Blob area
		Rect rect = boundingRect(contours[i]);           // Bounding box
		int radius = rect.width / 2;                     // Approximate radius

		// Look for round shaped blob
		if (area >= 30 &&
			abs(1 - ((double)rect.width / (double)rect.height)) <= 0.5 && area <= 10000)
		{
			Point3d color(0, 0, 255);
			Point c(rect.x + radius, rect.y + radius);
			imgplus(c, source,5,color);				
			source += temp;
			//circle(src, Point(rect.x + radius, rect.y + radius), radius, CV_RGB(255, 0, 0), 2);
		}
	}
}


//               EYES DETECTION     //

vector<Point> eye_M1(Mat eye_mat)
{
	vector<Point> eye_loc_haar, null_eye_M1;
	std::vector<Rect> eyes;

	String eyes_cascade_name = "C:/opencv/sources/data/haarcascades/haarcascade_eye.xml";
	CascadeClassifier eyes_cascade;

	eye_mat = eye_mat(Rect(0, 0, eye_mat.cols, eye_mat.rows * 0.55));
	//eye_mat = intensity_variation(eye_mat);
	eyes_cascade.load(eyes_cascade_name);
	eyes_cascade.detectMultiScale(eye_mat, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	if (eyes.size() == 0)
	{
		return null_eye_M1;
	}
	for (size_t j = 0; j < eyes.size(); j++)
	{

		Point center1(eyes[j].x + eyes[j].width*0.5, eyes[j].y + eyes[j].height*0.5);
		int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
		if (j == 1 && radius<45 && radius>25)
		{
			eye2 = center1;
			Point3d color(0, 255, 255);
			imgplus(eye2, eye_mat, 5, color);
			eye_mat += temp;
			circle(eye_mat, eye2, radius, Scalar(0, 255, 0), 4, 8, 0);

		}
		else if (j == 0 && radius<45 && radius>25)
		{

			eye1 = center1;
			Point3d color(0, 255, 255);
			imgplus(eye1, eye_mat, 5, color);
			eye_mat += temp;
			circle(eye_mat, eye1, radius, Scalar(0, 255, 0), 4, 8, 0);
		}
		//cout << "r = " << radius << endl;
		if (j < 2 && radius<45 && radius>25)
		{
			eye_loc_haar.push_back(Point(eyes[j].x, eyes[j].y));
			eye_loc_haar.push_back(Point(eyes[j].width, eyes[j].height));

		}
	}
	imshow("eye_mat", eye_mat);
	//waitKey(10);
	return eye_loc_haar;
}

vector<Point> eye_M2(Mat input)
{
	vector<vector<Point>> contours;
	vector<Point> null_eye, eye_loc_m2;
	vector<Vec4i> hierarchy;
	int k = 0;
	int *arr = new int[k];
	int *arr2 = new int[k];
	Mat input2, input3, input4;

	Mat input1 = input.clone();
	Rect roi(0, 0, input.cols, input.rows*0.6);

	input = input(roi);
	//input = intensity_variation(input);
	Mat input5 = input.clone();
	cvtColor(input, input, CV_BGR2YCrCb);
	for (int i = 0; i < 3; i++)
	{
		for (int y = 0; y < input.rows; y++)
		{
			for (int x = 0; x < input.cols; x++)
			{
				input.at<Vec3b>(y, x)[i] = ((input.at<Vec3b>(y, x)[2] ^ 2) + ((255 - input.at<Vec3b>(y, x)[1]) ^ 2) + (input.at<Vec3b>(y, x)[2] / input.at<Vec3b>(y, x)[1])) / 3;
			}

		}
	}
	Mat splt[3];
	split(input, splt);
	Mat eyec;
	eyec = splt[0];
	equalizeHist(eyec, eyec);

	erode(input5, input2, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)));
	dilate(input5, input3, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

	Mat eyei = eyec.clone();
	for (int y = 0; y < eyec.rows; y++)
	{
		for (int x = 0; x < eyec.cols; x++)
		{

			eyei.at<uchar>(y, x) = input3.at<uchar>(y, x) / (1 + input2.at<uchar>(y, x));
		}

	}

	for (int y = 0; y < input.rows; y++)
	{
		for (int x = 0; x < input.cols; x++)
		{
			eyei.at<uchar>(y, x) *= eyec.at<uchar>(y, x);
		}

	}

	threshold(eyei, eyei, 200, 255, THRESH_BINARY);
	findContours(eyei, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	int j = 0;
	for (int i = 0; i < contours.size(); i++)
	{

		double pq = contourArea(contours[i], 0);
		if (pq > 300 && pq < 2500)
		{
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = boundingRect(Mat(contours_poly[i]));
			int width = boundRect[i].br().x - boundRect[i].tl().x;
			int height = boundRect[i].br().y - boundRect[i].tl().y;

			if (width - height < 50 && width - height>-3)
			{
				k = j;
				arr[j] = boundRect[i].tl().y;
				arr2[j] = i;
				j++;
			}
		}

	}

	int max1 = 0, max0 = 0, max2 = 0, eye1y = 0, eye2y = 0;

	for (int k = 0; k < j; k++)
	{
		if (arr[k]>max0)
		{
			max1 = k;
			max0 = arr[k];

		}
	}
	max0 = 0;
	eye1y = arr2[max1];

	arr[max1] = 0;

	for (int k = 0; k < j; k++)
	{
		if (arr[k]>max0)
		{
			max0 = arr[k];
			max2 = k;
		}
	}

	eye2y = arr2[max2];
	double angle1 = 0;
	if (j > 1)
	{
		angle1 = atan((double)(boundRect[eye2y].tl().y - boundRect[eye1y].tl().y) / (double)(boundRect[eye2y].tl().x - boundRect[eye1y].tl().x)) * 180 / PI;
		if (angle1 > 30)
			return null_eye;
	}
	if (j == 0)
	{
		return null_eye;
	}
	int width2 = boundRect[eye2y].br().x - boundRect[eye2y].tl().x;
	int height2 = boundRect[eye2y].br().y - boundRect[eye2y].tl().y;
	Point3d color1(255, 0, 0);
	if (abs(angle1) < 30 && j > 1)
	{
		int width1 = boundRect[eye1y].br().x - boundRect[eye1y].tl().x;
		int height1 = boundRect[eye1y].br().y - boundRect[eye1y].tl().y;
		
		Point c1(boundRect[eye1y].tl().x + (width1 / 2), boundRect[eye1y].tl().y + (height1 / 2));
		imgplus(c1, input1, 5, color1);
		input1 += temp;

		Point c2(boundRect[eye2y].tl().x + (width2 / 2), boundRect[eye2y].tl().y + (height2 / 2));
		imgplus(c2, input1, 5, color1);
		input1 += temp;
		circle(input1, c1, width1/2, Scalar(0, 255, 0), 4, 8, 0);
		circle(input1, c2, width2/2, Scalar(0, 255, 0), 4, 8, 0);
		//rectangle(input1, boundRect[eye1y].tl(), boundRect[eye1y].br(), (0, 0, 255), 2, 8, 0);
		//rectangle(input1, boundRect[eye2y].tl(), boundRect[eye2y].br(), (0, 0, 255), 2, 8, 0);
		eye_loc_m2.push_back(boundRect[eye1y].tl());
		eye_loc_m2.push_back(Point(width1, height1));
		eye_loc_m2.push_back(boundRect[eye2y].tl());
		eye_loc_m2.push_back(Point(width2, height2));

	}

	else if (j>0 && j == 1)
	{
		Point c2(boundRect[eye2y].tl().x + (width2 / 2), boundRect[eye2y].tl().y + (height2 / 2));
		imgplus(c2, input1, 5, color1);
		input1 += temp;
		circle(input1, c2, width2/2, Scalar(0, 255, 0), 4, 8, 0);
		//rectangle(input1, boundRect[eye2y].tl(), boundRect[eye2y].br(), (0, 0, 255), 2, 8, 0);
		eye_loc_m2.push_back(boundRect[eye2y].tl());
		eye_loc_m2.push_back(Point(width2, height2));
	}
	else
	{
		return null_eye;
	}
	imshow("input1", input1);

	waitKey(20);
	return eye_loc_m2;
}

void eye_extract(Mat input)
{
	
	Rect eye_roi;
	if (m1 == 1)
	{
		if (eye_loc_m1.size() == 2)
		{
			eye_roi.x = eye_loc_m1[0].x-5;
			eye_roi.y = eye_loc_m1[0].y-5;
			eye_roi.width = eye_loc_m1[1].x+5;
			eye_roi.height = eye_loc_m1[1].y+5;
			IplImage* eye_clone = new IplImage(input);
			cvSetImageROI(eye_clone, eye_roi);
			cvResize(eye_clone, eye_clone, CV_INTER_LINEAR);
			Mat eye(eye_clone);
			eyes_detect.push_back(eye);
		}
		else if (eye_loc_m1.size() == 4)
		{
			eye_roi.x = eye_loc_m1[0].x-5;
			eye_roi.y = eye_loc_m1[0].y-5;
			eye_roi.width = eye_loc_m1[1].x+5;
			eye_roi.height = eye_loc_m1[1].y+5;
			IplImage* eye_clone = new IplImage(input);
			cvSetImageROI(eye_clone, eye_roi);
			cvResize(eye_clone, eye_clone, CV_INTER_LINEAR);
			Mat eye(eye_clone);

			eyes_detect.push_back(eye);

			eye_roi.x = eye_loc_m1[2].x-5;
			eye_roi.y = eye_loc_m1[2].y-5;
			eye_roi.width = eye_loc_m1[3].x+5;
			eye_roi.height = eye_loc_m1[3].y+5;
			IplImage* eye_clone1 = new IplImage(input);
			cvSetImageROI(eye_clone1, eye_roi);
			cvResize(eye_clone1, eye_clone1, CV_INTER_LINEAR);
			Mat eye1(eye_clone1);

			eyes_detect.push_back(eye1);
		}
	}
	else if (m2 == 1)
	{
		if (eye_loc.size() == 2)
		{
			eye_roi.x = eye_loc[0].x-5;
			eye_roi.y = eye_loc[0].y-5;
			eye_roi.width = eye_loc[1].x+5;
			eye_roi.height = eye_loc[1].y+5;
			IplImage* eye_clone2 = new IplImage(input);
			cvSetImageROI(eye_clone2, eye_roi);
			cvResize(eye_clone2, eye_clone2, CV_INTER_LINEAR);
			Mat eye2(eye_clone2);
			eyes_detect.push_back(eye2);
		}
		else if (eye_loc.size() == 4)
		{
			eye_roi.x = eye_loc[0].x-5;
			eye_roi.y = eye_loc[0].y-5;
			eye_roi.width = eye_loc[1].x+5;
			eye_roi.height = eye_loc[1].y+5;
			IplImage* eye_clone2 = new IplImage(input);
			cvSetImageROI(eye_clone2, eye_roi);
			cvResize(eye_clone2, eye_clone2, CV_INTER_LINEAR);
			Mat eye2(eye_clone2);

			eyes_detect.push_back(eye2);

			eye_roi.x = eye_loc[2].x-5;
			eye_roi.y = eye_loc[2].y-5;
			eye_roi.width = eye_loc[3].x+5;
			eye_roi.height = eye_loc[3].y+5;
			IplImage* eye_clone3 = new IplImage(input);
			cvSetImageROI(eye_clone3, eye_roi);
			cvResize(eye_clone3, eye_clone3, CV_INTER_LINEAR);
			Mat eye3(eye_clone3);

			eyes_detect.push_back(eye3);
		}
	}
	else if (both_eyes == 1 && m1 == 0 && m2 == 0)
	{
		eye_roi.x = eye_loc_m1[0].x - 5;
		eye_roi.y = eye_loc_m1[0].y - 5;
		eye_roi.width = eye_loc_m1[1].x + 5;
		eye_roi.height = eye_loc_m1[1].y + 5;
		IplImage* eye_clone_both = new IplImage(input);
		cvSetImageROI(eye_clone_both, eye_roi);
		cvResize(eye_clone_both, eye_clone_both, CV_INTER_LINEAR);
		Mat eye5(eye_clone_both);

		eyes_detect.push_back(eye5);

		eye_roi.x = eye_loc_m1[2].x - 5;
		eye_roi.y = eye_loc_m1[2].y - 5;
		eye_roi.width = eye_loc_m1[3].x + 5;
		eye_roi.height = eye_loc_m1[3].y + 5;
		IplImage* eye_clone_both1 = new IplImage(input);
		cvSetImageROI(eye_clone_both1, eye_roi);
		cvResize(eye_clone_both1, eye_clone_both1, CV_INTER_LINEAR);
		Mat eye6(eye_clone_both1);

		eyes_detect.push_back(eye6);
	}
}

void eye_M3(Mat input)
{
	int a[4] = {0,0,0,0}, i = 0;
	Mat input1 = input.clone();
	Mat input2 = input.clone();
	//input1 = intensity_variation(input1);
	Eyebrow_ROI(input1);
	if (eyebrows_roi.empty() != 1)
	{
		for (int d = 0; d < eyebrows_roi.size(); d++)
		{
			ymin1 = 1080;
			ymax1 = 0;

			Mat_<uchar> image_exp = exponentialTransform(CRTransform(eyebrows_roi[d]));
			Mat_<uchar> image_binary = binaryThresholding(image_exp, returnImageStats(image_exp));
			morphologyEx(image_binary, image_binary, 3, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

			// A clone image is required because findContours() modifies the input image
			Mat image_binary_clone = image_binary.clone();
			vector<vector<Point> > contours;
			findContours(image_binary_clone, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

			// Initialize blank image (for drawing contours)
			Mat_<uchar> image_contour(image_binary.size());
			for (int i = 0; i < image_contour.rows; ++i)
			{
				for (int j = 0; j < image_contour.cols; ++j)
					image_contour.at<uchar>(i, j) = 0;
			}


			// Draw largest contour on the blank image
			int largest_contour_idx = returnLargestContourIndex(contours);

			if (largest_contour_idx != -1)
			{
				for (int i = 0; i < contours[largest_contour_idx].size(); i++)
				{
					if (contours[largest_contour_idx][i].y < ymin1)
					{
						ymin1 = contours[largest_contour_idx][i].y;
					}
					if (contours[largest_contour_idx][i].y > ymax1)
						ymax1 = contours[largest_contour_idx][i].y;
				}

				for (int i = 0; i < contours[largest_contour_idx].size(); ++i)
				{
					Point_<int> pt = contours[largest_contour_idx][i];
					image_contour.at<uchar>(pt.y, pt.x) = 255;
				}

				if (contourArea(contours[largest_contour_idx]) < (eyebrows_roi[d].rows*eyebrows_roi[d].cols / 2) && contourArea(contours[largest_contour_idx]) > (eyebrows_roi[d].rows*eyebrows_roi[d].cols / 20) && (ymax1 - ymin1 < image_contour.rows*0.6))
				{
					eye_loc_m3[d] = 1;
					a[i] = d;
					i++;
					
					m1 = 1;
					eye_extract(input2);
					m1 = 0;
					m2 = 1;
					eye_extract(input2);
					m2 = 0;
					char n[40];
					for (int u = 0; u < 2; u++)
					{
						sprintf(n,"pupil%d",u);
						pupil_detection(eyes_detect[u]);
						imshow(n, source);
					}
				}
				else
				{ 
					eye_loc_m3[d] = 0;
				   // eyebrows_roi.erase(eyebrows_roi.begin()+d);
				}
			}
			else
			{

				eye_loc_m3[d] = 0;
				//eyebrows_roi.erase(eyebrows_roi.begin() + d);
			}
			//cout << "eye_loc_m3 = " << eye_loc_m3[d] << endl;
		}
	}
	else
	{
		cout << " NO EYES FOUND " << endl;
	}

}

void eye_detect(Mat input)
{
	Mat frame;
	frame = roi_cont(input, cont);
	Mat frame_clone = frame.clone();
	Mat frame_clone2 = frame.clone();
	Mat frame_clone3 = frame.clone();

	//        METHOD 1 EYE_LOC_M1            //

	eye_loc = eye_M2(frame);

	if (eye_loc.size() == 2 && eye_loc[0].x != 0 && eye_loc[1].x != 0)
	{
		center1_m2 = Point(eye_loc[0].x + eye_loc[1].x*0.5, eye_loc[0].y + eye_loc[1].y*0.5);
	}
	else if (eye_loc.size() == 4)
	{
		center1_m2 = Point(eye_loc[0].x + eye_loc[1].x*0.5, eye_loc[0].y + eye_loc[1].y*0.5);
		center2_m2 = Point(eye_loc[2].x + eye_loc[3].x*0.5, eye_loc[2].y + eye_loc[3].y*0.5);
		if (center2_m2.x < center1_m2.x)
			swap(center2_m2, center1_m2);	
	}
	else
	{ }

	//          METHOD 2 EYE_LOC             //

	eye_loc_m1 = eye_M1(frame_clone);
	
	if (eye_loc_m1.size() == 4)
	{
		center1_m1 = Point(eye_loc_m1[0].x + eye_loc_m1[1].x*0.5, eye_loc_m1[0].y + eye_loc_m1[1].y*0.5);
		center2_m1 = Point(eye_loc_m1[2].x + eye_loc_m1[3].x*0.5, eye_loc_m1[2].y + eye_loc_m1[3].y*0.5);
		if (center2_m1.x < center1_m1.x)
			swap(center2_m1, center1_m1);
	}
	else if (eye_loc_m1.size() == 2)
	{
		center1_m1 = Point(eye_loc_m1[0].x + eye_loc_m1[1].x*0.5, eye_loc_m1[0].y + eye_loc_m1[1].y*0.5);
	}
	else
	{ }

	//          EYES CONDITIONS           //     	
	//      eye_loc_m1 --- HAAR 
	//      eye_loc    --- Intensity Variation
	int m = 0;
	if (eye_loc.size() + eye_loc_m1.size() == 8)
	{
		if (abs(center1_m1.x - center1_m2.x) < 26 && abs(center1_m1.y - center1_m2.y) < 15)
		{

			if (abs(center2_m1.x - center2_m2.x) < 26 && abs(center2_m1.y - center2_m2.y) < 15)
			{
				// extracting HAAR eyes and deleting M2 detected eyes

            	eye_loc.erase(eye_loc.begin(), eye_loc.end());
				both_eyes = 1;
				eye_extract(frame_clone2);
				char n[40];
				for (int i = 0; i < eyes_detect.size();i++)
				{ 
					pupil_detection(eyes_detect[i]);
					sprintf(n,"eye%d", i+1);
					imshow(n, source);
				}
				both_eyes = 0;
			}
			else
			{
				eye_loc.erase(eye_loc.begin(), eye_loc.end() - 2);
				eye_M3(frame_clone2);
			}
		}
		else
		{
			eye_loc.erase(eye_loc.begin() + 2, eye_loc.end());
			eye_M3(frame_clone2);
		}
		m = 0;
	}
	else if (eye_loc.size() + eye_loc_m1.size() == 6)
	{
		if (eye_loc_m1.size() == 4)
		{
			if (abs(center1_m1.x - center1_m2.x) < 22 && abs(center1_m1.y - center1_m2.y) < 15)
			{
				eye_loc.erase(eye_loc.begin(), eye_loc.end());
				eye_M3(frame_clone2);
			}
			else if (abs(center2_m1.x - center1_m2.x) < 22 && abs(center2_m1.y - center1_m2.y) < 15)
			{

				eye_loc.erase(eye_loc.begin(), eye_loc.end());
				eye_M3(frame_clone2);
			}
			else
			{
				eye_M3(frame_clone2);
			}
		}
		else if (eye_loc.size() == 4)
		{

			if (abs(center1_m2.x - center1_m1.x) < 22 && abs(center1_m2.y - center1_m1.y) < 15)
			{

				eye_loc_m1.erase(eye_loc_m1.begin(), eye_loc_m1.end());
				eye_M3(frame_clone2);
			}
			else if (abs(center2_m2.x - center1_m1.x) < 22 && abs(center2_m2.y - center1_m1.y) < 15)
			{

				eye_loc_m1.erase(eye_loc_m1.begin(), eye_loc_m1.end());
			    eye_M3(frame_clone2);
			}
			else
			{
				eye_M3(frame_clone2);
			}
		}
		m = 0;
	}
	else if (eye_loc.size() + eye_loc_m1.size() == 4)
	{
		if (eye_loc.size() == 0 || eye_loc_m1.size() == 0)
		{
			eye_M3(frame_clone2);
			m = 0;
		}
		else if (eye_loc.size() == 2 && eye_loc_m1.size() == 2 && abs(eye_loc[1].x - eye_loc_m1[1].x) > 30 && abs(eye_loc[1].y - eye_loc_m1[1].y) < 40)
		{
			eye_M3(frame_clone2);
			m = 0;
		}
		else
		{
			//           ALERT        //
			alert_count++;
			m = 1;
		}
	}
	else
	{
		//              ALERT       //
		alert_count++;
		m = 1;
	}
	if (alert_count < 10)
		alert[alert_count] = m;
	else
	{
		int g = 0;
		while (g < 9)
		{
			alert[g] = alert[g + 1];
			g++;
		}
		alert[9] = m;
	}
	if (alert_count < 12)
		alert_count++;
	int alert_sum = 0;

	for (int j = 0; j < 10; j++)
		alert_sum += alert[j];
	if (alert_sum > 8 && display2 == 1)
	{
		cout << "ALERT!!!!!!!!!!" << endl;
		waitKey(1000);
		display2 = 0;
	}
	if (f2 > 10)
	{
		for (int j = 0; j < 10; j++)
			alert[j] = 0;
		display2 = 1;
		f2 = 0;
	}
}


//           MOUTH DETECTION        //

void mouth_M1(Mat imgOriginal)
{
	Mat face, mouth, src;
	vector<Point> cont2;
	src = skin_segmentation(imgOriginal);
	cont2 = contour(src);
	face = roi_cont(imgOriginal, cont2);

	waitKey(10);
	lab = 1;
	mouth = roi_cont(imgOriginal, cont2);
	imshow("mouth", mouth);
	waitKey(10);

	Mat imgHSV, imgYCrCb, imgGray;
	cvtColor(mouth, imgHSV, COLOR_BGR2HSV);
	Mat imgThresholded1, imgThresholded2, outThresh;
	inRange(imgHSV, Scalar(0, 0, 0), Scalar(130, 90, 21), imgThresholded1);
	threshold(imgThresholded1, outThresh, 100, 255, THRESH_BINARY);
	erode(imgThresholded1, imgThresholded1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	dilate(imgThresholded1, imgThresholded1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	dilate(imgThresholded1, imgThresholded1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	erode(imgThresholded1, imgThresholded1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

	//imshow("Thresholded Image", imgThresholded1);
	contm = contour1(imgThresholded1);
	//cout << contm.size() << endl;
	waitKey(10);
}

void mouth_detect(Mat input)
{
	mouth_M1(input);
	waitKey(10);
	xmin = 1920;
	ymin = 1080;
	xmax = 0;
	ymax = 0;
	for (int i = 0; i < contm.size(); i++)
	{
		for (int j = 0; j < contm[i].size(); j++)
		{
			if (contm[i][j].x < xmin)
				xmin = contm[i][j].x;

			if (contm[i][j].y < ymin)
				ymin = contm[i][j].y;

			if (contm[i][j].x > xmax)
				xmax = contm[i][j].x;

			if (contm[i][j].y > ymax)
				ymax = contm[i][j].y;

		}
		//cout << xmax << "," << xmin << endl;
		Point ctr(xmin + (xmax - xmin) / 2, ymin + (ymax - ymin) / 2 + mo);
		Point c1, c2;
		//cout << "ctr = " << Point(ctr.x,ctr.y+mo) << endl;
		//cout << "eye = " << Point(eye_loc_m1[0].x + (eye_loc_m1[1].x / 2), eye_loc_m1[0].y + (eye_loc_m1[1].y / 2)) << endl;
		if (eye_loc.size() == 4)
		{
			c1 = Point(eye_loc[0].x + (eye_loc[1].x / 2), eye_loc[0].y + (eye_loc[1].y / 2));
			c2 = Point(eye_loc[2].x + (eye_loc[3].x / 2), eye_loc[2].y + (eye_loc[3].y / 2));
		}
		else if (eye_loc_m1.size() == 4)
		{
			c1 = Point(eye_loc_m1[0].x + (eye_loc_m1[1].x / 2), eye_loc_m1[0].y + (eye_loc_m1[1].y / 2));
			c2 = Point(eye_loc_m1[2].x + (eye_loc_m1[3].x / 2), eye_loc_m1[2].y + (eye_loc_m1[3].y / 2));
		}
		else if (eye_loc.size() == 2 && eye_loc_m1.size() == 2)
		{
			c1 = Point(eye_loc[0].x + (eye_loc[1].x / 2), eye_loc[0].y + (eye_loc[1].y / 2));
			c2 = Point(eye_loc_m1[0].x + (eye_loc_m1[1].x / 2), eye_loc_m1[0].y + (eye_loc_m1[1].y / 2));
		}
		if (abs(ctr.x - ((c1.x + c2.x) / 2)<15))
		{
			//      YAWNING          //

			int n;
			float a;
			a = (float)(ctr.y - c1.y) / (float)(c1.x - c2.x);
			if (a < 2.5 && a > 1.5)
			{
				n = 1;
			}
			else
			{
				n = 0;
			}
			if (y_count < 10)
				yawn[y_count] = n;
			else
			{
				int h = 0;
				while (h < 9)
				{
					yawn[h] = yawn[h + 1];
					h++;
				}
				yawn[9] = n;
			}
			if (y_count < 12)
				y_count++;
			int yawn_sum = 0;

			for (int j = 0; j < 10; j++)
				yawn_sum += yawn[j];
			if (yawn_sum > 5 && display == 1)
			{
				cout << "YAWNING!!!!!!!!" << endl;
				waitKey(1000);
				display = 0;
			}
			if (f > 50)
			{
				for (int j = 0; j < 10; j++)
					yawn[j] = 0;
				display = 1;
				f = 0;
			}
		}

	}
}


////            MAIN                ////

int main()
{
	Mat in, in1, src2;
	CvCapture* capture;
	int a = 0;
	capture = cvCaptureFromFile(IMAGE_PATH);
	//capture = cvCaptureFromCAM(CV_CAP_ANY);
	while (true)
	{
		f++;
		f2++;
		in = cvQueryFrame(capture);
		in1 = in.clone();
		if (!in.empty())
		{
			src2 = skin_segmentation(in);
			cont = contour(src2);

			if (!cont.empty())
			{
				eye_detect(in);

				xmin = 1920, ymin = 1080, xmax = 0, ymax = 0, in2 = 0, ymin1 = 1080, ymax1 = 0, lab = 0;
				center1_m1 = Point(0, 0), center2_m1 = Point(0, 0), center1_m2 = Point(0, 0), center2_m2 = Point(0, 0);
				cont.erase(cont.begin(), cont.end());
				eyebrows_roi.erase(eyebrows_roi.begin(), eyebrows_roi.end()), eyes_detect.erase(eyes_detect.begin(), eyes_detect.end());
				contours.erase(contours.begin(), contours.end());
				waitKey(10);

				if (eye_loc.size() + eye_loc_m1.size() == 4)
					mouth_detect(in1);
				waitKey(200);
			}
			else
			{
				f1++;
				if (f1 = 50)
				{
					cout << "NO SKIN FOUND !!!!!!" << endl;
					f1 = 0;
				}
			}
		}
		else
		{
			f3++;
			if (f3 = 50)
			{
				cout << "NO IMAGE FOUND !!!!!!" << endl;
				f3 = 0;
			}
		}
		xmin = 1920, ymin = 1080, xmax = 0, ymax = 0, in2 = 0, ymin1 = 1080, ymax1 = 0, lab = 0;
		center1_m1 = Point(0, 0), center2_m1 = Point(0, 0), center1_m2 = Point(0, 0), center2_m2 = Point(0, 0);
		cont.erase(cont.begin(), cont.end()), eye_loc.erase(eye_loc.begin(), eye_loc.end()), eye_loc_m1.erase(eye_loc_m1.begin(), eye_loc_m1.end());
		eyebrows_roi.erase(eyebrows_roi.begin(), eyebrows_roi.end()), eyes_detect.erase(eyes_detect.begin(), eyes_detect.end());
		contours.erase(contours.begin(), contours.end());
		contm.erase(contm.begin(), contm.end());
		eye1 = Point(0, 0), eye2 = Point(0, 0);
	}
	system("pause");
	return 0;
}
*/