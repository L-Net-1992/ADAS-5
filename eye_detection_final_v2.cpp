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
#define IMAGE_PATH "C:/Users/Nitin/Documents/Visual Studio 2013/files/Face/Yawn/3.3.jpg"

using namespace std;
using namespace cv;

//        GLOBAL VARIABLES          //

int xmin = 1920, ymin = 1080, xmax = 0, ymax = 0, in2 = 0, ymin1=1080 ,ymax1=0;
Point center1_m1(0, 0), center2_m1(0, 0), center1_m2(0, 0), center2_m2(0, 0);
vector<Point> cont, eye_loc, eye_loc_m1;
vector<Mat> eyebrows_roi;
int eye_loc_m3[4] = { 0, 0, 0, 0 };

Point eye1, eye2;
Mat threshold_output, imgHSV, imgGray, imgThresholded1, outThresh;
vector<vector<Point>> contours;
vector<Vec4i> hierarchy;

//        FUNCTION DECLARATIONS     //


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
	    
		int q=0;
		int m;
		if (eye_loc.size() + eye_loc_m1.size() == 8)
			m = 2;
		else if (eye_loc.size() + eye_loc_m1.size() == 6 || eye_loc.size() + eye_loc_m1.size() == 4)
			m = 1;
		for (unsigned int j = 0; j < (eye_loc.size()+eye_loc_m1.size())/2; j+=m)
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
				if (e2.y - e2.height*1.25>0)
					eyebrow_bbox_y = (e2.y - e2.height*1.25);
				else
					eyebrow_bbox_y = (e2.y - e2.height*0.8);
				int eyebrow_bbox_height = (e2.height);
				int eyebrow_bbox_width = round((double)e2.width * 1.2);
				if (eyebrows_roi.empty()!=1)
				eyebrows_roi.push_back(input(Rect(eyebrow_bbox_x, eyebrow_bbox_y, eyebrow_bbox_width, eyebrow_bbox_height)));
				//rectangle(input, Point(eyebrow_bbox_x, eyebrow_bbox_y), Point(eyebrow_bbox_x + eyebrow_bbox_width, eyebrow_bbox_y + eyebrow_bbox_height), Scalar(0, 255, 0), 1, 4);
				
			}
			q++;
			
		}
	
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
	vector<Point> null_vec;
	threshold(input, threshold_output, 60, 255, THRESH_BINARY);
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	vector<vector<Point> > contours_poly(contours.size());

	
	in2 = contours.size();
	if (in2 == 0)
	{ 
		//null_vec.push_back(Point(0, 0));
		return null_vec;
	}
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
	
	//cout << "Emptiness = " <<contours_poly.empty() << endl;

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
	Rect face_roi(abs(xmin), abs(ymin), abs(xmax - xmin) + 10, abs(ymax - ymin) + 20);

	IplImage* frame_clone = new IplImage(frame);
	cvSetImageROI(frame_clone, face_roi);
	cvResize(frame_clone, frame_clone, CV_INTER_LINEAR);
	Mat frame3(frame_clone);

	return frame3;
}



vector<Point> eye_M1(Mat eye_mat)
{
	vector<Point> eye_loc;
	std::vector<Rect> eyes;

	String eyes_cascade_name = "E:/opencv/sources/data/haarcascades/haarcascade_eye.xml";
	CascadeClassifier eyes_cascade;

	eye_mat = eye_mat(Rect(0, 0, eye_mat.cols, eye_mat.rows * 0.6));
	eyes_cascade.load(eyes_cascade_name);
	eyes_cascade.detectMultiScale(eye_mat, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	cout << "no. of eyes_M1 = " << eyes.size() << endl;
	for (size_t j = 0; j < eyes.size(); j++)
	{

		Point center1(eyes[j].x + eyes[j].width*0.5, eyes[j].y + eyes[j].height*0.5);
		int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
		if (j == 1 && radius<45)
		{
			eye2 = center1;

			circle(eye_mat, eye2, radius, Scalar(0, 255, 0), 4, 8, 0);

		}
		else if (j == 0 && radius<45)
		{

			eye1 = center1;
			circle(eye_mat, eye1, radius, Scalar(0, 255, 0), 4, 8, 0);
		}
		//cout << "r = " << radius << endl;
		if (j < 2 && radius<45)
		{
			eye_loc.push_back(Point(eyes[j].x, eyes[j].y));
			eye_loc.push_back(Point(eyes[j].width, eyes[j].height));

		}
	}
	imshow("eye_mat", eye_mat);
	waitKey(1000);
	return eye_loc;
}

vector<Point> eye_M2(Mat input)
{
	//cout << "chk" << endl;
	vector<vector<Point>> contours;
	vector<Point> eye_loc,null_eye;
	vector<Vec4i> hierarchy;
	//int arr[10] = { 0, 0, 0, 0}, arr2[4] = { 0, 0, 0, 0 };
	int k = 0;
	int *arr = new int[k];
	int *arr2 = new int[k];
	Mat input2, input3, input4;

	Mat input1 = input.clone();
	Rect roi(0, 0, input.cols, input.rows*0.6);

	input = input(roi);
	Mat input5 = input.clone();
	cvtColor(input, input, CV_BGR2YCrCb);
	//cout << "chk1" << endl;
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
	//cout << "chk2" << endl;
	findContours(eyei, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	int j = 0;
	for (int i = 0; i < contours.size(); i++)
	{

		double pq = contourArea(contours[i], 0);
		if (pq > 300 && pq < 2500)
		{
			//cout << "chk2.5" << endl;
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = boundingRect(Mat(contours_poly[i]));
			int width = boundRect[i].br().x - boundRect[i].tl().x;
			int height = boundRect[i].br().y - boundRect[i].tl().y;

			if (width - height < 50 && width - height>-3)
			{
				//cout << "chk3" << endl;
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
		cout << "angle1 = " << angle1 << endl;
	}
	if (j == 0)
	{
		return null_eye;
	}
	int width2 = boundRect[eye2y].br().x - boundRect[eye2y].tl().x;
	int height2 = boundRect[eye2y].br().y - boundRect[eye2y].tl().y;
	
	if (abs(angle1) < 30 && j > 1)
	{
		int width1 = boundRect[eye1y].br().x - boundRect[eye1y].tl().x;
		int height1 = boundRect[eye1y].br().y - boundRect[eye1y].tl().y;



		rectangle(input1, boundRect[eye1y].tl(), boundRect[eye1y].br(), (0, 0, 255), 2, 8, 0);
		rectangle(input1, boundRect[eye2y].tl(), boundRect[eye2y].br(), (0, 0, 255), 2, 8, 0);
		eye_loc.push_back(boundRect[eye1y].tl());
		eye_loc.push_back(Point(width1, height1));
		eye_loc.push_back(boundRect[eye2y].tl());
		eye_loc.push_back(Point(width2, height2));

	}
	
	else 
	{
		
		rectangle(input1, boundRect[eye2y].tl(), boundRect[eye2y].br(), (0, 0, 255), 2, 8, 0);
		eye_loc.push_back(boundRect[eye2y].tl());
		eye_loc.push_back(Point(width2, height2));
	}
	imshow("input1", input1);

	waitKey(20);
	return eye_loc;
}

void eye_M3(Mat input)
{
	Mat input1 = input.clone();
	
	
	Eyebrow_ROI(input1);
	if (eyebrows_roi.empty() != 1)
	{
		for (int d = 0; d < eyebrows_roi.size(); d++)
		{
			ymin1 = 1080;
			ymax1 = 0;

			//imshow("a = ", eyebrows_roi[0]);
			Mat_<uchar> image_exp = exponentialTransform(CRTransform(eyebrows_roi[d]));
			Mat_<uchar> image_binary = binaryThresholding(image_exp, returnImageStats(image_exp));
			morphologyEx(image_binary, image_binary, 3, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
			//imshow("hed", image_binary);
			// A clone image is required because findContours() modifies the input image
			Mat image_binary_clone = image_binary.clone();
			vector<vector<Point> > contours;
			findContours(image_binary_clone, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

			char nname[40];
			sprintf(nname, "brow %d ", d);
			imshow(nname, image_binary);

			imshow(nname + 'a', eyebrows_roi[d]);
			waitKey(10);
			// Initialize blank image (for drawing contours)
			Mat_<uchar> image_contour(image_binary.size());
			for (int i = 0; i < image_contour.rows; ++i)
			{
				for (int j = 0; j < image_contour.cols; ++j)
					image_contour.at<uchar>(i, j) = 0;
			}


			// Draw largest contour on the blank image
			cout << "Size of the contour image: " << image_contour.rows << " X " << image_contour.cols << "\n";
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
					//				cout << contours[largest_contour_idx][i] << endl;
				}

				waitKey(10);
				for (int i = 0; i < contours[largest_contour_idx].size(); ++i)
				{
					Point_<int> pt = contours[largest_contour_idx][i];
					image_contour.at<uchar>(pt.y, pt.x) = 255;
				}
				if (contourArea(contours[largest_contour_idx]) < (eyebrows_roi[d].rows*eyebrows_roi[d].cols / 2) && contourArea(contours[largest_contour_idx]) > (eyebrows_roi[d].rows*eyebrows_roi[d].cols / 20) && (ymax1 - ymin1 < image_contour.rows*0.6))
					eye_loc_m3[d] = 1;
				else
					eye_loc_m3[d] = 0;
			}
			else
			{

				eye_loc_m3[d] = 0;
			}
			cout << "eye_loc_m3 = " << eye_loc_m3[d] << endl;
		}
		imshow("aaa",eye_loc_m3[0]);
		waitKey(10);
		imshow("bbb", eye_loc_m3[1]);
		waitKey(10);
	}
	else
	{
		cout << " NO EYES FOUND " << endl;
	}
	
}


//            MAIN                //

int main()
{
	Mat in, frame, src2;
	
	in = imread(IMAGE_PATH);
	char name[50];
	//for (int t = 0; t < 20; t++)
	//{
	//sprintf(name, "D:/DRIVER DROWSINESS/images/positive3/image%d.jpg", t);
	//in = imread(name);
	//imshow("in", in);
	src2 = skin_segmentation(in);
	
	cont = contour(src2);
	if (!cont.empty())
	{
		frame = roi_cont(in, cont);
		Mat frame_clone = frame.clone();
		Mat frame_clone2 = frame.clone();
		if (!frame.empty())
		{
			eye_loc = eye_M2(frame);
//			cout <<"size1 = "<< eye_loc.size() << endl;
			cout << " Method2 " << endl;
			//cout << "image no. = " << t << endl;
			cout << "eyes = " << eye_loc.size() / 2 << endl;
			if (eye_loc.size() == 2 && eye_loc[0].x != 0 && eye_loc[1].x != 0 )
			{
				//cout << "eye_loc_m2 = " << eye_loc << endl;
				center1_m2 = Point(eye_loc[0].x + eye_loc[1].x*0.5, eye_loc[0].y + eye_loc[1].y*0.5);
				cout << "center1_m2 = " << center1_m2 << endl;

			}
			else if (eye_loc.size() == 4)
			{
				//cout << "eye_loc_m2 = " << eye_loc << endl;
				center1_m2 = Point(eye_loc[0].x + eye_loc[1].x*0.5, eye_loc[0].y + eye_loc[1].y*0.5);
				center2_m2 = Point(eye_loc[2].x + eye_loc[3].x*0.5, eye_loc[2].y + eye_loc[3].y*0.5);
				if (center2_m2.x < center1_m2.x)
					swap(center2_m2, center1_m2);
				cout << "center1_m2 = " << center1_m2 << endl;
				cout << "center2_m2 = " << center2_m2 << endl;
			}
			else
			{
				cout << "NO EYE FOUND M2" << endl;
			}
			cout << endl;

			//imshow("frame", frame_clone);
			waitKey(10);
			eye_loc_m1 = eye_M1(frame_clone);
			cout << " Method1 " << endl;


			if (eye_loc_m1.size() == 4)
			{
				center1_m1 = Point(eye_loc_m1[0].x + eye_loc_m1[1].x*0.5, eye_loc_m1[0].y + eye_loc_m1[1].y*0.5);
				center2_m1 = Point(eye_loc_m1[2].x + eye_loc_m1[3].x*0.5, eye_loc_m1[2].y + eye_loc_m1[3].y*0.5);
				if (center2_m1.x < center1_m1.x)
					swap(center2_m1, center1_m1);
				cout << "center1_m1 = " << center1_m1 << endl;
				cout << "center2_m1 = " << center2_m1 << endl;
			}
			else if (eye_loc_m1.size() == 2)
			{
				center1_m1 = Point(eye_loc_m1[0].x + eye_loc_m1[1].x*0.5, eye_loc_m1[0].y + eye_loc_m1[1].y*0.5);
				cout << "center1_m1 = " << center1_m1 << endl;
			}
			else
			{
				cout << "NO EYE FOUND M1" << endl;
			}
			cout << endl;

			//          EYES CONDITIONS           //     	
			if (eye_loc.size() + eye_loc_m1.size() == 8)
			{
				if (abs(center1_m1.x - center1_m2.x) < 26 && abs(center1_m1.y - center1_m2.y) < 15)
				{

					if (center2_m1.y != 0 && center2_m2.y != 0)
					{
						if (abs(center2_m1.x - center2_m2.x) < 26 && abs(center2_m1.y - center2_m2.y) < 15)
						{
							cout << "Both Eyes Detected" << endl;
						}
						else
						{
							waitKey(10);
							eye_M3(frame_clone2);
						}
					}

				}
			}
			else if (eye_loc.size() + eye_loc_m1.size() == 6)
			{
				if (eye_loc_m1.size() == 4)
				{
					if (abs(center1_m1.x - center1_m2.x) < 22 && abs(center1_m1.y - center1_m2.y) < 15)
					{
						eye_loc.erase(eye_loc.begin(), eye_loc.end());
						cout << "eye_loc = " << eye_loc.size() << endl;
						waitKey(10);
						eye_M3(frame_clone2);
					}
					else if (abs(center2_m1.x - center1_m2.x) < 22 && abs(center2_m1.y - center1_m2.y) < 15)
					{
						
						eye_loc.erase(eye_loc.begin(), eye_loc.end());
						cout << "eye_loc = " << eye_loc.size() << endl;
						waitKey(10);
						eye_M3(frame_clone2);
					}
					else
					{
						waitKey(10);
						cout << "eye_loc = " << eye_loc.size() << endl;
						eye_M3(frame_clone2);
					}
				}
				else if (eye_loc.size() == 4)
				{

					if (abs(center1_m2.x - center1_m1.x) < 22 && abs(center1_m2.y - center1_m1.y) < 15)
					{

						eye_loc_m1.erase(eye_loc_m1.begin(), eye_loc_m1.end());
						cout << "eye_loc_m1 = " << eye_loc_m1.size() << endl;
						waitKey(10);
						eye_M3(frame_clone2);
					}
					else if (abs(center2_m2.x - center1_m1.x) < 22 && abs(center2_m2.y - center1_m1.y) < 15)
					{

						eye_loc_m1.erase(eye_loc_m1.begin(), eye_loc_m1.end());
						cout << "eye_loc_m1 = " << eye_loc_m1.size() << endl;
						waitKey(30);
						eye_M3(frame_clone2);
					}
					else
					{

						cout << "eye_loc = " << eye_loc_m1.size() << endl;
						waitKey(10);
						eye_M3(frame_clone2);
					}
				}
			}
			else if (eye_loc.size() + eye_loc_m1.size() == 4)
			{
				if (eye_loc.size() == 0 || eye_loc_m1.size() == 0)
				{
					waitKey(10);
					eye_M3(frame_clone2);
				}
				else
				{
					cout << "    ALERT !!!!!!!!!!!!!!! " << endl;
				}
			}
			else
			{
				cout << "    ALERT !!!!!!!!!!!!!!! " << endl;
			}
		}
	}
	else
	{
		cout << "NO SKIN FOUND !!!!!!" << endl; 
	}
	cout << endl;
	system("pause");
	return 0;

}
*/