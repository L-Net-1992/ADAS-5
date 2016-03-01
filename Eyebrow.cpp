/*
//#include "eyebrow_roi.h"
#include <iostream>
#include <utility>
#include "opencv2/imgproc/imgproc.hpp"
#include<opencv2\opencv.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<stdio.h>
using namespace std;
using namespace cv;

class EyebrowROI
{
private:
	Mat image;
	string face_cascade_path = "E:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
	string eye_cascade_path = "E:/opencv/sources/data/haarcascades/haarcascade_eye.xml";
	CascadeClassifier face_cascade;
	CascadeClassifier eye_cascade;


public:
	Mat face_roi;
	vector<Mat> eyebrows_roi;
	vector<Rect_<int> > faces;
	vector<Rect_<int> > eyes;

	EyebrowROI(const Mat& _image, const string& _face_cascade_path,
		const string& _eye_cascade_path);
	EyebrowROI(const EyebrowROI& _obj);
	void detectFace();
	void detectEyebrows();
	vector<Mat> displayROI();
};

EyebrowROI::EyebrowROI(const Mat& _image, const string& _face_cascade_path,
	const string& _eye_cascade_path)
	:image(_image), face_cascade_path(_face_cascade_path), eye_cascade_path(_eye_cascade_path)
{
	face_cascade.load(face_cascade_path);
	eye_cascade.load(eye_cascade_path);
}

EyebrowROI::EyebrowROI(const EyebrowROI& _obj)
{
	image = _obj.image;
	face_cascade_path = _obj.face_cascade_path;
	eye_cascade_path = _obj.eye_cascade_path;
	face_cascade = _obj.face_cascade;
	eye_cascade = _obj.eye_cascade;
}

void EyebrowROI::detectFace()
{
	face_cascade.detectMultiScale(image, faces, 1.15, 3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	return;
}

void EyebrowROI::detectEyebrows()
{
	detectFace();
	for (unsigned int i = 0; i < faces.size(); ++i)
	{
		Rect_<int> face = faces[i];
		face_roi = image(Rect(face.x, face.y, face.width, face.height));

		eye_cascade.detectMultiScale(face_roi, eyes, 1.20, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	}
	return;
}

vector<Mat> EyebrowROI::displayROI()
{
	for (unsigned int i = 0; i < faces.size(); ++i)
	{
		Rect_<int> face = faces[i];
		
		

		//rectangle(image, Point(face.x, face.y), Point(face.x+face.width, face.y+face.height),
		//Scalar(0, 0, 255), 1, 4);
		
		
		for (unsigned int j = 0; j < eyes.size(); ++j)
		{
			Rect_<int> e = eyes[j];

			// Calculate parameters for eyebrow bounding box from those of eye bounding box
			int eyebrow_bbox_x = e.x;
			int eyebrow_bbox_y = (e.y - e.height / 5);

			int eyebrow_bbox_height = (e.height * 3) / 5;
			int eyebrow_bbox_width = round((double)e.width * 1.2 );

			// Save and mark eyebrow region
			eyebrows_roi.push_back(face_roi(Rect(eyebrow_bbox_x, eyebrow_bbox_y,eyebrow_bbox_width, eyebrow_bbox_height)));
			rectangle(face_roi, Point(eyebrow_bbox_x, eyebrow_bbox_y),Point(eyebrow_bbox_x+eyebrow_bbox_width, eyebrow_bbox_y+eyebrow_bbox_height),Scalar(255, 0, 0), 1, 4);

		}
	}
	imshow("Eyebrow_Detection", image);
	return eyebrows_roi;
}


string input_image_path;
string face_cascade_path, eye_cascade_path;
Mat_<uchar> CRTransform(const Mat& image);
Mat_<uchar> exponentialTransform(const Mat_<uchar>& image);
pair<double, double> returnImageStats(const Mat_<uchar>& image);
Mat_<uchar> binaryThresholding(const Mat_<uchar>& image, const pair<double, double>& stats);
int returnLargestContourIndex(vector<vector<Point> > contours);


int main(int argc, char** argv)
{
//	if (argc != 4)
//	{
//		cout << "Parameters missing!\n";
//		return 1;
//	}

	input_image_path = "C:/Users/Nitin/Documents/Visual Studio 2013/files/P00001.bmp";
	face_cascade_path = "E:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
	eye_cascade_path = "E:/opencv/sources/data/haarcascades/haarcascade_eye.xml";

	Mat_<Vec3b> image_BGR = imread(input_image_path);

	// Detect faces and eyebrows in image
	EyebrowROI eyebrow_detector(image_BGR, face_cascade_path, eye_cascade_path);
	eyebrow_detector.detectEyebrows();
	vector<Mat> eyebrows_roi = eyebrow_detector.displayROI();
	imshow("sn", eyebrows_roi[0]);
	// Mat_<uchar> image_exp = exponentialTransform(CRTransform(image_BGR));
	Mat_<uchar> image_exp = exponentialTransform(CRTransform(eyebrows_roi[0]));
	imshow("exp", image_exp);
	Mat_<uchar> image_binary = binaryThresholding(image_exp, returnImageStats(image_exp));
	imshow("hed", image_binary);
	// A clone image is required because findContours() modifies the input image
	Mat image_binary_clone = image_binary.clone();
	cout << "chk1" << endl;
	vector<vector<Point> > contours;
	findContours(image_binary_clone, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	cout << "chk2" << endl;
	// Initialize blank image (for drawing contours)
	Mat_<uchar> image_contour(image_binary.size());
	cout << "chk3 "<< endl;
	for (int i = 0; i < image_contour.rows; ++i)
	{
		for (int j = 0; j < image_contour.cols; ++j)
			image_contour.at<uchar>(i, j) = 0;
	}
	
	// Draw largest contour on the blank image
	cout << "Size of the contour image: " << image_contour.rows << " X " << image_contour.cols << "\n";
	int largest_contour_idx = returnLargestContourIndex(contours);
	

	cout << largest_contour_idx << endl;
//	cout << contours[largest_contour_idx].size() << endl;
	if (largest_contour_idx != -1)
	{
		
		for (int i = 0; i < contours[largest_contour_idx].size(); ++i)
		{
		
			Point_<int> pt = contours[largest_contour_idx][i];
			image_contour.at<uchar>(pt.y, pt.x) = 255;
		}
	}
	
	cout << "chk5" << endl;
	cout << "Area = " << contourArea(contours[largest_contour_idx]) << endl;
	imshow("Binary-Image", image_binary);
//	imshow("Contour", image_contour);

	waitKey(0);
	return 0;
}

Mat_<uchar> CRTransform(const Mat& image)
{
//	Mat_<Vec3b> _image = image;
//	imshow("_image", _image);
	Mat_<uchar> CR_image(image.size());
	for (int i = 0; i < image.rows; ++i)
	{ 
		for (int j = 0; j < image.cols; ++j)
			//CR_image.at<uchar>(i, j) = (255 - _image(i, j)[2]);
			CR_image.at<uchar>(i, j) = (255 - image.at<Vec3b>(i, j)[2]);
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

	double Z = 0.9;
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
	cout << "contours.size() = " << contours.size() << endl;
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
*/