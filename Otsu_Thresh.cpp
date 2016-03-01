/*
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <limits>

using namespace cv;
using namespace std;


// This function makes use of the Mat << operator.

void dump(const Mat &mat, const char* fname)
{
	ofstream filestream;
	filestream.open(fname);
	filestream << mat << endl << endl;
	filestream.close();
}
// Used to avoid noise in the image.

void applyGaussian(cv::Mat &input, cv::Mat &output) {
	double sigma = 1.5;
	cv::Mat gaussKernel = cv::getGaussianKernel(9, sigma, CV_32F);
	cv::GaussianBlur(input, output, cv::Size(3, 3), 1.5);
}


// This is similar to the implementation of Robert Laganière.
// See his book: OpenCV 2 Computer Vision Application Programming Cookbook.

cv::Mat showHistogram(const cv::Mat &inImage){

	cv::MatND hist;
	// For a gray scale [0:255] we have 256 bins
	const int bins[1] = { 256 };
	const float hranges[2] = { 0.0, 255.0 };
	const float* ranges[1] = { hranges };
	const int channels[1] = { 1 };
	
	cv::calcHist(&inImage,
		1,             // histogram from 1 image only
		channels,
		cv::Mat(),     // no mask is used
		hist,            // the output histogram
		1,             // 1D histogram
		bins,
		ranges         // pixel value ranges
		);

	// Get min and max bin values
	double maxVal = 0;
	double minVal = 0;
	cv::minMaxLoc(hist, &minVal, &maxVal, 0, 0);
	// The image to display the histogram
	cv::Mat histImg(bins[0], bins[0], CV_8U, cv::Scalar(255));

	// Map the highest point to 95% of the histogram height to leave some
	// empty space at the top
	const int histHeight = bins[0];
	const int maxHeight = 0.95 * histHeight;

	cv::Mat_<float>::iterator it = hist.begin<float>();
	cv::Mat_<float>::iterator itend = hist.end<float>();

	int barPosition = 0;
	for (; it != itend; ++it) {
		float histValue = (*it);
		int barHeight = (histValue * maxHeight) / maxVal;
		cv::line(histImg,
			// start the line from the bottom, and go up based on the barHeight
			// Remember the (0,0) is the top left corner
			cv::Point(barPosition, histHeight),
			cv::Point(barPosition, histHeight - barHeight),
			cv::Scalar::all(0));
		barPosition++;
	}

	return histImg;
}

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

int main(int argc, char *argv[])
{
	cout << "Compiled with OpenCV version " << CV_VERSION << endl;
//	Mat inImage = cv::imread("C:/Users/Nitin/Documents/Visual Studio 2013/files/3.21.jpg");
	Mat inImage = cv::imread("C:/Users/Nitin/Documents/Visual Studio 2013/files/Yawn/1.a.png");
	Mat inImage2 = cv::imread("C:/Users/Nitin/Documents/Visual Studio 2013/files/Yawn/1.b.png");
	Mat inImage3 = cv::imread("C:/Users/Nitin/Documents/Visual Studio 2013/files/eyes6aC.jpg");

	imshow("Original", inImage);


//	Mat histgram2 = showHistogram(inImage2);
//	imshow("Histogram2", histgram2);
//	Mat histgram3 = showHistogram(inImage3);
//	imshow("Histogram3", histgram3);

	Mat histgram = showHistogram(inImage);
	imshow("Histogram", histgram);

	Mat histgram2 = showHistogram(inImage2);
	imshow("Histogram2", histgram2);

	Mat grayImage,imgHSV;

	//cvtColor(inImage, imgHSV, COLOR_BGR2HSV);
	//inRange(imgHSV, Scalar(0, 33, 31), Scalar(15, 171, 195), grayImage);
	//cvtColor(grayImage, grayImage,CV_HSV2BGR);
	
	cvtColor(inImage, grayImage, CV_BGR2GRAY);
//	imshow("grayImage", grayImage);
//	equalizeHist(grayImage, grayImage);
//	imshow("EqualHist", grayImage);
	applyGaussian(grayImage, grayImage);



	int threshVal = 0;
//	namedWindow("My Window", CV_WINDOW_AUTOSIZE);
//	createTrackbar("Brightness", "My Window", &threshVal, 255);
//	imshow("Blur", grayImage);
	// The Otsu thresholding algorithm works well when the histogram has a bimodal distribution.
	// It will find the threshold value that maximizes the extra-class variance while
	// keeping a low intra-class variance.
	cv::Mat binaryImage;
	while (1)
	{
		cv::threshold(grayImage, binaryImage
			, threshVal    // the value doesn't matter for Otsu thresholding
			, 255  // we could choose any non-zero value. 255 (white) makes it easy to see the binary image
			, cv::THRESH_OTSU | cv::THRESH_BINARY);

//		imshow("Thresh", binaryImage);

		applyClosing(binaryImage, 2);

		cv::Mat outImage = cv::Mat::zeros(inImage.rows, inImage.cols, inImage.type());
		inImage.copyTo(outImage, binaryImage);
		waitKey(300);
	}
	

	return EXIT_SUCCESS;
}
*/