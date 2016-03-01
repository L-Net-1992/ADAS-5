/*
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <cstring>
#include <extractFeatures.h>
#include <learningAlgorithms.h>
#include <exceptions.h>

using namespace cv;
using namespace std;

#define IMAGE_FILE_JPG "jpg"
#define M_PI 3.14159265359

CvMat** mGabor = NULL;
bool kernelsDefined = false;

////////GaborFeaturesExtraction
double MeanVector(double* v, int vSize)
{
	int i;
	double mean = 0.0;
	double* ptr = v;

	for (i = 0; i<vSize; i++)
	{

		mean += *ptr;
		ptr++;
	}
	mean /= (double)vSize;
	return mean;
}
void ZeroMeanUnitLength(double* v, int vSize)
{

	double sqsum = 0.0;
	double mean = MeanVector(v, vSize);
	double* ptr = v;
	int i;

	for (i = 0; i<vSize; i++)
	{

		(*ptr) -= mean;
		sqsum += (*ptr)*(*ptr);
		ptr++;
	}
	double a = 1.0f / (double)(sqrt(sqsum));
	ptr = v;
	for (i = 0; i<vSize; i++)
	{

		(*ptr) *= a;
		ptr++;
	}
}
int gabor_extraction(IplImage* img, double* object, CvMat** mGabor)
{
	int w, h;
	w = 128;
	h = 128;
	CvSize img_size = cvGetSize(img);
	IplImage* imtmp = cvCreateImage(img_size, IPL_DEPTH_64F, 0);

	cvConvertScale(img, imtmp, 1.0, 0);

	int i, j, x, y, n;

	int dft_M = cvGetOptimalDFTSize(w + h - 1);
	int dft_N = cvGetOptimalDFTSize(w + h - 1);

	CvMat* imdft = cvCreateMat(dft_M, dft_N, CV_64FC1);
	cvZero(imdft);
	for (i = 0; i<h; i++)
		for (j = 0; j<w; j++)
			((double*)(imdft->data.ptr + (imdft->step)*i))[j] = ((double*)(imtmp->imageData + imtmp->widthStep*i))[j];

	cvDFT(imdft, imdft, CV_DXT_FORWARD, w);
	n = w*h / 64;


	for (i = 0; i<5; i++)
	{
		for (j = 0; j<8; j++)
		{
			CvMat* gout = cvCreateMatHeader(dft_M, dft_N, CV_64FC1);
			cvCreateData(gout);


			//			cvMulSpectrums(imdft, mGabor[i * 8 + j], gout, 0);				//tesing might not work cause this func. is causing runtime error so commented.

			cvDFT(gout, gout, CV_DXT_INVERSE, w + h - 1);

			//downsample sacle factor 64
			for (x = 4; x<w; x += 8)
				for (y = 4; y<h; y += 8)
				{
					double sum = ((double*)(gout->data.ptr + gout->step*(x + h / 2)))[(y + w / 2) * 2] *
						((double*)(gout->data.ptr + gout->step*(x + h / 2)))[(y + w / 2) * 2] +
						((double*)(gout->data.ptr + gout->step*(x + h / 2)))[(y + w / 2) * 2 + 1] *
						((double*)(gout->data.ptr + gout->step*(x + h / 2)))[(y + w / 2) * 2 + 1];

					object[(i * 8 + j)*n + x / 8 * h / 8 + y / 8] = sqrt(sum);
				}

			cvReleaseMat(&gout);
			ZeroMeanUnitLength(object, n);
		}
	}

	cvReleaseImage(&imtmp);
	cvReleaseMat(&imdft);

	return(1);
}
void extractGaborFeatures(const IplImage* img, Mat& gb)
{
	unsigned long long nsize = NUM_GABOR_FEATURES;
	CvSize size = cvSize(128, 128);
	CvSize img_size = cvGetSize(img);
	IplImage*	ipl = cvCreateImage(img_size, 8, 0);
	if (img->nChannels == 3)
	{
		cvCvtColor(img, ipl, CV_BGR2GRAY);
	}
	else
	{
		cvCopy(img, ipl, 0);
	}

	gb.release();
	gb = Mat::zeros(1, NUM_GABOR_FEATURES, CV_32FC1);
	if ((size.width != img_size.width) || (size.height != img_size.height))
	{
		IplImage* tmpsize = cvCreateImage(size, 8, 0);
		cvResize(ipl, tmpsize, CV_INTER_LINEAR);
		cvReleaseImage(&ipl);
		ipl = cvCreateImage(size, 8, 0);
		cvCopy(tmpsize, ipl, 0);
		cvReleaseImage(&tmpsize);
	}

	double* object = (double*)malloc(nsize*sizeof(double));
	IplImage* tmp = cvCreateImage(size, IPL_DEPTH_64F, 0);

	cvConvertScale(ipl, tmp, 1.0, 0);
	if (!kernelsDefined) {
		//	mGabor = LoadGaborFFT(GABOR_DATA_DIR_PATH);
		kernelsDefined = true;
	}
	//Gabor wavelet
	gabor_extraction(tmp, object, mGabor);
	ZeroMeanUnitLength(object, nsize);
	cvReleaseImage(&tmp);
	cvReleaseImage(&ipl);

	//UnloadGaborFFT(mGabor);
	int actualSize = 0;

	for (int i = 0; i<NUM_GABOR_FEATURES; i++) {
		gb.at<float>(0, actualSize++) = static_cast<float>(object[i]);
	}
	free(object);
}

////////HAAR_Features_Extraction
float getIntegralRectValue(IplImage* img, int top, int left, int bottom, int right)
{
	float res = static_cast<float>(((double*)(img->imageData + img->widthStep*bottom))[right]);
	res -= static_cast<float>(((double*)(img->imageData + img->widthStep*bottom))[left]);
	res -= static_cast<float>(((double*)(img->imageData + img->widthStep*top))[right]);
	res += static_cast<float>(((double*)(img->imageData + img->widthStep*top))[left]);
	return res;
}
void extractHaarFeatures(const IplImage* img, Mat& haar)
{
	CvSize size = cvSize(IMAGE_RESIZE, IMAGE_RESIZE);
	//cout << size.height << endl;
	CvSize size2 = cvSize(INTEGRAL_SIZE, INTEGRAL_SIZE);
	CvSize img_size = cvGetSize(img);
	IplImage*	ipl = cvCreateImage(img_size, 8, 0);
	if (img->nChannels == 3)
	{
		cvCvtColor(img, ipl, CV_BGR2GRAY);
	}
	else
	{
		cvCopy(img, ipl, 0);
	}
	//cvShowImage("iplOri", ipl);
	if ((size.width != img_size.width) || (size.height != img_size.height))
	{
		IplImage* tmpsize = cvCreateImage(size, IPL_DEPTH_8U, 0);
		cvResize(ipl, tmpsize, CV_INTER_LINEAR);
		cvReleaseImage(&ipl);
		ipl = cvCreateImage(size, IPL_DEPTH_8U, 0);
		cvCopy(tmpsize, ipl, 0);
		cvReleaseImage(&tmpsize);
		//cvShowImage("ipl", ipl);
	}
	IplImage* temp = cvCreateImage(size, IPL_DEPTH_64F, 0);
	//cvShowImage("temp", temp);
	cvCvtScale(ipl, temp);							////////////////
	//cvShowImage("temp2", temp);
	
	cvNormalize(temp, temp, 0, 1, CV_MINMAX);		////////////////
	//cvShowImage("temp3", temp);
	haar.release();
	
	haar = Mat::zeros(1, NUM_HAAR_FEATURES, CV_32FC1);
	//imshow("haar1", haar);
	IplImage* integral = cvCreateImage(size2, IPL_DEPTH_64F, 0);
	CvMat * sqSum = cvCreateMat(temp->height + 1, temp->width + 1, CV_64FC1);

	
	cvIntegral(temp, integral, sqSum);				//////////////////
	
	cvReleaseMat(&sqSum);

	int actualSize = 0;
	// top left
	for (int i = 0; i < 100; i += 10) {
		for (int j = 0; j < 100; j += 10) {
			// bottom right
			for (int m = i + 10; m <= 100; m += 10) {
				for (int n = j + 10; n <= 100; n += 10) {
					haar.at<float>(0, actualSize++) = getIntegralRectValue(integral, i, j, m, n);
				}
			}
		}
	}
	cout << actualSize << endl;
	cvReleaseImage(&ipl);
	cvReleaseImage(&temp);
	cvReleaseImage(&integral);
}

////////HOG_Features_Extraction
void extractPHoG(const Mat& img, Mat& PHOG)
{
	int bins = NUM_BINS; //64
	int div = NUM_DIVS;  //8

	if (!img.data)
		throw EMPTY_IMAGE_EXCEPTION;
	
	PHOG.release();
	PHOG = Mat::zeros(1, div*div*bins, CV_32FC1);
	Mat dx, dy;

	Sobel(img, dx, CV_16S, 1, 0, 3);
	Sobel(img, dy, CV_16S, 0, 1, 3);

	unsigned int area_img = img.rows*img.cols;

	float _dx, _dy;
	float angle;
	float grad_value;

	//Loop to find gradients
	int l_x = img.cols / div, l_y = img.rows / div; //l_x = 106, l_y = 142

	
	for (int m = 0; m<div; m++)
	{
		
		for (int n = 0; n<div; n++)
		{
			
			for (int i = 0; i<l_x; i++)
			{
			
				for (int j = 0; j<l_y; j++)
				{
					//testing might not work cause x and y cordinates are interchanged to remove runtime error and div were 3 and bin were 16
					_dx = static_cast<float>(dx.at<int16_t>(n*l_y + j,m*l_x + i));		
					_dy = static_cast<float>(dy.at<int16_t>(n*l_y + j, m*l_x + i));
					grad_value = static_cast<float>(std::sqrt(1.0*_dx*_dx + _dy*_dy)/ area_img);
					angle = std::atan2(_dy, _dx);
					if (angle < 0)
						angle += 2 * CV_PI;
					angle *= bins / (2 * CV_PI);
					PHOG.at<float>(0, (m*div + n)*bins + static_cast<int>(angle)) += grad_value;
				}
			}
		}
	}
	float max = 0;
	for (int i = 0; i < bins; i++)
	{
		if (PHOG.at<float>(0, i) > max)
			max = PHOG.at<float>(0, i);
	}

	for (int i = 0; i < bins; i++)
	{
		PHOG.at<float>(0, i) /= max;
	}

	dx.release();
	dy.release();
}


void extractFeatures(const Mat& inputImage, Mat& featureVector,featureExtractor fEx)
{
	if (inputImage.empty()) throw EMPTY_IMAGE_EXCEPTION;
	switch (fEx)
	{
	case HAAR:
	{
		IplImage img = inputImage;
		extractHaarFeatures(&img, featureVector);
		break;
	}
	case GABOR:
	{
		IplImage img = inputImage;
		extractGaborFeatures(&img, featureVector);
		break;
	}
	case PHOG:
		extractPHoG(inputImage, featureVector);
		break;
	default:
		throw(INVALID_FEATURE_EXTRACTOR);
	}
}

////////main
int main(int argc, char** argv)
{
	IplImage* img = cvLoadImage("C:/Users/Nitin/Documents/Visual Studio 2013/files/Face/P0010.bmp", 0);
	Mat a, b, c;
	if (img != NULL)
	{
		extractFeatures(img, a, GABOR);
		extractFeatures(img, b, PHOG);
		extractFeatures(img, c, HAAR);
	}
	cout << "Gabor Vector size = "<< a.size() << endl;
	cout << "HOG Vector size = " << b.size() << endl;
	cout << "HAAR Vector size = " << c.size() << endl;
	for (;;)
		if (waitKey(30) == 32)
			break;
	
	return EXIT_SUCCESS;
}
*/