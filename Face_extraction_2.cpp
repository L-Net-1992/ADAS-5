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
char name5[40];
#define IMAGE_FILE_JPG "jpg"
#define IMAGE_DIR "C:/Users/Nitin/Documents/Visual Studio 2013/files/Face/gabor/images/images"
#define M_PI 3.14159265359
RNG rng(12345);
CvMat** mGabor = NULL;
bool kernelsDefined = false;

void shuffle(Mat &data, Mat &catg)
{
	Mat data_backup = data.clone();
	Mat catg_backup = catg.clone();

	data.pop_back(data_backup.rows);
	catg.pop_back(catg_backup.rows);

	vector<int> myvector;
	vector<int>::iterator it;
	for (int k = 0; k<data_backup.rows; k++)
	{
		myvector.push_back(k);
	}
	random_shuffle(myvector.begin(), myvector.end());

	for (it = myvector.begin(); it != myvector.end(); ++it)
	{
		data.push_back(data_backup.row(*it));
		catg.push_back(catg_backup.row(*it));
	}

	data_backup.release();
	catg_backup.release();
}

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
int pl = 0;
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

	//cvShowImage("integral", temp);

	cvReleaseMat(&sqSum);

	int actualSize = 0;
	// top left
	for (int i = 0; i < 100; i += 10) {
		for (int j = 0; j < 100; j += 10) {
			// bottom right
			for (int m = i + 10; m <= 100; m += 10) {
				for (int n = j + 10; n <= 100; n += 10) {
					haar.at<float>(0, actualSize++) = getIntegralRectValue(temp, i, j, m, n);
				}
			}
		}
	}
	//imshow("haar", haar);
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
					_dx = static_cast<float>(dx.at<int16_t>(n*l_y + j, m*l_x + i));
					_dy = static_cast<float>(dy.at<int16_t>(n*l_y + j, m*l_x + i));
					grad_value = static_cast<float>(std::sqrt(1.0*_dx*_dx + _dy*_dy) / area_img);
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


void extractFeatures(const Mat& inputImage, Mat& featureVector, featureExtractor fEx)
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


////////SVM 
void learningAlgorithmTrain(CvStatModel* model,const Mat& trainData,const Mat& responses,int numCategories,learningAlgorithm lA)
{
	switch (lA)
	{
	case ANN:
	{
		Mat tmpOp = Mat::zeros(responses.rows, numCategories, CV_32F);
		for (int i = 0; i < responses.rows; i++)
			tmpOp.at<float>(i, static_cast<int>(responses.at<float>(i, 0))) = 1;
		Mat annWt = Mat::ones(responses.rows, 1, CV_32F);
		reinterpret_cast<CvANN_MLP*>(model)->train(trainData, tmpOp, annWt);
		break;
	}
	case SVM_ML:
	{
		SVMParams svmParams;
		svmParams.kernel_type = CvSVM::LINEAR;
		reinterpret_cast<CvSVM*>(model)->train(trainData, responses,Mat(), Mat(), svmParams);
		break;
	}
	case RT:
	{
		reinterpret_cast<CvRTrees*>(model)->train(trainData, CV_ROW_SAMPLE,
			responses);
		break;
	}
	default:
		throw INVALID_LEARNING_ALGORITHM;
	}
}

void learningAlgorithmPredict(CvStatModel* model,const Mat& featureData,Mat& responses,int numCategories,learningAlgorithm lA)
{
	responses.create(featureData.rows, 1, CV_32F);
	switch (lA)
	{
	case ANN:
	{

		Mat tmpOps(featureData.rows, numCategories, CV_32F);
		reinterpret_cast<CvANN_MLP*>(model)->predict(featureData, tmpOps);
		for (int i = 0; i < featureData.rows; i++)
		{
			float max = -HUGE_VAL;
			float prediction = 0;
			for (int j = 0; j < numCategories; j++)
			{
				if (tmpOps.at<float>(i, j) > max)
				{
					max = tmpOps.at<float>(i, j);
					prediction = j;
				}
			}
			responses.at<float>(i, 0) = prediction;
		}
		break;
	}
	case SVM_ML:
	{
		CvMat fData = featureData;
		CvMat resp = responses;
		
		reinterpret_cast<CvSVM*>(model)->predict(&fData, &resp);
		break;
	}
	case RT:
	{
		for (int i = 0; i < featureData.rows; i++)
		{
			responses.at<float>(i, 0) = reinterpret_cast<CvRTrees*>(model)
				->predict(featureData.row(i));
		}
		break;
	}
	default:
		throw INVALID_LEARNING_ALGORITHM;
	}
}



////////main
int main(int argc, char** argv)
{


	--

		IplImage* img = cvLoadImage("C:/Users/Nitin/Documents/Visual Studio 2013/files/Face/P0010.bmp", 0);
		Mat a, b, c;
		if (img != NULL)
		{
			extractFeatures(img, a, GABOR);
			extractFeatures(img, b, PHOG);
			extractFeatures(img, c, HAAR);
		}
		cout << "Gabor Vector size = " << a.size() << endl;
		cout << "HOG Vector size = " << b.size() << endl;
		cout << "HAAR Vector size = " << c.size() << endl;
	///////////////////////////////////////////////////////////////////////////////////////
		
	
	

	Mat imageFeatureData;
	Mat categoryData;

	int numCategories = 0;
	string filename;
	int label;
	label = 0;
	char name[40];
	Mat m;
	cout << "chk1" << endl;
	for (int i = 01; i < 355; i++)
	{
		if (i < 109)
			sprintf(name, "C:/Users/Nitin/Documents/Visual Studio 2013/files/Face/positive2/Clipboard%d.jpg", i);
		else if (i >= 109 && i < 255)
			sprintf(name, "C:/Users/Nitin/Documents/Visual Studio 2013/files/Face/positive2/image%d.jpg", i - 108);
		else if (i >= 255 && i < 355)
			sprintf(name, "C:/Users/Nitin/Documents/Visual Studio 2013/files/Face/negetive2/image%d.jpg", i - 254);
		
		
		IplImage* img_a = cvLoadImage(name, 0);

		if (!img_a)
			continue;

		if (i < 109)
			label = 1;
		else if (i >= 109 && i < 255)
			label = 1;
		else if (i >= 255)
			label = 0;

		if (img_a != NULL)
		{
			extractFeatures(img_a, m, HAAR);
			imageFeatureData.push_back(m);
			categoryData.push_back(static_cast<float>(label));
			numCategories += 1;
		}
		imshow("imageFeatureDataShuffeled", imageFeatureData);
		cvReleaseImage(&img_a);
	}
	imshow("label1", categoryData);
	//	imshow("imageFeatureData1", imageFeatureData);
	shuffle(imageFeatureData, categoryData);
	imshow("imageFeatureDataShuffeled", imageFeatureData);
	imshow("labelShuffeled", categoryData);

	Mat responses;
	cout << "Featur Data size = " << imageFeatureData.size() << endl;
	cout << "Label matrix size = " << categoryData.size() << endl;
	SVMParams svmParams;
	svmParams.kernel_type = CvSVM::RBF;
	svmParams.gamma = pow(2, -15);
	svmParams.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	

	--

//	system("pause");

	CvBoost boost;
//	boost.train(imageFeatureData, CV_ROW_SAMPLE, categoryData);

	CvSVM model;
//	model.train(imageFeatureData, categoryData, Mat(), Mat(), svmParams);

//	int c1 = model.get_support_vector_count();
//	cout << "SVM support Vectors = " << c1 << endl;

//	model.save("C:/Users/Nitin/Documents/Visual Studio 2013/files/result/model_HAAR.xml");
//	model.save("C:/Users/Nitin/Documents/Visual Studio 2013/files/result/model_GABOR.xml");
//	model.save("C:/Users/Nitin/Documents/Visual Studio 2013/files/result/model_PHOG_SVM.xml");
//	boost.save("C:/Users/Nitin/Documents/Visual Studio 2013/files/result/model_PHOG_ADA.xml");
//	CvSVM model;
	boost.load("C:/Users/Nitin/Documents/Visual Studio 2013/files/Face/result/model_PHOG_ADA.xml");
//	model.load("C:/Users/Nitin/Documents/Visual Studio 2013/files/result/model_HAAR.xml");
//	model.load("C:/Users/Nitin/Documents/Visual Studio 2013/files/result/model_PHOG_SVM.xml");
//	model.load("C:/Users/Nitin/Documents/Visual Studio 2013/files/result/model_GABOR.xml");
	waitKey(1000);
	

	


	char name2[40];
	Mat src;
	for (int i = 1; i < 10;i++)
	{
		if (i == 1 || i == 2 || i == 4 || i == 5)
			continue;
//		sprintf(name2, "C:/Users/Nitin/Documents/Visual Studio 2013/files/Face/test/img%d.jpg", i);
		sprintf(name2, "C:/Users/Nitin/Documents/Visual Studio 2013/files/Face/test2/image%d.jpg", i);
		IplImage* test = cvLoadImage(name2, 0);
		if (test != NULL)
		{
			cout << "image number == " << i << endl;
			Mat src_gray, imgHSV, imgThresholded1;
			src = imread(name2, 1);
			cvtColor(src, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
			inRange(imgHSV, Scalar(0, 30, 33), Scalar(20, 194, 130), imgThresholded1); //Threshold the image Face2 values

			erode(imgThresholded1, imgThresholded1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
			dilate(imgThresholded1, imgThresholded1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

			dilate(imgThresholded1, imgThresholded1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
			erode(imgThresholded1, imgThresholded1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

			blur(imgThresholded1, imgThresholded1, Size(3, 3));

			Mat threshold_output;
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;

			/// Detect edges using Threshold
			threshold(imgThresholded1, threshold_output, 60, 255, THRESH_BINARY);
			/// Find contours
			findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

			/// Approximate contours to polygons + get bounding rects and circles
			vector<vector<Point> > contours_poly(contours.size());
			vector<Rect> boundRect(contours.size());
			vector<Point2f>center(contours.size());
			vector<float>radius(contours.size());
			
			for (int i = 0; i < contours.size(); i++)
			{
				approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
				boundRect[i] = boundingRect(Mat(contours_poly[i]));
				minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);
			}

			/// Draw polygonal contour + bonding rects + circles
			cout << contours.size() << endl;
			Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
			Mat m2;
			int jp = 0;
			char name3[40];
			for (int i = 0; i < contours.size(); i++)
			{
				Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
				double pq = contourArea(contours[i], 0);
				if (pq > 3200 && pq < 200000)
				{
					jp++;
					cout << "pq = " << pq << endl;
					drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
					
					cout << "x = " << boundRect[i].tl() << endl;
					cout << "y = " << boundRect[i].br() << endl;
					int width = boundRect[i].br().x - boundRect[i].tl().x;
					cout << "width = " << width << endl;
					int height = -boundRect[i].tl().y + boundRect[i].br().y;
					cout << "height = " << height << endl;

					Rect roi(boundRect[i].tl().x, boundRect[i].tl().y, width, height);
					cvSetImageROI(test, roi);
					cvResize(test, test, CV_INTER_LINEAR);
					sprintf(name2, "Area_%d_%d",(int)pq,i);
					cvShowImage(name2, test);

					extractFeatures(test, m2, PHOG);
//					float resp1 = model.predict(m2);
//					cout << "resp1 = " << resp1 << endl;
					float resp2 = boost.predict(m2);
					cout << "resp 2 =" << resp2 << endl;
				if (resp2 == 1)
					{
						rectangle(src, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
					}
					cvResetImageROI(test);
				}
				//		circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);
			}
			cout << "jp = " << jp << endl;
			jp = 0;
		}
		cout << "   /////////////////////////////////////////////////" << endl;
		sprintf(name5, "src_%d", i);
//		imshow(name5, src);
	}

	

	for (;;)
		if (waitKey(30) == 32)
			goto end;
end:
	return 0;
}
*/