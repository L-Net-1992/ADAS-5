//						www.MahdiRezaei.com
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;


String face_cascade_name = "E:/Users/Nitin/Documents/Face Detection - 775/MyCascade/haarcascade_frontalface_alt2.xml";
CascadeClassifier face_cascade;

CvHaarClassifierCascade *cascade;
CvMemStorage *storage;

void detectFaces(IplImage *newframe);

int key;
int iImg;
int N_Images = 20;

int main(int argc, char** argv)
{


	// (1) Defining the path and Loading the Classifier (C format)
	//char      *CascadeFile = "MyCascade/haarcascade_frontalface_alt2.xml";
	//cascade = (CvHaarClassifierCascade*)cvLoad(CascadeFile);


	// (2) Defining the path and Loading the Classifier in (C++ format)
	//string RootCascade = "MyCascade/";
	//string Extension = ".xml";
	//string filename = RootCascade + "haarcascade_frontalface_alt2" + Extension;

	if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading\n"); return -1; };

	//cascade = (CvHaarClassifierCascade*)cvLoad(filename.c_str());



	// (3)  Creating Buffer
	storage = cvCreateMemStorage(0);



	// (4)  Loads a single Image into 
	IplImage* inImage = cvLoadImage("E:/Users/Nitin/Documents/Face Detection - 775/MyImages/GT.jpg", 1);   // 1 for color images, 0 for gray sclae, and CV_LOAD_IMAGE_UNCHANGED if you don't know


	// (5) Check for proper initialization
	if (!storage || !inImage)
	{
		printf("Initialization Failed: %s\n",
			(!storage) ? " Not memmory allocated or not enough memory !\n" :
			" The input file can not be found!\n");
		system("pause");

		return 0;
	}


	// (6) face detection
	detectFaces(inImage);						// Calling the face detection function for the input image-inImage 
	cvShowImage("Face Detection", inImage);		// Shows the final image after Face detection, in Window "Face Detection"
	cvReleaseImage(&inImage);					// Release the current image from memory
	cvWaitKey(0);								// Wait to press a key


	// (7) Load a set of multiple images

	char buf[22];

	for (iImg = 0; iImg <= N_Images; iImg++)  // WHOLE DATABASE LOAD
	{
		sprintf(buf, "E:/Users/Nitin/Documents/Face Detection - 775/MyImages/P%04d.bmp", iImg);
		printf("\n");
		inImage = cvLoadImage(buf, CV_LOAD_IMAGE_UNCHANGED);
		printf("Input Image = P%04d.bmp\n", iImg);

		detectFaces(inImage);

		cvShowImage("Face Detection", inImage);
		cvReleaseImage(&inImage);

		key = cvWaitKey(0);								// Exit if the key q or Q is pressed
		if (key == 'q' || key == 'Q')
			return(0);
	}


	//(8) Releasing the resources (Cascade and Buffer)
	cvReleaseHaarClassifierCascade(&cascade);
	cvReleaseMemStorage(&storage);

	return 0;
}

//======================================== Function DETECT & DRAW =======================================
// (9)
void detectFaces(IplImage *newframe)
{
	CvSeq *faces = cvHaarDetectObjects(newframe, cascade, storage,
		1.4, 1,
		0, //CV_HAAR_DO_CANNY_PRUNING 
		cvSize(100, 100));
	// Looking for better detection?! Try these parameters: 
	// 1.15, 5, 0, cvSize(30 x 30)

	for (int i = 0; i < (faces ? faces->total : 0); i++)
	{
		CvRect *r = (CvRect *)cvGetSeqElem(faces, i);

		cvRectangle(newframe,
			cvPoint(r->x, r->y),
			cvPoint(r->x + r->width, r->y + r->height),
			CV_RGB(0, 255, 0), 2, 8, 0);
	}
}


//END OF CODE