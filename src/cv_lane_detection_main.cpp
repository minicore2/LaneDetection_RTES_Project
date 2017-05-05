#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <sched.h>
#include <unistd.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <sys/time.h>
#include <time.h>
#include <bits/stdc++.h>

using namespace cv;
using namespace std;

#define NUM_THREADS                     	1
//#define VIDEO_CAPTURE_THREAD            	1
//#define FEATURE_EXTRACTION_THREAD       	2
//#define LANE_DETECTION_THREAD           	3
//#define ALERT_SERVICE_THREAD            	4

#define FEATURE_EXTRACTION_THREAD       	1

#define FEATURE_EXTRACTION_SOFT_DEADLINE_MS   	100    /* Corresponding to an average frame rate of 8.7 fps + ~20 ms margin */
//#define VIDEO_CAPTURE_SOFT_DEADLINE_MS          190    /* Corresponding to an average frame rate of 5.85 fps + ~20 ms margin */
//#define LANE_DETECTION_SOFT_DEADLINE_MS       	170    /* Corresponding to an average frame rate of 6.5 fps + ~20 ms margin */
//#define ALERT_SERVICE_SOFT_DEADLINE_MS        	170    /* Corresponding to an average frame rate of 6.5 fps + ~20 ms margin */
//#define COMBINED_SOFT_DEADLINE_MS       	500

#define COMBINED_SOFT_DEADLINE_MS       	100


pthread_t threads[NUM_THREADS];
pthread_attr_t rt_sched_attr[NUM_THREADS];
int rt_max_prio, rt_min_prio;
struct sched_param rt_param[NUM_THREADS];
struct sched_param nrt_param;
struct timeval tv;
int temp=0;

double feat_ext_start_time = 0, feat_ext_stop_time = 0;
double temp_time=0, prev_frame_time=0;
double ave_framedt=0.0, ave_frame_rate=0.0, fc=0.0, framedt=0.0;

int lowThreshold = 30;
int const max_lowThreshold = 100;
int kernel_size = 3;
int edgeThresh = 1;
int threshold_ratio = 3;

CvCapture* capture;
IplImage* frame; /* C structure to store the image in the memory */

const char fileName[] = "../test_images/lane_detection_image.jpeg";
//const char fileName[] = "../test_images/lane_image_1.jpg";
//const char fileName[] = "../test_images/lane_image_2.jpg";
//const char fileName[] = "../test_images/lane_detection_image_2.bmp";
// Transform display window
char display_window_name[] = "Display Window";

Mat image;
Mat result_image, gray_image, canny_image, roi_image;

void CannyThreshold(int, void*);
void ApplyHoughtransform(int, void*, Mat &in_image);
void define_region_of_interest(Mat &in_image);
void DrawHoughLines(Mat &in_image);
void Create_Threads(void);
void Destroy_Threads(void);
void *feature_extraction_fun(void* threadID);

int main(void)
{
	/* Using a reference image and working with that for now */
	image = imread(fileName, CV_LOAD_IMAGE_COLOR);

	if(!image.data)                              // Check for invalid input
	{
		cout << "could not open " << fileName << endl;
		return -1;
	}

	//namedWindow(display_window_name, WINDOW_AUTOSIZE );// Create a window for display.
	//imshow(display_window_name, image );                   // Show our image inside it.

	waitKey(0);                                          // Wait for a keystroke in the window
	Create_Threads();

	Destroy_Threads();

	printf("Feature Extraction thread took %d ms\n", (int)(feat_ext_stop_time - feat_ext_start_time));

}

double readTOD(void)
{
    double ft=0.0;
    if ( gettimeofday(&tv, NULL) != 0 )
    {
        perror("readTOD");
        return 0.0;
    }
    else
    {
        ft = ((double)(((double)tv.tv_sec) + (((double)tv.tv_usec) /1000000.0)));
        ft = ft*1000;
    }
    return ft;
}

void display_timestamp()
{
    if(temp > 2)
    {
        fc=(double)temp;
        ave_framedt=((fc-1.0)*ave_framedt + framedt)/fc;
        ave_frame_rate=1.0/(ave_framedt/1000.0);
    }

    framedt=temp_time - prev_frame_time;
    prev_frame_time=temp_time;
    printf("Frame @ %u sec, %lu usec, dt=%5.2lf msec, avedt=%5.2lf msec, rate=%5.2lf fps, Jitter = %d ms\n",
           (unsigned)tv.tv_sec,
           (unsigned long)tv.tv_usec,
           framedt, ave_framedt, ave_frame_rate,
           (int)(COMBINED_SOFT_DEADLINE_MS - ave_framedt));
}

void Create_Threads(void)
{
	struct timeval timeNow;
	int rc;

	printf("Creating thread %d\n", FEATURE_EXTRACTION_THREAD);
	rc = pthread_create(&threads[FEATURE_EXTRACTION_THREAD], NULL, feature_extraction_fun, (void *)FEATURE_EXTRACTION_THREAD);

	if (rc)
	{
		printf("ERROR; pthread_create() rc is %d\n", rc);
		perror(NULL);
		exit(-1);
	}

	gettimeofday(&timeNow, NULL);
	printf("Feature Extraction Thread spawned at %d sec, %d usec\n", timeNow.tv_sec, (double)timeNow.tv_usec);

#if 0
	printf("Creating thread %d\n", HOUGH_LINE_THREAD);
	rc = pthread_create(&threads[HOUGH_LINE_THREAD], NULL, HoughLineFunc, (void *)HOUGH_LINE_THREAD);

	if (rc)
	{
		printf("ERROR; pthread_create() rc is %d\n", rc);
		perror(NULL);
		exit(-1);
	}

	gettimeofday(&timeNow, NULL);
	printf("Hough Line Thread spawned at %d sec, %d usec\n", timeNow.tv_sec, (double)timeNow.tv_usec);

	printf("Creating thread %d\n", HOUGH_CIRCLE_THREAD);
	rc = pthread_create(&threads[HOUGH_CIRCLE_THREAD], NULL, HoughCircleFunc, (void *)HOUGH_CIRCLE_THREAD);

	if (rc)
	{
		printf("ERROR; pthread_create() rc is %d\n", rc);
		perror(NULL);
		exit(-1);
	}

	gettimeofday(&timeNow, NULL);
	printf("Hough Circle Thread spawned at %d sec, %d usec\n", timeNow.tv_sec, (double)timeNow.tv_usec);
#endif

}

void *feature_extraction_fun(void* threadID)
{
    feat_ext_start_time = readTOD();
    
    CannyThreshold(0, 0);

    define_region_of_interest(canny_image);
 
    DrawHoughLines(roi_image);

    feat_ext_stop_time = readTOD();

    pthread_exit(NULL);
}


void Destroy_Threads(void)
{
    if(pthread_join(threads[FEATURE_EXTRACTION_THREAD], NULL) == 0)
        printf("Feature Extraction Thread done\n");
    else
        perror("Feature Extraction Thread");

#if 0
    if(pthread_join(threads[HOUGH_LINE_THREAD], NULL) == 0)
        printf("Hough Line Thread done\n");
    else
        perror("Hough Line Thread");

    if(pthread_join(threads[HOUGH_CIRCLE_THREAD], NULL) == 0)
        printf("Hough Circle Thread done\n");
    else
        perror("Hough Circle Thread");

#endif
}

void CannyThreshold(int, void*)
{
    //Mat inv_image;

    cvtColor(image, gray_image, CV_RGB2GRAY);

    /// Reduce noise with a kernel 3x3
    blur(gray_image, canny_image, Size(3,3) );

    /// Canny detector
    Canny(canny_image, canny_image, lowThreshold, lowThreshold*threshold_ratio, kernel_size );

    //threshold(canny_image, canny_image, 128,255,THRESH_BINARY_INV);

    /// Using Canny's output as a mask, we display our result
    result_image = Scalar::all(0);

    image.copyTo(result_image, canny_image);

    //namedWindow("Canny Display Window", WINDOW_AUTOSIZE );// Create a window for display.
    //imshow("Canny Display Window", result_image);
    
    //imshow(display_window_name, result_image);

}

void define_region_of_interest(Mat &in_image)
{
   Mat black(in_image.rows, in_image.cols, in_image.type(), cv::Scalar::all(0));
   Mat mask(in_image.rows, in_image.cols, CV_8UC1, cv::Scalar(0));

   Point P1(in_image.cols*0.4, in_image.rows*0.6);
   Point P2(in_image.cols*0.6, in_image.rows*0.6);
   Point P3(in_image.cols*0.9, in_image.rows*0.95);
   Point P4(in_image.cols*0.1, in_image.rows*0.95);

   vector< vector<Point> >  co_ordinates;
   co_ordinates.push_back(vector<Point>());
   co_ordinates[0].push_back(P1);
   co_ordinates[0].push_back(P2);
   co_ordinates[0].push_back(P3);
   co_ordinates[0].push_back(P4);
   drawContours(mask,co_ordinates,0, Scalar(255),CV_FILLED, 8 );
   
   //black.copyTo(in_image, mask);

   bitwise_and(in_image, mask, roi_image); 

   /// Show in a window
   //namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
   //imshow( "Contours", roi_image );

}

void DrawHoughLines(Mat &in_image)
{
    vector<Vec4i> lines;
    Mat mat_image(image);
    
    HoughLinesP(in_image, lines, 1, CV_PI/180, 20, 20, 300);

    for( size_t i = 0; i < lines.size(); i++ )
    {
	    Vec4i l = lines[i];
	    line(mat_image, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
    }

   //namedWindow( "Hough", CV_WINDOW_AUTOSIZE );
   //imshow( "Hough", mat_image );
   //imwrite("feat_ext_out.jpg", mat_image);

}

