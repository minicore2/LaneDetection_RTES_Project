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

#define USE_VIDEO

#define HRES 640        //Horizontal resolution i.e. width
#define VRES 480        //Vertical resolution i.e. height

#define NUM_ITERATIONS						1

#define NUM_THREADS                     	3
#define VIDEO_CAPTURE_THREAD            	1
#define FEATURE_EXTRACTION_THREAD       	2
#define LANE_DETECTION_THREAD           	3
//#define ALERT_SERVICE_THREAD            	4

#define VIDEO_CAPTURE_SOFT_DEADLINE_MS          190    /* Corresponding to an average frame rate of 5.85 fps + ~20 ms margin */
#define FEATURE_EXTRACTION_SOFT_DEADLINE_MS   	100    /* Corresponding to an average frame rate of 8.7 fps + ~20 ms margin */
#define LANE_DETECTION_SOFT_DEADLINE_MS       	170    /* Corresponding to an average frame rate of 6.5 fps + ~20 ms margin */
//#define ALERT_SERVICE_SOFT_DEADLINE_MS        	170    /* Corresponding to an average frame rate of 6.5 fps + ~20 ms margin */
//#define COMBINED_SOFT_DEADLINE_MS       	500

#define COMBINED_SOFT_DEADLINE_MS       	500

pthread_t threads[NUM_THREADS];
pthread_attr_t rt_sched_attr[NUM_THREADS];
int rt_max_prio, rt_min_prio;
struct sched_param rt_param[NUM_THREADS];
struct sched_param nrt_param;
struct timeval tv;
int temp=0;

double video_capt_start_time = 0, video_capt_stop_time = 0;
double feat_ext_start_time = 0, feat_ext_stop_time = 0;
double lane_detect_start_time = 0, lane_detect_stop_time = 0;
double video_proc_start_time = 0, video_proc_stop_time = 0;
double temp_time=0, prev_frame_time=0;
double ave_framedt=0.0, ave_frame_rate=0.0, fc=0.0, framedt=0.0;

int lowThreshold = 50;
int const max_lowThreshold = 100;
int kernel_size = 3;
int edgeThresh = 1;
int threshold_ratio = 3;

char str[200];

CvCapture* capture;

const char fileName[] = "../test_images/lane_detection_image.jpeg";
//const char fileName[] = "../test_images/lane_image_1.jpg";
//const char fileName[] = "../test_images/lane_image_2.jpg";
//const char fileName[] = "../test_images/lane_detection_image_2.bmp";
// Transform display window
char display_window_name[] = "Display Window";

#if defined(USE_VIDEO)
IplImage *image;
#else
Mat image;
#endif
Mat result_image, gray_image, canny_image, roi_image;

void CannyThreshold(int, void*);
void ApplyHoughtransform(int, void*, Mat &in_image);
void define_region_of_interest(Mat &in_image);
void DrawHoughLines(Mat &in_image);
void Create_Threads(void);
void Destroy_Threads(void);
void *feature_extraction_fun(void* threadID);
double readTOD(void);
void display_timestamp();
void *video_capture_fun(void* threadID);
void *feature_extraction_fun(void* threadID);
void *lane_detection_fun(void* threadID);

sem_t sem_video_capt;
sem_t sem_feat_ext;
sem_t sem_lane_detect;
sem_t sem_display_timestamp;

int main(void)
{
#if !defined(USE_VIDEO)
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
#else

    //namedWindow( display_window_name, CV_WINDOW_AUTOSIZE );

	/* Start capturing frames from camera */
	capture = (CvCapture *)cvCreateFileCapture("../test_images/pikes_peak.mp4");
	/* Set height property */
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, HRES);
	/* Set width property */
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, VRES);

	/* Initializing bianry semaphores */
    sem_init(&sem_video_capt, 0, 0);

	sem_init(&sem_feat_ext, 0, 0);

	/* Deliberately setting the initial value of this semaphore to 0, so that canny detector thread
	   is executed first, followed by hough line thread and hough line thread releases the
	   sem_hough_circle semaphore */
	sem_init(&sem_lane_detect, 0, 0);

	sem_init(&sem_display_timestamp, 0, 1);

	Create_Threads();

	video_proc_start_time = readTOD();
	
	sem_post(&sem_video_capt);
	
    video_proc_stop_time = readTOD();
	
 	printf("Video Process took %d ms for %d iteration(s)\n", 
		(int)(video_proc_stop_time - video_proc_start_time), NUM_ITERATIONS);

	Destroy_Threads();
	
 	/* Releases the CvCapture structure */
    cvReleaseCapture(&capture);

    /* Destroy created window */
    //cvDestroyWindow(display_window_name);

#endif

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

	printf("Creating thread %d\n", VIDEO_CAPTURE_THREAD);
	rc = pthread_create(&threads[VIDEO_CAPTURE_THREAD], NULL, video_capture_fun, (void *)VIDEO_CAPTURE_THREAD);

	if (rc)
	{
		printf("ERROR; pthread_create() rc is %d\n", rc);
		perror(NULL);
		exit(-1);
	}

	gettimeofday(&timeNow, NULL);
	printf("Video Capture thread spawned at %d sec, %d usec\n", timeNow.tv_sec, (double)timeNow.tv_usec);
	
	printf("Creating thread %d\n", FEATURE_EXTRACTION_THREAD);
	rc = pthread_create(&threads[FEATURE_EXTRACTION_THREAD], NULL, feature_extraction_fun, (void *)FEATURE_EXTRACTION_THREAD);

	if (rc)
	{
		printf("ERROR; pthread_create() rc is %d\n", rc);
		perror(NULL);
		exit(-1);
	}

	gettimeofday(&timeNow, NULL);
	printf("Feature Extraction thread spawned at %d sec, %d usec\n", timeNow.tv_sec, (double)timeNow.tv_usec);

	printf("Creating thread %d\n", LANE_DETECTION_THREAD);
	rc = pthread_create(&threads[LANE_DETECTION_THREAD], NULL, lane_detection_fun, (void *)LANE_DETECTION_THREAD);

	if (rc)
	{
		printf("ERROR; pthread_create() rc is %d\n", rc);
		perror(NULL);
		exit(-1);
	}

	gettimeofday(&timeNow, NULL);
	printf("Lane Detection thread spawned at %d sec, %d usec\n", timeNow.tv_sec, (double)timeNow.tv_usec);

}

void *video_capture_fun(void* threadID)
{
	while(temp < NUM_ITERATIONS)
	{
		video_capt_start_time = readTOD();

		sem_wait(&sem_video_capt);    

		temp_time = readTOD();
		display_timestamp();
		printf("Capturing Video Frame %d\n", temp+1);
		image = cvQueryFrame(capture);
		temp++;
	
		video_capt_stop_time = readTOD();

		/* Release the semaphore for feature extraction thread */
		sem_post(&sem_feat_ext);

		printf("Video Capture thread instance %d ran for %d ms\n", temp, (int)(video_capt_stop_time - video_capt_start_time));
	}

	pthread_exit(NULL);
}

void *feature_extraction_fun(void* threadID)
{
	while(temp < NUM_ITERATIONS)
	{
		feat_ext_start_time = readTOD();

		sem_wait(&sem_feat_ext);    

		CannyThreshold(0, 0);

		define_region_of_interest(canny_image);

		feat_ext_stop_time = readTOD();

		printf("Feature extraction thread instance %d ran for %d ms\n", temp, (int)(feat_ext_stop_time - feat_ext_start_time));
	
		/* Release the semaphore for lane detection thread */
		sem_post(&sem_lane_detect);
	}

	pthread_exit(NULL);
}

void *lane_detection_fun(void* threadID)
{
	while(temp < NUM_ITERATIONS)
	{
		lane_detect_start_time = readTOD();

		sem_wait(&sem_lane_detect);    

		DrawHoughLines(roi_image);

		lane_detect_stop_time = readTOD();

		printf("Lane Detection thread instance %d ran for %d ms\n", temp, (int)(lane_detect_stop_time - lane_detect_start_time));
	
		/* Release the semaphore for timestamp display function and video capture thread */	
		sem_post(&sem_video_capt);

		sem_post(&sem_display_timestamp);
	}

	pthread_exit(NULL);
}

void Destroy_Threads(void)
{
    if(pthread_join(threads[VIDEO_CAPTURE_THREAD], NULL) == 0)
        printf("Video Capture Thread done\n");
    else
        perror("Video Capture Thread");
    
	if(pthread_join(threads[FEATURE_EXTRACTION_THREAD], NULL) == 0)
        printf("Feature Extraction Thread done\n");
    else
        perror("Feature Extraction Thread");

    if(pthread_join(threads[LANE_DETECTION_THREAD], NULL) == 0)
        printf("Lane Detection Thread done\n");
    else
        perror("Lane Detection Thread");

}

void CannyThreshold(int, void*)
{
    //Mat inv_image;

	Mat canny_frame(image);
    cvtColor(canny_frame, gray_image, CV_RGB2GRAY);

    /// Reduce noise with a kernel 3x3
    blur(gray_image, canny_image, Size(3,3));

    /// Canny detector
    Canny(canny_image, canny_image, lowThreshold, lowThreshold*threshold_ratio, kernel_size );

    //threshold(canny_image, canny_image, 128,255,THRESH_BINARY_INV);

    /// Using Canny's output as a mask, we display our result
    result_image = Scalar::all(0);

    canny_frame.copyTo(result_image, canny_image);

    //namedWindow("Canny Display Window", WINDOW_AUTOSIZE );// Create a window for display.
    //imshow("Canny Display Window", result_image);
    
    //imshow(display_window_name, result_image);

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);

	sprintf(str, "Image_%llu.png", (unsigned long long)readTOD());   
	try 
	{
		printf("Writing to file : %s\n", str);
		imwrite(str, result_image, compression_params);
	}
	catch (runtime_error& ex) {
		fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
	}


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
  
#if 0 
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);

	sprintf(str, "Image_%llu.png", (unsigned long long)readTOD());   
	try 
	{
		printf("Writing to file : %s\n", str);
		imwrite(str, mat_image, compression_params);
	}
	catch (runtime_error& ex) {
		fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
	}
#endif	

}

