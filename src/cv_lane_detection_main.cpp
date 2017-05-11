#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <sched.h>
#include <unistd.h>
#include <iostream>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <math.h>

#include <sys/time.h>
#include <time.h>
#include <bits/stdc++.h>
#include "cv_lane_detection_main.h"

using namespace cv;
using namespace std;

#define HRES 640        //Horizontal resolution i.e. width
#define VRES 480        //Vertical resolution i.e. height

#define NUM_ITERATIONS						1

#define NUM_THREADS                     	3
#define VIDEO_CAPTURE_THREAD            	1
#define FEATURE_EXTRACTION_THREAD       	2
#define LANE_DETECTION_THREAD           	3
#define DISPLAY_THREAD           			4

#define VIDEO_CAPTURE_SOFT_DEADLINE_MS          190    /* Corresponding to an average frame rate of 5.85 fps + ~20 ms margin */
#define FEATURE_EXTRACTION_SOFT_DEADLINE_MS   	100    /* Corresponding to an average frame rate of 8.7 fps + ~20 ms margin */
#define LANE_DETECTION_SOFT_DEADLINE_MS       	170    /* Corresponding to an average frame rate of 6.5 fps + ~20 ms margin */
#define DISPLAY_SOFT_DEADLINE_MS       			100    /* Corresponding to an average frame rate of 6.5 fps + ~20 ms margin */
//#define COMBINED_SOFT_DEADLINE_MS       	500

#define COMBINED_SOFT_DEADLINE_MS       	600

pthread_t threads[NUM_THREADS];
pthread_attr_t rt_sched_attr[NUM_THREADS];
int rt_max_prio, rt_min_prio;
struct sched_param rt_param[NUM_THREADS];
struct sched_param nrt_param;
struct timeval tv;
int temp = 0;

double video_capt_start_time = 0, video_capt_stop_time = 0;
double feat_ext_start_time = 0, feat_ext_stop_time = 0;
double lane_detect_start_time = 0, lane_detect_stop_time = 0;
double video_proc_start_time = 0, video_proc_stop_time = 0;
double temp_time=0, prev_frame_time=0;
double ave_framedt=0.0, ave_frame_rate=0.0, fc=0.0, framedt=0.0;

char str[200];

// Transform display window
char display_window_name[] = "Display Window";

CvCapture *input_video;

int key_pressed = 0;
IplImage *frame = NULL;
IplImage *temp_frame;
IplImage *grey;
IplImage *edges;
IplImage *half_frame;

CvSize frame_size;;
CvMemStorage* houghStorage;

void Create_Threads(void);
void Destroy_Threads(void);
void *feature_extraction_fun(void* threadID);
double readTOD(void);
void display_timestamp();
void *video_capture_fun(void* threadID);
void *feature_extraction_fun(void* threadID);
void *lane_detection_fun(void* threadID);
void *display_thread_fun(void* threadID);

double start_init=0, start_capture, stop_capture, start_feature, stop_feature, start_lane, stop_lane;

sem_t sem_video_capt;
sem_t sem_feat_ext;
sem_t sem_lane_detect;
sem_t sem_display_thread;
sem_t sem_display_timestamp;

int main(void)
{
#if defined(USE_VIDEO)
	input_video = cvCreateFileCapture("../test_images/road.avi");
#else
	input_video = cvCaptureFromCAM(0);
#endif

    if (input_video == NULL) {
        fprintf(stderr, "Error: Can't open video\n");
        return -1;
    }

    CvSize video_size;
    video_size.height = (int) cvGetCaptureProperty(input_video, CV_CAP_PROP_FRAME_HEIGHT);
    video_size.width = (int) cvGetCaptureProperty(input_video, CV_CAP_PROP_FRAME_WIDTH);
    
    frame_size = cvSize(video_size.width, video_size.height/2);
    temp_frame = cvCreateImage(frame_size, IPL_DEPTH_8U, 3);
    grey = cvCreateImage(frame_size, IPL_DEPTH_8U, 1);
    edges = cvCreateImage(frame_size, IPL_DEPTH_8U, 1);
    half_frame = cvCreateImage(cvSize(video_size.width/2, video_size.height/2), IPL_DEPTH_8U, 3);

	houghStorage = cvCreateMemStorage(0);
	
	/* Initializing bianry semaphores */
    sem_init(&sem_video_capt, 0, 0);
	
	sem_init(&sem_feat_ext, 0, 0);

	/* Deliberately setting the initial value of this semaphore to 0, so that canny detector thread
	   is executed first, followed by hough line thread and hough line thread releases the
	   sem_hough_circle semaphore */
	sem_init(&sem_lane_detect, 0, 0);

	sem_init(&sem_display_timestamp, 0, 0);
	
	sem_init(&sem_display_thread, 0, 0);

	Create_Threads();

	video_proc_start_time = readTOD();
	start_init = readTOD();
	start_capture = readTOD();
	
#if defined(ENABLE_DEBUG)
	printf("Capture Service Requested at time stamp %lf msec\n",(double)(start_capture - start_init)*1000);
#endif
	sem_post(&sem_video_capt);
	
    video_proc_stop_time = readTOD();
	
#if defined(ENABLE_DEBUG)
 	//printf("Video Process took %d ms for %d iteration(s)\n", 
//		(int)(video_proc_stop_time - video_proc_start_time), NUM_ITERATIONS);
#endif

	Destroy_Threads();
	
    /* Destroy created window */
    //cvDestroyWindow(display_window_name);

    cvReleaseMemStorage(&houghStorage);
    
	cvReleaseImage(&grey);
    cvReleaseImage(&edges);
    cvReleaseImage(&temp_frame);
    cvReleaseImage(&half_frame);

    cvReleaseCapture(&input_video);

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
  //      ft = ft*1000;
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
#if defined(ENABLE_DEBUG)
   /* printf("Frame @ %u sec, %lu usec, dt=%5.2lf msec, avedt=%5.2lf msec, rate=%5.2lf fps, Jitter = %d ms\n",
           (unsigned)tv.tv_sec,
           (unsigned long)tv.tv_usec,
           framedt, ave_framedt, ave_frame_rate,
           (int)(COMBINED_SOFT_DEADLINE_MS - ave_framedt));
	*/
#endif
}

void Create_Threads(void)
{
	struct timeval timeNow;
	int rc;

	printf("Spawning all the required threads\n");

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
	
	printf("Creating thread %d\n", DISPLAY_THREAD);
	rc = pthread_create(&threads[DISPLAY_THREAD], NULL, display_thread_fun, (void *)DISPLAY_THREAD);

	if (rc)
	{
		printf("ERROR; pthread_create() rc is %d\n", rc);
		perror(NULL);
		exit(-1);
	}

	gettimeofday(&timeNow, NULL);
	printf("Display thread spawned at %d sec, %d usec\n", timeNow.tv_sec, (double)timeNow.tv_usec);

}

void *video_capture_fun(void* threadID)
{

		while (key_pressed != 27)
		{
			temp++;
			video_capt_start_time = readTOD();
			
			sem_wait(&sem_video_capt);    
			
			start_capture = readTOD();
#if defined(ENABLE_DEBUG)
			printf("Capture Service started execution at time stamp %lf msec\n",(double)(start_capture - start_init)*1000);
#endif
			
			temp_time = readTOD();
			display_timestamp();
			
			frame = cvQueryFrame(input_video);
			
			video_capt_stop_time = readTOD();

#if defined(ENABLE_DEBUG)
			printf("Capture Service Completed Execution at time stamp %lf msec\n",(double)(video_capt_stop_time - start_init)*1000);
			printf("Capture Service Execution Time = %lf msec\n",(double)(video_capt_stop_time - start_capture)*1000);
#endif
			/* Release the semaphore for feature extraction thread */
			start_feature = readTOD();

#if defined(ENABLE_DEBUG)
			printf("Feature Service requested execution at time stamp %lf msec\n",(double)(start_feature - start_init)*1000);
#endif
			
        	key_pressed = cvWaitKey(5);
			
			sem_post(&sem_feat_ext);
#if defined(ENABLE_DEBUG)
			printf("Video Capture thread instance %d ran for %d ms\n", temp, (int)(video_capt_stop_time - video_capt_start_time));
#endif		
		}
	
	pthread_exit(NULL);
}

void *feature_extraction_fun(void* threadID)
{
	while(key_pressed != 27)
	{
		feat_ext_start_time = readTOD();

		sem_wait(&sem_feat_ext);    
		start_feature = readTOD();
		
#if defined(ENABLE_DEBUG)
		printf("Feature Service started execution at time stamp %lf msec\n",(double)(start_feature - start_init)*1000);
#endif
		/* Perform downsampling step of Gaussian pyramid decomposition. First convolve the source
        ** image with the specified filter and then downsample the image by rejecting even rows
        ** and columns */
        cvPyrDown(frame, half_frame, CV_GAUSSIAN_5x5);

        /* we're interested only in road below horizont - so crop top image portion off */
        crop(frame, temp_frame, cvRect(0,frame_size.height,frame_size.width,frame_size.height));

        /* Convert to grayscale */
        cvCvtColor(temp_frame, grey, CV_BGR2GRAY);

        /* Perform gaussian blur using a 5x5 kernel */
        cvSmooth(grey, grey, CV_GAUSSIAN, 5, 5);

        /* Perform edge detection using canny edge detector */
        cvCanny(grey, edges, CANNY_MIN_TRESHOLD, CANNY_MAX_TRESHOLD);
		
		feat_ext_stop_time = readTOD();

#if defined(ENABLE_DEBUG)
		printf("Feature Service Completed Execution at time stamp %lf msec\n",(double)(feat_ext_stop_time - start_init)*1000);
		printf("Feature Service Execution Time = %lf msec\n",(double)(feat_ext_stop_time - start_feature)*1000);
#endif

		start_lane = readTOD();

#if defined(ENABLE_DEBUG)
		printf("Lane Detection Service requested execution at time stamp %lf msec\n",(double)(start_lane - start_init)*1000);
		//	printf("Feature extraction thread instance %d ran for %d ms\n", temp, (int)(feat_ext_stop_time - feat_ext_start_time));
#endif	
		/* Release the semaphore for lane detection thread */
		sem_post(&sem_lane_detect);

	}

	pthread_exit(NULL);
}

void *lane_detection_fun(void* threadID)
{
	
	while(key_pressed != 27)
	{
		lane_detect_start_time = readTOD();

		sem_wait(&sem_lane_detect);    
		start_lane = readTOD();
	
#if defined(ENABLE_DEBUG)
		printf("Lane Detection Service started execution at time stamp %lf msec\n",(double)(start_lane - start_init)*1000);
#endif

        /* Perform Hough transform to find lanes */ 
        double rho = 1;
        double theta = CV_PI/180;
        CvSeq* lines = cvHoughLines2(edges, houghStorage, CV_HOUGH_PROBABILISTIC, 
                rho, theta, HOUGH_TRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP);

        /* Compute the lanes */
        processLanes(lines, edges, temp_frame);

        // show middle line
        //cvLine(temp_frame, cvPoint(frame_size.width/2,0), 
        //cvPoint(frame_size.width/2,frame_size.height), CV_RGB(255, 255, 0), 1);

		lane_detect_stop_time = readTOD();

#if defined(ENABLE_DEBUG)
		printf("Lane Detection Service Completed Execution at time stamp %lf msec\n",(double)(lane_detect_stop_time - start_init)*1000);
        printf("Lane Detection Service Execution Time = %lf msec\n",(double)(lane_detect_stop_time - start_feature)*1000); 
	//	printf("Lane Detection thread instance %d ran for %d ms\n", temp, (int)(lane_detect_stop_time - lane_detect_start_time));
#endif		
		/* Release the semaphore for display thread */
		sem_post(&sem_display_thread);

	}

	pthread_exit(NULL);
}

void *display_thread_fun(void* threadID)
{

	while(key_pressed != 27)
	{
		sem_wait(&sem_display_thread);    
        
		cvShowImage("Grey", grey);
        cvShowImage("Edges", edges);
        cvShowImage("Color", temp_frame);

        cvMoveWindow("Grey", 0, 0); 
        cvMoveWindow("Edges", 0, frame_size.height+25);
        cvMoveWindow("Color", 0, 2*(frame_size.height+25));
	
		sem_post(&sem_display_timestamp);
		
		/* Release the semaphore for timestamp display function and video capture thread */	
		sem_post(&sem_video_capt);
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

    if(pthread_join(threads[DISPLAY_THREAD], NULL) == 0)
        printf("Display Thread done\n");
    else
        perror("Display Thread");
	
	printf("Destroyed all the spawned threads\n");

}

void crop(IplImage* src,  IplImage* dest, CvRect rect) 
{
    cvSetImageROI(src, rect); 
    cvCopy(src, dest); 
    cvResetImageROI(src); 
}

void processSide(std::vector<Lane> lanes, IplImage *edges, bool right) 
{
    Status* side = right ? &laneR : &laneL;

    /* response search */
    int w = edges->width;
    int h = edges->height;
    const int BEGINY = 0;
    const int ENDY = h-1;
    const int ENDX = right ? (w-BORDERX) : BORDERX;
    unsigned char* ptr = (unsigned char*)edges->imageData;

	int midx = w/2;
	int midy = edges->height/2;	
		
    /* show responses */
    int* votes = new int[lanes.size()];
    for(int i=0; i<lanes.size(); i++) votes[i++] = 0;

    for(int y=ENDY; y>=BEGINY; y-=SCAN_STEP) {
        std::vector<int> rsp;
        FindResponses(edges, midx, ENDX, y, rsp);

        if (rsp.size() > 0) {
            int response_x = rsp[0]; // use first reponse (closest to screen center)

            float dmin = 9999999;
            float xmin = 9999999;
            int match = -1;
            for (int j=0; j<lanes.size(); j++) {
                /* compute response point distance to current line */
                float d = dist2line(
                        cvPoint2D32f(lanes[j].p0.x, lanes[j].p0.y), 
                        cvPoint2D32f(lanes[j].p1.x, lanes[j].p1.y), 
                        cvPoint2D32f(response_x, y));

                /* point on line at current y line */
                int xline = (y - lanes[j].b) / lanes[j].k;
                int dist_mid = abs(midx - xline); // distance to midpoint

                /* pick the best closest match to line & to screen center */
                if (match == -1 || (d <= dmin && dist_mid < xmin)) {
                    dmin = d;
                    match = j;
                    xmin = dist_mid;
                    break;
                }
            }

            /* vote for each line */
            if (match != -1) {
                votes[match] += 1;
            }
        }
    }

    int bestMatch = -1;
    int mini = 9999999;
    for (int i=0; i<lanes.size(); i++) {
        int xline = (midy - lanes[i].b) / lanes[i].k;
        int dist = abs(midx - xline); // distance to midpoint

        if (bestMatch == -1 || (votes[i] > votes[bestMatch] && dist < mini)) {
            bestMatch = i;
            mini = dist;
        }
    }

    if (bestMatch != -1) {
        Lane* best = &lanes[bestMatch];
        float k_diff = fabs(best->k - side->k.get());
        float b_diff = fabs(best->b - side->b.get());

        bool update_ok = (k_diff <= K_VARY_FACTOR && b_diff <= B_VARY_FACTOR) || side->reset;

        //printf("side: %s, k vary: %.4f, b vary: %.4f, lost: %s\n", 
        //        (right?"RIGHT":"LEFT"), k_diff, b_diff, (update_ok?"no":"yes"));

        if (update_ok) 
        {
            /* update is in valid bounds */
            side->k.add(best->k);
            side->b.add(best->b);
            side->reset = false;
            side->lost = 0;
        } 
        else 
        {
            /* can't update, lanes flicker periodically, start counter for partial reset! */
            side->lost++;
            if (side->lost >= MAX_LOST_FRAMES && !side->reset) {
                side->reset = true;
            }
        }

    } 
    else 
    {
        //printf("no lanes detected - lane tracking lost! counter increased\n");
        side->lost++;
        if (side->lost >= MAX_LOST_FRAMES && !side->reset) {
            /* do full reset when lost for more than N frames */ 
            side->reset = true;
            side->k.clear();
            side->b.clear();
        }
    }

    delete[] votes;
}

void processLanes(CvSeq* lines, IplImage* edges, IplImage* temp_frame)
{
    /* classify lines to left/right side */ 
    std::vector<Lane> left, right;
    static std::vector<Lane> left_initial, right_initial;
	CvPoint midpoint1, midpoint2;

    for(int i = 0; i < lines->total; i++ )
    {
        CvPoint* Point = (CvPoint*)cvGetSeqElem(lines,i);
        int dx = Point[1].x - Point[0].x;
        int dy = Point[1].y - Point[0].y;
        float angle = atan2f(dy, dx) * 180/CV_PI;

        if (fabs(angle) <= LINE_REJECT_DEGREES) 
		{ 
			/* reject near horizontal lines */
            continue;
        }

        /* assume that vanishing point is close to the image horizontal center
           calculate line parameters: y = kx + b; */ 
        dx = (dx == 0) ? 1 : dx; /* prevent division by zero error */  
        float k = dy/(float)dx;
        float b = Point[0].y - k*Point[0].x;

        /* assign lane's side based by its midpoint position */
        int midx = (Point[0].x + Point[1].x) / 2;
        if (midx < temp_frame->width/2) 
		{
            left.push_back(Lane(Point[0], Point[1], angle, k, b));
        } 
		else if (midx > temp_frame->width/2) 
		{
            right.push_back(Lane(Point[0], Point[1], angle, k, b));
        }

    }

    /* show Hough lines */ 
    for(int i=0; i<right.size(); i++) 
	{
        cvLine(temp_frame, right[i].p0, right[i].p1, CV_RGB(0, 0, 255), 2);
	}

    for(int i=0; i<left.size(); i++) 
	{
        cvLine(temp_frame, left[i].p0, left[i].p1, CV_RGB(255, 0, 0), 2);
    }
    
    processSide(left, edges, false);
    processSide(right, edges, true);

	/* show computed lanes */ 
    int x = temp_frame->width * 0.55f;
    int x2 = temp_frame->width;

	cvLine(temp_frame, cvPoint(x, laneR.k.get()*x + laneR.b.get()), 
					cvPoint(x2, laneR.k.get() * x2 + laneR.b.get()), CV_RGB(255, 0, 255), 2);
	
	x = temp_frame->width * 0;
	x2 = temp_frame->width * 0.45f;

	cvLine(temp_frame, cvPoint(x, laneL.k.get()*x + laneL.b.get()), 
					cvPoint(x2, laneL.k.get() * x2 + laneL.b.get()), CV_RGB(255, 0, 255), 2);

}

void FindResponses(IplImage *img, int startX, int endX, int y, std::vector<int>& list)
{
    /* scan for single response */
    const int row = y * img->width * img->nChannels;
    unsigned char* ptr = (unsigned char*)img->imageData;

    int step = (endX < startX) ? -1: 1;
    int range = (endX > startX) ? endX-startX+1 : startX-endX+1;

    for(int x = startX; range>0; x += step, range--)
    {
        if(ptr[row + x] <= BW_TRESHOLD) 
        {
            /* skip black: loop until white pixels show up */
            continue;
        }

        /* first response found */
        int idx = x + step;

        /* skip same response(white) pixels */
        while(range > 0 && ptr[row+idx] > BW_TRESHOLD){
            idx += step;
            range--;
        }

        /* reached black again */
        if(ptr[row+idx] <= BW_TRESHOLD) {
            list.push_back(x);
        }

        /* begin from new position */
        x = idx;
    }
}

unsigned char pixel(IplImage* img, int x, int y) 
{
    return (unsigned char)img->imageData[(y*img->width+x)*img->nChannels];
}
