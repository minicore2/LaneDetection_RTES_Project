#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

#define HRES 640
#define VRES 480

int lowThreshold = 30;
int const max_lowThreshold = 100;
int kernel_size = 3;
int edgeThresh = 1;
int threshold_ratio = 3;

const char fileName[] = "./lane_detection_image.jpeg";
//const char fileName[] = "./lane_image_1.jpg";
//const char fileName[] = "./lane_image_2.jpg";
//const char fileName[] = "./lane_detection_image_2.bmp";
// Transform display window
char display_window_name[] = "Display Window";

Mat image;
Mat result_image, gray_image, canny_image, roi_image;

void CannyThreshold(int, void*);
void ApplyHoughtransform(int, void*, Mat &in_image);
void define_region_of_interest(Mat &in_image);
void draw_lines(Mat &in_image);
void DrawHoughLines(Mat &in_image);

int main(void)
{
    /* Using a reference image and working with that for now */
    image = imread(fileName, CV_LOAD_IMAGE_COLOR);

    if(!image.data)                              // Check for invalid input
    {
        cout << "could not open " << fileName << endl;
        return -1;
    }

    namedWindow(display_window_name, WINDOW_AUTOSIZE );// Create a window for display.
    imshow(display_window_name, image );                   // Show our image inside it.

    waitKey(0);                                          // Wait for a keystroke in the window

    CannyThreshold(0, 0);

    waitKey(0);

    define_region_of_interest(canny_image);
 
    waitKey(0);

    //DrawHoughLines(canny_image);
    DrawHoughLines(roi_image);

    waitKey(0);
    
    //draw_lines(image);

    waitKey(0);

    return 0;
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
   namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
   imshow( "Contours", roi_image );

}

void DrawHoughLines(Mat &in_image)
{
    vector<Vec4i> lines;
    Mat mat_image(image);
    
    HoughLinesP(in_image, lines, 1, CV_PI/180, 20, 20, 10);

    for( size_t i = 0; i < lines.size(); i++ )
    {
	    Vec4i l = lines[i];
	    line(mat_image, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
    }

   namedWindow( "Hough", CV_WINDOW_AUTOSIZE );
   imshow( "Hough", mat_image );

#if 0
    Mat dst;

    dst = Scalar::all(0);

    Mat addweight;
    image.copyTo( dst); // copy part of src image according the canny output, canny is used as mask
    cvtColor(res, res, CV_GRAY2BGR); // convert canny image to bgr
    addWeighted( image, 0.5, res, 0.5, 0.0, addweight); // blend src image with canny image
    image += res; // add src image with canny image

    imshow("addition", image );
    imshow("copyTo", dst );
    imshow("addwweighted", addweight );
    waitKey(0);
#endif
}

void draw_lines(Mat &in_image, Vector<Vec4i> &line, int thickness)
{
}
