#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include "linefinder.h"

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
// Transform display window
char display_window_name[] = "Display Window";

Mat image;
Mat result_image, gray_image, canny_image;
Mat res;

void CannyThreshold(int, void*);
void ApplyHoughtransform(int, void*, Mat &in_image);
void define_region_of_interest(Mat &in_image);
void draw_lines(Mat &in_image);
void DrawHoughLines(Mat &in_image);

int crestCount = 0, frameSkip = 0;;

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

    
    //waitKey(0);

    DrawHoughLines(canny_image);


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
   
   black.copyTo(in_image, mask);

   bitwise_and(canny_image, mask, res); 

   /// Show in a window
   //namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
   //imshow( "Contours", res );

}

void DrawHoughLines(Mat &in_image)
{
#if 0

    vector<Vec4i> lines;
    Mat mat_image(in_image);
    
    HoughLinesP(canny_image, lines, 1, CV_PI/180, 50, 20, 10);

#if 0
    for( size_t i = 0; i < lines.size(); i++ )
    {
	    Vec4i l = lines[i];
	    line(mat_image, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
    }
#endif

   namedWindow( "Hough Transform", CV_WINDOW_AUTOSIZE );
   imshow( "Hough Transform", in_image );
#endif

   int houghVote = 200;
   int top = 0;
   int left = 0;
   int width = 800;
   int height = 600;

   vector<Vec2f> lines;
   if (houghVote < 1 or lines.size() > 2) { // we lost all lines. reset
	houghVote = 300;
   }
   else
   { 
	houghVote += 25;
   }

   while(lines.size() < 4 && houghVote > 0){
	HoughLines(in_image,lines,1,CV_PI/180, houghVote);
	houghVote -= 5;
   }
   
   cout << houghVote << "\n";
  
   define_region_of_interest(image);    
   
   Mat result(res.size(),CV_8U,Scalar(255));
   res.copyTo(result);

   // Draw the lines
   vector<Vec2f>::const_iterator it= lines.begin();
   Mat hough(res.size(),CV_8U,Scalar(0));
   while (it!=lines.end()) {

	   float rho= (*it)[0];   // first element is distance rho
	   float theta= (*it)[1]; // second element is angle theta

	   if ( (theta > 0.09 && theta < 1.48) || (theta < 3.14 && theta > 1.66) ) { // filter to remove vertical and horizontal lines

		   // point of intersection of the line with first row
		   Point pt1(rho/cos(theta),0);
		   // point of intersection of the line with last row
		   Point pt2((rho-result.rows*sin(theta))/cos(theta),result.rows);
		   // draw a line: Color = Scalar(R, G, B), thickness
		   line( result, pt1, pt2, Scalar(255,255,255), 1);
		   line( hough, pt1, pt2, Scalar(255,255,255), 1);
	   }

	   //std::cout << "line: (" << rho << "," << theta << ")\n";
	   ++it;
   }

   //namedWindow("Detected Lines with Hough");
   //imshow("Detected Lines with Hough",hough);

   LineFinder ld;
   // Set probabilistic Hough parameters
   ld.setLineLengthAndGap(10,60); // min accepted length and gap
   ld.setMinVote(15); // sit > 3 to get rid of "spiderweb"

   // Detect lines
   std::vector<Vec4i> li= ld.findLines(canny_image);
   Mat houghP(res.size(),CV_8U,Scalar(0));
   ld.setShift(0,0);
   ld.drawDetectedLines(houghP);
   std::cout << "First Hough" << endl; 

   // bitwise AND of the two hough images
   bitwise_and(houghP,hough,houghP);
   Mat houghPinv(res.size(),CV_8U,Scalar(0));
   Mat dst(res.size(),CV_8U,Scalar(0));
   threshold(houghP,houghPinv,150,255,THRESH_BINARY_INV); // threshold and invert to black lines
   
   //namedWindow("Detected Lines with Hough");
   //imshow("Detected Lines with Hough",houghP);
   
   ld.setLineLengthAndGap(5,2);
   ld.setMinVote(1);
   ld.setShift(top, left);

   // draw point on image where line intersection occurs
   int yShift = 25;
   int allowableFrameSkip = 5;
   ld.drawDetectedLines(image);
   cv::Point iPnt = ld.drawIntersectionPunto(image, 2);

   // track hill crest
   int gap = 20;
   cv::Point lptl(0, image.rows / 2 + yShift);
   cv::Point lptr(gap, image.rows / 2 + yShift);
   line(image, lptl, lptr, Scalar(255, 255, 255), 1);// left mid line

   cv::Point rptl(image.cols - gap, image.rows / 2 + yShift);
   cv::Point rptr(image.cols, image.rows / 2 + yShift);
   line(image, rptl, rptr, Scalar(255, 255, 255), 1);// right mid line

   cv::Point ulpt(0, image.rows / 2 - 50 + yShift);
   cv::Point urpt(image.cols, image.rows / 2 - 50 + yShift);
   //     line(image, ulpt, urpt, Scalar(255, 255, 255), 1);// upper line

   bool hillCrestFound = (iPnt.y < (image.rows / 2 + yShift)) && (iPnt.y > (image.rows / 2 - 50 + yShift));
   if(hillCrestFound) {
	   crestCount++;
	   frameSkip = 0;
   } else if(crestCount != 0 && frameSkip < allowableFrameSkip)
	   frameSkip++;
   else {
	   crestCount = 0;
	   frameSkip = 0;
   }

   cv::Point txtPt(image.cols / 2 - 31, image.rows / 2 - 140);
   if(crestCount > 3)
	   putText(image, "tracking", txtPt, FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255), 2, 8);

   std::stringstream stream;
   stream << "Lines Segments: " << lines.size();

   putText(image, stream.str(), Point(10,image.rows-10), 1, 0.8, Scalar(0,255,0),0);
   imshow(display_window_name, image);

}

void draw_lines(Mat &in_image, Vector<Vec4i> &line, int thickness)
{
}
