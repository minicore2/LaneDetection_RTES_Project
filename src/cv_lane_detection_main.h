#pragma once

#include <list>

#undef MIN
#undef MAX
#define MAX(a,b) ((a)<(b)?(b):(a))
#define MIN(a,b) ((a)>(b)?(b):(a)

#define GREEN CV_RGB(0,255,0)
#define RED CV_RGB(255,0,0)
#define BLUE CV_RGB(255,0,255)
#define PURPLE CV_RGB(255,0,255)

#define K_VARY_FACTOR 0.2f
#define B_VARY_FACTOR 20
#define MAX_LOST_FRAMES 30

#define USE_VIDEO 1
//#define ENABLE_DEBUG 1

class ExpMovingAverage {
    private:
        double alpha; // [0;1] less = more stable, more = less stable
        double oldValue;
        bool unset;
    public:
        ExpMovingAverage() {
            this->alpha = 0.2;
            unset = true;
        }

        void clear() {
            unset = true;
        }

        void add(double value) {
            if (unset) {
                oldValue = value;
                unset = false;
            }
            double newValue = oldValue + alpha * (value - oldValue);
            oldValue = newValue;
        }

        double get() {
            return oldValue;
        }
};


struct Lane {
    Lane(){}
    Lane(CvPoint a, CvPoint b, float angle, float kl, float bl): p0(a),p1(b),angle(angle),
    votes(0),visited(false),found(false),k(kl),b(bl) { }

    CvPoint p0, p1;
    int votes;
    bool visited, found;
    float angle, k, b;
};

struct Status {
    Status():reset(true),lost(0){}
    ExpMovingAverage k, b;
    bool reset;
    int lost;
};

enum{
    SCAN_STEP = 5,  // in pixels
    LINE_REJECT_DEGREES = 10, // in degrees
    BW_TRESHOLD = 250,  // edge response strength to recognize for 'WHITE'
    BORDERX = 10,  // px, skip this much from left & right borders
    MAX_RESPONSE_DIST = 5,  // px

    CANNY_MIN_TRESHOLD = 1,  // edge detector minimum hysteresis threshold
    CANNY_MAX_TRESHOLD = 100, // edge detector maximum hysteresis threshold

    HOUGH_TRESHOLD = 50,// line approval vote threshold
    HOUGH_MIN_LINE_LENGTH = 50,// remove lines shorter than this treshold
    HOUGH_MAX_LINE_GAP = 100,   // join lines to one with smaller than this gaps

    CAR_DETECT_LINES = 4,    // minimum lines for a region to pass validation as a 'CAR'
    CAR_H_LINE_LENGTH = 10,  // minimum horizontal line length from car body in px

    MAX_VEHICLE_SAMPLES = 30,      // max vehicle detection sampling history
    CAR_DETECT_POSITIVE_SAMPLES = MAX_VEHICLE_SAMPLES-2, // probability positive matches for valid car
    MAX_VEHICLE_NO_UPDATE_FREQ = 15 // remove car after this much no update frames
 };



Status laneR, laneL;

CvPoint2D32f sub(CvPoint2D32f b, CvPoint2D32f a) { return cvPoint2D32f(b.x-a.x, b.y-a.y); }
CvPoint2D32f mul(CvPoint2D32f b, CvPoint2D32f a) { return cvPoint2D32f(b.x*a.x, b.y*a.y); }
CvPoint2D32f add(CvPoint2D32f b, CvPoint2D32f a) { return cvPoint2D32f(b.x+a.x, b.y+a.y); }
CvPoint2D32f mul(CvPoint2D32f b, float t) { return cvPoint2D32f(b.x*t, b.y*t); }
float dot(CvPoint2D32f a, CvPoint2D32f b) { return (b.x*a.x + b.y*a.y); }
float dist(CvPoint2D32f v) { return sqrtf(v.x*v.x + v.y*v.y); }

CvPoint2D32f point_on_segment(CvPoint2D32f line0, CvPoint2D32f line1, CvPoint2D32f pt){
    CvPoint2D32f v = sub(pt, line0);
    CvPoint2D32f dir = sub(line1, line0);
    float len = dist(dir);
    float inv = 1.0f/(len+1e-6f);
    dir.x *= inv;
    dir.y *= inv;

    float t = dot(dir, v);
    if(t >= len) return line1;
    else if(t <= 0) return line0;

    return add(line0, mul(dir,t));
}

float dist2line(CvPoint2D32f line0, CvPoint2D32f line1, CvPoint2D32f pt){
    return dist(sub(point_on_segment(line0, line1, pt), pt));

}

void crop(IplImage* src,  IplImage* dest, CvRect rect); 
void processSide(std::vector<Lane> lanes, IplImage *edges, bool right); 
void processLanes(CvSeq* lines, IplImage* edges, IplImage* temp_frame);
void FindResponses(IplImage *img, int startX, int endX, int y, std::vector<int>& list);
unsigned char pixel(IplImage* img, int x, int y); 
