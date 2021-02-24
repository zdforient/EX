#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

string file1 = "./../LK1.png";
string file2 = "./../LK2.png";

class OpticalFlowTracker{
public:
    OpticalFlowTracker(
        const Mat &img1_,
        const Mat &img2_,
        const vector<KeyPoint> &kp1_,
        vector<KeyPoint> &kp2_,
        vector<bool> &success_,
        bool inverse_ = true,
        bool has_initial_ = false) : 
        img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), inverse(inverse_),
        has_initial(has_initial_){}
    
    void calculateOpticalFlow(const Range &range);
    
private:
    const Mat &img1;
    const Mat &img2;
    const vector<KeyPoint> &kp1;
    vector<KeyPoint> &kp2;
    vector<bool> &success;
    bool inverse = true;
    bool has_initial = false;
};

/*****
 * signle level optical flow tracker
 * img1:    first image
 * img2:    second image
 * kp1:     keypoints in img1
 * kp2:     keypoints in img2
 * success: true if kp is tracked
 * inverse use inverse formulation
 * ****/

void OpticalFlowSingleLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2_,
    vector<bool> &success_,
    bool inverse_ = true,
    bool has_initial_ = false);


void OpticalFlowMultiLevel(
        const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2_,
    vector<bool> &success_,
    bool inverse_ = false
);

inline float GetPixelValue(const Mat &img, float x, float y){
    //boundary check
    if(x < 0) x = 0;
    if(y < 0) y = 0;
    if(x > img.cols - 1) x = img.cols - 1;
    if(y > img.rows - 1) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] + 
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] + 
        xx * yy * data[img.step + 1]
           );
}

int main(int argc, char ** argv){
    
}




























