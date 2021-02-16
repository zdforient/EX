#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <sophus/se3.hpp>
#include <chrono>

using namespace std;
using namespace cv;

void find_feature_matches(const Mat &img_1, const Mat &img_2, std::vector<KeyPoint> &keypoints_1, std::vector<KeyPoint> &keypoints_2, std::vector<DMatch> &matches);

Point2d pixel12cam(const Point2d &p, const Mat &k);


void bundleAdjustment(const vector<Point3f> &pts1, const vector<Point3f> &pts2, Mat &R, Mat &t);

void pose_estimation_3d3d(const vector<Point3f> &pts1, const vector<Point3f> &pts2, Mat &R, Mat &t);

class VertexPose:public g2o::BaseVertex<6, Sophus::SE3d>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    virtual void setToOriginImpl() override{
        _estimate = Sophus::SE3d();
    }
    
    virtual void oplusImpl(const double *update) override{
        Eigen::Matrix<double ,6,1> update_eigen;
        update_eigen<<update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }

    virtual bool read(istream &in) override{};
    virtual bool write(ostream &out) const override{};
};

class EdgeProjectRGBD: public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPose>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    EdgeProjectRGBD(const Eigen::Vector3d &pos) : _pos3d(pos){}
    
    virtual void computeError() override{
        const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
        _error = _measurement - v->estimate() * _pos3d;
    }
    
    virtual void linearizeOplus() override{
        const VertexPose * v = static_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d xyz_trans = T * _pos3d;
        _jacobianOplusXi.block<3,3>(0,0) = -Eigen::Matrix3d::Identity();
        _jacobianOplusXi.block<3,3>(0,3) = Sophus::SO3d::hat(xyz_trans);
    }
    virtual bool read(istream &in) override{}
    virtual bool write(ostream &out) const override{}
    
private:
    Eigen::Vector3d _pos3d;
};


int main(int argc, char ** argv){
    string s1,s2,s3,s4;
    if(argc != 5){
        s1 = "./../1.png";
        s2 = "./../2.png";
        s3 = "./../1_depth.png";
        s4 = "./../2_depth.png";
    }
    else{
        s1 = argv[1];
        s2 = argv[2];
        s3 = argv[3];
        s4 = argv[4];
    }
    Mat img_1 = imread(s1, CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(s2, CV_LOAD_IMAGE_COLOR);
    assert(img_1.data && img_2.data &&"cannot load images\n");
    
    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "found "<<matches.size() << "matches"<<endl;
    
    Mat depth1 = imread(s3, CV_LOAD_IMAGE_UNCHANGED);
    Mat depth2 = imread(s4, CV_LOAD_IMAGE_UNCHANGED);
    Mat K = (Mat_<double>(3,3)<<520.9, 0, 325.1, 0, 521.0, 249.7, 0,0,1);
    
    vector<Point3f> pts1, pts2;
    
    for(DMatch m:matches){
        ushort d1 = depth1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        ushort d2 = depth2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
        if(!d1 || !d2)
            continue;
        float dd1 = d1 / 5000.0;
        float dd2 = d2 / 5000.0;
        Point2d p1 = pixel12cam(keypoints_1[m.queryIdx].pt, K);
        Point2d p2 = pixel12cam(keypoints_2[m.trainIdx].pt, K);
        
        pts1.push_back(Point3f(p1.x * dd1, p1.y * dd1, dd1));
        pts2.push_back(Point3f(p2.x * dd1, p2.y * dd2, dd2));
    }
    
    cout<<"3d-3d pairs: "<<pts1.size()<<endl;
    
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    Mat Re,te, R, t;
    pose_estimation_3d3d(pts1, pts2, Re, te);
    
    cout<<"ICP via SVD results:"<<endl;    
    cout<<"R = "<<endl<<Re<<endl;
    cout<<"t = "<<endl<<te<<endl;
    cout<<"R_inv = "<<Re.t()<<endl;
    cout<<"t_inv = "<<-Re.t() * te<<endl;
    
    cout << "calling bundle adjustment by Leven Berg"<<endl;
    
    
    bundleAdjustment(pts1, pts2, R,t);
    
    //verify p1 = R* p2 + t;
    for(int i = 0; i < 5; i++){
        cout<<"p1 = "<<pts1[i]<<endl;
        cout<<"p2 = "<<pts2[i]<<endl;
        cout<<"R*p2+t = "<<R * (Mat_<double>(3,1)<<pts2[i].x, pts2[i].y, pts2[i].z) + t<<endl;
        cout<<"Re*p2+te = "<<Re * (Mat_<double>(3,1)<<pts2[i].x, pts2[i].y, pts2[i].z) + te<<endl;
        cout<<endl;
    }
    
    return 0;
}
        
void find_feature_matches(const cv::Mat& img_1, const cv::Mat& img_2, std::vector<KeyPoint>& keypoints_1, std::vector<KeyPoint>& keypoints_2, std::vector<DMatch>& matches){
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);
    
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);
    
    vector<DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);
    double min_dist = 10000, max_dist = 0;
    
    for(int i = 0; i < descriptors_1.rows; i++){
        double dist = match[i].distance;
        if(dist < min_dist)
            min_dist = dist;
        if(dist > max_dist)
            max_dist = dist;
    }
    
    printf("--Max dist : %f \n", max_dist);
    printf("--Min dist : %f \n", min_dist);
    
    for(int i = 0; i < descriptors_1.rows; i++){
        if(match[i].distance <= max(2* min_dist, 30.0)){
            matches.push_back(match[i]);
        }
    }
    
}

Point2d pixel12cam(const cv::Point2d& p, const cv::Mat& k){
    return Point2d((p.x - k.at<double>(0,2)) / k.at<double>(0,0),
                   (p.y - k.at<double>(1,2)) / k.at<double>(1,1));
}

void pose_estimation_3d3d(const vector<Point3f> &pts1, const vector<Point3f> &pts2, Mat &R, Mat &t){
    Point3f p1, p2;
    int N = pts1.size();
    for(int i = 0; i < N; i++){
        p1 += pts1[i];
        p2 += pts2[i];
    }
    
    p1 = Point3f(Vec3f(p1) / N);
    p2 = Point3f(Vec3f(p2) / N);
    
    vector<Point3f> q1(N), q2(N);
    
    for(int i = 0; i < N; i++){
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }
    
    //compute q1*q2^T;
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for(int i = 0; i < N; i++){
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }
    
    //SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    
    cout<<"U=" <<U<<endl;
    cout<<"V=" <<V<<endl;
    
    Eigen::Matrix3d R_ = U * (V.transpose());
    if(R_.determinant() < 0){
        R_ = - R_;
    }
    
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);
    
    R = (Mat_<double>(3,3)<<R_(0,0),R_(0,1),R_(0,2),R_(1,0),R_(1,1),R_(1,2),R_(2,0),R_(2,1),R_(2,2));
    t = (Mat_<double>(3,1) << t_(0,0), t_(1,0), t_(2,0));
    
}


void bundleAdjustment(const vector<Point3f> &pts1, const vector<Point3f> &pts2, Mat &R, Mat &t){
    //estiblish graph optimization;
    typedef g2o::BlockSolverX BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
    
    auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);
    
    VertexPose * vertex_pose = new VertexPose();
    vertex_pose->setId(0);
    vertex_pose->setEstimate(Sophus::SE3d());
    optimizer.addVertex(vertex_pose);
    
    for(int i = 0; i < pts1.size(); i++){
        EdgeProjectRGBD *edge = new EdgeProjectRGBD(Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));
        edge->setVertex(0, vertex_pose);
        edge->setMeasurement(Eigen::Vector3d(pts1[i].x, pts1[i].y, pts1[i].z));
        edge->setInformation(Eigen::Matrix3d::Identity());
        optimizer.addEdge(edge);
    }
    
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout<<"optimization costs time by g2o::\n"<<time_used.count()<<" seconds"<<endl;
    cout<<"pose estimated by g2o = \n" << vertex_pose->estimate().matrix()<<endl;
    
    
    Eigen::Matrix3d R_ = vertex_pose->estimate().rotationMatrix();
    Eigen::Vector3d t_ = vertex_pose->estimate().translation();
    
    R = (Mat_<double>(3,3)<<R_(0,0),R_(0,1),R_(0,2),R_(1,0),R_(1,1),R_(1,2),R_(2,0),R_(2,1),R_(2,2));
    t = (Mat_<double>(3,1) << t_(0,0), t_(1,0), t_(2,0));
}

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
