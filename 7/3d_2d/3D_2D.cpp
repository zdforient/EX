#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <sophus/se3.hpp>
#include <chrono>

using namespace std;
using namespace cv;

void find_feature_matches(const Mat &img_1, const Mat &img_2, std::vector<KeyPoint> &keypoints_1, std::vector<KeyPoint> &keypoints_2, std::vector<DMatch> &matches);
Point2d pixel12cam(const Point2d &p, const Mat &k);

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

void bundleAdjustmentG2O(const VecVector3d &points_3d, const VecVector2d &points_2d, const Mat &K, Sophus::SE3d &pose);

//gauss-optimization_algorithm_gauss_newton
void bundleAdjustmentGaussNewton(const VecVector3d &points_3d, const VecVector2d &points_2d, const Mat &K, Sophus::SE3d &pose);

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
    
    Mat d1 = imread(s3, CV_LOAD_IMAGE_UNCHANGED);
    Mat K = (Mat_<double>(3,3)<<520.9, 0, 325.1, 0, 521.0, 249.7, 0,0,1);
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;
    
    for(DMatch m:matches){
        ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        if(!d)
            continue;
        float dd = d/5000.0;
        Point2d p1 = pixel12cam(keypoints_1[m.queryIdx].pt, K);
        pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
        pts_2d.push_back(keypoints_2[m.trainIdx].pt);
    }
    
    cout<<"3d-2d pairs: "<<pts_3d.size()<<endl;
    
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    Mat r,t;
    solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false);
    Mat R;
    cv::Rodrigues(r,R);//r is rotation matrix vector format, to matrix format
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout<<"solve pnp in opencv solution time: " << time_used.count() << "seconds."<<endl;
    
    cout<<"R = "<<endl<<R<<endl;
    cout<<"t = "<<endl<<t<<endl;
    
    VecVector3d pts_3d_eigen;
    VecVector2d pts_2d_eigen;
    
    for(size_t i = 0; i < pts_3d.size(); i++){
        pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
        pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    }
    cout<<"calling bundle adjectment by gauss newton"<<endl;
    Sophus::SE3d pose_gn;
    t1 = chrono::steady_clock::now();
    bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout<<"solve pnp by gauss newton cost time: "<<time_used.count()<<" seconds"<<endl;
    
    cout<<"calling bundle adjustment by g2o"<<endl;
    Sophus::SE3d pose_g2o;
    t1 = chrono::steady_clock::now();
    bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout<<"colve pnp by g2o cost time: "<<time_used.count()<<" seconds"<<endl;
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

void bundleAdjustmentGaussNewton(const VecVector3d &points_3d, const VecVector2d &points_2d, const Mat &K, Sophus::SE3d &pose){
    typedef Eigen::Matrix<double, 6,1> Vector6d;
    const int iterations = 10;
    double cost = 0, lastCost = 0;
    double fx = K.at<double>(0,0), fy = K.at<double>(1,1);
    double cx = K.at<double>(0,2), cy = K.at<double>(1,2);
    
    for(int iter = 0; iter < iterations; iter++){
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6,6>::Zero();
        Vector6d b = Vector6d::Zero();
        cost = 0;
        for(int i = 0; i < points_3d.size(); i++){
            Eigen::Vector3d pc = pose * points_3d[i];
            double inv_z = 1.0 / pc[2];
            double inv_z2 = inv_z * inv_z;
            Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);
            
            Eigen::Vector2d e = points_2d[i] - proj;
            cost += e.squaredNorm();
            Eigen::Matrix<double, 2,6> J;
            J <<
            -fx*inv_z,
            0,
            fx*pc[0]*inv_z2,
            fx * pc[0] * pc[1] * inv_z2,
            -fx - fx * pc[0] * pc[0] * inv_z2,
            fx * pc[1] * inv_z,
            0,
            -fy * inv_z,
            fy * pc[1] * inv_z2,
            fy + fy * pc[1] * pc[1] * inv_z2,
            -fy * pc[0] * pc[1] * inv_z2,
            -fy * pc[0] * inv_z;
            H += J.transpose() * J;
            b += -J.transpose() * e;
        }
        Vector6d dx;
        dx = H.ldlt().solve(b);
        
        if(isnan(dx[0])){
            cout<<"result of gaussian newton is nan!"<<endl;
            break;
        }
        if(iter > 0 && cost >= lastCost){
            cout<<"cost: "<<cost <<", last cost: "<<lastCost<<endl;
            break;
        }
        pose = Sophus::SE3d::exp(dx) * pose;
        lastCost = cost;
        cout<<"Iteration "<<iter<<" cost = "<<std::setprecision(12)<<cost<<endl;
        if(dx.norm() < 1e-16){
            //converage
            break;
        }
    }
    cout << "pose by g-n: \n"<<pose.matrix()<<endl;
}

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

class EdgeProjection: public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K) : _pos3d(pos), _K(K){}
    
    virtual void computeError() override{
        const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_pixel = _K * (T * _pos3d);
        pos_pixel /= pos_pixel[2];
        _error = _measurement - pos_pixel.head<2>();
    }
    
    virtual void linearizeOplus() override{
        const VertexPose * v = static_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_cam = T * _pos3d;
        double fx = _K(0,0);
        double fy = _K(1,1);
        double cx = _K(0,2), cy = _K(1,2);
        double X = pos_cam[0], Y = pos_cam[1], Z = pos_cam[2];
        double Z2 = Z * Z;
        _jacobianOplusXi<<-fx / Z, 0, fx * X / Z2, fx * X * Y / Z2, -fx - fx* X *X /Z2, fx*Y/Z,
        0, -fy / Z, fy * Y / Z2, fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
    }
    virtual bool read(istream &in) override{}
    virtual bool write(ostream &out) const override{}
    
private:
    Eigen::Vector3d _pos3d;
    Eigen::Matrix3d _K;
};

void bundleAdjustmentG2O(const VecVector3d &points_3d, const VecVector2d &points_2d, const Mat &K, Sophus::SE3d &pose){
    //estiblish graph optimization;
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
    
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);
    
    VertexPose * vertex_pose = new VertexPose();
    vertex_pose->setId(0);
    vertex_pose->setEstimate(Sophus::SE3d());
    optimizer.addVertex(vertex_pose);
    
    Eigen::Matrix3d K_eigen;
    vector<double> v_temp;
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            v_temp.push_back(K.at<double>(i,j));
        }
    }
    K_eigen<<v_temp[0],v_temp[1],v_temp[2],v_temp[3],v_temp[4],v_temp[5],v_temp[6],v_temp[7],v_temp[8];
    
    int index = 1;
    for(int i = 0; i < points_2d.size(); i++){
        auto p2d = points_2d[i];
        auto p3d = points_3d[i];
        EdgeProjection *edge = new EdgeProjection(p3d, K_eigen);
        edge->setId(index);
        edge->setVertex(0, vertex_pose);
        edge->setMeasurement(p2d);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        index++;
    }
    
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout<<"optimization costs time by g2o::\n"<<time_used.count()<<" seconds"<<endl;
    cout<<"pose estimated by g2o = \n" << vertex_pose->estimate().matrix()<<endl;
    pose = vertex_pose->estimate();
}

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
