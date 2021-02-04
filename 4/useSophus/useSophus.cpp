#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

using namespace std;
using namespace Eigen;

//the script is used to describe ways to use Sophus

int main(int argv, char **argc){
    //90 degree along Z direction
    Matrix3d R = AngleAxisd(M_PI / 2, Vector3d(0,0,1)).toRotationMatrix();
    //quaternion
    Quaterniond q(R);
    Sophus::SO3d SO3_R(R);
    Sophus::SO3d SO3_q(q);
    cout<<"SO(3) from matrix:\n" << SO3_R.matrix() << endl;
    cout<<"SO(3) from quaternion:\n" << SO3_q.matrix() << endl;
    cout<<"they should be equal."<<endl;
    
    Vector3d so3 = SO3_R.log();
    cout << "so3 = "<<so3.transpose()<<endl;//lie algbra of SO(3)
    cout << "so3 hat = \n" << Sophus::SO3d::hat(so3) << endl;//anti symentry
    cout << "so3 har vee = "<<Sophus::SO3d::vee(Sophus::SO3d::hat(so3)).transpose()<<endl;
    
    Vector3d update_so3(1e-4,0,0);//update small value
    Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3) * SO3_R;
    cout << "SO3 updated = \n" << SO3_updated.matrix()<<endl;
    
    cout<<"****************"<<endl;
    
    Vector3d t(1,0,0); // shift along x axis
    Sophus::SE3d SE3_RT(R, t);
    Sophus::SE3d SE3_qt(q, t);
    cout << "SE3 from R,t = \n" << SE3_RT.matrix() << endl;
    cout << "SE3 from Q,t = \n" << SE3_qt.matrix() << endl;
    
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    Vector6d se3 = SE3_RT.log();
    cout << "se3 = " << se3.transpose() << endl;
    
    cout << "se3 hat = \n" << Sophus::SE3d::hat(se3) << endl;
    cout << "se3 hat vee = " << Sophus::SE3d::vee(Sophus::SE3d::hat(se3)).transpose() << endl;
    
    Vector6d update_se3;
    update_se3.setZero();
    update_se3(0,0) = 1e-4;
    Sophus::SE3d SE3_updated = Sophus::SE3d::exp(update_se3) * SE3_RT;
    cout << "SE3 updated = " << endl << SE3_updated.matrix() << endl;
    return 0;
}
