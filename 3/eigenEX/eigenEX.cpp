#include <iostream>
using namespace std;

#include<ctime>

#include<Eigen/Core>

#include<Eigen/Dense>//for algbra calculations

using namespace Eigen;
#define MATRIX_SIZE 50


int main(int argc, char** argv){
    Matrix<float, 2,3> matrix_23;
    cout<<matrix_23<<endl;
    Vector3d v_3d1;
    Matrix<float, 3, 1> v_3d2;
    Matrix3d matrix_33 = Matrix3d::Zero();
    
    Matrix<double, Dynamic, Dynamic> matrix_dynamic;
    
    MatrixXd matrix_x;
    
    matrix_23<<1,2,3,4,5,6;
    cout<<"matrix_23: "<<matrix_23<<endl;
    
    v_3d1 << 3,2,1;
    v_3d2 << 4,5,6;
    
    Matrix<double, 2,1> result = matrix_23.cast<double>() * v_3d1;
    cout<<matrix_23<<"*"<<v_3d1<<"="<<result<<endl;
    
    matrix_33 = Matrix3d::Random();
    cout<<"random matrix_33: "<<matrix_33<<endl;
    cout<<"transpose: "<<matrix_33.transpose()<<endl;
    cout<<"sum: "<<matrix_33.sum()<<endl;
    cout<<"trace "<<matrix_33.trace()<<endl;
    cout<<"times "<<matrix_33*10<<endl;
    cout<<"inverse:"<<matrix_33.inverse()<<endl;
    cout<<"det: " <<matrix_33.determinant()<<endl;
    
    //eigen value and eigen Vector;
    SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);
    cout << "Eigenvalues = \n"<<eigen_solver.eigenvalues()<<endl;
    cout << "Eigenvactors = \n"<<eigen_solver.eigenvectors()<<endl;
    
    //find -1 of a matrix;
    Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN = MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    matrix_NN = matrix_NN * matrix_NN.transpose();
    Matrix<double, MATRIX_SIZE, 1> v_Nd = MatrixXd::Random(MATRIX_SIZE, 1);
    
    clock_t time_stt = clock();//start counting time_stt;
    
    Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
    cout<<"time for normal inverse is: "<<1000*(clock() - time_stt) / (double) CLOCKS_PER_SEC<<"ms"<<endl;
    cout<<"x = "<<x.transpose()<<endl;
    
    //QR method
    time_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout<<"time of QR mathod is: "<<1000*(clock() - time_stt) / (double) CLOCKS_PER_SEC<<"ms"<<endl;
    cout<<"x = "<<x.transpose()<<endl;
    
    time_stt = clock();
    x = matrix_NN.ldlt().solve(v_Nd);
    cout << "time of ldlt decomposition is "
    <<1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC <<"ms" <<endl;
    cout<<"x = "<<x.transpose() <<endl;
    
    return 0;
}
