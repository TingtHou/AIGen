#pragma once
#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#include <iostream>
class LinearRegression
{
public:
	~LinearRegression();
//	LinearRegression(float ** y, float ** x, bool intercept, int yrows, int xrows, int xcols);
//	LinearRegression(float * y, float ** x, bool intercept, int yrows, int xrows, int xcols);
	LinearRegression(Eigen::VectorXf &Y, Eigen::MatrixXf &X, bool intercept);
	Eigen::VectorXf GetB();
	void MLE();
	float GetMSE();
private:
 	Eigen::VectorXf Matrix_Y;
// 	Eigen::VectorXf P;
 	Eigen::MatrixXf X;
	Eigen::MatrixXf Matrix_X;
// 	Eigen::MatrixXf Fisher;
// 	Eigen::VectorXf T_statistic;
 	Eigen::VectorXf B;
 	Eigen::VectorXf fitted;
 	Eigen::VectorXf res;
// 	float *Y;
// 	float **X;
//	float **Fisher;
	//float *B;
	bool intercept;
// 	float *fitted;
// 	float *res;
	int ynrow = 0;
	int yncol = 0;
	int xnrow = 0;
	int xncol = 0;
	float mse=0;
//	void initial(float **x, bool intercept);

	void initial(Eigen::MatrixXf &x);
	
};
