#pragma once
#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#include <iostream>
class LinearRegression
{
public:
	~LinearRegression();
	LinearRegression(Eigen::VectorXf &Y, Eigen::MatrixXf &X, bool intercept);
	Eigen::VectorXf GetB();
	void MLE();
	float GetMSE();
private:
 	Eigen::VectorXf Matrix_Y;
 	Eigen::MatrixXf X;
	Eigen::MatrixXf Matrix_X;
 	Eigen::VectorXf B;
 	Eigen::VectorXf fitted;
 	Eigen::VectorXf res;
	bool intercept;

	int ynrow = 0;
	int yncol = 0;
	int xnrow = 0;
	int xncol = 0;
	float mse=0;
	void initial(Eigen::MatrixXf &x);
	
};
