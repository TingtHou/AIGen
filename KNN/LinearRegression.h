#pragma once
#include <Eigen/Dense>
#include <iostream>
class LinearRegression
{
public:
	~LinearRegression();
//	LinearRegression(double ** y, double ** x, bool intercept, int yrows, int xrows, int xcols);
//	LinearRegression(double * y, double ** x, bool intercept, int yrows, int xrows, int xcols);
	LinearRegression(Eigen::VectorXd &Y, Eigen::MatrixXd &X, bool intercept);
	Eigen::VectorXd GetB();
	void MLE();
	double GetMSE();
private:
 	Eigen::VectorXd Matrix_Y;
// 	Eigen::VectorXd P;
 	Eigen::MatrixXd X;
	Eigen::MatrixXd Matrix_X;
// 	Eigen::MatrixXd Fisher;
// 	Eigen::VectorXd T_statistic;
 	Eigen::VectorXd B;
 	Eigen::VectorXd fitted;
 	Eigen::VectorXd res;
// 	double *Y;
// 	double **X;
//	double **Fisher;
	//double *B;
	bool intercept;
// 	double *fitted;
// 	double *res;
	int ynrow = 0;
	int yncol = 0;
	int xnrow = 0;
	int xncol = 0;
	double mse=0;
//	void initial(double **x, bool intercept);

	void initial(Eigen::MatrixXd &x);
	
};
