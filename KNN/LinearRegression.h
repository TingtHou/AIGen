#include <Eigen/Dense>
#include <iostream>
class LinearRegression
{
public:
	~LinearRegression();
//	LinearRegression(double ** y, double ** x, bool intercept, int yrows, int xrows, int xcols);
	LinearRegression(double * y, double ** x, bool intercept, int yrows, int xrows, int xcols);
	LinearRegression(Eigen::VectorXd &Y, Eigen::MatrixXd &X, bool intercept);
	double* GetB();
	void MLE();
	double GetMSE();
private:
// 	Eigen::VectorXd Y;
// 	Eigen::VectorXd P;
// 	Eigen::MatrixXd X;
// 	Eigen::MatrixXd Fisher;
// 	Eigen::VectorXd T_statistic;
// 	Eigen::VectorXd B;
// 	Eigen::VectorXd fitted;
// 	Eigen::VectorXd res;
	double *Y;
	double *P;
	double **X;
	double **Fisher;
	double *T_statistic;
	double *B;
	bool intercept;
	double *fitted;
	double *res;
	int ynrow = 0;
	int yncol = 0;
	int xnrow = 0;
	int xncol = 0;
	void initial(double **x, bool intercept);


	
};
