#include "pch.h"
#include "LinearRegression.h"

LinearRegression::~LinearRegression()
{
}

// LinearRegression::LinearRegression(double ** y, double ** x, bool intercept)
// {
// 	ynrow = sizeof(y) / sizeof(y[0]);
// 	yncol = sizeof(y[0]) / sizeof(y[0][0]);
// 	xnrow = sizeof(x) / sizeof(x[0]);
// 	xncol = sizeof(x[0]) / sizeof(x[0][0]);
// 	Y = new double[ynrow];
// 	for (int i = 0; i < ynrow; i++) {
// 		Y[i] = y[i][0];
// 	}
// 	initial(x, intercept);
// }

// LinearRegression::LinearRegression(double * y, double ** x, bool intercept, int yrows,int xrows,int xcols)
// {
// 	ynrow = yrows;
// 	xnrow = xrows;
// 	xncol = xcols;
// 	Y = new double[ynrow];
// 	for (int i = 0; i < ynrow; i++) {
// 		Y[i] = y[i];
// 	}
// 	initial(x, intercept);
// }

LinearRegression::LinearRegression(Eigen::VectorXd &Y, Eigen::MatrixXd &X, bool intercept)
{
	Matrix_Y = Y;
	this->intercept = intercept;
	ynrow = Y.size();
	xnrow = X.rows();
	xncol = X.cols();
	initial(X);
// 	this->Y = new double[ynrow];
// 	for (int i = 0; i < ynrow; i++) {
// 		this->Y[i] = Y[i];
// 	}
// 	double **x;
// 	x = (double**)calloc(xnrow, sizeof(double*));
// 	for (int i=0;i<xnrow;i++)
// 	{
// 		x[i] = (double*)calloc(xncol, sizeof(double));
// 		for (int j=0;j<xncol;j++)
// 		{
// 			x[i][j] = X(i, j);
// 		}
// 	}
// 	initial(x, intercept);
// 	for (int i = 0; i < xnrow; i++)
// 	{
// 		delete x[i];
// 	}
// 	delete x;
}

Eigen::VectorXd LinearRegression::GetB()
{
	return B;
}
// 
// void LinearRegression::initial(double ** x, bool intercept)
// {
// 	this->intercept = intercept;
// 	int u = intercept ? 1 : 0;
// 	X = (double**)calloc(xnrow, sizeof(double*));
// 	for (int i = 0; i < xnrow; i++) {
// 		X[i] = new double[xncol + u];
// 		if (intercept) {
// 			X[i][0] = u;
// 		}
// 	for (int j=0;j<xncol;j++)
// 	{
// 		X[i][j + u] = x[i][j];
// 	}
// 	}
// 	B = new double[xncol + u];
// }

void LinearRegression::initial(Eigen::MatrixXd & x)
{
	int u = intercept ? 1 : 0;
	Matrix_X.resize(xnrow, xncol + u);
	if (intercept)
	{
		Eigen::MatrixXd ones(xnrow, 1);
		ones.setOnes();
		Matrix_X << ones, x;
	}
	else
	{
		Matrix_X << x;
	}
	B.resize(xncol + u);
}


void LinearRegression::MLE()
{
	using namespace Eigen;
// 	VectorXd Matrix_Y(ynrow);
// 	MatrixXd Matrix_X(xnrow,xncol+(intercept?1:0));
// 	for (int i = 0; i < Matrix_X.rows(); i++)
// 	{
// 		for (int j = 0; j < Matrix_X.cols(); j++)
// 		{
// 			Matrix_X(i, j) = X[i][j];
// 		}
// 		Matrix_Y[i] = Y[i];
// 	}
	MatrixXd Matrix_XT = Matrix_X.transpose();
	MatrixXd Matrix_XT_X = Matrix_XT * Matrix_X;
	//JacobiSVD<MatrixXd> svd(Matrix_XT_X, ComputeThinU | ComputeThinV);
	MatrixXd Matrix_XT_X_Ivt = Matrix_XT_X.inverse();
	MatrixXd Matrix_XT_X_Ivt_XT = Matrix_XT_X_Ivt * Matrix_XT;
	B = Matrix_XT_X_Ivt_XT * Matrix_Y;
	fitted= Matrix_X * B;
	res = Matrix_Y - fitted;
	for (int i=0;i<ynrow;i++)
	{
		mse += res(i)*res(i);
	}
	mse /= Matrix_Y.size() - Matrix_X.cols();
/*
	MatrixXd Matrix_Fisher = Matrix_XT_X_Ivt*mse;
	int Matrix_Fisher_rows = Matrix_Fisher.rows();
	Fisher = (double**)malloc(sizeof(double*)*Matrix_Fisher_rows);
	for (int i=0;i<Matrix_Fisher_rows;i++)
	{
		Fisher[i] = new double[Matrix_Fisher_rows];
	}
	for (int i=0;i<Matrix_Fisher_rows;i++)
	{
		for (int j=0;j<Matrix_Fisher_rows;j++)
		{
			Fisher[i][j] = Matrix_Fisher(i, j);
		}
	}*/
}

double LinearRegression::GetMSE()
{
	return mse;
}

// void main() 
// {
// 	Eigen::VectorXd Y(20);
// 	Eigen::MatrixXd X = Eigen::MatrixXd::Random(20, 3);
// 	X.col(0).setOnes();
// 	Eigen::VectorXd a(3);
// 	a << 1, 0.5, 1;
// 	Eigen::MatrixXd Xi(20, 2);
// 	Xi << X.col(1), X.col(2);
// 	std::cout << "X: \n" << X << std::endl;
// 	std::cout << "Theta: " << a.transpose() << std::endl;
// 	Y = X * a;
// 	LinearRegression LR(Y, Xi, true);
// 	LR.MLE();
// 	Eigen::VectorXd ahat = LR.GetB();
// 	std::cout << ahat << std::endl;
// 
// }

