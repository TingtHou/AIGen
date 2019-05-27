#include "pch.h"
#include "LinearRegression.h"

LinearRegression::~LinearRegression()
{
	delete Y;
	delete P;
	delete T_statistic;
	delete B;
	delete fitted;
	delete res;
	int nrow = sizeof(X) / sizeof(X[0]);
	int ncol = sizeof(X[0]) / sizeof(X[0][0]);
	for (int i = 0; i<nrow; i++)
	{
		delete X[i];
		delete Fisher[i];
	}
	delete X;
	delete Fisher;
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

LinearRegression::LinearRegression(double * y, double ** x, bool intercept, int yrows,int xrows,int xcols)
{
	ynrow = yrows;
	xnrow = xrows;
	xncol = xcols;
	Y = new double[ynrow];
	for (int i = 0; i < ynrow; i++) {
		Y[i] = y[i];
	}
	initial(x, intercept);
}

LinearRegression::LinearRegression(Eigen::VectorXd &Y, Eigen::MatrixXd &X, bool intercept)
{
	ynrow = Y.size();
	xnrow = X.rows();
	xncol = X.cols();
	this->Y = new double[ynrow];
	for (int i = 0; i < ynrow; i++) {
		this->Y[i] = Y[i];
	}
	double **x;
	x = (double**)malloc(sizeof(double*)*xnrow);
	for (int i=0;i<xnrow;i++)
	{
		x[i] = (double*)malloc(sizeof(double)*xncol);
		for (int j=0;j<xncol;j++)
		{
			x[i][j] = X(i, j);
		}
	}
	initial(x, intercept);
}

double* LinearRegression::GetB()
{
	return B;
}

void LinearRegression::initial(double ** x, bool intercept)
{
	P = new double[ynrow];
	this->intercept = intercept;
	int u = intercept ? 1 : 0;
	X = new double*[xnrow];
	for (int i = 0; i < xnrow; i++) {
		X[i] = new double[xncol + u];
		if (intercept) {
			X[i][0] = u;
		}
	for (int j=0;j<xncol;j++)
	{
		X[i][j + u] = x[i][j];
	}
	}
	B = new double[xncol + u];
}


void LinearRegression::MLE()
{
	using namespace Eigen;
	VectorXd Matrix_Y(ynrow);
	MatrixXd Matrix_X(xnrow,xncol+(intercept?1:0));
	for (int i = 0; i < Matrix_X.rows(); i++)
	{
		for (int j = 0; j < Matrix_X.cols(); j++)
		{
			Matrix_X(i, j) = X[i][j];
		}
		Matrix_Y[i] = Y[i];
	}
	MatrixXd Matrix_XT = Matrix_X.transpose();
	MatrixXd Matrix_XT_X = Matrix_XT * Matrix_X;
	//JacobiSVD<MatrixXd> svd(Matrix_XT_X, ComputeThinU | ComputeThinV);
	MatrixXd Matrix_XT_X_Ivt = Matrix_XT_X.inverse();
	MatrixXd Matrix_XT_X_Ivt_XT = Matrix_XT_X_Ivt * Matrix_XT;
	VectorXd b = Matrix_XT_X_Ivt_XT * Matrix_Y;
	VectorXd Fitted = Matrix_X * b;
	VectorXd Res = Matrix_Y - Fitted;
	double mse = 0;
	for (int i=0;i<ynrow;i++)
	{
		mse += Res(i)*Res(i);
	}
	mse /= Matrix_Y.size() - Matrix_X.cols();
	MatrixXd Matrix_Fisher = Matrix_XT_X_Ivt*mse;
	int Matrix_Fisher_rows = Matrix_Fisher.rows();
	Fisher = new double *[Matrix_Fisher_rows];
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
	}
	res = new double[Res.rows()];
	fitted = new double[Fitted.rows()];
	for (int i = 0; i < ynrow; i++)
	{
		res[i] = Res[i];
		fitted[i] = Fitted[i];
	}
	for (int i=0;i<b.rows();i++)
	{
		B[i] = b[i];
	}
	
	
}

double LinearRegression::GetMSE()
{
	double mse = 0;
	for (int  i = 0; i < ynrow; i++)
	{
		mse = res[i] * res[i];
	}
	mse /=ynrow -xncol;
	return mse;
}

// void main() 
// {
// 	double *Y = new double[10];
// 	double X[10][3] = { { 1, 15, 4 },{ 1, 30, 14 },{ 1, 31, 16 },{ 1, 31, 11 },{ 1, 32, 17 },{ 1, 29, 10 },
// 	{ 1, 30, 8 },{ 1, 31, 12 },{ 1, 32, 6 },{ 1, 40, 7 } };
// 	double **x;
// 	x = new double *[10];
// 	for (int i=0;i<10;i++)
// 	{
// 		x[i] = new double[3];
// 		x[i] = X[i];
// 		Y[i] = 0.3*x[i][0] + 0.5*x[i][1] + 1 * x[i][2];
// 		
// 	}
// 	std::shared_ptr<double> r;
// 	{
// 		LinearRegression LR(Y, x, false, 10, 10, 3);
// 		LR.MLE();
// 		r = LR.GetB();
// 		std::cout << r.use_count() << std::endl;
// 	}
// 	std::cout << r.use_count() << std::endl;
// 	for (int i = 0; i < 3; i++) {
// 		std::cout << r.get()[i] << std::endl;
// 	}
// 	std::cout<< std::endl;
// }

