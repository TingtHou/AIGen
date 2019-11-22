#include "../include/LinearRegression.h"

LinearRegression::~LinearRegression()
{
}

// LinearRegression::LinearRegression(float ** y, float ** x, bool intercept)
// {
// 	ynrow = sizeof(y) / sizeof(y[0]);
// 	yncol = sizeof(y[0]) / sizeof(y[0][0]);
// 	xnrow = sizeof(x) / sizeof(x[0]);
// 	xncol = sizeof(x[0]) / sizeof(x[0][0]);
// 	Y = new float[ynrow];
// 	for (int i = 0; i < ynrow; i++) {
// 		Y[i] = y[i][0];
// 	}
// 	initial(x, intercept);
// }

// LinearRegression::LinearRegression(float * y, float ** x, bool intercept, int yrows,int xrows,int xcols)
// {
// 	ynrow = yrows;
// 	xnrow = xrows;
// 	xncol = xcols;
// 	Y = new float[ynrow];
// 	for (int i = 0; i < ynrow; i++) {
// 		Y[i] = y[i];
// 	}
// 	initial(x, intercept);
// }

LinearRegression::LinearRegression(Eigen::VectorXf &Y, Eigen::MatrixXf &X, bool intercept)
{
	Matrix_Y = Y;
	this->intercept = intercept;
	ynrow = Y.size();
	xnrow = X.rows();
	xncol = X.cols();
	initial(X);
// 	this->Y = new float[ynrow];
// 	for (int i = 0; i < ynrow; i++) {
// 		this->Y[i] = Y[i];
// 	}
// 	float **x;
// 	x = (float**)calloc(xnrow, sizeof(float*));
// 	for (int i=0;i<xnrow;i++)
// 	{
// 		x[i] = (float*)calloc(xncol, sizeof(float));
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

Eigen::VectorXf LinearRegression::GetB()
{
	return B;
}
// 
// void LinearRegression::initial(float ** x, bool intercept)
// {
// 	this->intercept = intercept;
// 	int u = intercept ? 1 : 0;
// 	X = (float**)calloc(xnrow, sizeof(float*));
// 	for (int i = 0; i < xnrow; i++) {
// 		X[i] = new float[xncol + u];
// 		if (intercept) {
// 			X[i][0] = u;
// 		}
// 	for (int j=0;j<xncol;j++)
// 	{
// 		X[i][j + u] = x[i][j];
// 	}
// 	}
// 	B = new float[xncol + u];
// }

void LinearRegression::initial(Eigen::MatrixXf & x)
{
	int u = intercept ? 1 : 0;
	Matrix_X.resize(xnrow, xncol + u);
	if (intercept)
	{
		Eigen::MatrixXf ones(xnrow, 1);
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
// 	VectorXf Matrix_Y(ynrow);
// 	MatrixXf Matrix_X(xnrow,xncol+(intercept?1:0));
// 	for (int i = 0; i < Matrix_X.rows(); i++)
// 	{
// 		for (int j = 0; j < Matrix_X.cols(); j++)
// 		{
// 			Matrix_X(i, j) = X[i][j];
// 		}
// 		Matrix_Y[i] = Y[i];
// 	}
	MatrixXf Matrix_XT = Matrix_X.transpose();
	MatrixXf Matrix_XT_X = Matrix_XT * Matrix_X;
	//JacobiSVD<MatrixXf> svd(Matrix_XT_X, ComputeThinU | ComputeThinV);
	MatrixXf Matrix_XT_X_Ivt = Matrix_XT_X.inverse();
	MatrixXf Matrix_XT_X_Ivt_XT = Matrix_XT_X_Ivt * Matrix_XT;
	B = Matrix_XT_X_Ivt_XT * Matrix_Y;
	fitted= Matrix_X * B;
	res = Matrix_Y - fitted;
	for (int i=0;i<ynrow;i++)
	{
		mse += res(i)*res(i);
	}
	mse /= Matrix_Y.size() - Matrix_X.cols();
/*
	MatrixXf Matrix_Fisher = Matrix_XT_X_Ivt*mse;
	int Matrix_Fisher_rows = Matrix_Fisher.rows();
	Fisher = (float**)malloc(sizeof(float*)*Matrix_Fisher_rows);
	for (int i=0;i<Matrix_Fisher_rows;i++)
	{
		Fisher[i] = new float[Matrix_Fisher_rows];
	}
	for (int i=0;i<Matrix_Fisher_rows;i++)
	{
		for (int j=0;j<Matrix_Fisher_rows;j++)
		{
			Fisher[i][j] = Matrix_Fisher(i, j);
		}
	}*/
}

float LinearRegression::GetMSE()
{
	return mse;
}

// void main() 
// {
// 	Eigen::VectorXf Y(20);
// 	Eigen::MatrixXf X = Eigen::MatrixXf::Random(20, 3);
// 	X.col(0).setOnes();
// 	Eigen::VectorXf a(3);
// 	a << 1, 0.5, 1;
// 	Eigen::MatrixXf Xi(20, 2);
// 	Xi << X.col(1), X.col(2);
// 	std::cout << "X: \n" << X << std::endl;
// 	std::cout << "Theta: " << a.transpose() << std::endl;
// 	Y = X * a;
// 	LinearRegression LR(Y, Xi, true);
// 	LR.MLE();
// 	Eigen::VectorXf ahat = LR.GetB();
// 	std::cout << ahat << std::endl;
// 
// }

