#pragma once
#define EIGEN_USE_MKL_ALL
#include <vector>
#include <Eigen/Dense>
#include <stdio.h>
#include <iostream>
#include <mkl.h>
using namespace std;
class ToolKit
{
public:
	static void ArraytoVector(double ** a, int n, int m, vector<vector<double>>& v, bool Transpose = false);
	static void Array2toArrat1(double **a, int n, int m, double *b, bool Colfirst=true);
	static void Vector2toArray1(vector<vector<double>>& v, double *b, bool Colfirst = true);
	static void Stringsplit(string &org, vector<string> & splited, string delim);
	static void dec2bin(int num, int *bin);
	static bool Inv_Cholesky(Eigen::MatrixXd & Ori_Matrix);
	static bool Inv_LU(Eigen::MatrixXd & Ori_Matrix, Eigen::MatrixXd & Inv_Matrix);
	static bool Inv_SVD(Eigen::MatrixXd & Ori_Matrix,  bool allowPseudoInverse);
	static bool Inv_QR(Eigen::MatrixXd & Ori_Matrix,  bool allowPseudoInverse);
	static bool comput_inverse_logdet_LDLT_mkl(Eigen::MatrixXd &Vi);
	static bool comput_inverse_logdet_LU_mkl(Eigen::MatrixXd &Vi);
	static bool comput_inverse_logdet_QR_mkl(Eigen::MatrixXd& Vi);
	static bool comput_inverse_logdet_SVD_mkl(Eigen::MatrixXd& Vi);
};

