#pragma once
#define EIGEN_USE_MKL_ALL
#include <vector>
#include <Eigen/Dense>
#include <stdio.h>
#include <iostream>
#include <mkl.h>
class ToolKit
{
public:
	static void ArraytoVector(float ** a, int n, int m, std::vector< std::vector<float>>& v, bool Transpose = false);
	static void Array2toArrat1(float **a, int n, int m, float *b, bool Colfirst=true);
	static void Vector2toArray1(std::vector< std::vector<float>>& v, float *b, bool Colfirst = true);
	static void Stringsplit(std::string &org, std::vector< std::string> & splited, std::string delim);
	static void dec2bin(int num, int *bin);
	static bool Inv_Cholesky(Eigen::MatrixXf & Ori_Matrix);
	static bool Inv_LU(Eigen::MatrixXf & Ori_Matrix, Eigen::MatrixXf & Inv_Matrix);
	static bool Inv_SVD(Eigen::MatrixXf & Ori_Matrix,  bool allowPseudoInverse);
	static bool Inv_QR(Eigen::MatrixXf & Ori_Matrix,  bool allowPseudoInverse);
	static bool comput_inverse_logdet_LDLT_mkl(Eigen::MatrixXf &Vi);
	static bool comput_inverse_logdet_LU_mkl(Eigen::MatrixXf &Vi);
	static bool comput_inverse_logdet_QR_mkl(Eigen::MatrixXf& Vi);
	static bool comput_inverse_logdet_SVD_mkl(Eigen::MatrixXf& Vi);
	static bool comput_inverse_logdet_LDLT_mkl(Eigen::MatrixXd& Vi);
	static bool comput_inverse_logdet_LU_mkl(Eigen::MatrixXd& Vi);
	static bool comput_inverse_logdet_QR_mkl(Eigen::MatrixXd& Vi);
	static bool comput_inverse_logdet_SVD_mkl(Eigen::MatrixXd& Vi);
};

