#pragma once
#include "pch.h"
#include <Eigen/Dense>
#define EIGEN_USE_MKL_ALL
enum MatrixDecompostionOptions :int
{
	Cholesky = 0,
	LU = 1,
	QR = 2,
	SVD = 3
};

struct MinqueOptions
{
	int iterate = 20;
	double tolerance = 1e-6;
	int MatrixDecomposition = Cholesky;
	int altMatrixDecomposition = SVD;
	bool allowPseudoInverse = true;
};


bool Inverse(Eigen::MatrixXd & Ori_Matrix, Eigen::MatrixXd & Inv_Matrix,int DecompositionMode, int AltDecompositionMode, bool allowPseudoInverse);
double Variance(Eigen::VectorXd &Y);
double mean(Eigen::VectorXd &Y);
double isNum(std::string line);
