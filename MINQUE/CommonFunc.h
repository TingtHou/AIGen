#pragma once
#include "pch.h"
#include <map>
#include <string>
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
	int iterate = 100;
	double tolerance = 1e-6;
	int MatrixDecomposition = 0;
	int altMatrixDecomposition = 3;
	bool allowPseudoInverse = true;
};

struct KernelData
{
	std::map<int, std::string> fid_iid;
	std::map<std::string, int> rfid_iid;
	Eigen::MatrixXd kernelMatrix;
	Eigen::MatrixXd VariantCountMatrix;
};

struct PhenoData
{
	std::map<int, std::string> fid_iid;
	Eigen::VectorXd Phenotype;
};

int Inverse(Eigen::MatrixXd & Ori_Matrix, Eigen::MatrixXd & Inv_Matrix,int DecompositionMode, int AltDecompositionMode, bool allowPseudoInverse);
double Variance(Eigen::VectorXd &Y);
double mean(Eigen::VectorXd &Y);
double isNum(std::string line);
std::string GetBaseName(std::string pathname);
std::string GetParentPath(std::string pathname);
