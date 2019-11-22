#pragma once
#define EIGEN_USE_MKL_ALL
#include <map>
#include <string>
#include <boost/bimap.hpp>
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include "ToolKit.h"


enum MatrixDecompostionOptions :int
{
	Cholesky = 0,
	LU = 1,
	QR = 2,
	SVD = 3
};


enum KernelNames :int
{
	CAR = 0,
	Identity = 1,
	Product = 2,
	Polymonial = 3,
	Gaussian = 4,
	IBS = 5
};

struct MinqueOptions
{
	int iterate = 200;
	double tolerance = 1e-6;
	int MatrixDecomposition = 0;
	int altMatrixDecomposition = 3;
	bool allowPseudoInverse = true;
};

struct KernelData
{
	boost::bimap<int, std::string> fid_iid;
	Eigen::MatrixXd kernelMatrix;
	Eigen::MatrixXd VariantCountMatrix;
};

struct PhenoData
{
	boost::bimap<int, std::string> fid_iid;
	Eigen::VectorXd Phenotype;
};

struct GenoData
{
	std::map<int, std::string> fid_iid;
	Eigen::MatrixXd Geno;               //individual mode, row: individual; col: SNP;

};


int Inverse(Eigen::MatrixXd & Ori_Matrix,int DecompositionMode, int AltDecompositionMode, bool allowPseudoInverse);
double Variance(Eigen::VectorXd &Y);
double mean(Eigen::VectorXd &Y);
double isNum(std::string line);
std::string GetBaseName(std::string pathname);
std::string GetParentPath(std::string pathname);
void stripSameCol(Eigen::MatrixXd &Geno);
void stdSNPmv(Eigen::MatrixXd &Geno);
void set_difference(boost::bimap<int, std::string> &map1, boost::bimap<int, std::string> &map2, std::vector<std::string> &overlap);
void GetSubMatrix(Eigen::MatrixXd &oMatrix, Eigen::MatrixXd &subMatrix, std::vector<int> rowIds, std::vector<int> colIDs);
void GetSubVector(Eigen::VectorXd &oVector, Eigen::VectorXd &subVector, std::vector<int> IDs);

