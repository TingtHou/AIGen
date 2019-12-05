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
	float tolerance = 1e-6;
	int MatrixDecomposition = 0;
	int altMatrixDecomposition = 3;
	bool allowPseudoInverse = true;
};

struct KernelData
{
	boost::bimap<int, std::string> fid_iid;
	Eigen::MatrixXf kernelMatrix;
	Eigen::MatrixXf VariantCountMatrix;
};

struct PhenoData
{
	boost::bimap<int, std::string> fid_iid;
	Eigen::VectorXf Phenotype;
};

struct GenoData
{
	std::map<int, std::string> fid_iid;
	Eigen::MatrixXf Geno;               //individual mode, row: individual; col: SNP;

};


int Inverse(Eigen::MatrixXf & Ori_Matrix,int DecompositionMode, int AltDecompositionMode, bool allowPseudoInverse);
int Inverse(Eigen::MatrixXd & Ori_Matrix, int DecompositionMode, int AltDecompositionMode, bool allowPseudoInverse);
float Variance(Eigen::VectorXf &Y);
float mean(Eigen::VectorXf &Y);
float isNum(std::string line);
std::string GetBaseName(std::string pathname);
std::string GetParentPath(std::string pathname);
void stripSameCol(Eigen::MatrixXf &Geno);
void stdSNPmv(Eigen::MatrixXf &Geno);
void set_difference(boost::bimap<int, std::string> &map1, boost::bimap<int, std::string> &map2, std::vector<std::string> &overlap);
void GetSubMatrix(Eigen::MatrixXf &oMatrix, Eigen::MatrixXf &subMatrix, std::vector<int> rowIds, std::vector<int> colIDs);
void GetSubVector(Eigen::VectorXf &oVector, Eigen::VectorXf &subVector, std::vector<int> IDs);

