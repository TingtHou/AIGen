#pragma once
#define EIGEN_USE_MKL_ALL
#include "pch.h"
#include <map>
#include <string>
#include <boost/bimap.hpp>
#include <Eigen/Dense>
#include "ToolKit.h"
#include <boost/algorithm/string.hpp>


#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */


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

