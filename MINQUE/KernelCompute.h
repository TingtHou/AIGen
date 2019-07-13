#pragma once
#include <Eigen/dense>
#include "CommonFunc.h"
#include <vector>
#include "PlinkReader.h"
#include "CommonFunc.h"
#include "ToolKit.h"
#include "KernelManage.h"
class KernelCompute
{
public:
	KernelCompute(GenoData & gd, int KernelName, Eigen::VectorXd weights, double constant = 1, double deg = 2, double sigmma = 1);
	KernelCompute();
	void BuildBin(std::string prefix);
	~KernelCompute();
	void test();
private:
	KernelData kernels;
	void getCAR(Eigen::MatrixXd &Geno, Eigen::VectorXd &weights, Eigen::MatrixXd &kernel);
	void getIdentity(Eigen::MatrixXd &Geno, Eigen::MatrixXd &kernel);
	void getProduct(Eigen::MatrixXd &Geno, Eigen::VectorXd &weights, Eigen::MatrixXd &kernel);
	void getPolynomial(Eigen::MatrixXd &Geno, Eigen::VectorXd &weights, double constant, double deg, Eigen::MatrixXd & kernel);
	void getGaussian(Eigen::MatrixXd &Geno, Eigen::VectorXd &weights, double sigmma, Eigen::MatrixXd & kernel);
	void getIBS(Eigen::MatrixXd & Geno, Eigen::VectorXd & weights, Eigen::MatrixXd & kernel);
};
