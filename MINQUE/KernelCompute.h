#pragma once
#include <Eigen/dense>
#include "CommonFunc.h"
class KernelCompute
{
public:
	KernelCompute(Eigen::MatrixXd &Geno, std::string KernelName, double maf = 0, double weights = 1, double constant = 1, double deg = 2, double sigmma = 1);
	KernelCompute();
	~KernelCompute();
	void test();
private:
	KernelData kernels;
	void getCAR(Eigen::MatrixXd &Geno, double maf, double weights);
	void getIdentity(Eigen::MatrixXd &Geno);
	void getProduct(Eigen::MatrixXd &Geno, double weights);
	void getPolynomial(Eigen::MatrixXd &Geno, double weights, double constant, double deg);
	void getGaussian(Eigen::MatrixXd &Geno, double weights, double sigmma);
	void getIBS(Eigen::MatrixXd &Geno, double weights);
	void stripSameCol(Eigen::MatrixXd &Geno);
	void stdSNPmv(Eigen::MatrixXd &Geno);
}