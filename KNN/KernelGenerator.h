#pragma once
#include <Eigen/Dense>
#include "CommonFunc.h"
#include <vector>
#include "PlinkReader.h"
#include "CommonFunc.h"
#include "ToolKit.h"
#include "KernelManage.h"
#include <cmath>
#include <fstream>
class KernelGenerator
{
public:
	KernelGenerator(GenoData & gd, int KernelName, Eigen::VectorXd &weights, double scale, double constant = 1, double deg = 2, double sigmma = 1);
	KernelGenerator();
	void BuildBin(std::string prefix);
	KernelData getKernel() { return kernels; };
	~KernelGenerator();
	void test();
private:
	bool scale;
	KernelData kernels;
	void getCAR(Eigen::MatrixXd &Geno, Eigen::VectorXd &weights, Eigen::MatrixXd &kernel);
	void getIdentity(Eigen::MatrixXd &Geno, Eigen::MatrixXd &kernel);
	void getProduct(Eigen::MatrixXd &Geno, Eigen::VectorXd &weights, Eigen::MatrixXd &kernel);
	void getPolynomial(Eigen::MatrixXd &Geno, Eigen::VectorXd &weights, double constant, double deg, Eigen::MatrixXd & kernel);
	void getGaussian(Eigen::MatrixXd &Geno, Eigen::VectorXd &weights, double sigmma, Eigen::MatrixXd & kernel);
	void getIBS(Eigen::MatrixXd & Geno, Eigen::VectorXd & weights, Eigen::MatrixXd & kernel);
};
