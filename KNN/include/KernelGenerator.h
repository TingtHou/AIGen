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
	KernelGenerator(GenoData & gd, int KernelName, Eigen::VectorXf &weights, float scale, float constant = 1, float deg = 2, float sigmma = 1);
	KernelGenerator();
	void BuildBin(std::string prefix);
	KernelData getKernel() { return kernels; };
	~KernelGenerator();
	void test();
private:
	bool scale;
	KernelData kernels;
	void getCAR(Eigen::MatrixXf &Geno, Eigen::VectorXf &weights, Eigen::MatrixXf &kernel);
	void getIdentity(Eigen::MatrixXf &Geno, Eigen::MatrixXf &kernel);
	void getProduct(Eigen::MatrixXf &Geno, Eigen::VectorXf &weights, Eigen::MatrixXf &kernel);
	void getPolynomial(Eigen::MatrixXf &Geno, Eigen::VectorXf &weights, float constant, float deg, Eigen::MatrixXf & kernel);
	void getGaussian(Eigen::MatrixXf &Geno, Eigen::VectorXf &weights, float sigmma, Eigen::MatrixXf & kernel);
	void getIBS(Eigen::MatrixXf & Geno, Eigen::VectorXf & weights, Eigen::MatrixXf & kernel);
};
