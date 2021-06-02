#pragma once
#define EIGEN_USE_MKL_ALL
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
	std::shared_ptr<KernelData> getKernel() { return kernels; };
	~KernelGenerator();
	void test();
private:
	bool scale;
	std::shared_ptr<KernelData> kernels;
	void getCAR(Eigen::MatrixXf &Geno, Eigen::VectorXf &weights, std::shared_ptr<KernelData> kernel);
	void getIdentity(Eigen::MatrixXf &Geno, std::shared_ptr<KernelData> kernel);
	void getProduct(Eigen::MatrixXf &Geno, Eigen::VectorXf &weights, std::shared_ptr<KernelData> kernel);
	void getPolynomial(Eigen::MatrixXf &Geno, Eigen::VectorXf &weights, float constant, float deg, std::shared_ptr<KernelData> kernel);
	void getGaussian(Eigen::MatrixXf &Geno, Eigen::VectorXf &weights, float sigmma, std::shared_ptr<KernelData> kernel);
	void getIBS(Eigen::MatrixXf & Geno, Eigen::VectorXf & weights, std::shared_ptr<KernelData> kernel);
};
