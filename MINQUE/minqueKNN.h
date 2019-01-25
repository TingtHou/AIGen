#pragma once
#include <Eigen/Dense>
#include <vector>
#include "cuMatrix.h"
class MINQUEKNN
{
public:
	void importY(Eigen::VectorXd Y);
	void importU_pushback(Eigen::MatrixXd Ui);
	void setGPU(bool isGPU);

	MINQUEKNN();
	~MINQUEKNN();
private:
	int nind = 0;
	int nKernel = 0;
	bool GPU = true;
	Eigen::VectorXd Y;
	std::vector<Eigen::MatrixXd> U;
	Eigen::MatrixXd Kernelsum_INV;
	Eigen::MatrixXd Gamma;
	Eigen::MatrixXd Gamma_INV;
	Eigen::VectorXd Eta;
	Eigen::VectorXd theta;
	Eigen::MatrixXd Ai;
	void Cal_Eta(int i);
	void Calc_KernelSum_Inv();
	void Calc_Gamma();
	void Calc_Ai(int i);
	void start();
};

