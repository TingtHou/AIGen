#pragma once
#include <Eigen/Dense>
#include <vector>
#include "cuMatrix.h"
class MINQUE
{
public:
	void importY(Eigen::VectorXd Y);
	void Vi_pushback(Eigen::MatrixXd Ui);
	void setGPU(bool isGPU);
	void start();
	~MINQUE();
private:
	int nind = 0;
	int nVi = 0;
	bool GPU = true;
	Eigen::VectorXd Y;
	std::vector<Eigen::MatrixXd> V;
	Eigen::MatrixXd Vsum_INV;
	Eigen::MatrixXd Gamma;
	Eigen::MatrixXd Gamma_INV;
	Eigen::VectorXd Eta;
	Eigen::VectorXd theta;
	Eigen::MatrixXd Ai;
	void Cal_Eta(int i);
	void Calc_Vsum_Inv();
	void Calc_Gamma();
	void Calc_Ai(int i);
	
};

