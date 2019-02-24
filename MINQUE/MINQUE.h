#pragma once
#include <Eigen/Dense>
#include <vector>
#define EIGEN_USE_MKL_ALL
class MINQUE
{
public:
	void importY(Eigen::VectorXd Y);
	void pushback_Vi(Eigen::MatrixXd Ui);
	void estimate();
	Eigen::VectorXd Gettheta();
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

