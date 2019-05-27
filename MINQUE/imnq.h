#pragma once
#include "MinqueBase.h"
#include "CommonFunc.h"
#define EIGEN_USE_MKL_ALL

class imnq :
	public MinqueBase
{
public:
	void estimate();
	void setOptions(MinqueOptions mnqoptions);
private:
	double tol = 1e-5; //convergence tolerence (def=1e-5)
	int itr = 20;	//iterations allowed (def=20)
	int Decomposition = Cholesky;
	int altDecomposition = QR;
	bool allowpseudoinverse = true;
private:
	Eigen::VectorXd initVCS(); //initialize variance components
	void Iterate();

};

