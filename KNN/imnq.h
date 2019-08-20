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
	int getIterateTimes();
private:
	double tol = 1e-5; //convergence tolerence (def=1e-5)
	int itr = 20;	//iterations allowed (def=20)
	int Decomposition = Cholesky;
	int altDecomposition = QR;
	bool allowpseudoinverse = true;
	int initIterate = 0;
private:
	Eigen::VectorXd initVCS(); //initialize variance components
	void Iterate();

};

