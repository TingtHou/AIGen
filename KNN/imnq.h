#pragma once
#define EIGEN_USE_MKL_ALL
#include "MinqueBase.h"
#include "CommonFunc.h"
#include "LinearRegression.h"
#include "rln_mnq.h"
#include <sstream>
#include <iomanip>
#include <thread>
#include "logger.h"

class imnq :
	public MinqueBase
{
public:
	void estimate();
	void setOptions(MinqueOptions mnqoptions);
	int getIterateTimes();
	void isEcho(bool isecho);
private:
	double tol = 1e-5; //convergence tolerence (def=1e-5)
	int itr = 20;	//iterations allowed (def=20)
	int Decomposition = Cholesky;
	int altDecomposition = QR;
	bool allowpseudoinverse = true;
	int initIterate = 0;
	bool isecho=true;
private:
	Eigen::VectorXd initVCS(); //initialize variance components
	void Iterate();

};

