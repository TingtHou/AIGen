#pragma once
#define EIGEN_USE_MKL_ALL
#include "MinqueBase.h"
#include "CommonFunc.h"
#include "LinearRegression.h"
#include "MINQUE1.h"
#include <sstream>
#include <iomanip>
#include <thread>
#include "easylogging++.h"

//Iterate MINQUE(1)
//MINQUE(1) when iterate time is set to 1
//usage:
//	imnq varest;
//	varest.setOptions(minque);
//	varest.isEcho(isecho);
//	varest.importY(phe.Phenotype);
//	varest.pushback_X(Covs, false);
//	varest.pushback_Vi(Kernels);
//	varest.estimate();
class imnq :
	public MinqueBase
{
public:
	void estimateVCs();										//starting estimation
	void setOptions(MinqueOptions mnqoptions);				//
	int getIterateTimes();
	void isEcho(bool isecho);
private:
	float tol = 1e-5; //convergence tolerence (def=1e-5)
	int itr = 20;	//iterations allowed (def=20)

	int initIterate = 0;
	bool isecho=true;
	bool MINQUE1 = false;
private:
	Eigen::VectorXf initVCS(); //initialize variance components
	void Iterate();
};

