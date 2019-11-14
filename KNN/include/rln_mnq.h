#pragma once
#define EIGEN_USE_MKL_ALL

#include "MinqueBase.h"
#include "CommonFunc.h"
#include "ToolKit.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include "easylogging++.h"
#include <mkl.h>
#include <thread>
class rln_mnq :
	public MinqueBase
{
public:
	rln_mnq(int DecompositionMode, int altDecompositionMode, bool allowPseudoInverse);
	void estimate();
	~rln_mnq();
private:
	int Decomp = 0;
	int altDecomp = 0;
	bool allowPseudoInverse = true;

	void CheckInverseStatus(int status);
	static void Calc(double* vi, double* rvi, double* inv_vw_mkl, double* p_mkl, int nind);
};

