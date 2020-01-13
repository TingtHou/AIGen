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
#include "mkl_types.h"
#include "mkl_cblas.h"

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

};

