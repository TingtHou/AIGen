#pragma once
#include "pch.h"
#include "MinqueBase.h"
#include "CommonFunc.h"
#include "ToolKit.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#define EIGEN_USE_MKL_ALL
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
};

