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

class minque1 :
	public MinqueBase
{
public:
	minque1(int DecompositionMode, int altDecompositionMode, bool allowPseudoInverse);
	void estimateVCs();
	Eigen::VectorXf estimateVCs_Null(std::vector<int> DropIndex);
	~minque1();
private:
};

