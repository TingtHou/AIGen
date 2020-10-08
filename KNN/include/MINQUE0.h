#pragma once
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
//#include <boost/multiprecision/mpfr.hpp>

class MINQUE0 :
	public MinqueBase
{
public:
	MINQUE0(int DecompositionMode, int altDecompositionMode, bool allowPseudoInverse);
	void estimateVCs();
	~MINQUE0();
private:
};

