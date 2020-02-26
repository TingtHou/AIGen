#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <mkl.h>
#include <chrono>
#include "helper_cuda.h"
#include "cuToolkit.h"
#include "ToolKit.h"
#include "cuMinqueBase.h"
class cuMINQUE1:
	public cuMinqueBase
{
public:
	cuMINQUE1(int DecompositionMode, int altDecompositionMode, bool allowPseudoInverse);
	~cuMINQUE1();
	void estimateVCs();
private:
};


