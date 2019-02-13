#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <malloc.h>
#include <cusolverDn.h>
#include <iostream>
class cuToolkit
{
public:
	static int cuMatrixInv(double * d_A, double *d_A_INV, int N);
};

