#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <malloc.h>
#include <cusolverDn.h>
#include <iostream>
#include "cuMatrix.cuh"
void InverseTest();
class cuToolkit
{
public:
	static bool cuLU(double * d_A, double *d_A_INV, int N);
	static bool cuSVD(double * d_A, double *d_A_INV, int N);
	static bool cuQR(double * d_A, double *d_A_INV, int N);
	static bool cuCholesky(double * d_A, double *d_A_INV, int N);
	static void cuGetGPUinfo();
};

