#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <malloc.h>
#include <cusolverDn.h>
#include <iostream>
#include "cuMatrix.cuh"
#include <Eigen/Dense>
void InverseTest();
class cuToolkit
{
public:
	static bool cuLU(float* d_A, float* d_A_INV, int N);
	static bool cuSVD(double * d_A, double *d_A_INV, int N);
	static bool cuQR(double * d_A, double *d_A_INV, int N);
	static bool cuCholesky(float* d_A, int N);
	static void cuGetGPUinfo();
};

