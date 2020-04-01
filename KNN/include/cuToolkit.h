#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "cuMatrix.cuh"
//void InverseTest();
int cuInverse(float* Ori_Matrix, int N, int DecompositionMode, int AltDecompositionMode, bool allowPseudoInverse);
class cuToolkit
{
public:
	static bool cuLU(float* d_A, int N);
	static bool cuSVD(float* d_A, int N);
	static bool cuQR(float* d_A, int N);
	static bool cuCholesky(float* d_A, int N);
//	static void cuGetGPUinfo();
};

