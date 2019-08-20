#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>

//calculate matrix trace
//A is a pointer of input matrix, which is colmun-major
//nrow is row number of matrix A
//Result is a return value
__global__ void cuMatrixTrace_Kernel(const double *A, const int nrows, double *Result)
{
	int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
	(*Result) = 0;
	if (thread_idx==0)
	{
		for (int id = 0; id < nrows; id++)
		{
			(*Result) += A[id*nrows + id];
		}
	}
}

//inverse diagonal matrix
//A is a pointer of input matrix, which is colmun-major
//nrow is row number of matrix A
//Result is a return value
__global__ void cuDiagMatrixinv_kernel(double *A, double *A_inv,const int nrows)
{
	int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (thread_idx == 0)
	{
		double  pinvtoler = 1.e-20; // choose your tolerance wisely
		for (int id = 0; id < nrows; id++)
		{
			if (!A[id])
			{
				A_inv[id*nrows + id] = 0;
			}		
			else
			{
				A_inv[id*nrows + id] = 1 / A[id];
			}
		}
	}
}

//call function for cuMatrixTrace_Kernel function
extern "C" void cuMatrixTrace(const double *A, const int nrows, double *Result)
{
	int nBlocks = 1;
	int nThreads = 1;
	cuMatrixTrace_Kernel <<<nBlocks, nThreads>>>(A, nrows, Result);
}

extern "C" void cuDiagMatrixinv(double *A, double *A_inv, const int nrows)
{
	int nBlocks = 1;
	int nThreads = 1;
	cuDiagMatrixinv_kernel << <nBlocks, nThreads >> > (A, A_inv, nrows);
}
