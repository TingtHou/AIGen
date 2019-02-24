#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>


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


extern "C" void cuMatrixTrace(const double *A, const int nrows, double *Result)
{
	int nBlocks = 1;
	int nThreads = 1;
	cuMatrixTrace_Kernel <<<nBlocks, nThreads>>>(A, nrows, Result);
}
