#include "cuMatrix.cuh"
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
__global__ void cuElementInverse_kernel(double *A, double *A_inv,const int nrows)
{
	int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
	double  pinvtoler = 1.e-20; // choose your tolerance wisely
	//printf("ID: %d\n",thread_idx);
	if (thread_idx>=nrows)
	{
		return;
	}
	if (A[thread_idx]<pinvtoler)
	{
		A_inv[thread_idx*nrows + thread_idx] = 0;
	}
	else
	{
		A_inv[thread_idx*nrows + thread_idx] = 1 / A[thread_idx];
	}
}



void cuElementInverse(double *A, double *A_inv, const int nrows)
{
	int nthread = nrows < 1024 ? nrows : 1024;
	int nblock = 1 + (nrows / nthread);
	dim3 nThread(nthread, 1, 1);
	dim3 nBlock(nblock, 1, 1);
	cuElementInverse_kernel <<<nBlock,nThread>>> (A, A_inv, nrows);
	return;
}

__global__ void cuGetupperTriangular_kernel(double * A, double *A_Tri, int nrows)
{
	int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
	A_Tri[thread_idx] = A[thread_idx];
	for (int i = 0; i < nrows; i++)
	{
		int start = i * nrows + i;
		int end = (i+1) * nrows;
		if (thread_idx>start&&thread_idx<end)
		{
			A_Tri[thread_idx] = 0;
		}
	}
	A_Tri[thread_idx] = fabs(A_Tri[thread_idx])<1e-10?0: A_Tri[thread_idx];
	return;
}

void cuGetupperTriangular(double * A, double *A_Tri, const int nrows)
{
	int totalThread = nrows * nrows;
	int nthread = totalThread < 1024 ? totalThread : 1024;
	int nblock = 1 + (totalThread / nthread);
	dim3 nThread(nthread, 1, 1);
	dim3 nBlock(nblock, 1, 1);
	cuGetupperTriangular_kernel <<<nBlock, nThread >> > (A, A_Tri,nrows);
	return;
}


