#include "../include/cuMatrix.cuh"

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
__global__ void cuElementInverse_kernel(float*A, float*A_inv,const int nrows)
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



void cuElementInverse(float*A, float*A_inv, const int nrows)
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

__global__ void cuVectorCwiseProduct_kernel(float* A, float* B, float* C, int size)
{
	int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (thread_idx >= size)
	{
		return;
	}
	C[thread_idx] = A[thread_idx] * B[thread_idx];
}

void cuVectorCwiseProduct(float* A, float* B, float* C, int size)
{
	int nthread = size < 1024 ? size : 1024;
	int nblock = 1 + (size / nthread);
	dim3 nThread(nthread, 1, 1);
	dim3 nBlock(nblock, 1, 1);
	cuVectorCwiseProduct_kernel << <nBlock, nThread >> > (A, B, C , size);
	return;
}

__global__ void cuVectorCwiseMinus_kernel(float* A, float* B, float* C, int size)
{
	int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (thread_idx >= size)
	{
		return;
	}
	C[thread_idx] = A[thread_idx] - B[thread_idx];
}

void cuVectorCwiseMinus(float* A, float* B, float* C, int size)
{
	int nthread = size < 1024 ? size : 1024;
	int nblock = 1 + (size / nthread);
	dim3 nThread(nthread, 1, 1);
	dim3 nBlock(nblock, 1, 1);
	cuVectorCwiseMinus_kernel << <nBlock, nThread >> > (A, B, C, size);
	return;
}

__global__ void cuTranspose_kernel(float* A, float* At, int width, int height)
{
	int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int row = thread_idx / height;
	const int column = thread_idx % height;
	//printf("[%d,%d],threadId %d threadIdx.x %d blockIdx.x %d blockDim.x %d \n", 
	//	row,column, thread_idx, (int)threadIdx.x, (int)blockIdx.x, (int)blockDim.x);
	if (column > width || row > height)
	{
		return;
	}
	At[row + height * column] = A[row + height * column];
	if (column <= row)
	{
		return;
	}
	float tmp = At[row + height * column];
	At[row + height * column] = At[column + height * row];
	At[column + height * row] = tmp;
	return ;
}

void cuTranspose(float* A, float* At, int width, int height)
{
	int size=  width * height;
	int nthread_x = size < 1024 ? size : 1024;
	int nblock_x = (size + nthread_x - 1) / nthread_x;
	dim3 nThread(nthread_x, 1, 1);
	dim3 nBlock(nblock_x, 1, 1);
	cuTranspose_kernel << <nBlock, nThread >> > (A, At,  width,  height);
	return;
}



__global__ void cuFillValue_kernel(float* x, int size, float value)
{
	int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (thread_idx<size)
	{
		x[thread_idx] = value;
	}
	return;
}

void cuFillValue(float* x, int size, float value)
{
	int nthread = size < 1024 ? size : 1024;
	int nblock = 1 + (size / nthread);
	dim3 nThread(nthread, 1, 1);
	dim3 nBlock(nblock, 1, 1);
	cuFillValue_kernel << <nBlock, nThread >> > (x,size,value);
	return;
}

__global__ void cuFillValue_kernel(double* x, int size, double value)
{
	int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (thread_idx < size)
	{
		x[thread_idx] = value;
	}
	return;
}

void cuFillValue(double* x, int size, double value)
{
	int nthread = size < 1024 ? size : 1024;
	int nblock = 1 + (size / nthread);
	dim3 nThread(nthread, 1, 1);
	dim3 nBlock(nblock, 1, 1);
	cuFillValue_kernel << <nBlock, nThread >> > (x, size, value);
	return;
}

__global__ void cuFloat2Double_kernel(float* idata, double* odata, int size)
{
	int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (thread_idx < size)
	{
		odata[thread_idx] = idata[thread_idx];
	}
	return ;
}

void cuFloat2Double(float* idata, double* odata, int size)
{
	int nthread = size < 1024 ? size : 1024;
	int nblock = 1 + (size / nthread);
	dim3 nThread(nthread, 1, 1);
	dim3 nBlock(nblock, 1, 1);
	cuFloat2Double_kernel << <nBlock, nThread >> > (idata, odata, size);
	return;
}

__global__ void cuAsDiag_kernel(float* idata, int Dim)
{
	int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int row = thread_idx / Dim;
	const int column = thread_idx % Dim;
	if (column > Dim || row > Dim)
	{
		return;
	}
	if (column == row)
	{
		idata[thread_idx] = 1;
	}
	else
	{
		idata[thread_idx] = 0;
	}
	return;
}

void cuAsDiag(float* idata, int Dim)
{
	int nthread = Dim* Dim < 1024 ? Dim * Dim : 1024;
	int nblock = 1 + (Dim * Dim / nthread);
	dim3 nThread(nthread, 1, 1);
	dim3 nBlock(nblock, 1, 1);
	cuAsDiag_kernel << <nBlock, nThread >> > (idata, Dim);
	return;
}