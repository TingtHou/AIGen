#include "pch.h"
#include "cuMatrix.h"
#include <iostream>
#include <assert.h>
//C=A*B
//A is an Dim 1 array, whose size is M*N, means M rows and N colnums;
//B is an Dim 1 array, whose size is N*S, means N rows and S colnums;
//C is an Dim 1 array, whose size is M*S, means M rows and S colnums;
void cuMatrixMult(double** A, double** B, double** C, int m, int n, int s)
{
	unsigned int Size_A = m * n * sizeof(double);
	unsigned int Size_B = n * s * sizeof(double);
	unsigned int Size_C = m * s * sizeof(double);
	double *Matrix_A = (double *)malloc(Size_A);
	double *Matrix_B = (double *)malloc(Size_B);
	double *Matrix_C = (double *)malloc(Size_C);
	int id = 0;
	for (int j = 0; j < n; j++)
	{
		for (int i = 0; i < m; i++)
		{

			Matrix_A[id++] = A[i][j];
		}
	}
	id = 0;
	for (int j=0;j<s;j++)
	{
		for (int i = 0; i < n; i++)
		{

			Matrix_B[id++] = B[i][j];
		}
	}
	double *D_A, *D_B, *D_C;
	cudaMalloc((void**)&D_A, Size_A);
	cudaMalloc((void**)&D_B, Size_B);
	cudaMalloc((void**)&D_C, Size_C);
	cudaMemcpy(D_A, Matrix_A, Size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(D_B, Matrix_B, Size_B, cudaMemcpyHostToDevice);
	cublasHandle_t handle;
	const double alpha = 1.0f;
	const double beta = 0.0f;
	cublasCreate(&handle);
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, s, n, &alpha, D_A, m, D_B, n, &beta, D_C, m);
	cublasDestroy(handle);
	cudaMemcpy(Matrix_C, D_C, Size_C, cudaMemcpyDeviceToHost);
	id = 0;
	for (int j=0;j<s;j++)
	{
		for (int i = 0; i < m; i++)
		{

			C[i][j] = Matrix_C[id++];
		}
	}
	free(Matrix_A);
	free(Matrix_B);
	free(Matrix_C);
	cudaFree(D_A);
	cudaFree(D_B);
	cudaFree(D_C);
}

//C=A*B
//cublas using the colmun first
//eg if Matrix A=[1, 2, 3,
//				  4, 5, 6,
//				  7, 8, 9],
//then the Input array A=[1,4,7,2,5,8,3,6,9];
void cuMatrixMult(double* A, double* B, double* C, int m, int n, int s)
{
	unsigned int Size_A = m * n * sizeof(double);
	unsigned int Size_B = n * s * sizeof(double);
	unsigned int Size_C = m * s * sizeof(double);
	double *D_A, *D_B, *D_C;
	cublasHandle_t handle;
	cudaMalloc((void**)&D_A, Size_A);
	cudaMalloc((void**)&D_B, Size_B);
	cudaMalloc((void**)&D_C, Size_C);
	cudaMemcpy(D_A, A, Size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(D_B, B, Size_B, cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
	const double alpha = 1.0f;
	const double beta = 0.0f;
	cublasCreate(&handle);
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, s, n, &alpha, D_A, m, D_B, n, &beta, D_C, m);
	cudaThreadSynchronize();
	cudaMemcpy(C, D_C, Size_C, cudaMemcpyDeviceToHost);
	cublasDestroy(handle);
	cudaFree(D_A);
	cudaFree(D_B);
	cudaFree(D_C);
	cudaDeviceReset();
}

//A is a matrix with n rows and n cols
void cuMatrixInv(double *A, double *A_INV, int N)
{
	int sizeA = N * N * sizeof(double);
	cusolverDnHandle_t handle;	cusolverDnCreate(&handle);	double *d_A, *d_B, *d_Work;	int *dLUPivots, *dLUInfo, Lwork;	double *B = (double*)malloc(N*N * sizeof(double));	memset(B, 0, N*N * sizeof(double));	for (int i=0;i<N;i++)
	{
		B[(N + 1)*i] = 1;
	}	cudaMalloc((void **)&d_A, N*N * sizeof(double));	cudaMalloc((void **)&d_B, N*N * sizeof(double));	cudaMalloc((void **)& dLUPivots, N * sizeof(int));	cudaMalloc((void **)& dLUInfo, sizeof(int));	cudaMemcpy(d_A, A, N*N * sizeof(double), cudaMemcpyHostToDevice);	cudaMemcpy(d_B, B, N*N * sizeof(double), cudaMemcpyHostToDevice);	cusolverDnDgetrf_bufferSize(handle, N, N,d_A, N, &Lwork);	cudaMalloc((void **)& d_Work, Lwork * sizeof(double));	cusolverDnDgetrf(handle, N, N, d_A, N, d_Work, dLUPivots, dLUInfo);
	cudaDeviceSynchronize();
	int * test = (int*)malloc(N * sizeof(double));
	cudaMemcpy(test, dLUPivots, N * sizeof(double), cudaMemcpyDeviceToHost);
	cusolverDnDgetrs(handle, CUBLAS_OP_N, N, N,d_A, N, dLUPivots, d_B, N, dLUInfo);
	cudaDeviceSynchronize();
	cudaMemcpy(A_INV, d_B, sizeA, cudaMemcpyDeviceToHost);
	free(B);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(dLUInfo);
	cudaFree(dLUPivots);
	cudaFree(d_Work);
	cusolverDnDestroy(handle);
	cudaDeviceReset();
}