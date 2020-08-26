#pragma once
#include <cuda.h>
#include <device_launch_parameters.h>
__global__ void cuElementInverse_kernel(float*A, float*A_inv, const int nrows);
//__global__ double cuMatrixCompare_kernel(double *A, double *B, double *residual);
void cuElementInverse(float*A, float*A_inv, const int nrows); //api
//bool cuMatrixCompare(double *A, double *B, int nrow, int row);
//__global__ void cuGetupperTriangular_kernel(double *A, int nrows);
//void cuGetupperTriangular(double * A, double *A_Tri, int nrows); //api

__global__ void cuVectorCwiseProduct_kernel(float* A, float* B, float *C, int size);
//__global__ double cuMatrixCompare_kernel(double *A, double *B, double *residual);
void cuVectorCwiseProduct(float* A, float* B, float* C, int size); //api

__global__ void cuVectorCwiseMinus_kernel(float* A, float* B, float* C, int size);
//__global__ double cuMatrixCompare_kernel(double *A, double *B, double *residual);
void cuVectorCwiseMinus(float* A, float* B, float* C, int size); //api

__global__ void cuTranspose_kernel(float* A, float* At, int width, int height);
void cuTranspose(float* A, float* At, int width, int height);


__global__ void cuFillValue_kernel(float *x, int size, float value);
void  cuFillValue(float* x, int size, float value);

__global__ void cuFillValue_kernel(double* x, int size, double value);
void  cuFillValue(double* x, int size, double value);

__global__ void cuFloat2Double_kernel(float* idata, double* odata, int size);
void cuFloat2Double(float* idata, double* odata, int size);

__global__ void cuAsDiag_kernel(float* idata, int Dim);
void cuAsDiag(float* idata, int Dim);