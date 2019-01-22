#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <malloc.h>
#include <cusolverDn.h>
void cuMatrixMult(double** A, double** B, double** C, int m, int n, int s);
void cuMatrixMult(double* A, double* B, double* C, int m, int n, int s);
void cuMatrixInv(double *A, double *A_INV, int N);