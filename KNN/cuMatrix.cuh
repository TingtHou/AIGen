#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
__global__ void cuElementInverse_kernel(double *A, double *A_inv, const int nrows);
//__global__ double cuMatrixCompare_kernel(double *A, double *B, double *residual);
void cuElementInverse(double *A, double *A_inv, const int nrows); //api
//bool cuMatrixCompare(double *A, double *B, int nrow, int row);
__global__ void cuGetupperTriangular_kernel(double *A, int nrows);
void cuGetupperTriangular(double * A, double *A_Tri, int nrows); //api